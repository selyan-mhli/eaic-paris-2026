"""Copie de distill.py (Yihuan, unlearning/distill.py, non modifié) avec une cible mélangée.

Ajouts c3 : --mix a → cible = a·consensus + (1−a)·p_clean (T=2), renormalisée, pour viser une
distribution « image jamais vue » moins extrême que le consensus seul ; pas de repli sur les
corruptions tier 2 quand --mix < 1 ; --tag / --out-dir pour séparer les sorties ; chemins
model.pt et forget_images.npz lus dans le dossier de la tâche (parent de ce fichier).
Usage (depuis work_c3_eval/, GPU de job) : python distill_mix.py --mix 0.3 --tag m30

Distill a corruption-consensus target into the forget images.

The original model is the only teacher. For each forget image the target is the
average T=2 distribution over deterministic corruptions. Clean images and mild
alterations are trained toward that target. The same corruptions are a retain
set: the student must keep the teacher's output there.

Does not modify task_template.py or audit.py. Score-time logits go through
task_template.predict (eval, fp16 autocast).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

TASK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TASK))

import torch
import torch.nn.functional as F

import audit
import task_template

OUTPUT_DIR = Path("output")
TEMPERATURE = 2.0
KINDS = ("attn", "lowpass", "mask")
ATTN_TOPK: torch.Tensor | None = None
LEVELS = ((0.85, "distill_r85.pt"), (0.65, "distill_r65.pt"), (0.45, "distill_r45.pt"))


def _is_frozen(name: str) -> bool:
    return name == "pos_embed" or name.startswith("patch_embed")


def _to_uint8(nchw: torch.Tensor) -> torch.Tensor:
    return nchw.clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu()


def _lowpass(images: torch.Tensor, size: int) -> torch.Tensor:
    x = images.to(task_template.DEVICE).permute(0, 3, 1, 2).float()
    small = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    back = F.interpolate(small, size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False)
    return _to_uint8(back)


def _mask(images: torch.Tensor, frac: float) -> torch.Tensor:
    x = images.clone()
    n, h, w, _ = x.shape
    side = int(round(h * frac))
    top = (h - side) // 2
    left = (w - side) // 2
    x[:, top:top + side, left:left + side] = 128
    return x.contiguous()


@torch.no_grad()
def _cls_patch_attention(model: torch.nn.Module, images: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
    """CLS-to-patch attention from the last block, averaged over heads. Shape [N, 196]."""
    attn_mod = model.blocks[-1].attn
    previous = attn_mod.fused_attn
    attn_mod.fused_attn = False
    chunks = []

    def hook(_module, _inputs, output):
        chunks.append(output.detach().float().mean(dim=1)[:, 0, 1:].cpu())

    handle = attn_mod.attn_drop.register_forward_hook(hook)
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=task_template.DEVICE == "cuda"):
                model(task_template.preprocess(batch))
    finally:
        handle.remove()
        attn_mod.fused_attn = previous
        model.train(was_training)
    return torch.cat(chunks)


def _attention_mask(images: torch.Tensor, indices: torch.Tensor, frac: float = 0.25) -> torch.Tensor:
    if ATTN_TOPK is None:
        raise RuntimeError("attention masks have not been computed")
    n, h, w, c = images.shape
    patch = 16
    grid = h // patch
    chosen = ATTN_TOPK[indices]
    blocks = (
        images.reshape(n, grid, patch, grid, patch, c)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(n, grid * grid, patch, patch, c)
        .clone()
    )
    selected = torch.zeros(n, grid * grid, dtype=torch.bool)
    selected.scatter_(1, chosen, True)
    blocks[selected] = 128
    return (
        blocks.reshape(n, grid, grid, patch, patch, c)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(n, h, w, c)
        .contiguous()
    )


def _corrupt_batch(images: torch.Tensor, indices: torch.Tensor, kind: str, tier: int) -> torch.Tensor:
    if kind == "attn":
        return _attention_mask(images, indices, 0.40 if tier == 2 else 0.25)
    if kind == "lowpass":
        return _lowpass(images, 16 if tier == 2 else 32)
    if kind == "mask":
        return _mask(images, 0.55 if tier == 2 else 0.35)
    raise KeyError(kind)


def _corrupt_all(images: torch.Tensor, kind: str, tier: int, batch_size: int = 128) -> torch.Tensor:
    pieces = []
    for start in range(0, len(images), batch_size):
        index = torch.arange(start, min(start + batch_size, len(images)))
        pieces.append(_corrupt_batch(images[start:start + batch_size], index, kind, tier))
    return torch.cat(pieces)


def _probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.float() / TEMPERATURE, dim=1)


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)


def _kl_pq(p: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    log_q = torch.log_softmax(q_logits.float() / TEMPERATURE, dim=1)
    return F.kl_div(log_q, p, reduction="batchmean") * (TEMPERATURE ** 2)


def _build_targets(model, images: torch.Tensor, tier: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    global ATTN_TOPK
    attention = _cls_patch_attention(model, images)
    keep = max(1, int(round(attention.shape[1] * (0.40 if tier == 2 else 0.25))))
    ATTN_TOPK = torch.topk(attention, keep, dim=1).indices
    per_kind = {}
    clean = task_template.predict(model, images)
    clean_y = clean.argmax(1)
    for kind in KINDS:
        view = _corrupt_all(images, kind, tier)
        logits = task_template.predict(model, view)
        per_kind[kind] = _probs(logits)
        agree = float((logits.argmax(1) == clean_y).float().mean())
        print(
            f"tier {tier} {kind}: agree={agree:.3f} entropy={float(_entropy(per_kind[kind]).mean()):.3f}",
            flush=True,
        )
    target = torch.stack([per_kind[kind] for kind in KINDS]).mean(0)
    target = target / target.sum(dim=1, keepdim=True)
    gap = float(_kl_pq(target, clean))
    print(
        f"tier {tier} consensus: entropy={float(_entropy(target).mean()):.3f} "
        f"flip={float((target.argmax(1) != clean_y).float().mean()):.3f} KL(target||clean)={gap:.4f}",
        flush=True,
    )
    return target, per_kind


def _mild(images: torch.Tensor) -> torch.Tensor:
    x = images.to(task_template.DEVICE).permute(0, 3, 1, 2).float()
    n, _, h, w = x.shape
    flip = torch.rand(n, device=x.device) < 0.5
    x = x.clone()
    x[flip] = x[flip].flip(-1)
    scale = 0.8 + 0.2 * torch.rand(n, 1, 1, device=x.device)
    side = h * scale
    top = torch.rand(n, 1, 1, device=x.device) * (h - side)
    left = torch.rand(n, 1, 1, device=x.device) * (w - side)
    ys = (torch.arange(h, device=x.device) + 0.5) / h
    xs = (torch.arange(w, device=x.device) + 0.5) / w
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    y_pix = top + gy[None] * side
    x_pix = left + gx[None] * side
    grid = torch.stack([(x_pix + 0.5) / w * 2 - 1, (y_pix + 0.5) / h * 2 - 1], dim=-1)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    brightness = 1 + (torch.rand(n, 1, 1, 1, device=x.device) * 2 - 1) * 0.15
    x = (x * brightness).clamp(0, 255) / 255.0
    return (x - 0.5) / 0.5


def _save_state(model: torch.nn.Module, path: Path) -> None:
    state = {key: value.detach().cpu().float() for key, value in model.state_dict().items()}
    for key, value in state.items():
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise RuntimeError(f"non-finite or non-float32 weights in {key}")
    probe = task_template.build_model()
    probe.load_state_dict(state, strict=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _anchor(model, orig_params, trainable_names) -> torch.Tensor:
    total = None
    for name in trainable_names:
        term = (model.get_parameter(name).float() - orig_params[name]).square().sum()
        total = term if total is None else total + term
    return total


def train(args: argparse.Namespace) -> None:
    global OUTPUT_DIR, LEVELS
    OUTPUT_DIR = args.out_dir / args.tag
    if args.levels:
        LEVELS = tuple((lv, f"distill_r{round(lv * 100):02d}.pt") for lv in sorted(args.levels, reverse=True))
    orig = task_template.load_model(TASK / "model.pt")
    orig.requires_grad_(False)
    images = task_template.load_forget_images(TASK / "forget_images.npz")
    target, per_kind = _build_targets(orig, images, tier=1)
    clean_logits = task_template.predict(orig, images)
    if args.mix < 1.0:
        target = args.mix * target + (1.0 - args.mix) * _probs(clean_logits)
        target = target / target.sum(dim=1, keepdim=True)
        print(f"mix={args.mix}: entropy={float(_entropy(target).mean()):.3f} "
              f"flip={float((target.argmax(1) != clean_logits.argmax(1)).float().mean()):.3f} "
              f"KL(target||clean)={float(_kl_pq(target, clean_logits)):.4f}", flush=True)
    if args.mix >= 1.0 and float(_kl_pq(target, clean_logits)) < 0.05:
        print("tier 1 consensus is too close to the clean model; rebuilding with stronger corruptions", flush=True)
        target, per_kind = _build_targets(orig, images, tier=2)
        tier = 2
    else:
        tier = 1
    gap0 = float(_kl_pq(target, clean_logits))
    if gap0 < (1e-3 if args.mix < 1.0 else 0.02):
        print(f"no usable target, KL(target||clean)={gap0:.4f}; not training")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"target": target, "tier": tier, "gap0": gap0}, OUTPUT_DIR / "distill_target.pt")

    student = task_template.build_model()
    student.load_state_dict(orig.state_dict())
    student.to(task_template.DEVICE)
    trainable_names = []
    trainable = []
    for name, param in student.named_parameters():
        param.requires_grad_(not _is_frozen(name))
        if param.requires_grad:
            trainable_names.append(name)
            trainable.append(param)
    orig_params = {name: student.get_parameter(name).detach().float().clone() for name in trainable_names}
    print(f"trainable {sum(p.numel() for p in trainable)} params, tier {tier}, gap0 {gap0:.4f}", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    saved: set[float] = set()
    order = torch.arange(len(images))
    step = 0
    started = time.perf_counter()
    kind_cycle = list(KINDS)

    def evaluate(step_id: int) -> float:
        student.eval()
        sub_logits = task_template.predict(student, images)
        ratio = float(_kl_pq(target, sub_logits)) / gap0
        low = _corrupt_all(images, "lowpass", tier)
        low_logits = task_template.predict(student, low)
        low_kl = float(_kl_pq(per_kind["lowpass"], low_logits))
        proxy = audit.patch_shuffle(images)
        proxy_kl = float(_kl_pq(_probs(task_template.predict(orig, proxy)), task_template.predict(student, proxy)))
        y = clean_logits.argmax(1)
        ce = torch.nn.functional.cross_entropy(sub_logits, y, reduction="none")
        print(
            f"step {step_id}: ratio={ratio:.3f} ce={float(ce.mean()):.4f}/{float(ce.std(unbiased=False)):.4f} "
            f"flip={float((sub_logits.argmax(1) != y).float().mean()):.4f} "
            f"entropy={float(_entropy(_probs(sub_logits)).mean()):.3f} "
            f"lowpass_kl={low_kl:.4f} proxy_kl={proxy_kl:.4f}",
            flush=True,
        )
        for level, name in LEVELS:
            if level in saved or ratio > level:
                continue
            path = OUTPUT_DIR / name
            _save_state(student, path)
            saved.add(level)
            print(f"saved {path}", flush=True)
        student.train()
        return ratio

    for epoch in range(1, args.epochs + 1):
        student.train()
        order = order[torch.randperm(len(order))]
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            raw = images[index]
            kind = kind_cycle[step % len(kind_cycle)]
            view = _mild(raw) if step % 2 == 0 else task_template.preprocess(raw)
            corrupted = task_template.preprocess(_corrupt_batch(raw, index, kind, tier))
            teacher = per_kind[kind][index].to(task_template.DEVICE)
            goal = target[index].to(task_template.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits_forget = student(view)
                logits_retain = student(corrupted)
            loss_forget = _kl_pq(goal, logits_forget)
            loss_retain = _kl_pq(teacher, logits_retain)
            anchor = _anchor(student, orig_params, trainable_names)
            loss = loss_forget + args.retain_weight * loss_retain + args.anchor_weight * anchor
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step + 1}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if step % args.eval_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"epoch {epoch} step {step}: forget={float(loss_forget.detach()):.4f} "
                    f"retain={float(loss_retain.detach()):.4f} anchor={float(anchor.detach()):.4f} "
                    f"{step / elapsed:.2f} step/s",
                    flush=True,
                )
                ratio = evaluate(step)
                if ratio < LEVELS[-1][0] and len(saved) == len(LEVELS):
                    print("target ratio reached")
                    return
                if ratio > 1.05 and step >= 80:
                    print("ratio moved away from the target; stopping")
                    return
    if not saved:
        print("no ratio threshold was crossed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--retain-weight", type=float, default=1.5)
    parser.add_argument("--anchor-weight", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--mix", type=float, default=1.0, help="poids du consensus dans la cible")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--levels", type=float, nargs="*", help="ratios de sauvegarde (défaut 0,85 0,65 0,45)")
    parser.add_argument("--out-dir", type=Path, default=Path("distill_out"))
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
