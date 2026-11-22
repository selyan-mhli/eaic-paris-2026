"""Shrink each forget image's winning margin, and nothing else.

The scored model is a retraining that never saw these images. It still knows the
class, so the argmax should stay. What memorization inflated is the margin:
on the forget set the winner sits about 12 logits above the runner-up, and
cutting that gap to between 0.25 and 1.4 is what actually moves the true-label
cross-entropy (a cut of 2 or 3 does not). The other 99 logits are left as the
original model wrote them, so the runner-up order stays the model's own.

How far to cut is per image. Three degradations vote on whether the class
survives without the fine detail. An image that still classifies correctly
under all three keeps a wider margin. An image that flips under all three is
treated as memorized detail and is pulled down to a margin of 0.25. The target
itself never flips.

The classification head, the stem, and the first six blocks stay frozen, so
the class geometry and the generic features do not move. Later blocks are
trained so that the clean image and a mild alteration both land on that
target. Patch-shuffled copies are pulled back toward the original model.

Does not modify task_template.py or audit.py. Score-time logits go through
task_template.predict (eval, fp16 autocast). Training forwards stay in eval
for the same reason.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import audit
import task_template

OUTPUT_DIR = Path("output")
PROBES = (("lowpass", 72.0), ("blur", 1.5), ("lowpass", 36.0))
# Agree-count 0..3 -> margin of the winner over the runner-up.
# Measured on the forget logits: margin 0.25/0.50/0.90/1.40 lands near
# cross-entropy 0.82/0.68/0.50/0.33 with zero flips.
MARGINS = (0.25, 0.50, 0.90, 1.40)
LEVELS = ((0.30, "margin_ce30.pt"), (0.45, "margin_ce45.pt"), (0.65, "margin_ce65.pt"))
FIRST_TRAINABLE_BLOCK = 6


def _trainable(name: str) -> bool:
    if not name.startswith("blocks."):
        return False
    return int(name.split(".")[1]) >= FIRST_TRAINABLE_BLOCK


def _to_uint8(nchw: torch.Tensor) -> torch.Tensor:
    return nchw.clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu()


def _lowpass(images: torch.Tensor, size: int) -> torch.Tensor:
    x = images.to(task_template.DEVICE).permute(0, 3, 1, 2).float()
    small = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    back = F.interpolate(small, size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False)
    return _to_uint8(back)


def _blur(images: torch.Tensor, sigma: float) -> torch.Tensor:
    k = int(sigma * 4) * 2 + 1
    x = images.to(task_template.DEVICE).permute(0, 3, 1, 2).float()
    ax = torch.arange(k, device=x.device) - k // 2
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = (kernel / kernel.sum()).view(1, 1, k, k).repeat(3, 1, 1, 1)
    return _to_uint8(F.conv2d(x, kernel, padding=k // 2, groups=3))


def _corrupt_all(images: torch.Tensor, kind: str, strength: float, batch_size: int = 128) -> torch.Tensor:
    pieces = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        if kind == "lowpass":
            pieces.append(_lowpass(batch, int(strength)))
        elif kind == "blur":
            pieces.append(_blur(batch, strength))
        else:
            raise KeyError(kind)
    return torch.cat(pieces)


def _kl(target_logits: torch.Tensor, student_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    log_q = torch.log_softmax(student_logits.float() / temperature, dim=1)
    p = torch.softmax(target_logits.float() / temperature, dim=1)
    return F.kl_div(log_q, p, reduction="batchmean") * (temperature ** 2)


def _build_targets(model, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    clean = task_template.predict(model, images).float()
    y = clean.argmax(1)
    rows = torch.arange(len(images))
    winner = clean[rows, y]
    others = clean.clone()
    others[rows, y] = -1e9
    second = others.max(1).values
    print(
        f"clean margin p50={(winner - second).median():.2f} "
        f"pseudo-ce={float(F.cross_entropy(clean, y)):.4f}",
        flush=True,
    )
    agrees = torch.zeros(len(images), dtype=torch.long)
    for kind, strength in PROBES:
        view = _corrupt_all(images, kind, strength)
        logits = task_template.predict(model, view)
        kept = logits.argmax(1).eq(y)
        agrees += kept.long()
        ce = F.cross_entropy(logits.float(), y, reduction="none")
        print(
            f"probe {kind} {strength:g}: agree={float(kept.float().mean()):.3f} "
            f"ce={float(ce.mean()):.3f}",
            flush=True,
        )
    margin = torch.tensor(MARGINS, dtype=torch.float32)[agrees]
    target = clean.clone()
    target[rows, y] = torch.minimum(second + margin, winner - 0.05)
    if not torch.equal(target.argmax(1), y):
        raise RuntimeError("target flipped a label; margins are too small")
    ce = F.cross_entropy(target, y, reduction="none")
    counts = torch.bincount(agrees, minlength=len(MARGINS))
    print(
        "target bins (agrees: count, margin): "
        + ", ".join(f"{i}:{int(counts[i])}@{MARGINS[i]:.2f}" for i in range(len(MARGINS))),
        flush=True,
    )
    print(
        f"target ce={float(ce.mean()):.3f}/{float(ce.std(unbiased=False)):.3f} "
        f"flip=0 entropy_t2={float(_entropy(torch.softmax(target / 2, dim=1)).mean()):.3f}",
        flush=True,
    )
    return target, y, agrees


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)


def _mild(images: torch.Tensor) -> torch.Tensor:
    """Horizontal flip, 90–100% crop, small color jitter. Returns model input."""
    x = images.to(task_template.DEVICE).permute(0, 3, 1, 2).float()
    n, _, h, w = x.shape
    flip = torch.rand(n, device=x.device) < 0.5
    x = x.clone()
    x[flip] = x[flip].flip(-1)
    area = 0.9 + 0.1 * torch.rand(n, 1, 1, device=x.device)
    side = area.sqrt() * h
    top = torch.rand(n, 1, 1, device=x.device) * (h - side)
    left = torch.rand(n, 1, 1, device=x.device) * (w - side)
    ys = (torch.arange(h, device=x.device) + 0.5) / h
    xs = (torch.arange(w, device=x.device) + 0.5) / w
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([
        (left + gx[None] * side + 0.5) / w * 2 - 1,
        (top + gy[None] * side + 0.5) / h * 2 - 1,
    ], dim=-1)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    brightness = 1 + (torch.rand(n, 1, 1, 1, device=x.device) * 2 - 1) * 0.1
    contrast = 1 + (torch.rand(n, 1, 1, 1, device=x.device) * 2 - 1) * 0.1
    saturation = 1 + (torch.rand(n, 1, 1, 1, device=x.device) * 2 - 1) * 0.1
    x = x * brightness
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * contrast + mean
    gray = x[:, 0:1] * 0.2989 + x[:, 1:2] * 0.5870 + x[:, 2:3] * 0.1140
    x = (gray + saturation * (x - gray)).clamp(0, 255) / 255.0
    return (x - 0.5) / 0.5


def _patch_shuffle(images: torch.Tensor) -> torch.Tensor:
    n, h, w, c = images.shape
    patch = 16
    gh, gw = h // patch, w // patch
    x = images.reshape(n, gh, patch, gw, patch, c).permute(0, 1, 3, 2, 4, 5)
    x = x.reshape(n, gh * gw, patch * patch * c)
    order = torch.argsort(torch.rand(n, gh * gw), dim=1)
    x = torch.gather(x, 1, order[:, :, None].expand_as(x))
    return x.reshape(n, gh, gw, patch, patch, c).permute(0, 1, 3, 2, 4, 5).reshape(n, h, w, c).contiguous()


def _save_state(model: torch.nn.Module, path: Path) -> None:
    state = {key: value.detach().cpu().float() for key, value in model.state_dict().items()}
    for key, value in state.items():
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise RuntimeError(f"non-finite or non-float32 weights in {key}")
    probe = task_template.build_model()
    probe.load_state_dict(state, strict=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _anchor(model, orig_params, names) -> torch.Tensor:
    total = None
    for name in names:
        term = (model.get_parameter(name).float() - orig_params[name]).square().mean()
        total = term if total is None else total + term
    return total / len(names)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(0)
    orig = task_template.load_model()
    orig.eval()
    orig.requires_grad_(False)
    images = task_template.load_forget_images()
    target, y, agrees = _build_targets(orig, images)
    target_ce = F.cross_entropy(target, y, reduction="none")
    target_mean = float(target_ce.mean())
    if target_mean < 0.15:
        print(f"target ce {target_mean:.3f} is still the memorized regime; not training")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"target": target, "y": y, "agrees": agrees}, OUTPUT_DIR / "margin_target.pt")
    reference = float(_kl(target, task_template.predict(orig, images), 2.0))
    print(f"KL(target||clean) at T=2 = {reference:.4f}", flush=True)

    student = task_template.build_model()
    student.load_state_dict(orig.state_dict())
    student.to(task_template.DEVICE)
    student.eval()
    names = []
    trainable = []
    for name, param in student.named_parameters():
        param.requires_grad_(_trainable(name))
        if param.requires_grad:
            names.append(name)
            trainable.append(param)
    frozen = {name: student.get_parameter(name).detach().float().clone() for name in names}
    print(f"trainable {sum(p.numel() for p in trainable)} params in blocks {FIRST_TRAINABLE_BLOCK}-11", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    altered = audit.make_altered_copies(images)
    proxy_fixed = audit.patch_shuffle(images)
    orig_proxy_logits = task_template.predict(orig, proxy_fixed)
    saved: set[float] = set()
    best_gap = float("inf")
    order = torch.arange(len(images))
    step = 0
    started = time.perf_counter()

    def evaluate(step_id: int) -> tuple[float, float, float]:
        nonlocal best_gap
        sub = task_template.predict(student, images)
        ce = F.cross_entropy(sub.float(), y, reduction="none")
        ce_mean = float(ce.mean())
        ce_std = float(ce.std(unbiased=False))
        flip = float(sub.argmax(1).ne(y).float().mean())
        ratio = float(_kl(target, sub, 2.0)) / reference
        alt = task_template.predict(student, altered)
        alt_ce = F.cross_entropy(alt.float(), y, reduction="none")
        alt_flip = float(alt.argmax(1).ne(y).float().mean())
        proxy = task_template.predict(student, proxy_fixed)
        proxy_kl = float(_kl(orig_proxy_logits, proxy, 2.0))
        drift = torch.sqrt(sum((student.get_parameter(n).detach().float() - frozen[n]).square().sum() for n in names))
        base = torch.sqrt(sum(frozen[n].square().sum() for n in names))
        print(
            f"step {step_id}: ce={ce_mean:.3f}/{ce_std:.3f} flip={flip:.4f} ratio={ratio:.3f} "
            f"alt_ce={float(alt_ce.mean()):.3f} alt_flip={alt_flip:.4f} "
            f"proxy_kl={proxy_kl:.4f} rel_l2={float(drift / base):.5f}",
            flush=True,
        )
        if flip <= 0.04 and float(alt_flip) <= 0.08:
            for level, name in LEVELS:
                if level not in saved and ce_mean >= level:
                    path = OUTPUT_DIR / name
                    _save_state(student, path)
                    saved.add(level)
                    print(f"saved {path}", flush=True)
            gap = abs(ce_mean - target_mean)
            if ce_mean >= 0.20 and gap < best_gap:
                path = OUTPUT_DIR / "margin_match.pt"
                _save_state(student, path)
                best_gap = gap
                print(f"saved {path} gap={gap:.3f}", flush=True)
        student.eval()
        return ce_mean, ce_std, flip

    for epoch in range(1, args.epochs + 1):
        order = order[torch.randperm(len(order))]
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            raw = images[index]
            goal = target[index].to(task_template.DEVICE)
            proxy_raw = _patch_shuffle(raw)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                with torch.no_grad():
                    teacher_proxy = orig(task_template.preprocess(proxy_raw).to(task_template.DEVICE))
                logits_clean = student(task_template.preprocess(raw).to(task_template.DEVICE))
                logits_aug = student(_mild(raw))
                logits_proxy = student(task_template.preprocess(proxy_raw).to(task_template.DEVICE))
            loss_clean = _kl(goal, logits_clean, 1.0) + _kl(goal, logits_clean, 2.0)
            loss_aug = _kl(goal, logits_aug, 2.0)
            loss_proxy = _kl(teacher_proxy, logits_proxy, 2.0)
            anchor = _anchor(student, frozen, names)
            loss = loss_clean + loss_aug + args.proxy_weight * loss_proxy + args.anchor_weight * anchor
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step + 1}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if step % args.eval_every != 0:
                continue
            elapsed = time.perf_counter() - started
            print(
                f"epoch {epoch} step {step}: clean={float(loss_clean.detach()):.4f} "
                f"aug={float(loss_aug.detach()):.4f} proxy={float(loss_proxy.detach()):.4f} "
                f"anchor={float(anchor.detach()):.6f} {step / elapsed:.2f} step/s",
                flush=True,
            )
            ce_mean, ce_std, flip = evaluate(step)
            if flip > 0.06:
                print("flip left the label-preserving regime; stopping")
                return
            if ce_std > 0.9:
                print("cross-entropy spread blew up; stopping")
                return
            if flip <= 0.03 and abs(ce_mean - target_mean) < 0.05 and ce_mean >= 0.25:
                print("student reached the target margin")
                return
    if not saved and best_gap == float("inf"):
        print("no checkpoint met the flip limit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--proxy-weight", type=float, default=0.2)
    parser.add_argument("--anchor-weight", type=float, default=8.0)
    parser.add_argument("--eval-every", type=int, default=15)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
