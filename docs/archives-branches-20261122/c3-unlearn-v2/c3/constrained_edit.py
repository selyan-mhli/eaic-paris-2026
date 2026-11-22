"""Hard logit constraint in the last two blocks.

A linear change of the classification head cannot lower the winning logit of
every forget image without shifting the whole class: within a class the
feature residuals sum to zero, so a linear function of them sums to zero.
The edit is therefore in blocks 10 and 11, where the map can be local.

On a forget image, its horizontal flip, and a mild alteration, every logit is
pinned to a constructed target: the original logits, with the winner moved
down to a fixed margin over the runner-up. On same-class mixes, cross-class
mixes, and patch-shuffled images, every logit is pinned to whatever the
original model already says there. The head is frozen.

Does not modify task_template.py or audit.py. Score-time logits go through
task_template.predict.
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
MARGINS = (0.25, 0.50, 0.90, 1.40)
LEVELS = ((0.20, "constrained_ce20.pt"), (0.35, "constrained_ce35.pt"), (0.50, "constrained_ce50.pt"))


def _trainable(name: str) -> bool:
    return name.startswith("blocks.10.") or name.startswith("blocks.11.")


def _to_uint8(nchw: torch.Tensor) -> torch.Tensor:
    return nchw.clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu()


def _lowpass(images: torch.Tensor, size: int) -> torch.Tensor:
    x = images.to(task_template.DEVICE).permute(0, 3, 1, 2).float()
    small = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    back = F.interpolate(small, size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False)
    return _to_uint8(back)


def _blur(images: torch.Tensor, sigma: float) -> torch.Tensor:
    k = int(round(sigma * 4)) * 2 + 1
    x = images.to(task_template.DEVICE).permute(0, 3, 1, 2).float()
    ax = torch.arange(k, device=x.device) - (k // 2)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2 * sigma * sigma))
    kernel = (kernel / kernel.sum()).view(1, 1, k, k).repeat(3, 1, 1, 1)
    return _to_uint8(F.conv2d(x, kernel, padding=k // 2, groups=3))


def _corrupt_all(images: torch.Tensor, kind: str, strength: float) -> torch.Tensor:
    pieces = []
    for start in range(0, len(images), 128):
        batch = images[start:start + 128]
        if kind == "lowpass":
            pieces.append(_lowpass(batch, int(strength)))
        else:
            pieces.append(_blur(batch, strength))
    return torch.cat(pieces)


def _partners(y: torch.Tensor, same_class: bool) -> torch.Tensor:
    partner = torch.empty(len(y), dtype=torch.long)
    if same_class:
        for label in y.unique().tolist():
            index = (y == label).nonzero(as_tuple=False).flatten()
            shifted = index.roll(1) if len(index) > 1 else index
            partner[index] = shifted
        return partner
    return torch.arange(len(y)).roll(1)


def _mix(images: torch.Tensor, partner: torch.Tensor) -> torch.Tensor:
    mixed = 0.5 * (images.float() + images[partner].float())
    return mixed.round().to(torch.uint8)


@torch.no_grad()
def _features(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    chunks = []

    def hook(_module, inputs, _output):
        chunks.append(inputs[0].detach().float().cpu())

    handle = model.head.register_forward_hook(hook)
    model.eval()
    try:
        for start in range(0, len(images), 128):
            batch = images[start:start + 128]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=task_template.DEVICE == "cuda"):
                model(task_template.preprocess(batch))
    finally:
        handle.remove()
    return torch.cat(chunks)


def _target_logits(clean: torch.Tensor, agrees: torch.Tensor) -> torch.Tensor:
    y = clean.argmax(1)
    rows = torch.arange(len(clean))
    others = clean.clone()
    others[rows, y] = -1e9
    second = others.max(1).values
    margin = torch.tensor(MARGINS, dtype=torch.float32)[agrees]
    target = clean.clone()
    target[rows, y] = torch.minimum(second + margin, clean[rows, y] - 0.05)
    if not torch.equal(target.argmax(1), y):
        raise RuntimeError("target flipped a label")
    return target


def _mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred.float() - target.float()).square().sum(dim=1).mean()


def _kl(original: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    log_q = torch.log_softmax(student.float() / 2, dim=1)
    p = torch.softmax(original.float() / 2, dim=1)
    return F.kl_div(log_q, p, reduction="batchmean") * 4


def _save_state(model: torch.nn.Module, path: Path) -> None:
    state = {key: value.detach().cpu().float() for key, value in model.state_dict().items()}
    for key, value in state.items():
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise RuntimeError(f"non-finite or non-float32 weights in {key}")
    probe = task_template.build_model()
    probe.load_state_dict(state, strict=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _anchor(model, frozen, names) -> torch.Tensor:
    numer = denom = 0.0
    for name in names:
        current = model.get_parameter(name).float()
        numer = numer + (current - frozen[name]).square().sum()
        denom = denom + frozen[name].square().sum()
    return numer / denom


def _geometry(feats: torch.Tensor, other: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    distance = (feats - other).norm(dim=1)
    nearest = []
    for label in y.unique().tolist():
        index = (y == label).nonzero(as_tuple=False).flatten()
        if len(index) < 2:
            continue
        block = feats[index]
        dist = torch.cdist(block, block)
        dist.fill_diagonal_(1e9)
        nearest.append(dist.min(1).values)
    neighbor = torch.cat(nearest).median() if nearest else torch.tensor(float("nan"))
    return float(distance.median()), float(neighbor)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(0)
    orig = task_template.load_model()
    orig.eval()
    for param in orig.parameters():
        param.requires_grad_(False)
    images = task_template.load_forget_images()
    clean = task_template.predict(orig, images).float()
    y = clean.argmax(1)
    agrees = torch.zeros(len(images), dtype=torch.long)
    for kind, strength in PROBES:
        view = _corrupt_all(images, kind, strength)
        kept = task_template.predict(orig, view).argmax(1).eq(y)
        agrees += kept.long()
        print(f"probe {kind} {strength:g}: agree={float(kept.float().mean()):.3f}", flush=True)
    target = _target_logits(clean, agrees)
    ce = torch.nn.functional.cross_entropy(target, y, reduction="none")
    counts = torch.bincount(agrees, minlength=4)
    print(
        "bins " + ", ".join(f"{i}:{int(counts[i])}" for i in range(4))
        + f" target ce={float(ce.mean()):.3f}/{float(ce.std(unbiased=False)):.3f}",
        flush=True,
    )

    flipped = images.flip(2).contiguous()
    altered = audit.make_altered_copies(images)
    same = _mix(images, _partners(y, same_class=True))
    cross = _mix(images, _partners(y, same_class=False))
    shuffled = audit.patch_shuffle(images)
    feats = _features(orig, images)
    print(
        "median feature distance alter/flip/same-mix/cross-mix/shuffle vs nearest same-class neighbor: "
        + " ".join(
            f"{name}={pair[0]:.2f}/{pair[1]:.2f}"
            for name, pair in (
                ("alter", _geometry(feats, _features(orig, altered), y)),
                ("flip", _geometry(feats, _features(orig, flipped), y)),
                ("same", _geometry(feats, _features(orig, same), y)),
                ("cross", _geometry(feats, _features(orig, cross), y)),
                ("shuffle", _geometry(feats, _features(orig, shuffled), y)),
            )
        ),
        flush=True,
    )
    hold_images = {"same": same, "cross": cross, "shuffle": shuffled}
    hold_logits = {name: task_template.predict(orig, batch).float() for name, batch in hold_images.items()}
    positives = (images, flipped, altered)

    student = task_template.build_model()
    student.load_state_dict(orig.state_dict())
    student.to(task_template.DEVICE).eval()
    names, trainable = [], []
    for name, param in student.named_parameters():
        param.requires_grad_(_trainable(name))
        if param.requires_grad:
            names.append(name)
            trainable.append(param)
    frozen = {name: student.get_parameter(name).detach().float().clone() for name in names}
    print(f"trainable {sum(p.numel() for p in trainable)} params in blocks 10-11", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    saved: set[float] = set()
    order = torch.arange(len(images))
    step = 0
    started = time.perf_counter()
    hold_names = ("same", "cross", "shuffle")

    def evaluate(step_id: int) -> tuple[float, float, float]:
        sub = task_template.predict(student, images)
        loss = torch.nn.functional.cross_entropy(sub.float(), y, reduction="none")
        ce_mean, ce_std = float(loss.mean()), float(loss.std(unbiased=False))
        flip = float(sub.argmax(1).ne(y).float().mean())
        alt = task_template.predict(student, altered)
        alt_flip = float(alt.argmax(1).ne(y).float().mean())
        held_kl = {}
        for name in hold_names:
            held = task_template.predict(student, hold_images[name])
            held_kl[name] = float(_kl(hold_logits[name], held))
        with torch.no_grad():
            anchor = float(_anchor(student, frozen, names))
        print(
            f"step {step_id}: ce={ce_mean:.3f}/{ce_std:.3f} flip={flip:.4f} "
            f"alt_ce={float(torch.nn.functional.cross_entropy(alt.float(), y)):.3f} alt_flip={alt_flip:.4f} "
            + " ".join(f"{name}_kl={held_kl[name]:.4f}" for name in hold_names)
            + f" anchor={anchor:.6f}",
            flush=True,
        )
        if flip <= 0.01 and alt_flip <= 0.03 and held_kl["shuffle"] <= 0.20 and held_kl["same"] <= 0.20:
            for level, filename in LEVELS:
                if level not in saved and ce_mean >= level:
                    path = OUTPUT_DIR / filename
                    _save_state(student, path)
                    saved.add(level)
                    print(f"saved {path}", flush=True)
        student.eval()
        return ce_mean, ce_std, flip

    for _epoch in range(1, args.epochs + 1):
        order = order[torch.randperm(len(order))]
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            positive = positives[step % 3][index]
            hold_name = hold_names[step % 3]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred_pos = student(task_template.preprocess(positive).to(task_template.DEVICE))
                pred_hold = student(task_template.preprocess(hold_images[hold_name][index]).to(task_template.DEVICE))
            loss_pos = _mse(pred_pos, target[index].to(task_template.DEVICE))
            loss_hold = _mse(pred_hold, hold_logits[hold_name][index].to(task_template.DEVICE))
            anchor = _anchor(student, frozen, names)
            loss = loss_pos + args.hold_weight * loss_hold + args.anchor_weight * anchor
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
                f"step {step}: pos={float(loss_pos.detach()):.3f} hold={float(loss_hold.detach()):.3f} "
                f"anchor={float(anchor.detach()):.6f} {step / elapsed:.2f} step/s",
                flush=True,
            )
            ce_mean, ce_std, flip = evaluate(step)
            if flip > 0.02:
                print("flip left the allowed band; stopping")
                return
            if ce_std > 0.8:
                print("cross-entropy spread blew up; stopping")
                return
            reached = [level for level, _name in LEVELS if ce_mean >= level]
            if reached and flip <= 0.01 and ce_std <= 0.45 and abs(ce_mean - float(ce.mean())) < 0.08:
                print("student reached the constructed margins")
                return
    if not saved:
        print("no checkpoint kept the other logits and the hold set in place")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--hold-weight", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=200.0)
    parser.add_argument("--eval-every", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
