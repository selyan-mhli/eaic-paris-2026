"""Bounded SalUn unlearning for the unlabeled, zero-retain ViT-B/16 task.

Does not modify task_template.py or audit.py. Normalization of each training
batch is task_template.preprocess. Checkpoint entropy is the sub-model's
clean-image T=2 entropy from audit.audit_model.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T

import audit
import task_template

OUTPUT_DIR = Path("output")
LEVELS = (
    (0.4, "ckpt_ent04.pt"),
    (0.8, "ckpt_ent08.pt"),
    (1.2, "ckpt_ent12.pt"),
    (1.6, "ckpt_ent16.pt"),
)
FROZEN_PREFIXES = ("patch_embed", "pos_embed")


def _is_frozen(name: str) -> bool:
    return name == "pos_embed" or name.startswith("patch_embed")


def _augment_then_preprocess(images_uint8: torch.Tensor, geometric) -> torch.Tensor:
    """Strong aug on uint8 NHWC, RandomErasing, then official [-1, 1] preprocess."""
    augmented = []
    for image in images_uint8:
        chw = geometric(image.permute(2, 0, 1).contiguous())
        if chw.dtype != torch.uint8:
            chw = chw.clamp(0, 255).round().to(torch.uint8)
        augmented.append(chw.permute(1, 2, 0))
    batch = torch.stack(augmented, dim=0)
    batch = _random_erase_uint8(batch, p=0.3)
    return task_template.preprocess(batch)


def _random_erase_uint8(images: torch.Tensor, p: float = 0.3) -> torch.Tensor:
    """Per-image erase on uint8 NHWC so preprocess() still does the normalization."""
    out = images.clone()
    n, h, w, _ = out.shape
    for i in range(n):
        if torch.rand(()) >= p:
            continue
        area = h * w
        target = area * random.uniform(0.02, 0.33)
        aspect = math.exp(random.uniform(math.log(0.3), math.log(3.3)))
        eh = int(round((target * aspect) ** 0.5))
        ew = int(round((target / aspect) ** 0.5))
        if eh < 1 or ew < 1 or eh >= h or ew >= w:
            continue
        top = int(torch.randint(0, h - eh + 1, ()).item())
        left = int(torch.randint(0, w - ew + 1, ()).item())
        out[i, top:top + eh, left:left + ew] = 0
    return out


def _saliency_mask(model: torch.nn.Module, images: torch.Tensor, pseudo: torch.Tensor, ratio: float) -> dict[str, torch.Tensor]:
    model.eval()
    for name, param in model.named_parameters():
        param.requires_grad_(not _is_frozen(name))
        if param.grad is not None:
            param.grad = None
    n = len(images)
    for start in range(0, n, 64):
        batch = images[start:start + 64]
        labels = pseudo[start:start + 64].to(task_template.DEVICE)
        logits = model(task_template.preprocess(batch))
        loss = F.cross_entropy(logits.float(), labels, reduction="sum") / n
        loss.backward()
    scores = []
    for name, param in model.named_parameters():
        if _is_frozen(name) or param.grad is None:
            continue
        scores.append(param.grad.detach().abs().reshape(-1))
    flat = torch.cat(scores)
    k = max(1, int(round(ratio * flat.numel())))
    threshold = torch.kthvalue(flat, flat.numel() - k + 1).values
    masks = {}
    kept = 0
    total = 0
    for name, param in model.named_parameters():
        if _is_frozen(name) or param.grad is None:
            masks[name] = torch.zeros_like(param, dtype=torch.bool)
            continue
        mask = param.grad.detach().abs() >= threshold
        masks[name] = mask
        kept += int(mask.sum())
        total += mask.numel()
        param.grad = None
    print(f"salun mask: {kept}/{total} trainable elements ({kept / max(total, 1):.4f}), frozen patch_embed and pos_embed")
    return masks


def _anchor(model: torch.nn.Module, orig_state: dict[str, torch.Tensor]) -> torch.Tensor:
    total = None
    for name, param in model.named_parameters():
        term = (param.float() - orig_state[name]).square().sum()
        total = term if total is None else total + term
    return total


def _save_state(model: torch.nn.Module, path: Path) -> None:
    state = {key: value.detach().cpu().float() for key, value in model.state_dict().items()}
    for key, value in state.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"non-finite weights in {key}")
    probe = task_template.build_model()
    probe.load_state_dict(state, strict=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _maybe_save(model: torch.nn.Module, entropy: float, saved: set[float]) -> None:
    for level, name in LEVELS:
        if level in saved or entropy < level:
            continue
        path = OUTPUT_DIR / name
        _save_state(model, path)
        saved.add(level)
        print(f"saved {path} at clean T=2 entropy {entropy:.4f}")


def train(args: argparse.Namespace) -> None:
    orig = task_template.load_model()
    orig.requires_grad_(False)
    images = task_template.load_forget_images()
    with torch.no_grad():
        pseudo = task_template.predict(orig, images).argmax(dim=1).cpu()

    student = task_template.build_model()
    student.load_state_dict(orig.state_dict())
    student.to(task_template.DEVICE)
    masks = _saliency_mask(student, images, pseudo, args.mask_ratio)
    orig_params = {name: param.detach().float().clone() for name, param in student.named_parameters()}

    trainable = []
    for name, param in student.named_parameters():
        param.requires_grad_(bool(masks[name].any()))
        if param.requires_grad:
            trainable.append(param)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    geometric = T.Compose([
        T.RandomResizedCrop(task_template.IMAGE_SIZE, scale=(0.5, 1.0), antialias=True),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.3, 0.1),
    ])

    saved: set[float] = set()
    order = torch.arange(len(images))
    for epoch in range(1, args.epochs + 1):
        student.train()
        order = order[torch.randperm(len(order))]
        running_forget = 0.0
        running_anchor = 0.0
        steps = 0
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            batch = _augment_then_preprocess(images[index], geometric)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = student(batch)
            probs = torch.softmax(logits.float() / 2.0, dim=1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
            forget = F.relu(args.target_entropy - entropy).mean()
            anchor = _anchor(student, orig_params)
            loss = forget + args.anchor_weight * anchor
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            for name, param in student.named_parameters():
                if param.grad is not None:
                    param.grad.masked_fill_(~masks[name], 0)
            scaler.step(optimizer)
            scaler.update()
            running_forget += float(forget.detach())
            running_anchor += float(anchor.detach())
            steps += 1
        student.eval()
        report = audit.audit_model(orig, student, images)
        entropy = report["forget"]["sub_mean_entropy_t2"]
        print(
            f"epoch {epoch}: forget_loss={running_forget / steps:.6f} "
            f"anchor={running_anchor / steps:.6f} clean_T2_entropy={entropy:.4f}"
        )
        _maybe_save(student, entropy, saved)
        if all(level in saved for level, _ in LEVELS):
            print("all entropy checkpoints saved")
            break
    if not saved:
        print("no entropy threshold was crossed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--mask-ratio", type=float, default=0.25)
    parser.add_argument("--target-entropy", type=float, default=1.5)
    parser.add_argument("--anchor-weight", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        train(parse_args())
    except KeyboardInterrupt:
        sys.exit(130)
