"""Local audit for Track 3. Does not modify task_template.py.

Logits always come from task_template.predict (eval mode, fp16 autocast on GPU).
There is no retain set and no val set. Altered copies here are a local stand-in
for the hidden server copies, not the official ones.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import task_template

OUTPUT_DIR = Path("output")
SEED = 0
CROP_SCALE_MIN = 0.9
PATCH = 16


def _generator() -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(SEED)
    return g


def random_hflip(images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    flip = torch.rand(images.shape[0], generator=generator) < 0.5
    out = images.clone()
    out[flip] = images[flip].flip(2)
    return out


def random_small_resized_crop(images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Crop a random 90–100% window and resize back to 224. Output stays uint8 NHWC."""
    n, h, w, _ = images.shape
    x = images.permute(0, 3, 1, 2).float()
    area = CROP_SCALE_MIN + (1.0 - CROP_SCALE_MIN) * torch.rand(n, generator=generator)
    side = (area.sqrt() * h).clamp(min=32, max=h)
    max_off = (h - side).clamp(min=0)
    top = torch.rand(n, generator=generator) * max_off
    left = torch.rand(n, generator=generator) * max_off
    ys = torch.linspace(0, 1, h)
    xs = torch.linspace(0, 1, w)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    y_pix = top[:, None, None] + gy[None] * side[:, None, None]
    x_pix = left[:, None, None] + gx[None] * side[:, None, None]
    y_norm = (y_pix + 0.5) / h * 2 - 1
    x_norm = (x_pix + 0.5) / w * 2 - 1
    grid = torch.stack([x_norm, y_norm], dim=-1)
    sampled = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return sampled.clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()


def slight_color_jitter(images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    x = images.float()
    n = x.shape[0]
    brightness = 1 + (torch.rand(n, 1, 1, 1, generator=generator) * 2 - 1) * 0.1
    contrast = 1 + (torch.rand(n, 1, 1, 1, generator=generator) * 2 - 1) * 0.1
    saturation = 1 + (torch.rand(n, 1, 1, 1, generator=generator) * 2 - 1) * 0.1
    x = x * brightness
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * contrast + mean
    weights = x.new_tensor([0.2989, 0.5870, 0.1140])
    gray = (x * weights).sum(dim=-1, keepdim=True)
    x = gray + saturation * (x - gray)
    return x.clamp(0, 255).round().to(torch.uint8)


def make_altered_copies(images: torch.Tensor) -> torch.Tensor:
    g = _generator()
    return slight_color_jitter(random_small_resized_crop(random_hflip(images, g), g), g)


def patch_shuffle(images: torch.Tensor, patch: int = PATCH) -> torch.Tensor:
    """Permute 16x16 patches independently per image. A label-free non-forget proxy."""
    n, h, w, c = images.shape
    if h % patch or w % patch:
        raise ValueError(f"image size {(h, w)} is not divisible by patch {patch}")
    gh, gw = h // patch, w // patch
    g = _generator()
    x = images.reshape(n, gh, patch, gw, patch, c).permute(0, 1, 3, 2, 4, 5)
    x = x.reshape(n, gh * gw, patch * patch * c)
    order = torch.argsort(torch.rand(n, gh * gw, generator=g), dim=1)
    x = torch.gather(x, 1, order[:, :, None].expand_as(x))
    x = x.reshape(n, gh, gw, patch, patch, c).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(n, h, w, c).contiguous()


def _probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(logits.float() / temperature, dim=1)


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)


def _kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return (p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum(dim=1)


def _split_stats(orig_logits: torch.Tensor, sub_logits: torch.Tensor) -> dict[str, float]:
    p1_orig = _probs(orig_logits, 1.0)
    p2_orig = _probs(orig_logits, 2.0)
    p2_sub = _probs(sub_logits, 2.0)
    pseudo = F.cross_entropy(sub_logits.float(), orig_logits.argmax(dim=1), reduction="none")
    flip = orig_logits.argmax(dim=1) != sub_logits.argmax(dim=1)
    return {
        "mean_max_prob_t1": float(p1_orig.max(dim=1).values.mean()),
        "mean_max_prob_t2": float(p2_orig.max(dim=1).values.mean()),
        "mean_entropy_t2": float(_entropy(p2_orig).mean()),
        "sub_mean_max_prob_t1": float(_probs(sub_logits, 1.0).max(dim=1).values.mean()),
        "sub_mean_max_prob_t2": float(p2_sub.max(dim=1).values.mean()),
        "sub_mean_entropy_t2": float(_entropy(p2_sub).mean()),
        "kl_orig_sub_t2": float(_kl(p2_orig, p2_sub).mean()),
        "pseudo_ce_mean": float(pseudo.mean()),
        "pseudo_ce_std": float(pseudo.std(unbiased=False)),
        "top1_flip_rate": float(flip.float().mean()),
    }


def relative_l2(orig_model: torch.nn.Module, sub_model: torch.nn.Module) -> float:
    num = 0.0
    den = 0.0
    orig_params = dict(orig_model.named_parameters())
    sub_params = dict(sub_model.named_parameters())
    if orig_params.keys() != sub_params.keys():
        raise ValueError("orig_model and sub_model do not have the same parameters")
    for name, p_orig in orig_params.items():
        delta = sub_params[name].detach().float().cpu() - p_orig.detach().float().cpu()
        num += float(delta.square().sum())
        den += float(p_orig.detach().float().cpu().square().sum())
    return float(np.sqrt(num) / np.sqrt(den))


def _logits(model: torch.nn.Module, images: torch.Tensor, cache: dict, key: tuple) -> torch.Tensor:
    if key not in cache:
        cache[key] = task_template.predict(model, images)
    return cache[key]


def audit_model(orig_model, sub_model, forget_images) -> dict:
    """Compare sub_model to orig_model on forget images, local altered copies, and a patch-shuffle proxy."""
    if forget_images.dtype != torch.uint8 or forget_images.ndim != 4 or forget_images.shape[-1] != 3:
        raise TypeError(
            "forget_images must be uint8 [N, H, W, 3], as returned by load_forget_images(). "
            f"Got dtype {forget_images.dtype} shape {tuple(forget_images.shape)}."
        )

    altered = make_altered_copies(forget_images)
    proxy = patch_shuffle(forget_images)
    cache: dict = {}
    same = orig_model is sub_model

    def pair(name: str, images: torch.Tensor) -> dict[str, float]:
        orig_logits = _logits(orig_model, images, cache, ("orig", name))
        if same:
            sub_logits = orig_logits
        else:
            sub_logits = _logits(sub_model, images, cache, ("sub", name))
        return _split_stats(orig_logits, sub_logits)

    report = {
        "forget": pair("forget", forget_images),
        "altered": pair("altered", altered),
        "proxy_patch_shuffle": {
            "kl_orig_sub_t2": pair("proxy", proxy)["kl_orig_sub_t2"],
        },
        "relative_l2": relative_l2(orig_model, sub_model),
        "same_module": same,
    }
    _print_report(report)
    return report


def _print_report(report: dict) -> None:
    print(f"same_module: {report['same_module']}")
    print(f"relative_l2 ||sub-orig||/||orig||: {report['relative_l2']:.6e}")
    for name in ("forget", "altered"):
        row = report[name]
        print(f"[{name}]")
        print(f"  orig mean max prob T=1: {row['mean_max_prob_t1']:.6f}")
        print(f"  orig mean max prob T=2: {row['mean_max_prob_t2']:.6f}")
        print(f"  orig mean entropy  T=2: {row['mean_entropy_t2']:.6f} nats")
        print(f"  sub  mean max prob T=1: {row['sub_mean_max_prob_t1']:.6f}")
        print(f"  sub  mean max prob T=2: {row['sub_mean_max_prob_t2']:.6f}")
        print(f"  sub  mean entropy  T=2: {row['sub_mean_entropy_t2']:.6f} nats")
        print(f"  KL(p_orig || p_sub) T=2: {row['kl_orig_sub_t2']:.6f}")
        print(f"  pseudo-CE mean/std: {row['pseudo_ce_mean']:.6f} / {row['pseudo_ce_std']:.6f}")
        print(f"  top-1 flip rate: {row['top1_flip_rate']:.6f}")
    print(f"[proxy 16x16 patch shuffle] KL(p_orig || p_sub) T=2: {report['proxy_patch_shuffle']['kl_orig_sub_t2']:.6f}")


def main() -> int:
    model = task_template.load_model()
    images = task_template.load_forget_images()
    print(
        f"baseline: sub_model is orig_model, n={len(images)} "
        f"shape={tuple(images.shape)} dtype={images.dtype}"
    )
    print("altered copies: horizontal flip, crop scale 0.9-1.0 resized to 224, color jitter 10%")
    audit_model(model, model, images)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
