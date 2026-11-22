"""NPO unlearning with a T=2 patch-shuffle retain anchor.

VERSION CORRIGÉE (docs/task3-work-20261121, copie de sources/unlearn_v2.py) : le log-ratio NPO
compare élève et référence sur LE MÊME tenseur augmenté (`same_view_log_ratio`). Dans l'original,
la référence venait des images propres (ligne 272) et l'élève voyait la vue augmentée (ligne 316),
si bien qu'avec des poids identiques le ratio n'était pas nul. Seul ce point change ; la sélection
de checkpoints (_consider_best) est conservée telle quelle et reste à revoir (cf. audit P1).
Coût : un forward de la référence en plus par pas (sans gradient).

Does not modify task_template.py or audit.py. Clean-image T=2 entropy,
pseudo-CE and flip rate come from audit.audit_model. Orig logits used by
the audit are cached here so repeated step checks do not re-forward the
frozen reference model.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

try:  # absents hors JURECA ; le test CPU n'utilise que same_view_log_ratio et _npo
    import audit
    import task_template
except ImportError:  # pragma: no cover
    audit = task_template = None

OUTPUT_DIR = Path("output")
LEVELS = (
    (0.3, "v2_ent03.pt"),
    (0.6, "v2_ent06.pt"),
    (0.9, "v2_ent09.pt"),
    (1.2, "v2_ent12.pt"),
)
TEMPERATURE = 2.0


def _is_frozen(name: str) -> bool:
    return name == "pos_embed" or name.startswith("patch_embed")


def _with_tag(filename: str, tag: str) -> str:
    if not tag:
        return filename
    if any(char in tag for char in "/\\"):
        raise ValueError(f"tag must be a filename suffix, got {tag!r}")
    suffix = tag if tag.startswith("_") else f"_{tag}"
    stem, ext = filename.rsplit(".", 1)
    return f"{stem}{suffix}.{ext}"


def _batch_resized_crop(x: torch.Tensor, scale: tuple[float, float] = (0.5, 1.0)) -> torch.Tensor:
    """Per-image random crop, one grid_sample for the whole batch. x is BCHW float."""
    n, _, h, w = x.shape
    device = x.device
    ratio = (3.0 / 4.0, 4.0 / 3.0)
    log_r0, log_r1 = math.log(ratio[0]), math.log(ratio[1])
    area = float(h * w)
    crop_h = torch.full((n,), float(h), device=device)
    crop_w = torch.full((n,), float(w), device=device)
    pending = torch.ones(n, dtype=torch.bool, device=device)
    for _ in range(10):
        if not bool(pending.any()):
            break
        idx = pending.nonzero(as_tuple=False).flatten()
        target = torch.empty(idx.numel(), device=device).uniform_(scale[0], scale[1]) * area
        aspect = torch.empty(idx.numel(), device=device).uniform_(log_r0, log_r1).exp()
        hh = (target / aspect).sqrt().round()
        ww = (target * aspect).sqrt().round()
        good = (hh >= 1) & (ww >= 1) & (hh <= h) & (ww <= w)
        crop_h[idx[good]] = hh[good]
        crop_w[idx[good]] = ww[good]
        pending[idx[good]] = False
    top = torch.rand(n, device=device) * (h - crop_h).clamp(min=0)
    left = torch.rand(n, device=device) * (w - crop_w).clamp(min=0)
    ys = (torch.arange(h, device=device) + 0.5) / h
    xs = (torch.arange(w, device=device) + 0.5) / w
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    y_pix = top[:, None, None] + gy[None] * crop_h[:, None, None]
    x_pix = left[:, None, None] + gx[None] * crop_w[:, None, None]
    y_norm = (y_pix + 0.5) / h * 2 - 1
    x_norm = (x_pix + 0.5) / w * 2 - 1
    grid = torch.stack([x_norm, y_norm], dim=-1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def _batch_hflip(x: torch.Tensor) -> torch.Tensor:
    flip = torch.rand(x.shape[0], device=x.device) < 0.5
    out = x.clone()
    out[flip] = x[flip].flip(-1)
    return out


def _shift_hue(x: torch.Tensor, hue: torch.Tensor) -> torch.Tensor:
    """x is BCHW in [0, 1]. hue is [B] in about [-0.5, 0.5]."""
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    maxc, max_idx = torch.max(x, dim=1)
    minc = torch.min(x, dim=1).values
    delta = maxc - minc
    safe = delta.clamp_min(1e-8)
    sector = torch.zeros_like(maxc)
    sector = torch.where(max_idx == 0, ((g - b) / safe) % 6, sector)
    sector = torch.where(max_idx == 1, (b - r) / safe + 2, sector)
    sector = torch.where(max_idx == 2, (r - g) / safe + 4, sector)
    hue_img = torch.where(delta > 1e-6, (sector / 6.0) % 1.0, torch.zeros_like(sector))
    hue_img = (hue_img + hue[:, None, None]) % 1.0
    sat = torch.where(maxc > 0, delta / maxc.clamp_min(1e-8), torch.zeros_like(maxc))
    val = maxc
    h6 = hue_img * 6.0
    sector_i = torch.floor(h6).long() % 6
    frac = h6 - torch.floor(h6)
    p = val * (1.0 - sat)
    q = val * (1.0 - sat * frac)
    t = val * (1.0 - sat * (1.0 - frac))
    out_r = torch.zeros_like(val)
    out_g = torch.zeros_like(val)
    out_b = torch.zeros_like(val)
    for index, (rr, gg, bb) in enumerate(((val, t, p), (q, val, p), (p, val, t), (p, q, val), (t, p, val), (val, p, q))):
        mask = sector_i == index
        out_r = torch.where(mask, rr, out_r)
        out_g = torch.where(mask, gg, out_g)
        out_b = torch.where(mask, bb, out_b)
    return torch.stack([out_r, out_g, out_b], dim=1)


def _batch_color_jitter(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    device = x.device
    brightness = 1 + (torch.rand(n, 1, 1, 1, device=device) * 2 - 1) * 0.3
    contrast = 1 + (torch.rand(n, 1, 1, 1, device=device) * 2 - 1) * 0.3
    saturation = 1 + (torch.rand(n, 1, 1, 1, device=device) * 2 - 1) * 0.3
    hue = (torch.rand(n, device=device) * 2 - 1) * 0.1
    x = x * brightness
    mean = x.mean(dim=(2, 3), keepdim=True)
    x = (x - mean) * contrast + mean
    gray = (x * x.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
    x = (gray + saturation * (x - gray)).clamp(0, 255)
    return _shift_hue(x / 255.0, hue) * 255.0


def _batch_erase(x: torch.Tensor, p: float = 0.3) -> torch.Tensor:
    n, _, h, w = x.shape
    device = x.device
    apply = torch.rand(n, device=device) < p
    area = float(h * w)
    target = torch.empty(n, device=device).uniform_(0.02, 0.33) * area
    aspect = torch.empty(n, device=device).uniform_(math.log(0.3), math.log(3.3)).exp()
    eh = (target * aspect).sqrt().round()
    ew = (target / aspect).sqrt().round()
    valid = (eh >= 1) & (ew >= 1) & (eh < h) & (ew < w)
    apply = apply & valid
    eh = eh.clamp(1, h - 1)
    ew = ew.clamp(1, w - 1)
    top = (torch.rand(n, device=device) * (h - eh)).floor()
    left = (torch.rand(n, device=device) * (w - ew)).floor()
    ys = torch.arange(h, device=device)
    xs = torch.arange(w, device=device)
    mask = (
        (ys[None, :, None] >= top[:, None, None])
        & (ys[None, :, None] < (top + eh)[:, None, None])
        & (xs[None, None, :] >= left[:, None, None])
        & (xs[None, None, :] < (left + ew)[:, None, None])
        & apply[:, None, None]
    )
    return x.masked_fill(mask[:, None], 0.0)


def _augment_batch(images_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 NHWC batch -> float NCHW in [-1, 1], cropped and jittered on GPU."""
    x = images_uint8.to(task_template.DEVICE, non_blocking=True)
    x = x.permute(0, 3, 1, 2).contiguous().float()
    x = _batch_resized_crop(x)
    x = _batch_hflip(x)
    x = _batch_color_jitter(x)
    x = _batch_erase(x)
    x = x.clamp(0, 255) / 255.0
    return (x - 0.5) / 0.5


def _install_orig_logit_cache(orig: torch.nn.Module, forget_logits: torch.Tensor) -> None:
    """Reuse frozen orig logits across audit_model calls. Student forwards stay live."""
    cache: dict[tuple, torch.Tensor] = {("orig", "forget"): forget_logits}
    real_logits = audit._logits

    def cached(model, images, local_cache, key):
        if key[0] == "orig" and model is orig:
            if key not in cache:
                cache[key] = task_template.predict(model, images)
            return cache[key]
        return real_logits(model, images, local_cache, key)

    audit._logits = cached


def _anchor(model: torch.nn.Module, orig_state: dict[str, torch.Tensor]) -> torch.Tensor:
    total = None
    for name, param in model.named_parameters():
        term = (param.float() - orig_state[name]).square().sum()
        total = term if total is None else total + term
    return total


def same_view_log_ratio(student: torch.nn.Module, reference: torch.nn.Module,
                        x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """log p_student(y|x) et log p_ref(y|x) sur le MÊME tenseur x [B, 3, H, W] ; y [B].

    Retourne (log_p_sub_y [B] avec gradient, log_p_ref_y [B] sans gradient). La référence
    doit être en mode eval (pas de dropout). Autocast fp16 seulement sur CUDA."""
    on_cuda = x.is_cuda
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=on_cuda):
        logits_sub = student(x)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=on_cuda):
        logits_ref = reference(x)
    log_p_sub_y = F.log_softmax(logits_sub.float(), dim=1).gather(1, y[:, None]).squeeze(1)  # B C -> B
    log_p_ref_y = F.log_softmax(logits_ref.float(), dim=1).gather(1, y[:, None]).squeeze(1)  # B C -> B
    return log_p_sub_y, log_p_ref_y


def _npo(log_p_sub_y: torch.Tensor, log_p_orig_y: torch.Tensor, beta: float) -> torch.Tensor:
    delta = log_p_sub_y - log_p_orig_y
    return -(2.0 / beta) * F.logsigmoid(-beta * delta).mean()


def _retain_proxy(sub_logits: torch.Tensor, orig_logits: torch.Tensor) -> torch.Tensor:
    """KL(p_orig || p_sub) at T=2, scaled by T^2, as specified."""
    log_q = F.log_softmax(sub_logits.float() / TEMPERATURE, dim=-1)
    p = F.softmax(orig_logits.float() / TEMPERATURE, dim=-1)
    return F.kl_div(log_q, p, reduction="batchmean") * (TEMPERATURE ** 2)


def _save_state(model: torch.nn.Module, path: Path) -> None:
    state = {key: value.detach().cpu().float() for key, value in model.state_dict().items()}
    for key, value in state.items():
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise RuntimeError(f"non-finite or non-float32 weights in {key}")
    probe = task_template.build_model()
    probe.load_state_dict(state, strict=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _maybe_save(model: torch.nn.Module, entropy: float, saved: set[float], tag: str) -> None:
    for level, name in LEVELS:
        if level in saved or entropy < level:
            continue
        path = OUTPUT_DIR / _with_tag(name, tag)
        _save_state(model, path)
        saved.add(level)
        print(f"saved {path} at clean T=2 entropy {entropy:.4f}", flush=True)


def _consider_best(model: torch.nn.Module, row: dict, proxy: float, best: dict, tag: str) -> None:
    """Keep the strongest forget point that is still inside the target box."""
    ce = row["pseudo_ce_mean"]
    flip = row["top1_flip_rate"]
    if proxy > 0.15 or not (0.5 <= ce <= 1.5) or not (0.15 <= flip <= 0.30):
        return
    if flip <= best.get("flip", -1.0):
        return
    path = OUTPUT_DIR / _with_tag("v2_best.pt", tag)
    _save_state(model, path)
    best.update(flip=flip, ce=ce, proxy=proxy, entropy=row["sub_mean_entropy_t2"])
    print(
        f"saved {path} ce={ce:.4f} flip={flip:.4f} proxy_KL={proxy:.4f}",
        flush=True,
    )


def _evaluate(orig, student, images, step: int, saved: set[float], best: dict, tag: str) -> dict:
    student.eval()
    report = audit.audit_model(orig, student, images)
    row = report["forget"]
    entropy = row["sub_mean_entropy_t2"]
    proxy = report["proxy_patch_shuffle"]["kl_orig_sub_t2"]
    print(
        f"step {step}: clean_T2_entropy={entropy:.4f} "
        f"pseudo_ce={row['pseudo_ce_mean']:.4f}/{row['pseudo_ce_std']:.4f} "
        f"flip={row['top1_flip_rate']:.4f} proxy_KL={proxy:.4f}",
        flush=True,
    )
    _maybe_save(student, entropy, saved, tag)
    _consider_best(student, row, proxy, best, tag)
    student.train()
    return row


def train(args: argparse.Namespace) -> None:
    orig = task_template.load_model()
    orig.requires_grad_(False)
    images = task_template.load_forget_images()
    with torch.no_grad():
        orig_logits = task_template.predict(orig, images)
    pseudo = orig_logits.argmax(dim=1).cpu()  # pseudo-labels sur les forget propres (inchangé)
    _install_orig_logit_cache(orig, orig_logits)

    student = task_template.build_model()
    student.load_state_dict(orig.state_dict())
    student.to(task_template.DEVICE)
    orig_params = {name: param.detach().float().clone() for name, param in student.named_parameters()}

    trainable = []
    frozen = 0
    for name, param in student.named_parameters():
        param.requires_grad_(not _is_frozen(name))
        if param.requires_grad:
            trainable.append(param)
        else:
            frozen += param.numel()
    n_train = sum(param.numel() for param in trainable)
    print(
        f"tag={args.tag!r} lr={args.lr:g} proxy_weight={args.proxy_weight:g} epochs={args.epochs} "
        f"trainable {n_train} params (blocks + head + cls), frozen embeddings {frozen}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    images_gpu = images.to(task_template.DEVICE, non_blocking=True)

    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    saved: set[float] = set()
    best: dict = {}
    order = torch.arange(len(images))
    step = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        student.train()
        order = order[torch.randperm(len(order))]
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            raw = images[index]
            batch = _augment_batch(images_gpu[index])
            proxy = task_template.preprocess(audit.patch_shuffle(raw))
            y = pseudo[index].to(task_template.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            # Correctif : référence et élève voient le même tenseur augmenté `batch`.
            log_p_sub_y, ref_log_y = same_view_log_ratio(student, orig, batch, y)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits_proxy = student(proxy)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                logits_orig_proxy = orig(proxy)
            loss_npo = _npo(log_p_sub_y, ref_log_y, args.beta)
            loss_proxy = _retain_proxy(logits_proxy, logits_orig_proxy)
            anchor = _anchor(student, orig_params)
            loss = loss_npo + args.proxy_weight * loss_proxy + args.anchor_weight * anchor
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step + 1}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if step % args.eval_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"epoch {epoch} step {step}: npo={float(loss_npo.detach()):.6f} "
                    f"proxy={float(loss_proxy.detach()):.6f} anchor={float(anchor.detach()):.6f} "
                    f"delta={(log_p_sub_y.detach() - ref_log_y).mean().item():.4f} "
                    f"{step / elapsed:.2f} step/s",
                    flush=True,
                )
                row = _evaluate(orig, student, images, step, saved, best, args.tag)
                if all(level in saved for level, _ in LEVELS):
                    print("all entropy checkpoints saved")
                    return
                # NPO suppresses the top-1 without flattening the other 99 classes,
                # so clean T=2 entropy plateaus near 0.35 while pseudo-CE keeps climbing.
                if best and row["pseudo_ce_mean"] > 1.5 and row["sub_mean_entropy_t2"] < 0.55:
                    print(
                        "entropy stayed under 0.55 after pseudo-CE left 1.5; "
                        f"{_with_tag('v2_best.pt', args.tag)} is the in-band submit. stopping.",
                        flush=True,
                    )
                    return
    if not saved:
        print("no entropy threshold was crossed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--proxy-weight", type=float, default=2.0)
    parser.add_argument("--anchor-weight", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--tag", type=str, default="", help="suffix added to checkpoint filenames")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        train(parse_args())
    except KeyboardInterrupt:
        sys.exit(130)
