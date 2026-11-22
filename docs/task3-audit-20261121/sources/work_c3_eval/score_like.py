"""Mesures c3 calquées sur les termes de la formule serveur (task_template.py), par checkpoint.

Tout en fp16 autocast et mode eval, comme le serveur. Pseudo-labels = argmax de l'origine sur
les forget propres. Pour l'origine et chaque candidat :
  forget (3 500 propres) et altéré (copies légères : flip, crop 0,9-1, luminosité/contraste ±10 %,
  graine fixe) : CE moyenne / écart-type (le serveur compare ces deux nombres à ceux des gold),
  part de top-1 changés, entropie T=2, part de la classe dominante ;
  retain : accord top-1 et 1/(1+KL(p_orig‖p_cand)) à T=2 sur patch-shuffle des forget
  (audit.patch_shuffle), passe-bas 32 et masque central 35 % (corruptions tier 1 de distill.py),
  et photos hors forget (offforget.npz) ;
  contrôles : format (eval.check_format : strict=True init_values=1e-4, float32, taille),
  logits finis en fp32 et fp16, part de top-1 changés fp32→fp16 sur les forget.
Rien de tout cela ne remplace les gold ni la validation cachée du serveur.

Usage : python score_like.py --original ../model.pt --off offforget.npz c1.pt c2.pt ...
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TASK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TASK))
import audit  # noqa: E402
import task_template as tt  # noqa: E402
from distill_mix import _lowpass, _mask  # noqa: E402  (copie figée : distill.py de Yihuan évolue)

from eval import check_format, load_state_dict, make_timm_model  # noqa: E402

T = 2.0


def mild(images: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """uint8 NHWC -> uint8 NHWC, altération légère déterministe."""
    g = torch.Generator().manual_seed(seed)
    x = images.permute(0, 3, 1, 2).float()
    n, _, h, w = x.shape
    out = []
    for i in range(n):
        im = x[i:i + 1]
        if torch.rand(1, generator=g).item() < 0.5:
            im = im.flip(3)
        s = 0.9 + 0.1 * torch.rand(1, generator=g).item()
        ch, cw = int(h * s), int(w * s)
        y0 = int(torch.randint(0, h - ch + 1, (1,), generator=g))
        x0 = int(torch.randint(0, w - cw + 1, (1,), generator=g))
        im = F.interpolate(im[:, :, y0:y0 + ch, x0:x0 + cw], size=(h, w), mode="bilinear", align_corners=False)
        b = 1 + (torch.rand(1, generator=g).item() * 2 - 1) * 0.1
        c = 1 + (torch.rand(1, generator=g).item() * 2 - 1) * 0.1
        m = im.mean()
        out.append(((im - m) * c + m) * b)
    return torch.cat(out).clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()


@torch.no_grad()
def logits(model, images: torch.Tensor, fp16: bool = True, bs: int = 128) -> torch.Tensor:
    model.eval()
    out = []
    for i in range(0, len(images), bs):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16):
            out.append(model(tt.preprocess(images[i:i + bs])).float().cpu())
    return torch.cat(out)


def kl(ref: torch.Tensor, cand: torch.Tensor) -> float:
    p = torch.softmax(ref / T, 1)
    return float((p * (torch.log_softmax(ref / T, 1) - torch.log_softmax(cand / T, 1))).sum(1).mean())


def forget_stats(lg: torch.Tensor, y: torch.Tensor) -> dict:
    ce = F.cross_entropy(lg, y, reduction="none")
    p = torch.softmax(lg / T, 1)
    return {"ce_mean": float(ce.mean()), "ce_std": float(ce.std(unbiased=False)),
            "flip": float((lg.argmax(1) != y).float().mean()),
            "ent_T2": float(-(p * p.clamp_min(1e-12).log()).sum(1).mean()),
            "top": float(torch.bincount(lg.argmax(1), minlength=lg.shape[1]).max() / len(lg))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", type=Path, required=True)
    ap.add_argument("--off", type=Path, required=True)
    ap.add_argument("candidates", type=Path, nargs="+")
    a = ap.parse_args()
    images = tt.load_forget_images(TASK / "forget_images.npz")
    views = {"forget": images, "altered": mild(images)}
    rviews = {"pshuffle": audit.patch_shuffle(images), "lowpass32": _lowpass(images, 32),
              "mask35": _mask(images, 0.35), "off": torch.from_numpy(np.load(a.off)["images"])}

    def load(path: Path):
        m = make_timm_model()
        m.load_state_dict(load_state_dict(path), strict=True)
        return m.to(tt.DEVICE)

    orig = load(a.original)
    ref = {k: logits(orig, v) for k, v in {**views, **rviews}.items()}
    y = ref["forget"].argmax(1)
    del orig
    cols = ["fmt", "fin", "fp16flip"] + [f"{v}_{s}" for v in views for s in ("ce_mean", "ce_std", "flip", "ent_T2", "top")] \
        + [f"{r}_{s}" for r in rviews for s in ("agree", "ret")]
    print("candidat\t" + "\t".join(cols), flush=True)
    seen = set()
    for c in [a.original, *a.candidates]:
        if c.resolve() in seen:
            continue
        seen.add(c.resolve())
        try:
            sd = load_state_dict(c)
            fmt = check_format(sd, make_timm_model, c)
            m = load(c)
            out = {k: logits(m, v) for k, v in {**views, **rviews}.items()}
            f32 = logits(m, images, fp16=False)
        except Exception as e:  # noqa: BLE001 — un candidat cassé ne doit pas tuer les suivants
            print(f"{c.stem}\tREJET\t{str(e).splitlines()[0][:200]}", flush=True)
            continue
        fin = bool(all(torch.isfinite(v).all() for v in out.values()) and torch.isfinite(f32).all())
        row = [str(fmt["format_ok"]), str(fin), f"{float((f32.argmax(1) != out['forget'].argmax(1)).float().mean()):.4f}"]
        for v in views:
            st = forget_stats(out[v], y)
            row += [f"{st[s]:.4f}" for s in ("ce_mean", "ce_std", "flip", "ent_T2", "top")]
        for r in rviews:
            agree = float((out[r].argmax(1) == ref[r].argmax(1)).float().mean())
            row += [f"{agree:.4f}", f"{1 / (1 + kl(ref[r], out[r])):.4f}"]
        print(c.stem + "\t" + "\t".join(row), flush=True)
        del m


if __name__ == "__main__":
    main()
