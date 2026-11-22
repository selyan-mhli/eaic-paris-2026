"""Canaux fins pour la tête F (trous = zones restaurées au propre), cartes 128x128 float16 par image.

  hfe   : moyenne locale de |Y - flou gaussien σ=1| (énergie haute fréquence)
  srm   : moyenne locale de |SRM KV 5x5 (Y)|
  jnd   : carte JND de PixelSeal (attenuation.heatmaps du kit) sur l'image observée
  ncc_m : corrélation normalisée locale (fenêtre 16 px à 512) entre le passe-haut de Y(x') et le résidu
          ré-embarqué r = Y(embed(x', m)) - Y(x'), m = message connu (méthode de l'image)
  ncc_r : idem avec un message aléatoire m2 (témoin : part de corrélation due au contenu)
  ncc_d : ncc_m - ncc_r
Idée (A1 = recollé) : hors trous, x' = x0 + w(x0) ; r ≈ w(x') ≈ w(x0) est corrélé au résidu présent ; dans
les trous (propres), seule la part liée au contenu reste, que ncc_r estime.

  srun python work_segf/hf_maps.py --split train --out output/segf_hf_train   (une part par tâche Slurm)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

KIT = Path("/p/scratch/training2666/mouhali1/SLZ/Hackathon_Watermark_Localisation_Task")
sys.path.insert(0, str(KIT))
from wmloc.watermarks import load_watermark  # noqa: E402

SRM_KV = torch.tensor([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
                       [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=torch.float32) / 12


def luma(x: torch.Tensor) -> torch.Tensor:  # B 3 H W -> B 1 H W
    return 0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def blur(y: torch.Tensor, s: float) -> torch.Tensor:
    r = int(3 * s + 0.5); k = torch.exp(-torch.arange(-r, r + 1, device=y.device).float() ** 2 / (2 * s * s)); k = k / k.sum()
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(y, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))


def box(t: torch.Tensor, w: int = 16) -> torch.Tensor:
    return F.avg_pool2d(F.pad(t, (w // 2, w - 1 - w // 2, w // 2, w - 1 - w // 2), mode="reflect"), w, 1)


def to128(t: torch.Tensor) -> np.ndarray:  # 1 1 512 512 -> 128 128
    return F.avg_pool2d(t, 4)[0, 0].float().cpu().numpy().astype(np.float16)


def ncc(hp: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    r = r - blur(r, 2.0)
    return box(hp * r) / torch.sqrt(box(hp * hp) * box(r * r) + 1e-12)


class Embedder:
    """Résidu d'embarquement en flottant quand c'est possible (PixelSeal, StegaStamp), uint8 sinon (TrustMark)."""

    def __init__(self, wm, method):
        self.wm, self.method = wm, method

    @torch.no_grad()
    def __call__(self, img_u8: np.ndarray, bits: np.ndarray, x: torch.Tensor) -> torch.Tensor:
        """-> image watermarquée float B=1 3 512 512 (device de x)."""
        if self.method == "pixelseal":
            msgs = torch.tensor(bits[None], dtype=torch.float32, device=x.device)
            return self.wm.model.embed(x, msgs=msgs, is_video=False)["imgs_w"].clamp(0, 1)
        if self.method == "stegastamp":
            h, w = img_u8.shape[:2]
            small = np.asarray(Image.fromarray(img_u8).resize((400, 400), Image.Resampling.BICUBIC), np.float32) / 255
            res = self.wm._run(small[None], ["residual"], np.asarray([bits], np.float32))[0][0] * 255
            res = res - res.mean(axis=(0, 1), keepdims=True)
            full = np.stack([np.asarray(Image.fromarray(res[:, :, c], mode="F").resize((w, h), Image.Resampling.BILINEAR))
                             for c in range(3)], 2)
            return (x + torch.from_numpy(full / 255).permute(2, 0, 1)[None].float().to(x.device)).clamp(0, 1)
        out = self.wm.embed(img_u8, bits)
        return torch.from_numpy(out).permute(2, 0, 1)[None].float().div(255).to(x.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train"); ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--only-f", action="store_true")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    rank = int(os.environ.get("SLURM_PROCID", 0)); world = int(os.environ.get("SLURM_NTASKS", 1))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.loads((KIT / "release" / a.split / "metadata.json").read_text())
    if a.only_f:
        meta = [m for m in meta if m.get("full_watermark")]
    meta = meta[rank::world]
    if a.limit:
        meta = meta[:a.limit]
    rng = np.random.default_rng(7 + rank)
    srm = SRM_KV.view(1, 1, 5, 5).to(dev)
    keys = ("hfe", "srm", "jnd", "ncc_m", "ncc_r", "ncc_d")
    res = {k: [] for k in keys}; names, methods = [], []
    t0 = time.time()
    ps = load_watermark("pixelseal", KIT / "models", dev)  # pour la JND (et l'embarquement PixelSeal)
    for meth in ("pixelseal", "trustmark", "stegastamp"):
        sub = [m for m in meta if m["method"] == meth]
        if not sub:
            continue
        wm = ps if meth == "pixelseal" else load_watermark(meth, KIT / "models", dev)
        emb = Embedder(wm, meth)
        for j, m in enumerate(sub):
            img = np.asarray(Image.open(KIT / "release" / a.split / "images" / f"{m['image_name']}.png").convert("RGB"))
            x = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255).to(dev)
            y = luma(x); hp = y - blur(y, 2.0)
            res["hfe"].append(to128((y - blur(y, 1.0)).abs()))
            res["srm"].append(to128(F.conv2d(F.pad(y, (2, 2, 2, 2), mode="reflect"), srm).abs()))
            res["jnd"].append(to128(ps.model.attenuation.heatmaps(x).mean(1, keepdim=True)))
            bits = np.array([int(c) for c in m["message"]], np.uint8)
            m2 = rng.integers(0, 2, len(bits)).astype(np.uint8)
            cm = ncc(hp, luma(emb(img, bits, x)) - y); cr = ncc(hp, luma(emb(img, m2, x)) - y)
            res["ncc_m"].append(to128(cm)); res["ncc_r"].append(to128(cr)); res["ncc_d"].append(to128(cm - cr))
            names.append(m["image_name"]); methods.append(meth)
            if j < 2 or j % 40 == 0:
                print(f"[{rank}] {meth} {j}/{len(sub)} ncc_m {float(cm.mean()):.3f} ncc_r {float(cr.mean()):.3f} "
                      f"{time.time() - t0:.0f}s", flush=True)
        if wm is not ps:
            del wm
        torch.cuda.empty_cache()
    out = f"{a.out}_{rank}.npz"
    np.savez_compressed(out, names=np.array(names), method=np.array(methods), **{k: np.stack(v) for k, v in res.items()})
    print(f"[{rank}] écrit {out} ({len(names)}) en {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
