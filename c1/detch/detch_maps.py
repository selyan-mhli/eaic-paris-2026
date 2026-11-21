"""Canal 0 (« détection ») du décodeur PixelSeal du kit, par position, sur train ou test.

Sorties (npz, une ligne par image PixelSeal ; les autres méthodes n'ont pas de canal de détection) :
  det256 : canal 0 par position, entrée redimensionnée à 256 (résolution de travail) -> 8x8
  det512 : canal 0 par position, entrée 512 native (ConvNeXt entièrement convolutif) -> 16x16
  agr256 : accord bit à bit par position avec le message connu (référence), 8x8
  lumvar : variance locale de luminance Y (fenêtre 16 px à 512) remontée en 128x128
Toutes les cartes sont aussi rendues en 128x128 (bilinéaire) sous le suffixe _128.

Usage : python work_detch/detch_maps.py --split train --out output/detch_maps_train.npz [--limit 10] [--device cpu]
"""
from __future__ import annotations

import argparse
import json
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


def per_pos_all(det, x: torch.Tensor) -> torch.Tensor:
    """x : B 3 H W dans [0,1] (déjà à la résolution voulue) -> B 257 h w (canal 0 + 256 bits)."""
    lat = det.convnext(det.preprocess(x))                     # B 768 H/32 W/32
    emb = det.pixel_decoder.output_upscaling(lat)             # B 768 H/32 W/32 (upscale_stages=[1])
    lin = det.pixel_decoder.linear
    return torch.einsum("bchw,oc->bohw", emb, lin.weight) + lin.bias[None, :, None, None]  # B 257 h w


def up128(m: torch.Tensor) -> np.ndarray:
    """m : B h w -> B 128 128 (bilinéaire)."""
    return F.interpolate(m[:, None].float(), size=(128, 128), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    meta = [m for m in json.load(open(KIT / "release" / a.split / "metadata.json")) if m["method"] == "pixelseal"]
    if a.limit:
        meta = meta[:a.limit]
    wm = load_watermark("pixelseal", KIT / "models", a.device)
    det = wm.det
    out = {k: [] for k in ("det256", "det512", "agr256", "det256_128", "det512_128", "agr256_128", "lumvar_128",
                           "logit_global0")}
    t0 = time.time()
    for i, m in enumerate(meta):
        img = np.asarray(Image.open(KIT / "release" / a.split / "images" / f"{m['image_name']}.png").convert("RGB"))
        x = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255).to(a.device)  # 1 3 512 512
        msg = torch.tensor([int(c) for c in m["message"]], device=a.device).float()      # 256
        p256 = per_pos_all(det, wm._resize(x))                     # 1 257 8 8
        p512 = per_pos_all(det, x)                                 # 1 257 16 16
        d256, d512 = p256[:, 0], p512[:, 0]                        # 1 8 8 / 1 16 16
        agr = ((p256[:, 1:] > 0).float() == msg[None, :, None, None]).float().mean(1)  # 1 8 8
        g0 = det(wm._resize(x))[:, 0]                              # 1 : canal 0 global (après mean-pool)
        y = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])[:, None]  # 1 1 512 512
        mu = F.avg_pool2d(y, 16, 4, padding=6)                     # 1 1 128 128 (approx.)
        var = (F.avg_pool2d(y * y, 16, 4, padding=6) - mu * mu).clamp_min(0)[:, 0]  # 1 128 128
        var = F.interpolate(var[:, None], size=(128, 128), mode="bilinear", align_corners=False)[:, 0]
        out["det256"].append(d256[0].cpu().numpy()); out["det512"].append(d512[0].cpu().numpy())
        out["agr256"].append(agr[0].cpu().numpy())
        out["det256_128"].append(up128(d256)[0]); out["det512_128"].append(up128(d512)[0])
        out["agr256_128"].append(up128(agr)[0]); out["lumvar_128"].append(var[0].cpu().numpy())
        out["logit_global0"].append(float(g0[0]))
        if i < 2 or i % 50 == 0:
            print(i, m["image_name"], "det256", d256.shape, float(d256.mean()), "det512", d512.shape,
                  "agr", float(agr.mean()), f"{time.time() - t0:.1f}s", flush=True)
    np.savez_compressed(a.out, names=np.array([m["image_name"] for m in meta]),
                        full=np.array([m["full_watermark"] for m in meta]),
                        **{k: np.asarray(v, np.float16 if k.endswith("_128") else np.float32) for k, v in out.items()})
    print("ok", a.out, len(meta), f"{time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
