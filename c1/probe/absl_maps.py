"""A2 — canal |logit| par position, sans le message (astuce de l'issue videoseal #36 : les bits sûrs ont de
grands |logits|).

Même schéma que `real_heats.heat_featmap` (fm_acc) : `features(x)["per_position_logits"]` (cellule 64 px en
512) sur 16 décalages de 0..48 px (pas 16), moyenne par pixel, rendue en 128x128 (avg_pool 4).
fm_absl = moyenne sur les bits de |logit| par position. Calculée pour PixelSeal et TrustMark ; StegaStamp n'a
pas de carte spatiale (NaN), comme fm_*.

Sortie : npz `names, method, full_watermark (train), fm_absl` (float16 N 128 128), dans l'ordre des métadonnées.
    python work_c1_probe/absl_maps.py --split train --out output/probe_absl_train.npz [--limit 6]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path.cwd()))
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import to_tensor  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402


@torch.no_grad()
def heat_absl(wm, img: torch.Tensor, step: int = 16, cell: int = 64) -> torch.Tensor:
    """img 1 3 512 512 -> 512 512 : moyenne sur les bits de |logit| par position, 16 décalages."""
    H = img.shape[-1]; n = H // cell
    acc = torch.zeros(H, H); cnt = torch.zeros(H, H)
    shifts = [(dy, dx) for dy in range(0, cell, step) for dx in range(0, cell, step)]
    xs = torch.cat([F.pad(img, (dx, 0, dy, 0), mode="reflect")[..., :H, :H] for dy, dx in shifts])  # 16 3 H H
    per = wm.features(xs)["per_position_logits"].float().cpu()  # 16 k 8 8
    assert per.shape[-1] == n, per.shape
    s = per.abs().mean(1)  # 16 8 8
    for (dy, dx), sk in zip(shifts, s):
        up = sk.repeat_interleave(cell, 0).repeat_interleave(cell, 1)
        acc[: H - dy, : H - dx] += up[dy:, dx:]
        cnt[: H - dy, : H - dx] += 1
    return acc / cnt.clamp(min=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--data", default="release")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = LocalisationDataset(a.data, a.split)
    idx = list(range(len(ds)))
    if a.limit:
        per = [[i for i in idx if ds.meta[i]["method"] == m][: a.limit // 3 + 1] for m in ("pixelseal", "trustmark", "stegastamp")]
        idx = sorted(sum(per, []))
    wms = {m: load_watermark(m, "models", dev) for m in ("pixelseal", "trustmark")}
    heat = np.full((len(idx), 128, 128), np.nan, np.float16)
    names, methods, full = [], [], []
    t0 = time.time()
    for j, i in enumerate(idx):
        m = ds.meta[i]; names.append(m["image_name"]); methods.append(m["method"]); full.append(bool(m.get("full_watermark", False)))
        if m["method"] in wms:
            it = ds[i]
            h = heat_absl(wms[m["method"]], to_tensor(it["image"]).to(dev))
            heat[j] = F.avg_pool2d(h[None, None], 4)[0, 0].numpy()
        if j < 3 or j % 100 == 0:
            print(f"[{j + 1}/{len(idx)}] {m['method']} mean={np.nanmean(heat[j].astype(np.float32)) if m['method'] in wms else float('nan'):.3f} "
                  f"{time.time() - t0:.0f}s", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, names=np.array(names), method=np.array(methods), full_watermark=np.array(full), fm_absl=heat)
    print("->", a.out, len(idx), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
