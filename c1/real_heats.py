"""Cartes de chaleur sur les vraies données (kit officiel, 3 décodeurs) + diagnostic d'incrustation.

À lancer depuis le dossier de la tâche sur JURECA (là où se trouvent wmloc/, models/, release/) :
    python work_c1/real_heats.py --split train --out output/heats_train.npz [--limit 5]

Par image (méthode et message connus) :
  - win128 / win256 : fenêtres glissantes à l'échelle d'origine, fond gris 0.5 hors fenêtre
    (`localize_sliding.heat_window`, bg="gray"), chaleur = bit accuracy moyenne des fenêtres.
  - fm_acc / fm_soft : `features(images)["per_position_logits"]` (PixelSeal, TrustMark ; cellule
    = 64 px en 512) sur 16 décalages de 0..48 px (pas 16) -> carte au pas de 16 px.
    acc = bits justes par position ; soft = moyenne de sigmoid(logit * signe du bit attendu).
    StegaStamp n'expose pas de carte spatiale : NaN.
  - diagnostic (train seulement) : bit accuracy de l'image entière, de la zone GT seule (reste gris),
    du rectangle englobant GT seul, et du complément (zone GT grisée).
Sorties en float16 128x128 dans un seul npz.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
TASK = Path.cwd()
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TASK))
from localize_sliding import heat_to_128, heat_window  # noqa: E402
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import to_tensor  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

METHODS = ("pixelseal", "trustmark", "stegastamp")


def acc_of(wm, x: torch.Tensor, msg: torch.Tensor) -> float:
    lg = torch.as_tensor(wm.logits(x)).float().cpu()[0]  # k
    return float(((lg > 0).float() == msg).float().mean())


@torch.no_grad()
def heat_featmap(wm, img: torch.Tensor, msg: torch.Tensor, step: int = 16, cell: int = 64):
    """img 1 3 512 512 -> (acc 512 512, soft 512 512). Carte par position décalée de (dy, dx)."""
    H = img.shape[-1]
    sign = (msg * 2 - 1)[None, :, None, None]  # 1 k 1 1
    n = H // cell  # 8 cellules par côté
    acc = torch.zeros(H, H); soft = torch.zeros(H, H); cnt = torch.zeros(H, H)
    for dy in range(0, cell, step):
        for dx in range(0, cell, step):
            # contenu décalé de (dy, dx) vers le bas/droite, même échelle (bord en réflexion)
            xs = F.pad(img, (dx, 0, dy, 0), mode="reflect")[..., :H, :H]  # 1 3 512 512
            per = wm.features(xs)["per_position_logits"].float().cpu()  # 1 k 8 8
            assert per.shape[-1] == n, per.shape
            a = ((per > 0).float() == msg[None, :, None, None]).float().mean(1)[0]  # 8 8
            s = torch.sigmoid(per * sign).mean(1)[0]  # 8 8
            up_a = a.repeat_interleave(cell, 0).repeat_interleave(cell, 1)  # 512 512 (repère décalé)
            up_s = s.repeat_interleave(cell, 0).repeat_interleave(cell, 1)
            # pixel (y, x) du repère décalé = pixel (y - dy, x - dx) d'origine
            acc[: H - dy, : H - dx] += up_a[dy:, dx:]
            soft[: H - dy, : H - dx] += up_s[dy:, dx:]
            cnt[: H - dy, : H - dx] += 1
    return acc / cnt.clamp(min=1), soft / cnt.clamp(min=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--data", default="release")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--wins", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--diag-only", action="store_true", help="diagnostic seulement (pas de cartes)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", dev, flush=True)

    ds = LocalisationDataset(args.data, args.split)
    idx = list(range(len(ds)))
    if args.limit:  # quelques images de chaque méthode
        per = {m: [i for i in idx if ds.meta[i]["method"] == m][: max(1, args.limit // 3 + 1)] for m in METHODS}
        idx = sorted(sum(per.values(), []))[: max(args.limit, 3)]
    wms = {m: load_watermark(m, "models", dev) for m in METHODS}

    keys = [f"win{w}" for w in args.wins] + ["fm_acc", "fm_soft"]
    heats = {k: np.full((len(idx), 128, 128), np.nan, np.float16) for k in keys}
    diag = {k: np.full(len(idx), np.nan, np.float32) for k in ("acc_full", "acc_gt", "acc_bbox", "acc_crop", "acc_out")}
    names, methods, fullwm = [], [], []
    t0 = time.time(); tm = {m: 0.0 for m in METHODS}
    for j, i in enumerate(idx):
        it = ds[i]; m = it["method"]; wm = wms[m]
        names.append(it["image_name"]); methods.append(m); fullwm.append(bool(it.get("full_watermark", False)))
        img = to_tensor(it["image"]).to(dev)  # 1 3 512 512
        msg = torch.from_numpy(it["message"]).float()  # k
        t1 = time.time()
        diag["acc_full"][j] = acc_of(wm, img, msg)
        if "mask" in it:
            g = torch.from_numpy(it["mask"]).float().to(dev)[None, None]  # 1 1 512 512
            gray = torch.full_like(img, 0.5)
            diag["acc_gt"][j] = acc_of(wm, gray + (img - gray) * g, msg)
            diag["acc_out"][j] = acc_of(wm, img + (gray - img) * g, msg)
            ys, xs = np.nonzero(it["mask"])
            if len(ys):
                b = torch.zeros_like(g); b[..., ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 1
                diag["acc_bbox"][j] = acc_of(wm, gray + (img - gray) * b, msg)
                # rectangle englobant recadré tel quel (le wrapper le redimensionne à sa résolution)
                diag["acc_crop"][j] = acc_of(wm, img[..., ys.min():ys.max() + 1, xs.min():xs.max() + 1], msg)
        if args.diag_only:
            tm[m] += time.time() - t1
            continue
        for w in args.wins:
            n_pos = 13 if w <= 128 else 9
            h = heat_window(wm, img, msg, win=w, n_pos=n_pos, bs=32, batched=True, bg="gray")
            heats[f"win{w}"][j] = heat_to_128(h)
        if m != "stegastamp":
            a, s = heat_featmap(wm, img, msg)
            heats["fm_acc"][j] = heat_to_128(a); heats["fm_soft"][j] = heat_to_128(s)
        tm[m] += time.time() - t1
        if j < 5 or j % 50 == 0:
            print(f"[{j + 1}/{len(idx)}] {m} full={fullwm[-1]} acc_full={diag['acc_full'][j]:.3f} "
                  f"acc_gt={diag['acc_gt'][j]:.3f} acc_crop={diag['acc_crop'][j]:.3f} acc_bbox={diag['acc_bbox'][j]:.3f} acc_out={diag['acc_out'][j]:.3f} "
                  f"{time.time() - t0:.0f}s", flush=True)
    cnt = {m: methods.count(m) for m in METHODS}
    print("temps par image :", {m: round(tm[m] / max(cnt[m], 1), 2) for m in METHODS}, "total", round(time.time() - t0), "s")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, names=np.array(names), method=np.array(methods), full_watermark=np.array(fullwm),
                        **heats, **diag)
    print("->", args.out, flush=True)


if __name__ == "__main__":
    main()
