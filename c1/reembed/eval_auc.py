"""AUC pixel des cartes re_* contre la vérité « trou » sur les F du train, par méthode (CPU).

    python work_c1/reembed/eval_auc.py --re 'output/re_pilot*.npz' [--data release] [--heats output/heats_train.npz]

Vérité : masks_128 du kit (1 = watermarqué, 0 = trou édité). Une carte forte là où le watermark survit
a une AUC > 0.5 pour « watermarqué » ; l'AUC « trou » vaut 1 - celle-ci (score = -carte).
Affiche par méthode, sur les F calculées : AUC poolée, médiane des AUC par image, et un mIoU F honnête :
seuil sur la carte min-max par image choisi sur une moitié (graine fixe) et appliqué à l'autre, puis
l'inverse ; comparé à « tout 1 ». --heats ajoute fm_acc/fm_soft/win128 sur les mêmes images (référence).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # c1/ (ou work_c1/)
from metric import miou_per_image  # noqa: E402

METHODS = ("pixelseal", "trustmark", "stegastamp")
THRS = np.round(np.arange(0.0, 1.02, 0.02), 2)


def auc(score, label) -> float:
    """AUC de Mann-Whitney (rangs, ex aequo moyennés)."""
    s = np.asarray(score, np.float64).ravel(); y = np.asarray(label, bool).ravel()
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2)[inv]
    npos = y.sum(); nneg = len(y) - npos
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else float("nan")


def rel(h: np.ndarray) -> np.ndarray:
    h = h.astype(np.float32); f = h.reshape(len(h), -1)
    lo = f.min(1)[:, None, None]; hi = f.max(1)[:, None, None]
    return (h - lo) / np.maximum(hi - lo, 1e-6)


def miou_2fold(h: np.ndarray, gt: np.ndarray, seed: int = 0) -> float:
    """Seuil (sur rel(h)) choisi sur une moitié, appliqué à l'autre ; moyenne des deux sens."""
    n = len(h); p = np.random.default_rng(seed).permutation(n); A, B = p[: n // 2], p[n // 2:]
    H = rel(h); tot = np.zeros(n)
    for fit, ev in ((A, B), (B, A)):
        if not len(fit) or not len(ev):
            continue
        sc = [miou_per_image(H[fit] >= t, gt[fit]).mean() for t in THRS]
        t = THRS[int(np.argmax(sc))]
        tot[ev] = miou_per_image(H[ev] >= t, gt[ev])
    return float(tot.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", required=True); ap.add_argument("--data", default="release")
    ap.add_argument("--heats", type=Path); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    meta = {x["image_name"]: x for x in json.loads((Path(a.data) / "train" / "metadata.json").read_text())}
    mk = np.load(Path(a.data) / "train" / "masks_128.npz"); gt_all = dict(zip(mk["names"].tolist(), mk["masks"]))
    maps: dict[str, dict[str, np.ndarray]] = {}
    for f in sorted(glob.glob(a.re)):
        z = np.load(f)
        for j, n in enumerate(z["names"].tolist()):
            if z["done"][j]:
                maps[n] = {k: z[k][j].astype(np.float32) for k in ("re_z", "re_zg", "re_c")}
    assert maps, a.re
    ref = {}
    if a.heats:
        h = np.load(a.heats); hi = {n: i for i, n in enumerate(h["names"].tolist())}
        for k in ("fm_acc", "fm_soft", "win128"):
            if k in h.files:
                ref[k] = (h[k], hi)
    print(f"{len(maps)} cartes ; AUC pixel 'watermarqué' (AUC 'trou' = 1 - x) sur les F du train")
    for m in METHODS:
        ns = [n for n in maps if meta[n]["method"] == m and meta[n].get("full_watermark", False)]
        if not ns:
            continue
        gt = np.stack([gt_all[n] for n in ns]).astype(bool)  # n 128 128
        print(f"\n{m} : {len(ns)} F, fraction trou {1 - gt.mean():.3f} ; mIoU tout 1 = {miou_per_image(np.ones_like(gt), gt).mean():.4f}")
        cand = {k: np.stack([maps[n][k] for n in ns]) for k in ("re_z", "re_zg", "re_c")}
        for k, (arr, hi) in ref.items():
            v = np.stack([arr[hi[n]].astype(np.float32) for n in ns])
            if np.isfinite(v).all():
                cand[k] = v
        for k, v in cand.items():
            per = [auc(v[i], gt[i]) for i in range(len(ns))]
            print(f"  {k:8s} AUC poolée {auc(v, gt):.3f} | médiane par image {np.nanmedian(per):.3f} | "
                  f"mIoU F 2-fold (rel) {miou_2fold(v, gt, a.seed):.4f}")


if __name__ == "__main__":
    main()
