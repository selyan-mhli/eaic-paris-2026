"""AUC pixel « watermarqué vs trou » de chaque canal candidat sur les vrais F du train, par méthode (numpy seul).

AUC > 0,5 : le canal est plus haut dans la zone watermarquée ; < 0,5 : plus haut dans les trous. La séparation
utile est |AUC - 0,5|. On donne l'AUC poolée et la médiane des AUC par image (le niveau absolu varie d'une
image à l'autre ; la médiane par image dit si un seuil relatif suffirait).

  python work_segf/chan_auc.py --hf 'output/segf_hf_train_*.npz' --heats output/heats_train.npz \
      --ff 'output/seg_ff_train_*.npz' [--extra 'output/segf_trufor_train_*.npz']
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

METHODS = ("pixelseal", "trustmark", "stegastamp")
FF_CH = {0: "ff_dense", 1: "ff_occ64", 2: "ff_occ128", 3: "ff_keep128", 4: "ff_keep256", 6: "ff_lumstd"}


def auc(score, label) -> float:
    s = np.asarray(score, np.float64).ravel(); y = np.asarray(label, bool).ravel()
    ok = np.isfinite(s); s, y = s[ok], y[ok]
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2)[inv]
    npos = y.sum(); nneg = len(y) - npos
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else float("nan")


def load(pattern) -> dict:
    fs = sorted(glob.glob(pattern)); assert fs, pattern
    zs = [dict(np.load(f)) for f in fs]
    return {k: np.concatenate([z[k] for z in zs]) for k in zs[0]}


def up(m32: np.ndarray) -> np.ndarray:  # n 32 32 -> n 128 128 (plus proche voisin)
    return m32.repeat(4, -2).repeat(4, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True); ap.add_argument("--heats"); ap.add_argument("--ff")
    ap.add_argument("--extra", nargs="*", default=[]); ap.add_argument("--data", default="release")
    a = ap.parse_args()
    meta = {m["image_name"]: m for m in json.loads((Path(a.data) / "train" / "metadata.json").read_text())}
    z = np.load(Path(a.data) / "train" / "masks_128.npz"); gt_all = dict(zip(z["names"].tolist(), z["masks"]))
    H = load(a.hf)
    names = [n for n in H["names"].tolist() if meta[n]["full_watermark"]]
    pos = {n: i for i, n in enumerate(H["names"].tolist())}
    sel = np.array([pos[n] for n in names])
    ch = {k: H[k][sel].astype(np.float32) for k in ("hfe", "srm", "jnd", "ncc_m", "ncc_r", "ncc_d")}
    ch["hfe/jnd"] = ch["hfe"] / (ch["jnd"] + 1e-4); ch["srm/jnd"] = ch["srm"] / (ch["jnd"] + 1e-4)
    for pat in a.extra:
        E = load(pat); pe = {n: i for i, n in enumerate(E["names"].tolist())}
        for k in E:
            if k not in ("names", "method") and E[k].ndim == 3:
                ch[k] = np.stack([E[k][pe[n]] for n in names]).astype(np.float32)
    if a.heats:
        d = np.load(a.heats); ph = {n: i for i, n in enumerate(d["names"].tolist())}
        for k in ("win128", "win256", "fm_acc", "fm_soft"):
            ch[k] = np.stack([d[k][ph[n]] for n in names]).astype(np.float32)
    if a.ff:
        Fz = load(a.ff); pf = {n: i for i, n in enumerate(Fz["names"].tolist())}
        fm = np.stack([Fz["fmaps"][pf[n]] for n in names]).astype(np.float32)  # n 7 32 32
        for c, k in FF_CH.items():
            ch[k] = up(fm[:, c])
    gt = np.stack([gt_all[n] for n in names]).astype(bool)
    meth = np.array([meta[n]["method"] for n in names])
    print(f"{len(names)} images F ; part de trous {1 - gt.mean():.3f}")
    print(f"{'canal':12s} " + " ".join(f"{m:>22s}" for m in METHODS))
    for k, v in ch.items():
        row = []
        for m in METHODS:
            s = meth == m
            if not np.isfinite(v[s]).any():
                row.append(f"{'-':>22s}"); continue
            per = [auc(v[i], gt[i]) for i in np.nonzero(s)[0]]
            row.append(f"{auc(v[s], gt[s]):.3f} (méd {np.nanmedian(per):.3f})".rjust(22))
        print(f"{k:12s} " + " ".join(row), flush=True)


if __name__ == "__main__":
    main()
