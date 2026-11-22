"""mIoU sur les vrais F seuls d'un canal seuillé (sens et seuil choisis hors-fold, 5 folds stratifiés par
méthode, graine 0), comparé à « tout 1 ». Le seuil est un quantile par image (part prédite « trou ») ou une
valeur absolue, choisi sur les 4 folds d'apprentissage.

  python work_segf/f_only.py --maps 'output/segf_trufor_train_*.npz' --keys tf_map tf_conf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chan_auc import METHODS, load


def miou(p, g):
    out = np.zeros(len(p))
    for c in (False, True):
        pc, gc = p == c, g == c
        u = (pc | gc).reshape(len(p), -1).sum(1); i = (pc & gc).reshape(len(p), -1).sum(1)
        out += np.where(u == 0, 1.0, i / np.maximum(u, 1))
    return out / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", required=True); ap.add_argument("--keys", nargs="+", required=True)
    ap.add_argument("--data", default="release")
    a = ap.parse_args()
    meta = {m["image_name"]: m for m in json.loads((Path(a.data) / "train" / "metadata.json").read_text())}
    z = np.load(Path(a.data) / "train" / "masks_128.npz"); gt_all = dict(zip(z["names"].tolist(), z["masks"]))
    M = load(a.maps)
    idx = [i for i, n in enumerate(M["names"].tolist()) if meta[n]["full_watermark"]]
    names = [M["names"][i] for i in idx]
    gt = np.stack([gt_all[n] for n in names]).astype(bool); meth = np.array([meta[n]["method"] for n in names])
    rng = np.random.default_rng(0); fold = np.zeros(len(names), int)
    for m in METHODS:
        ii = np.nonzero(meth == m)[0]; rng.shuffle(ii); fold[ii] = np.arange(len(ii)) % 5
    for k in a.keys:
        v = M[k][idx].astype(np.float32)
        for m in METHODS:
            s = meth == m; base = miou(np.ones_like(gt[s]), gt[s])
            cands = {}
            for sign in (1, -1):  # sign=1 : trou = score haut (TruFor : P(manipulé) haute)
                w = sign * v[s]
                for q in (0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3):  # part de l'image prédite trou
                    t = np.quantile(w.reshape(len(w), -1), 1 - q, axis=1)[:, None, None] if q else np.full((len(w), 1, 1), np.inf)
                    cands[("q", sign, q)] = miou(~(w >= t), gt[s])
                for t in np.quantile(w, [0.5, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995]):
                    cands[("abs", sign, float(t))] = miou(~(w >= t), gt[s])
            oof = np.zeros(s.sum()); fm = fold[s]; ch = []
            for f in range(5):
                best = max(cands, key=lambda c: cands[c][fm != f].mean()); ch.append(best[:3] if best[0] == "q" else best[:2])
                oof[fm == f] = cands[best][fm == f]
            print(f"{k:8s} {m:10s} F seuls : tout 1 {base.mean():.4f} ; seuillé hors-fold {oof.mean():.4f} "
                  f"({oof.mean() - base.mean():+.4f}) ; choix {ch}", flush=True)


if __name__ == "__main__":
    main()
