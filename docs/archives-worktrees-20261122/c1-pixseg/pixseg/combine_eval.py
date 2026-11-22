"""Score imbriqué global : segmentation pixseg pour certaines méthodes, règle gated_eval (full) pour les autres.

    python work_pixseg/combine_eval.py --cv 'output/pixseg_cv_ss6k_*.npz' --methods stegastamp \
        --heats output/heats_train.npz --boxes 'output/boxes_train_[0-9].npz'

Les fichiers --cv (train.py --cv, découpés par folds) sont fusionnés par image : probs et seuil du fold (seuil
choisi sur le fold de validation interne, jamais sur l'image notée). La règle gated_eval est refaite fold par
fold (ajustée sur les 4 autres folds, mêmes folds que train.py). Rapporte : règle seule, pixseg seul, et la
combinaison (pixseg pour --methods, règle ailleurs), par méthode x régime.
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd() / "work_c1")); sys.path.insert(0, str(Path.cwd()))
from gated_eval import Data, apply, fit, load_split  # noqa: E402
from real_eval import fast_miou, folds, group_line  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cv", required=True); ap.add_argument("--methods", default="stegastamp")
ap.add_argument("--heats", default="output/heats_train.npz"); ap.add_argument("--boxes", default="output/boxes_train_[0-9].npz")
ap.add_argument("--rule", default="full")
a = ap.parse_args()

d, names, boxes, accb = load_split(a.heats, a.boxes, "release", "train")
z = np.load("release/train/masks_128.npz"); M = dict(zip(z["names"].tolist(), z["masks"]))
gt = np.stack([M[n] for n in names]).astype(bool)
method, full = d["method"], d["full_watermark"]
D = Data(d, boxes, accb); n = D.n; idx = np.arange(n)
fo = folds(np.array([f"{m}|{f}" for m, f in zip(method, full)]), 5, 0)

probs = np.full((n, 128, 128), np.nan, np.float32); thr = np.full(n, np.nan, np.float32)
pos = {x: i for i, x in enumerate(names)}
for f in sorted(glob.glob(a.cv)):
    c = np.load(f)
    assert c["names"].tolist() == names, f"{f} : ordre des noms != heats"
    assert (c["fold"] == fo).all(), f"{f} : folds différents de gated_eval"
    ok = np.isfinite(c["thr"])
    assert not (ok & np.isfinite(thr)).any(), f"{f} : images déjà couvertes par un autre fichier"
    probs[ok] = c["probs"][ok].astype(np.float32); thr[ok] = c["thr"][ok]
    print(f"{f} : folds {sorted(set(fo[ok].tolist()))}", flush=True)
done = np.isfinite(thr)
print(f"pixseg disponible pour {done.sum()} / {n} images (folds {sorted(set(fo[done].tolist()))})")

rule = np.zeros((n, 128, 128), bool)
for k in range(5):
    r, _ = fit(D, gt, idx[fo != k], a.rule); te = idx[fo == k]
    rule[te] = apply(D, te, r)
seg = probs > thr[:, None, None]
seg05 = probs > 0.5
use = np.isin(method, a.methods.split(",")) & done
comb = np.where(use[:, None, None], seg, rule)
comb05 = np.where(use[:, None, None], seg05, rule)
s_rule, s_comb, s_comb05 = fast_miou(rule, gt), fast_miou(comb, gt), fast_miou(comb05, gt)
print(group_line(f"règle {a.rule} (imbriquée)", s_rule, method, full))
sel = done
print(group_line("pixseg seuil choisi (imbriqué), images couvertes", fast_miou(seg[sel], gt[sel]), method[sel], full[sel]))
print(group_line("pixseg seuil 0,5, images couvertes", fast_miou(seg05[sel], gt[sel]), method[sel], full[sel]))
print(group_line(f"COMBINÉ pixseg[{a.methods}] + règle (imbriqué)", s_comb, method, full))
print(group_line(f"combiné, pixseg seuil 0,5", s_comb05, method, full))
for k in range(5):
    q = fo == k
    print(f"  fold {k} : règle {s_rule[q].mean():.4f} combiné {s_comb[q].mean():.4f}")
