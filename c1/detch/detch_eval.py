"""Évaluation du canal 0 PixelSeal (cartes de detch_maps.py) : AUC pixel, mIoU officiel en CV 5 folds.

Usage : python c1/detch/detch_eval.py <dossier contenant detch_maps_train.npz, heats_train.npz, masks_128.npz>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

D = Path(sys.argv[1])
SEED, K = 0, 5


def miou_image(pred, gt) -> float:  # copie exacte de wmloc/utils.py
    p = np.asarray(pred).astype(np.uint8); g = np.asarray(gt).astype(np.uint8)
    s = 0.0
    for c in (0, 1):
        union = np.logical_or(p == c, g == c).sum()
        s += 1.0 if union == 0 else np.logical_and(p == c, g == c).sum() / union
    return s / 2


assert abs(miou_image(np.ones((2, 2)), np.array([[1, 1], [1, 0]])) - (0.75 + 0) / 2) < 1e-9  # cas à la main


def auc(score, label):
    """AUC de Mann-Whitney (rangs, ex aequo moyennés)."""
    s = np.asarray(score, np.float64).ravel(); y = np.asarray(label, bool).ravel()
    order = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); r = (csum - (cnt - 1) / 2)[inv]
    npos = y.sum(); nneg = len(y) - npos
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else float("nan")


dm = np.load(D / "detch_maps_train.npz"); hm = np.load(D / "heats_train.npz"); mk = np.load(D / "masks_128.npz")
gt_all = dict(zip(mk["names"], mk["masks"]))
hidx = {n: i for i, n in enumerate(hm["names"])}
names = dm["names"]; full = dm["full"]
gt = np.stack([gt_all[n] for n in names]).astype(bool)             # N 128 128
acc_full = np.array([hm["acc_full"][hidx[n]] for n in names])
fm_acc = np.stack([hm["fm_acc"][hidx[n]] for n in names]).astype(np.float32)  # accord par position (existant)
N = len(names)
print(f"PixelSeal : {N} images, F={full.sum()}, R={(~full).sum()}")

maps = {"det256": dm["det256_128"].astype(np.float32), "det512": dm["det512_128"].astype(np.float32),
        "agr256": dm["agr256_128"].astype(np.float32), "fm_acc": fm_acc}
lv = np.log(dm["lumvar_128"].astype(np.float32) + 1e-6)

# --- (a) AUC pixel
print("\n(a) AUC pixel watermarké vs propre (poolé ; médiane par image entre parenthèses)")
for k, m in maps.items():
    row = []
    for nm, sel in (("F", full), ("R", ~full), ("tous", np.ones(N, bool))):
        a = auc(m[sel], gt[sel])
        per = [auc(m[i], gt[i]) for i in np.where(sel)[0]]
        row.append(f"{nm} {a:.3f} ({np.nanmedian(per):.3f})")
    print(f"  {k:7s} " + " | ".join(row))
print(f"  logvarY  F {auc(lv[full], gt[full]):.3f} | R {auc(lv[~full], gt[~full]):.3f}")
# z-score par image (le niveau absolu varie d'une image à l'autre)
for k in ("det256", "det512"):
    m = maps[k]; z = (m - m.mean((1, 2), keepdims=True)) / (m.std((1, 2), keepdims=True) + 1e-6)
    print(f"  {k}_z F {auc(z[full], gt[full]):.3f} | R {auc(z[~full], gt[~full]):.3f}")
# niveau image : canal 0 global sépare-t-il F de R ?
print(f"  canal0 global (image) F vs R : AUC {auc(dm['logit_global0'], full):.3f} ; acc_full : {auc(acc_full, full):.3f}")

# --- folds stratifiés F/R (méthode unique ici)
rng = np.random.default_rng(SEED); fold = np.empty(N, int)
for sel in (full, ~full):
    idx = np.where(sel)[0]; rng.shuffle(idx); fold[idx] = np.arange(len(idx)) % K


def cv(predict, params, sel_imgs):
    """Choix du paramètre sur les folds d'entraînement, score sur le fold exclu. -> mIoU moyen sur sel_imgs."""
    sc = np.zeros(N)
    cache = {p: np.array([miou_image(predict(i, p), gt[i]) if sel_imgs[i] else 0 for i in range(N)]) for p in params}
    chosen = []
    for f in range(K):
        tr = sel_imgs & (fold != f); te = sel_imgs & (fold == f)
        best = max(params, key=lambda p: cache[p][tr].mean()); chosen.append(best)
        sc[te] = cache[best][te]
    return sc[sel_imgs].mean(), chosen


ones = lambda i, p: np.ones((128, 128), bool)
zeros = lambda i, p: np.zeros((128, 128), bool)
ref_F = np.mean([miou_image(ones(i, 0), gt[i]) for i in np.where(full)[0]])
ref_R = np.mean([miou_image(zeros(i, 0), gt[i]) for i in np.where(~full)[0]])
print(f"\n(b) références PixelSeal : F tout 1 = {ref_F:.4f} ; R tout 0 = {ref_R:.4f}")


def zmap(m):
    return (m - m.mean((1, 2), keepdims=True)) / (m.std((1, 2), keepdims=True) + 1e-6)


variants = {}
for k in ("det256", "det512", "agr256", "fm_acc"):
    variants[k] = maps[k]; variants[k + "_z"] = zmap(maps[k])
for k, m in variants.items():
    qs = np.quantile(m, np.linspace(0.0, 1.0, 41))
    params = [(s, float(t)) for s in (1, -1) for t in (qs if True else [])]
    pred = lambda i, p, m=m: (p[0] * m[i]) >= p[0] * p[1]
    sF, cF = cv(pred, params, full); sR, cR = cv(pred, params, ~full)
    print(f"  {k:9s} F {sF:.4f} ({sF - ref_F:+.4f})  R {sR:.4f} ({sR - ref_R:+.4f})  seuils F {cF[:2]}")

# --- (c) règle actuelle : acc_full >= 0.75 -> F ; dans F, on creuse les trous là où la carte est basse
isF = acc_full >= 0.75
print(f"\n(c) gate acc_full>=0.75 : F prédits {isF.sum()} ; accord avec la vérité {(isF == full).mean():.3f}")
allimg = np.ones(N, bool)
base = np.mean([miou_image(np.full((128, 128), isF[i]), gt[i]) for i in range(N)])
print(f"  base (F prédit -> tout 1, sinon tout 0) : {base:.4f}")
for k in ("det256_z", "det512_z", "agr256_z", "fm_acc_z", "det256", "fm_acc"):
    m = variants[k]
    params = [None] + [(s, float(t)) for s in (1, -1) for t in np.quantile(m[isF], np.linspace(0, 0.3, 31) if True else [])]
    def pred(i, p, m=m):
        if not isF[i]:
            return np.zeros((128, 128), bool)
        if p is None:
            return np.ones((128, 128), bool)
        return (p[0] * m[i]) >= p[0] * p[1]
    s, ch = cv(pred, params, allimg)
    print(f"  {k:9s} global {s:.4f} ({s - base:+.4f})  choix {ch}")

# --- (d) correction par variance locale : trou = carte basse ET zone texturée (logvar > v)
print("\n(d) trous = carte z basse ET log-variance Y > v (dans les F prédits)")
for k in ("det256_z", "det512_z", "fm_acc_z"):
    m = variants[k]
    params = [None] + [(float(t), float(v)) for t in np.quantile(m[isF], np.linspace(0, 0.3, 16))
                       for v in np.quantile(lv[isF], [0.0, 0.2, 0.4, 0.6])]
    def pred(i, p, m=m):
        if not isF[i]:
            return np.zeros((128, 128), bool)
        if p is None:
            return np.ones((128, 128), bool)
        return ~((m[i] < p[0]) & (lv[i] > p[1]))
    s, ch = cv(pred, params, allimg)
    print(f"  {k:9s} global {s:.4f} ({s - base:+.4f})  choix {ch}")
