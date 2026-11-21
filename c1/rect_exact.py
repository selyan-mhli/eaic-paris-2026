"""P1 — recherche exacte du rectangle de somme maximale sur une carte dense (régime R, rectangle seul).

Pour une carte s (128 x 128, accord bit à bit ou score, 0.5 = hasard) et un seuil t, le rectangle B qui maximise
sum_{B} (s - t) est trouvé exactement :
  - sans contrainte : Kadane 2D en O(n^3) (pour chaque ligne haute, sommes cumulées des colonnes puis sous-tableau
    maximal par préfixes : max_j P[j] - min_{i<j} P[i]) ;
  - avec rapport d'aspect <= r (TrustMark 2, PixelSeal/StegaStamp 16) : si l'optimum libre viole r, énumération
    exacte de toutes les tailles (h, w) admissibles par image intégrale.
Si la somme maximale est <= 0, aucun rectangle : masque tout 0.

Évaluation (CPU, sans GPU ni kit) sur les images R du train (full_watermark=False) : pour chaque méthode, la carte
source et le seuil t sont choisis en validation croisée imbriquée (5 folds stratifiés par méthode, fit sur 4 folds,
score sur le 5e). Comparaison, sur les mêmes images, avec la boîte de box_search.py et avec tout 0.

    python c1/rect_exact.py --heats heats_train.npz --feats 'seg_feats_train_*.npz' --boxes 'boxes_train_*.npz' \
        --masks c1/seg/data/masks_128.npz --meta c1/seg/data/train_metadata.json
"""
from __future__ import annotations

import argparse
import glob
import json
import time

import numpy as np

METHODS = ("pixelseal", "trustmark", "stegastamp")
MAX_RATIO = {"pixelseal": 16.0, "trustmark": 2.0, "stegastamp": 16.0}
G = 128
FEAT_CH = ("dense", "w128", "w256", "cmax", "cbest")


# ---------------------------------------------------------------- recherche exacte

def kadane2d(S: np.ndarray):
    """S (n, m) -> (somme, (y0, y1, x0, x1)) du rectangle de somme maximale (bornes exclusives)."""
    n, m = S.shape
    C = np.vstack([np.zeros((1, m)), np.cumsum(S, 0)])          # C[i] = somme des lignes < i
    best, box = -np.inf, (0, 1, 0, 1)
    for y0 in range(n):
        col = C[y0 + 1:] - C[y0]                                 # (n - y0, m) : sommes des lignes y0..y1-1 par colonne
        P = np.hstack([np.zeros((len(col), 1)), np.cumsum(col, 1)])
        Mn = np.minimum.accumulate(P[:, :-1], 1)                 # min_{i<j} P[i] pour j = 1..m
        V = P[:, 1:] - Mn
        k = int(np.argmax(V)); r, j = divmod(k, m)
        if V[r, j] > best:
            x1 = j + 1; x0 = int(np.argmin(P[r, :x1]))
            best, box = float(V[r, j]), (y0, y0 + r + 1, x0, x1)
    return best, box


def ratio_ok(box, r):
    h, w = box[1] - box[0], box[3] - box[2]
    return max(h / w, w / h) <= r + 1e-9


def enum_constrained(S: np.ndarray, r: float):
    """Optimum exact sous contrainte de rapport d'aspect, par image intégrale sur toutes les tailles admissibles."""
    n, m = S.shape
    I = np.zeros((n + 1, m + 1)); I[1:, 1:] = S.cumsum(0).cumsum(1)
    best, box = -np.inf, (0, 1, 0, 1)
    for h in range(1, n + 1):
        wmin = max(1, int(np.ceil(h / r - 1e-9))); wmax = min(m, int(np.floor(h * r + 1e-9)))
        A = I[h:, :] - I[:-h, :]                                 # (n - h + 1, m + 1) : sommes de bandes de hauteur h
        for w in range(wmin, wmax + 1):
            V = A[:, w:] - A[:, :-w]
            k = int(np.argmax(V))
            if V.flat[k] > best:
                y0, x0 = divmod(k, V.shape[1])
                best, box = float(V.flat[k]), (y0, y0 + h, x0, x0 + w)
    return best, box


def best_rect(S: np.ndarray, r: float):
    s, b = kadane2d(S)
    if not ratio_ok(b, r):
        s, b = enum_constrained(S, r)
    return s, b


# ---------------------------------------------------------------- métrique (identique à wmloc.utils.miou_image)

def box_mask(box, n=G):
    m = np.zeros((n, n), bool)
    if box is not None:
        y0, y1, x0, x1 = box; m[y0:y1, x0:x1] = True
    return m


def miou(pred: np.ndarray, gt: np.ndarray) -> float:
    s = 0.0
    for c in (0, 1):
        u = np.logical_or(pred == c, gt == c).sum()
        s += 1.0 if u == 0 else np.logical_and(pred == c, gt == c).sum() / u
    return s / 2


def box512_to_128(box) -> np.ndarray:
    """Boîte en pixels 512 -> masque 128 par la règle officielle (cellule 4x4 à 1 si >= 50 % de pixels à 1)."""
    m = np.zeros((512, 512), np.float32); y0, y1, x0, x1 = (int(v) for v in box); m[y0:y1, x0:x1] = 1
    return m.reshape(G, 4, G, 4).mean(axis=(1, 3)) >= 0.5


# ---------------------------------------------------------------- données

def load_maps(heats_path, feats_glob):
    h = np.load(heats_path)
    names = h["names"].tolist()
    maps = {}
    for k in ("win128", "win256", "fm_acc", "fm_soft"):
        a = h[k].astype(np.float32)
        maps[k] = np.where(np.isfinite(a), a, 0.5)
    fe = {}
    for f in sorted(glob.glob(feats_glob)):
        z = np.load(f)
        for j, n in enumerate(z["names"].tolist()):
            fe[n] = z["maps"][j].astype(np.float32)
    if fe:
        assert set(fe) == set(names), "seg_feats incomplet"
        F_ = np.stack([fe[n] for n in names])                      # n 5 32 32
        for c, k in enumerate(FEAT_CH):
            a = np.repeat(np.repeat(F_[:, c], 4, 1), 4, 2)          # 32 -> 128 (cellule 16 px = 4 cellules)
            maps["f_" + k] = np.where(np.isfinite(a), a, 0.5)
    return names, h["method"], h["full_watermark"], maps


def load_boxes(globs):
    d = {}
    for f in sorted(glob.glob(globs)):
        z = np.load(f)
        for j, n in enumerate(z["names"].tolist()):
            d[n] = z["box"][j]
    return d


def folds(strata, k=5, seed=0):
    rng = np.random.default_rng(seed); f = np.zeros(len(strata), int)
    for s in np.unique(strata):
        ii = np.nonzero(strata == s)[0]; rng.shuffle(ii); f[ii] = np.arange(len(ii)) % k
    return f


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heats", required=True); ap.add_argument("--feats", default="")
    ap.add_argument("--boxes", default=""); ap.add_argument("--masks", required=True); ap.add_argument("--meta", required=True)
    ap.add_argument("--thr", type=float, nargs="+", default=[round(x, 3) for x in np.arange(0.50, 0.905, 0.025)])
    ap.add_argument("--maps", nargs="+", default=None, help="sous-ensemble de cartes (défaut : toutes)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    names, method, full, maps = load_maps(a.heats, a.feats)
    z = np.load(a.masks); gtd = dict(zip(z["names"].tolist(), z["masks"]))
    meta = {m["image_name"]: m for m in json.load(open(a.meta))}
    assert all(bool(meta[n]["full_watermark"]) == bool(f) for n, f in zip(names, full))
    R = np.nonzero(~full.astype(bool))[0]
    keys = a.maps or sorted(maps)
    print(f"{len(R)} images R ; cartes : {keys} ; seuils : {a.thr[0]}..{a.thr[-1]} ({len(a.thr)})", flush=True)

    # table des scores : score[key][ti, i] (mIoU de la boîte exacte), boxes conservées pour diagnostic
    t0 = time.time()
    score = {k: np.zeros((len(a.thr), len(R))) for k in keys}
    zero = np.array([miou(np.zeros((G, G), bool), gtd[names[i]]) for i in R])
    for k in keys:
        for j, i in enumerate(R):
            gt = gtd[names[i]]; r = MAX_RATIO[str(method[i])]; s = maps[k][i]
            for ti, t in enumerate(a.thr):
                v, b = best_rect(s - t, r)
                score[k][ti, j] = miou(box_mask(b), gt) if v > 0 else zero[j]
        print(f"  {k:10s} fait ({time.time() - t0:.0f}s) ; meilleur non imbriqué {score[k].mean(1).max():.4f}", flush=True)

    bs = load_boxes(a.boxes) if a.boxes else {}
    bsc = np.array([miou(box512_to_128(bs[names[i]]), gtd[names[i]]) for i in R]) if bs else None

    # validation croisée imbriquée : par méthode, choix (carte, seuil) sur les folds d'ajustement
    mR = method[R].astype(str); fo = folds(mR, 5, a.seed)
    nested = np.zeros(len(R)); picks = []
    for m in METHODS:
        for f in range(5):
            fit = (mR == m) & (fo != f); ev = (mR == m) & (fo == f)
            if not ev.any():
                continue
            cands = [(score[k][ti, fit].mean(), k, ti) for k in keys for ti in range(len(a.thr))]
            cands.append((zero[fit].mean(), "tout0", -1))
            _, k, ti = max(cands)
            nested[ev] = zero[ev] if k == "tout0" else score[k][ti, ev]
            picks.append((m, f, k, a.thr[ti] if ti >= 0 else None))

    print("\nmIoU sur les images R (full_watermark=False), par méthode")
    print(f"{'méthode':<11} {'n':>3} {'tout0':>7} {'box_search':>10} {'P1 imbriqué':>11}")
    for m in METHODS:
        q = mR == m
        print(f"{m:<11} {q.sum():>3} {zero[q].mean():7.4f} {bsc[q].mean() if bsc is not None else float('nan'):10.4f} "
              f"{nested[q].mean():11.4f}")
    print(f"{'tous':<11} {len(R):>3} {zero.mean():7.4f} {bsc.mean() if bsc is not None else float('nan'):10.4f} "
          f"{nested.mean():11.4f}")
    print("\nchoix par fold :", picks)
    print("\nmeilleur (carte, seuil) non imbriqué par méthode (optimiste) :")
    for m in METHODS:
        q = mR == m
        best = max((score[k][ti, q].mean(), k, a.thr[ti]) for k in keys for ti in range(len(a.thr)))
        print(f"  {m:<11} {best[1]:<10} t={best[2]:.3f} -> {best[0]:.4f}")


def selftest():
    rng = np.random.default_rng(0)
    for trial in range(200):
        n = int(rng.integers(3, 10)); m = int(rng.integers(3, 10)); S = rng.normal(size=(n, m))
        bb, bs = -np.inf, None
        for y0 in range(n):
            for y1 in range(y0 + 1, n + 1):
                for x0 in range(m):
                    for x1 in range(x0 + 1, m + 1):
                        v = S[y0:y1, x0:x1].sum()
                        if v > bb:
                            bb, bs = v, (y0, y1, x0, x1)
        v, b = kadane2d(S)
        assert abs(v - bb) < 1e-9 and abs(S[b[0]:b[1], b[2]:b[3]].sum() - v) < 1e-9, (trial, v, bb)
        r = float(rng.choice([1.0, 1.5, 2.0]))
        cb = max(((S[y0:y1, x0:x1].sum(), (y0, y1, x0, x1)) for y0 in range(n) for y1 in range(y0 + 1, n + 1)
                  for x0 in range(m) for x1 in range(x0 + 1, m + 1) if ratio_ok((y0, y1, x0, x1), r)))
        v2, b2 = best_rect(S, r)
        assert abs(v2 - cb[0]) < 1e-9 and ratio_ok(b2, r), (trial, v2, cb)
    # rectangle planté dans du bruit sur 128 x 128
    S = rng.normal(0, 0.05, (G, G)) + 0.5; S[30:70, 40:100] += 0.15
    v, b = best_rect(S - 0.55, 16.0)
    assert b == (30, 70, 40, 100), b
    v, b = best_rect(S - 0.55, 1.2)
    assert ratio_ok(b, 1.2), b
    g = box_mask((30, 70, 40, 100)); assert miou(g, g) == 1.0
    print("selftest ok (Kadane 2D = force brute sur 200 cas, contrainte de rapport exacte, rectangle planté retrouvé)")


if __name__ == "__main__":
    main()
