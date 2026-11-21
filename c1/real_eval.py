"""Évaluation locale (CPU) des cartes de `real_heats.py` sur le train réel, et fichier de soumission test.

    python work_c1/real_eval.py --heats output/heats_train.npz
    python work_c1/real_eval.py --heats output/heats_train.npz --test-heats output/heats_test.npz \
        --write-sub output/sub_c1_v1.pt

- Vérifie que `c1/metric.py` == `wmloc.utils.score` sur les 510 masques (tout 0 / tout 1).
- Diagnostic d'incrustation par méthode x full_watermark.
- Seuil par méthode choisi en 5-fold stratifié (méthode x full_watermark) : score hors-fold.
  Variantes : carte (win128, win256, fm_acc, fm_soft) x normalisation (abs, rel = min-max par image).
  La grille de seuils va de 0 (= tout 1) à 1.01 (= tout 0), donc les baselines sont candidates.
- Soumission : par méthode, la variante au meilleur score hors-fold (choix légèrement optimiste,
  dit comme tel), seuil refit sur les 510 images. Le fichier n'est PAS soumis.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path.cwd()))
from metric import miou_per_image  # noqa: E402
from wmloc.utils import miou_image as off_miou_image, score as off_score  # noqa: E402

METHODS = ("pixelseal", "trustmark", "stegastamp")
THRS = np.round(np.arange(0.0, 1.02, 0.01), 2)
KEYS = ("win128", "win256", "fm_acc", "fm_soft")


def fast_miou(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """pred, gt (n, 128, 128) bool -> (n,) mIoU 2 classes (1 si union vide), comme le kit."""
    if len(pred) == 0:
        return np.zeros(0)
    p = pred.reshape(len(pred), -1).astype(bool); g = gt.reshape(len(gt), -1).astype(bool)
    out = np.zeros(len(p))
    for c in (False, True):
        pc, gc = p == c, g == c
        u = (pc | gc).sum(1); i = (pc & gc).sum(1)
        out += np.where(u == 0, 1.0, i / np.maximum(u, 1))
    return out / 2


def norm(h: np.ndarray, mode: str) -> np.ndarray:
    h = h.astype(np.float32)
    if mode == "abs":
        return h
    lo = h.reshape(len(h), -1).min(1)[:, None, None]; hi = h.reshape(len(h), -1).max(1)[:, None, None]
    return (h - lo) / np.maximum(hi - lo, 1e-6)


def best_thr(h: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    sc = [fast_miou(h >= t, gt).mean() for t in THRS]
    k = int(np.argmax(sc)); return float(THRS[k]), float(sc[k])


def folds(strata: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed); f = np.zeros(len(strata), int)
    for s in np.unique(strata):
        ii = np.nonzero(strata == s)[0]; rng.shuffle(ii); f[ii] = np.arange(len(ii)) % k
    return f


def group_line(name, s, method, full):
    line = f"{name:24s} all={s.mean():.4f} |"
    for m in METHODS:
        sel = method == m
        line += f" {m[:5]}={s[sel].mean():.3f}"
        for fw in (True, False):
            q = sel & (full == fw)
            if q.any():
                line += f" [{'F' if fw else 'R'} {s[q].mean():.3f} n={q.sum()}]"
    return line


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heats", type=Path, required=True)
    ap.add_argument("--data", default="release")
    ap.add_argument("--test-heats", type=Path)
    ap.add_argument("--write-sub", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    z = np.load(Path(args.data) / "train" / "masks_128.npz")
    allm = dict(zip(z["names"].tolist(), z["masks"]))
    gts_all = np.stack(list(allm.values()))
    # 1) métrique locale == officielle
    for nm, p in (("tout0", np.zeros_like(gts_all)), ("tout1", np.ones_like(gts_all))):
        a, b = miou_per_image(p, gts_all).mean(), off_score(p, gts_all)
        c = fast_miou(p.astype(bool), gts_all).mean()
        print(f"metrique {nm}: c1={a:.6f} kit={b:.6f} fast={c:.6f} n={len(gts_all)}")
        assert abs(a - b) < 1e-12 and abs(c - b) < 1e-12

    d = np.load(args.heats)
    names = d["names"].tolist(); method = d["method"]; full = d["full_watermark"]
    gt = np.stack([allm[n] for n in names])  # n 128 128
    n = len(names)
    print(f"\n{n} images ; couverture GT moyenne {gt.mean():.3f}")
    # 2) diagnostic
    print("\nDiagnostic bit accuracy (moyenne) par méthode x full_watermark (F = image entière watermarkée)")
    for m in METHODS:
        for fw in (True, False):
            q = (method == m) & (full == fw)
            if q.any():
                print(f"  {m:10s} {'F' if fw else 'R'} n={q.sum():3d} cov={gt[q].mean():.3f} "
                      + " ".join(f"{k}={np.nanmean(d[k][q]):.3f}" for k in ("acc_full", "acc_gt", "acc_bbox", "acc_crop", "acc_out") if k in d.files))
    # 3) baselines + k-fold
    print("\nmIoU (hors-fold pour les cartes ; F = full_watermark, R = rectangle seul)")
    print(group_line("tout0", fast_miou(np.zeros_like(gt, bool), gt), method, full))
    print(group_line("tout1", fast_miou(np.ones_like(gt, bool), gt), method, full))
    strata = np.array([f"{m}|{f}" for m, f in zip(method, full)])
    fo = folds(strata, 5, args.seed)
    oof_by = {}
    for key in KEYS:
        if key not in d.files:
            continue
        for mode in ("abs", "rel"):
            H = d[key].astype(np.float32)
            oof = np.full(n, np.nan)
            thr_info = {}
            for m in METHODS:
                sel = method == m
                if np.isnan(H[sel]).all():
                    continue
                Hm = norm(H[sel], mode); gm = gt[sel]; fm = fo[sel]; s = np.zeros(sel.sum())
                ts = []
                for k in range(5):
                    if not (fm == k).any() or (fm != k).sum() == 0:
                        continue
                    t, _ = best_thr(Hm[fm != k], gm[fm != k]); ts.append(t)
                    s[fm == k] = fast_miou(Hm[fm == k] >= t, gm[fm == k])
                oof[sel] = s; thr_info[m] = ts
            oof_by[(key, mode)] = oof
            ok = ~np.isnan(oof)
            print(group_line(f"{key}/{mode}", np.where(ok, oof, np.nan)[ok], method[ok], full[ok])
                  + "  seuils " + " ".join(f"{m[:2]}:{min(v):.2f}-{max(v):.2f}" for m, v in thr_info.items()))
    # 4) meilleure variante par méthode (choix sur le hors-fold : légèrement optimiste)
    choice, combo = {}, np.zeros(n)
    for m in METHODS:
        sel = method == m
        cands = {kv: o[sel].mean() for kv, o in oof_by.items() if not np.isnan(o[sel]).any()}
        kv = max(cands, key=cands.get)
        H = norm(d[kv[0]][sel].astype(np.float32), kv[1])
        t, s_in = best_thr(H, gt[sel])
        choice[m] = {"key": kv[0], "mode": kv[1], "thr": t, "oof": round(float(cands[kv]), 4), "in_sample": round(s_in, 4)}
        combo[sel] = oof_by[kv][sel]
    print("\nChoix par méthode :", json.dumps(choice))
    print(group_line("combo (hors-fold)", combo, method, full))
    # vérif croisée d'un échantillon avec la fonction officielle
    m0 = choice[METHODS[0]]; sel = np.nonzero(method == METHODS[0])[0][:20]
    H = norm(d[m0["key"]][sel].astype(np.float32), m0["mode"]) >= m0["thr"]
    assert np.allclose(fast_miou(H, gt[sel]), [off_miou_image(p, g) for p, g in zip(H, gt[sel])])

    if args.test_heats and args.write_sub:
        import torch
        dt = np.load(args.test_heats)
        tn = dt["names"].tolist(); tm = dt["method"]
        meta = json.loads((Path(args.data) / "test" / "metadata.json").read_text())
        assert sorted(tn) == sorted(x["image_name"] for x in meta) and len(set(tn)) == len(tn), "noms test"
        masks = np.zeros((len(tn), 128, 128), bool)
        for m, c in choice.items():
            sel = tm == m
            masks[sel] = norm(dt[c["key"]][sel].astype(np.float32), c["mode"]) >= c["thr"]
        torch.save({"image_names": tn, "masks": torch.from_numpy(masks)}, args.write_sub)
        back = torch.load(args.write_sub, weights_only=True)
        sz = args.write_sub.stat().st_size / 1e6
        print(f"\nsoumission -> {args.write_sub} {tuple(back['masks'].shape)} {back['masks'].dtype} "
              f"{sz:.1f} Mo, couverture prédite {masks.mean():.3f} (NON soumise)")
        assert back["masks"].shape == (1020, 128, 128) and sz <= 25


if __name__ == "__main__":
    main()
