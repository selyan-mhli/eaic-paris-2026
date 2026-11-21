"""Méthode combinée « tri F/R » : évaluation hors-fold sur le train réel et fichier de soumission test.

    python work_c1/gated_eval.py --heats output/heats_train.npz --boxes 'output/boxes_train_*.npz' \
        [--test-heats output/heats_test.npz --test-boxes 'output/boxes_test_*.npz' --write-sub output/sub.pt]

Par image (méthode et message connus) :
  1. tri : acc_full = bit accuracy de l'image entière ; >= --gate (0.75) -> régime F (image entière
     watermarkée puis zones restaurées), sinon régime R (rectangle seul).
  2. F : carte de chaleur (win128/win256/fm_*, abs/rel) >= seuil ; seuil 0 = tout 1 (candidat).
  3. R : rectangle de box_search.py si son acc >= tb, sinon tout 0 (tb > 1 = tout 0, candidat).
Seuils (et carte F) choisis par méthode en 5-fold stratifié méthode x full_watermark ; score hors-fold.
Le choix de la carte F parmi les variantes se fait sur le hors-fold (légèrement optimiste, dit comme tel).
La soumission refit les seuils sur les 510 images. Rien n'est soumis.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(Path.cwd()))
from real_eval import METHODS, fast_miou, folds, group_line, norm  # noqa: E402
from wmloc.utils import downsample_mask, miou_image as off_miou  # noqa: E402

KEYS = ("win128", "win256", "fm_acc", "fm_soft")
THR_F = np.round(np.arange(0.0, 1.02, 0.01), 2)
THR_B = np.round(np.arange(0.50, 1.02, 0.01), 2)


def load_boxes(pattern, names):
    files = sorted(glob.glob(pattern)); assert files, pattern
    bx, acc = {}, {}
    for f in files:
        z = np.load(f)
        for n, b, a in zip(z["names"].tolist(), z["box"], z["acc_box"]):
            bx[n] = b; acc[n] = a
    miss = [n for n in names if n not in bx]
    assert not miss, f"{len(miss)} images sans boîte"
    return np.stack([bx[n] for n in names]), np.array([acc[n] for n in names], np.float32)


def box_masks(boxes):
    out = np.zeros((len(boxes), 128, 128), bool)
    for i, (y0, y1, x0, x1) in enumerate(boxes):
        m = np.zeros((512, 512), np.uint8); m[y0:y1, x0:x1] = 1
        out[i] = downsample_mask(m).astype(bool)
    return out


def fit_thr(score_fn, thrs):
    sc = [score_fn(t) for t in thrs]; k = int(np.argmax(sc)); return float(thrs[k]), float(sc[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heats", type=Path, required=True); ap.add_argument("--boxes", required=True)
    ap.add_argument("--data", default="release"); ap.add_argument("--gate", type=float, default=0.75)
    ap.add_argument("--test-heats", type=Path); ap.add_argument("--test-boxes"); ap.add_argument("--write-sub", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    z = np.load(Path(a.data) / "train" / "masks_128.npz"); allm = dict(zip(z["names"].tolist(), z["masks"]))
    d = np.load(a.heats); names = d["names"].tolist(); method = d["method"]; full = d["full_watermark"]
    gt = np.stack([allm[n] for n in names]); n = len(names)
    af = d["acc_full"]; isF = af >= a.gate
    boxes, accb = load_boxes(a.boxes, names); BM = box_masks(boxes)
    print(f"{n} images ; tri F/R à {a.gate} : {int((isF != full).sum())} erreurs ({int((isF & ~full).sum())} R->F, "
          f"{int((~isF & full).sum())} F->R)")
    fo = folds(np.array([f"{m}|{f}" for m, f in zip(method, full)]), 5, a.seed)
    ones, zeros = np.ones_like(gt, bool), np.zeros_like(gt, bool)

    print(group_line("tout0", fast_miou(zeros, gt), method, full))
    gate_only = np.where(isF[:, None, None], ones, zeros)
    print(group_line("tri seul (F=1, R=0)", fast_miou(gate_only, gt), method, full))

    # --- partie R : rectangle si acc_box >= tb
    oofR = np.full(n, np.nan); tbs = {}
    def predR(idx, tb):
        return np.where((accb[idx] >= tb)[:, None, None], BM[idx], zeros[idx])
    for m in METHODS:
        for k in range(5):
            tr = np.nonzero((method == m) & ~isF & (fo != k))[0]; te = np.nonzero((method == m) & ~isF & (fo == k))[0]
            if not len(te):
                continue
            tb, _ = fit_thr(lambda t: fast_miou(predR(tr, t), gt[tr]).mean() if len(tr) else 0, THR_B)
            tbs.setdefault(m, []).append(tb); oofR[te] = fast_miou(predR(te, tb), gt[te])
    # --- partie F : carte >= t, variante choisie par méthode sur le hors-fold
    oofF_by = {}
    for key in KEYS:
        for mode in ("abs", "rel"):
            Hn = norm(d[key].astype(np.float32), mode); oof = np.full(n, np.nan)
            for m in METHODS:
                base = (method == m) & isF
                if np.isnan(Hn[base]).any():
                    continue
                for k in range(5):
                    tr = np.nonzero(base & (fo != k))[0]; te = np.nonzero(base & (fo == k))[0]
                    if not len(te):
                        continue
                    t, _ = fit_thr(lambda t: fast_miou(Hn[tr] >= t, gt[tr]).mean(), THR_F)
                    oof[te] = fast_miou(Hn[te] >= t, gt[te])
            oofF_by[(key, mode)] = oof
    choice = {}
    final = np.full(n, np.nan)
    for m in METHODS:
        selF = (method == m) & isF; selR = (method == m) & ~isF
        cands = {kv: o[selF].mean() for kv, o in oofF_by.items() if not np.isnan(o[selF]).any()}
        cands[("all1", "-")] = fast_miou(ones[selF], gt[selF]).mean()
        kv = max(cands, key=cands.get)
        final[selF] = oofF_by[kv][selF] if kv[0] != "all1" else fast_miou(ones[selF], gt[selF])
        final[selR] = oofR[selR]
        # refit sur tout le train
        if kv[0] != "all1":
            Hn = norm(d[kv[0]].astype(np.float32), kv[1]); ii = np.nonzero(selF)[0]
            t, _ = fit_thr(lambda t: fast_miou(Hn[ii] >= t, gt[ii]).mean(), THR_F)
        else:
            t = 0.0
        ii = np.nonzero(selR)[0]
        tb, _ = fit_thr(lambda q: fast_miou(predR(ii, q), gt[ii]).mean(), THR_B)
        choice[m] = {"F_key": kv[0], "F_mode": kv[1], "F_thr": t, "F_oof": round(float(cands[kv]), 4),
                     "F_all1": round(float(cands[("all1", "-")]), 4), "R_tb": tb, "R_tb_folds": tbs.get(m),
                     "R_oof": round(float(np.nanmean(oofR[selR])), 4), "R_all0": round(float(fast_miou(zeros[selR], gt[selR]).mean()), 4)}
    print(group_line("tri + R rectangles", np.where(isF, fast_miou(ones, gt), oofR), method, full))
    print(group_line("tri + F carte + R rect.", final, method, full))
    print("variantes F (hors-fold, images triées F) :",
          {f"{k}/{mo}": round(float(np.nanmean(o[isF])), 4) for (k, mo), o in oofF_by.items()})
    print("choix :", json.dumps(choice))
    # contrôle : fast_miou == métrique officielle sur 30 images R prédites
    ii = np.nonzero(~isF)[0][:30]
    assert np.allclose(fast_miou(predR(ii, 0.8), gt[ii]), [off_miou(p, g) for p, g in zip(predR(ii, 0.8), gt[ii])])

    if a.test_heats and a.write_sub:
        import torch
        dt = np.load(a.test_heats); tn = dt["names"].tolist(); tm = dt["method"]; taf = dt["acc_full"]
        meta = json.loads((Path(a.data) / "test" / "metadata.json").read_text())
        assert sorted(tn) == sorted(x["image_name"] for x in meta) and len(set(tn)) == len(tn) == 1020
        tbx, tab = load_boxes(a.test_boxes, tn); TBM = box_masks(tbx); tF = taf >= a.gate
        masks = np.zeros((len(tn), 128, 128), bool)
        for m, c in choice.items():
            sF = (tm == m) & tF; sR = (tm == m) & ~tF
            masks[sF] = True if c["F_key"] == "all1" else norm(dt[c["F_key"]][sF].astype(np.float32), c["F_mode"]) >= c["F_thr"]
            masks[sR] = TBM[sR] & (tab[sR] >= c["R_tb"])[:, None, None]
        torch.save({"image_names": tn, "masks": torch.from_numpy(masks)}, a.write_sub)
        back = torch.load(a.write_sub, weights_only=True); sz = a.write_sub.stat().st_size / 1e6
        assert back["masks"].shape == (1020, 128, 128) and back["masks"].dtype == torch.bool and sz <= 25
        print(f"soumission -> {a.write_sub} {sz:.1f} Mo ; test : {int(tF.sum())} F / {int((~tF).sum())} R ; "
              f"couverture prédite {masks.mean():.3f} (NON soumise)")


if __name__ == "__main__":
    main()
