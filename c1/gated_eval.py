"""Méthode « tri F/R » : évaluation en validation croisée imbriquée sur le train réel + fichier de soumission.

    python work_c1/gated_eval.py --heats output/heats_train.npz --boxes 'output/boxes_train_*.npz' \
        [--test-heats output/heats_test.npz --test-boxes 'output/boxes_test_*.npz' \
         --rule full --write-sub output/c1_<rule>_<date>.pt]

Règle, par image (méthode et message connus, rien d'autre) :
  1. tri : acc_full (bit accuracy de l'image entière) >= g -> régime F (image entière watermarkée puis
     zones restaurées), sinon R (rectangle seul). g dans GATES (<= gate de box_search, 0.75).
  2. F : rule=gate|gate_box -> tout 1 ; rule=full -> meilleure carte (win128/win256/fm_*, abs/rel) >= t,
     tout 1 inclus comme candidat (t = 0).
  3. R : rule=gate -> tout 0 ; sinon rectangle de box_search.py si acc_box >= tb, sinon tout 0.
Ajustement (fit) : g, puis par méthode variante + t et tb, en maximisant le mIoU moyen sur les images
d'ajustement. Évaluation : 5 folds externes stratifiés méthode x full_watermark ; le fit ne voit que les
4 folds d'ajustement, le score est celui du fold externe (score imbriqué). On affiche aussi le score
« non imbriqué » (fit et score sur les 510, optimiste). La soumission refit sur les 510.
Rien n'est soumis.
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
GATES = (0.60, 0.65, 0.70, 0.75)
BOX_GATE = 0.75  # box_search.py --gate (défaut) : au-delà, pas de recherche
THR_F = np.round(np.arange(0.0, 1.02, 0.01), 2)
THR_B = np.round(np.arange(0.50, 1.02, 0.01), 2)


def load_split(heats: Path, boxes_glob: str, data: str, split: str):
    meta = json.loads((Path(data) / split / "metadata.json").read_text())
    mm = {x["image_name"]: x["method"] for x in meta}
    d = np.load(heats); names = d["names"].tolist()
    assert len(names) == len(set(names)) == len(meta) and set(names) == set(mm), f"{split}: noms != métadonnées (sortie --limit ?)"
    assert all(mm[n] == m for n, m in zip(names, d["method"].tolist())), f"{split}: méthodes != métadonnées"
    assert np.isfinite(d["acc_full"]).all()
    files = sorted(glob.glob(boxes_glob)); assert files, boxes_glob
    bx, ab, seen = {}, {}, []
    for f in files:
        z = np.load(f); fn = z["names"].tolist(); seen += fn
        assert set(fn) <= set(mm), f"{f} : noms hors {split}"
        for n, b, a, af in zip(fn, z["box"], z["acc_box"], z["acc_full"]):
            bx[n] = b; ab[n] = a
    assert len(seen) == len(set(seen)), "doublons entre shards de boîtes"
    assert set(seen) == set(names), f"{split}: {len(set(names) - set(seen))} images sans boîte"
    boxes = np.stack([bx[n] for n in names]); accb = np.array([ab[n] for n in names], np.float32)
    af = d["acc_full"]
    searched = af < BOX_GATE
    full_box = (boxes == [0, 512, 0, 512]).all(1)
    assert full_box[~searched].all(), "images non cherchées sans boîte pleine : gate box_search différent ?"
    assert np.isfinite(accb).all()
    return d, names, boxes, accb


def box_masks(boxes):
    out = np.zeros((len(boxes), 128, 128), bool)
    for i, (y0, y1, x0, x1) in enumerate(boxes):
        m = np.zeros((512, 512), np.uint8); m[y0:y1, x0:x1] = 1
        out[i] = downsample_mask(m).astype(bool)
    return out


class Data:
    def __init__(self, d, boxes, accb):
        self.method = d["method"]; self.af = d["acc_full"].astype(np.float32); self.accb = accb
        self.BM = box_masks(boxes); self.n = len(self.af)
        self.H = {(k, mo): norm(d[k].astype(np.float32), mo) for k in KEYS if k in d.files for mo in ("abs", "rel")}

    def pred_F(self, idx, var, t):
        if var == ("all1", "-"):
            return np.ones((len(idx), 128, 128), bool)
        return self.H[var][idx] >= t

    def pred_R(self, idx, tb):
        return self.BM[idx] & (self.accb[idx] >= tb)[:, None, None]


def fit(D: Data, gt, idx, rule):
    """Choisit g puis, par méthode, (variante F, t) et tb sur les images idx. Renvoie (règle, score fit)."""
    best = None
    for g in GATES:
        r = {"rule": rule, "gate": g, "m": {}}; tot = 0.0
        for m in METHODS:
            iF = idx[(D.method[idx] == m) & (D.af[idx] >= g)]; iR = idx[(D.method[idx] == m) & (D.af[idx] < g)]
            # F
            cands = [(("all1", "-"), 0.0)]
            if rule == "full":
                for var, H in D.H.items():
                    if len(iF) and not np.isfinite(H[iF]).all():
                        continue  # carte indisponible pour cette méthode (StegaStamp : fm_*)
                    cands += [(var, float(t)) for t in THR_F]
            sF = [(fast_miou(D.pred_F(iF, v, t), gt[iF]).sum(), v, t) for v, t in cands] if len(iF) else [(0.0, ("all1", "-"), 0.0)]
            bF = max(sF, key=lambda x: x[0])
            # R
            tbs = [1.01] if rule == "gate" else list(THR_B)
            sR = [(fast_miou(D.pred_R(iR, tb), gt[iR]).sum(), float(tb)) for tb in tbs] if len(iR) else [(0.0, 1.01)]
            bR = max(sR, key=lambda x: x[0])
            r["m"][m] = {"F_var": list(bF[1]), "F_thr": bF[2], "R_tb": bR[1]}
            tot += bF[0] + bR[0]
        if best is None or tot > best[1]:
            best = (r, tot)
    return best[0], best[1] / len(idx)


def apply(D: Data, idx, r):
    out = np.zeros((len(idx), 128, 128), bool)
    for m, c in r["m"].items():
        sel = D.method[idx] == m
        isF = D.af[idx] >= r["gate"]
        a, b = np.nonzero(sel & isF)[0], np.nonzero(sel & ~isF)[0]
        if len(a):
            p = D.pred_F(idx[a], tuple(c["F_var"]), c["F_thr"]); out[a] = p
        if len(b):
            out[b] = D.pred_R(idx[b], c["R_tb"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heats", type=Path, required=True); ap.add_argument("--boxes", required=True)
    ap.add_argument("--data", default="release")
    ap.add_argument("--test-heats", type=Path); ap.add_argument("--test-boxes")
    ap.add_argument("--rule", choices=["gate", "gate_box", "full"], default="full", help="règle du fichier écrit")
    ap.add_argument("--write-sub", type=Path); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    d, names, boxes, accb = load_split(a.heats, a.boxes, a.data, "train")
    z = np.load(Path(a.data) / "train" / "masks_128.npz"); allm = dict(zip(z["names"].tolist(), z["masks"]))
    assert len(allm) == 510
    gt = np.stack([allm[n] for n in names]); assert gt.shape == (510, 128, 128) and set(np.unique(gt)) <= {0, 1}
    full = d["full_watermark"]; method = d["method"]
    D = Data(d, boxes, accb)
    fo = folds(np.array([f"{m}|{f}" for m, f in zip(method, full)]), 5, a.seed)
    allidx = np.arange(D.n)
    print(f"510 images train ; tri à 0.75 : {int(((D.af >= .75) != full).sum())} erreurs F/R")
    print(group_line("tout0", fast_miou(np.zeros_like(gt, bool), gt), method, full))
    rules = {}
    for rule in ("gate", "gate_box", "full"):
        nested = np.zeros(D.n); gs = []
        for k in range(5):
            tr, te = allidx[fo != k], allidx[fo == k]
            r, _ = fit(D, gt, tr, rule); gs.append(r["gate"])
            nested[te] = fast_miou(apply(D, te, r), gt[te])
        r_all, s_in = fit(D, gt, allidx, rule); rules[rule] = r_all
        print(group_line(f"{rule} imbriqué", nested, method, full) + f"  (g folds {gs})")
        print(f"{'':24s} non imbriqué (fit=score sur 510) = {s_in:.4f} ; règle refit : {json.dumps(r_all)}")
    # contrôle croisé fast_miou == officiel sur la règle complète
    p = apply(D, allidx[:40], rules["full"])
    assert np.allclose(fast_miou(p, gt[:40]), [off_miou(x, y) for x, y in zip(p, gt[:40])])

    if a.test_heats and a.write_sub:
        import torch
        dt, tn, tbx, tab = load_split(a.test_heats, a.test_boxes, a.data, "test")
        assert len(tn) == 1020
        T = Data(dt, tbx, tab); r = rules[a.rule]
        for m, c in r["m"].items():  # entrées utilisées finies
            if c["F_var"][0] != "all1":
                sel = (T.method == m) & (T.af >= r["gate"])
                assert np.isfinite(T.H[tuple(c["F_var"])][sel]).all(), f"NaN test {m}"
        masks = apply(T, np.arange(T.n), r)
        torch.save({"image_names": tn, "masks": torch.from_numpy(masks)}, a.write_sub)
        back = torch.load(a.write_sub, weights_only=True); sz = a.write_sub.stat().st_size / 1e6
        assert back["image_names"] == tn and back["masks"].shape == (1020, 128, 128) and back["masks"].dtype == torch.bool and sz <= 25
        isF = T.af >= r["gate"]
        print(f"soumission [{a.rule}] -> {a.write_sub} {sz:.1f} Mo ; test {int(isF.sum())} F / {int((~isF).sum())} R ; "
              f"couverture prédite {masks.mean():.3f} ; règle {json.dumps(r)} (NON soumise)")


if __name__ == "__main__":
    main()
