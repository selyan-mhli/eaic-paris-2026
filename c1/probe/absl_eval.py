"""A2 — évaluation du canal fm_absl (|logit| par position) avec la porte de `gated_eval.py` (sans la modifier).

1. Fusionne `fm_absl` (absl_maps.py) et `fm_comb` = (rel(fm_acc) + rel(fm_absl)) / 2 dans une copie du npz de
   cartes (`--write-merged`) : même format que heats_train.npz, donc lisible par `gated_eval.load_split`.
2. AUC pixel (masque GT) sur les F, par méthode : fm_acc, fm_soft, fm_absl, fm_comb.
3. mIoU imbriqué (5 folds externes stratifiés méthode x full_watermark, règle « full », graines 0..2) pour
   plusieurs jeux de cartes candidates (`gated_eval.KEYS` remplacé le temps de l'appel).

    python work_c1_probe/absl_eval.py --heats output/heats_train.npz --absl output/probe_absl_train.npz \
        --boxes 'output/boxes_train_*.npz' --write-merged output/probe_heats_train_absl.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, Path.cwd()):
    sys.path.insert(0, str(p))
import gated_eval as ge  # noqa: E402
from real_eval import fast_miou, folds, group_line, norm  # noqa: E402

BASE = ("win128", "win256", "fm_acc", "fm_soft")
SETS = {
    "base": BASE,
    "base+absl": BASE + ("fm_absl",),
    "base+comb": BASE + ("fm_comb",),
    "base+absl+comb": BASE + ("fm_absl", "fm_comb"),
    "acc seul": ("fm_acc",),
    "absl seul": ("fm_absl",),
    "comb seul": ("fm_comb",),
}


def auc(score, label) -> float:
    """AUC de Mann-Whitney (rangs moyens pour les ex aequo)."""
    s = np.asarray(score, np.float64).ravel(); y = np.asarray(label, bool).ravel()
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2)[inv]
    npos = y.sum(); nneg = len(y) - npos
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else float("nan")


def merge(d, absl) -> dict:
    """Ajoute fm_absl (aligné par nom) et fm_comb aux clés du npz de cartes."""
    pos = {n: i for i, n in enumerate(absl["names"].tolist())}
    names = d["names"].tolist()
    assert set(names) <= set(pos), "images sans fm_absl"
    out = {k: d[k] for k in d.files}
    fa = np.stack([absl["fm_absl"][pos[n]] for n in names]).astype(np.float32)
    out["fm_absl"] = fa.astype(np.float16)
    out["fm_comb"] = ((norm(d["fm_acc"].astype(np.float32), "rel") + norm(fa, "rel")) / 2).astype(np.float16)
    return out


def nested(D, gt, method, full, seed) -> np.ndarray:
    fo = folds(np.array([f"{m}|{f}" for m, f in zip(method, full)]), 5, seed)
    idx = np.arange(D.n); s = np.zeros(D.n)
    for k in range(5):
        tr, te = idx[fo != k], idx[fo == k]
        r, _ = ge.fit(D, gt, tr, "full")
        print(f"    règle graine {seed} fold {k} : " + json.dumps(r), flush=True)  # « jamais choisi » vérifiable
        s[te] = fast_miou(ge.apply(D, te, r), gt[te])
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heats", type=Path, required=True); ap.add_argument("--absl", type=Path, required=True)
    ap.add_argument("--boxes", required=True); ap.add_argument("--data", default="release")
    ap.add_argument("--write-merged", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    a = ap.parse_args()
    np.savez_compressed(a.write_merged, **merge(np.load(a.heats), np.load(a.absl)))
    d, names, boxes, accb = ge.load_split(a.write_merged, a.boxes, a.data, "train")  # format accepté par la porte
    z = np.load(Path(a.data) / "train" / "masks_128.npz"); allm = dict(zip(z["names"].tolist(), z["masks"]))
    gt = np.stack([allm[n] for n in names]).astype(bool)
    method, full = d["method"], d["full_watermark"]
    print("AUC pixel sur les F (poolée ; médiane par image)")
    for m in ("pixelseal", "trustmark"):
        sel = (method == m) & full
        for k in ("fm_acc", "fm_soft", "fm_absl", "fm_comb"):
            h = d[k].astype(np.float32)
            per = [auc(h[i], gt[i]) for i in np.nonzero(sel)[0]]
            print(f"  {m:10s} {k:8s} {auc(h[sel], gt[sel]):.3f} ({np.nanmedian(per):.3f})")
    res = {}
    for name in a.sets:
        ge.KEYS = SETS[name]
        D = ge.Data(d, boxes, accb)
        sc = np.stack([nested(D, gt, method, full, s) for s in a.seeds])  # graines x 510
        res[name] = sc
        ps = method == "pixelseal"
        print(group_line(f"{name}", sc.mean(0), method, full)
              + "  | pixelseal par graine " + " ".join(f"{x:.4f}" for x in sc[:, ps].mean(1)), flush=True)
    if "base" in res:
        b = res["base"]
        for name, sc in res.items():
            dd = {m: sc[:, method == m].mean(1) - b[:, method == m].mean(1) for m in ("pixelseal", "trustmark", "stegastamp")}
            print(f"Δ vs base {name:16s} all={np.mean(sc - b):+.4f} | "
                  + " ".join(f"{m[:5]}={v.mean():+.4f} [{v.min():+.4f}, {v.max():+.4f}]" for m, v in dd.items()))


if __name__ == "__main__":
    main()
