"""Lecture du pilote (CPU, après le job) : localisateur dense contre la référence head[0] du 0,5614, par image.

  python c1/pixseg/dense/compare.py --res res_pixelseal_head_<job>.npz res_trustmark_head_<job>.npz \
      --res-last res_pixelseal_last_<job>.npz res_trustmark_last_<job>.npz \
      --ref docs/task1-audit-20261121/segf_ens3_3.npz --masks c1/seg/data/masks_128.npz

Règle fixée AVANT lecture de la confirmation :
  mode retenu = head, sauf si le contrôle `head` échoue pour la méthode (report.control.pass) -> last.
  Puis, sur les folds 0-1 seulement : gain F moyen PS/TM (dense à seuil choisi en dev interne - head[0]) :
    >= +0,08, gain dans les DEUX méthodes, IoU des trous en hausse, pas d'effondrement < 1 % -> POURSUIVRE
    <  +0,03, ou contrôle des deux modes échoué                                         -> ABANDON
    entre les deux                                                                       -> DIAGNOSTIC
La référence head[0] est le score par image routé (régime prédit) ; le dense est évalué à régime F connu
(diagnostic). Le score global routé reste à calculer en remplaçant la branche F dans l'assemblage.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import data as pdata  # noqa: E402


def per_image(z, masks):
    names = z["names"]; gt = np.stack([masks[n] for n in names]).astype(bool)
    pred = z["probs_oof"].astype(np.float32) > z["thr"][:, None, None]
    m = pdata.fast_miou(pred, gt)
    p = pred.reshape(len(pred), -1); g = gt.reshape(len(gt), -1)
    u = (~p | ~g).sum(1); holes = np.where(u == 0, 1.0, (~p & ~g).sum(1) / np.maximum(u, 1))
    return names, m, holes, (1 - g.mean(1)) < 0.01


def boot(d, n=2000, seed=0):
    rng = np.random.default_rng(seed); b = rng.integers(0, len(d), (n, len(d)))
    return [float(np.percentile(d[b].mean(1), q)) for q in (2.5, 97.5)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", nargs="+", required=True); ap.add_argument("--res-last", nargs="*", default=[])
    ap.add_argument("--ref", required=True); ap.add_argument("--masks", required=True)
    ap.add_argument("--phase", default="confirm")
    a = ap.parse_args(argv)
    zm = np.load(a.masks); masks = dict(zip(zm["names"].tolist(), zm["masks"]))
    zr = np.load(a.ref); ref = dict(zip(zr["names"].tolist(), zr["head"][0]))
    last = {json.loads(str(np.load(f)["config"]))["method"]: f for f in a.res_last}
    out, gains, ok_all = {}, [], True
    for f in a.res:
        z = np.load(f); cfg = json.loads(str(z["config"])); rep = json.loads(str(z["report"]))
        meth, chosen = cfg["method"], f
        if not rep["control"].get("pass", True):
            chosen = last.get(meth)
            if chosen is None or not json.loads(str(np.load(chosen)["report"]))["control"].get("pass", False):
                out[meth] = "contrôle échoué dans les deux modes"; ok_all = False; continue
            z = np.load(chosen)
        names, m, holes, small = per_image(z, masks)
        sel = z["phase"] == a.phase
        r = np.array([ref[n] for n in names[sel]])
        d = m[sel] - r; gains.append(d.mean())
        out[meth] = {"fichier": str(chosen), "n": int(sel.sum()), "dense": float(m[sel].mean()), "ref_head0": float(r.mean()),
                     "gain": float(d.mean()), "ic95_apparié_descriptif": boot(d), "iou_trous_dense": float(holes[sel].mean()),
                     "n_<1%": int(small[sel].sum()),
                     "gain_<1%": float(d[small[sel]].mean()) if small[sel].any() else None}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if not ok_all or not gains:
        print("DÉCISION : ABANDON (contrôle)"); return out
    g = float(np.mean(gains))
    both = all(isinstance(v, dict) and v["gain"] > 0 for v in out.values())
    no_collapse = all(isinstance(v, dict) and (v["gain_<1%"] is None or v["gain_<1%"] > -0.02) for v in out.values())
    dec = "POURSUIVRE" if g >= 0.08 and both and no_collapse else ("ABANDON" if g < 0.03 else "DIAGNOSTIC")
    print(f"gain F moyen PS/TM ({a.phase}) {g:+.4f} ; deux méthodes en gain : {both} ; <1 % sans effondrement : "
          f"{no_collapse}\nDÉCISION : {dec} (vérifier aussi la hausse de l'IoU des trous contre l'assemblage de référence)")
    return out


if __name__ == "__main__":
    main()
