"""Mesure ROUTÉE du candidat « TrustMark dense seul » contre le champion, CPU, sans réseau.

Routage = celui du champion tel que disponible au test (aucune vérité F/R) : partition externe graine 0 de
`head_eval.py`, pour le fold k : image routée F <=> acc_full >= gate(seed 0, fold k) (lu dans `choices` du npz
de référence). Les TrustMark routés F (vrais F ET vrais R) reçoivent le masque dense au seuil FIGÉ enregistré
dans le npz dense (seuil interne du pilote) ; toutes les autres images gardent le score du champion.
Effet global = somme des écarts / 510 (images inchangées : écart 0), bootstrap apparié sur les 510 images.

  python measure_routed.py --dense res_trustmark_head_15903857.npz \
      --ref segf_sub_1_champion_oof.npz --ref2 segf_ens3_3.npz \
      --acc champion_boxes_train.npz --masks masks_128.npz --out mesure.json

--dense : un ou plusieurs npz (names, probs_oof|probs, thr) ; pour chaque image, le PREMIER fichier qui la
prédit (probabilités finies) est utilisé. Une image routée F sans prédiction est « manquante » : bornes [0, 1]
rapportées et décision INCOMPLET.
Règle de décision (fixée avant le job de mesure, référence = --ref, portée = --scope) :
  EXPORT_OK  si gain global >= +0,010 ET contribution des TM vrais R routés F >= -0,001 ET aucune manquante.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import data as pdata  # noqa: E402  c1/pixseg/data.py : fast_miou, real_folds (copies testées)

SCOPES = {"confirm": (0, 1), "all": (0, 1, 2, 3, 4)}
GAIN_MIN, R_CONTRIB_MIN = 0.010, -0.001


def load_dense(files):
    """-> dict nom -> (masque bool 128 128 au seuil figé, seuil, fichier)."""
    out = {}
    for f in files:
        z = np.load(f)
        P = z["probs_oof"] if "probs_oof" in z.files else z["probs"]
        for n, p, t in zip(z["names"].tolist(), P, z["thr"]):
            p = p.astype(np.float32)
            if n in out or not np.isfinite(t) or not np.isfinite(p).all():
                continue
            out[n] = (p > t, float(t), str(f))
    return out


def gates(ref):
    ch = json.loads(str(ref["choices"]))
    return {c["fold"]: c["gate"] for c in ch if c["seed"] == 0}


def boot(d, n=5000, seed=0):
    rng = np.random.default_rng(seed); b = rng.integers(0, len(d), (n, len(d)))
    return [float(np.percentile(d[b].mean(1), q)) for q in (2.5, 97.5)]


def measure(dense, ref, acc, masks, scope):
    names = ref["names"]; n = len(names)
    pos = {x: i for i, x in enumerate(acc["names"].tolist())}
    ia = np.array([pos[x] for x in names.tolist()])
    method, af = acc["method"][ia], acc["acc_full"][ia].astype(np.float32)
    full = acc["full_watermark"][ia].astype(bool)
    fold = pdata.real_folds(method, full, 0)  # = folds(method|full, 5, 0) de head_eval / analyze_champion
    g = gates(ref); routedF = af >= np.array([g[k] for k in fold])
    base = ref["head"][0].astype(np.float64)
    cand = base.copy()
    sel = (method == "trustmark") & routedF & np.isin(fold, SCOPES[scope])
    missing = [names[i] for i in np.nonzero(sel)[0] if names[i] not in dense]
    for i in np.nonzero(sel)[0]:
        if names[i] in dense:
            cand[i] = pdata.fast_miou(dense[names[i]][0][None], masks[names[i]][None].astype(bool))[0]
    d = cand - base
    res = {"scope": scope, "folds": list(SCOPES[scope]), "n_images": int(n), "gates_seed0": g,
           "ref_global": float(base.mean()), "cand_global": float(cand.mean()),
           "gain_global": float(d.mean()), "ic95_global_apparie": boot(d), "missing": missing}
    for lab, q in (("TM_F_routes_F", sel & full), ("TM_R_routes_F", sel & ~full)):
        ok = q & np.array([x in dense for x in names.tolist()])
        res[lab] = {"n": int(q.sum()), "n_predits": int(ok.sum()),
                    "ref": float(base[ok].mean()) if ok.any() else None,
                    "dense": float(cand[ok].mean()) if ok.any() else None,
                    "gain_moyen": float(d[ok].mean()) if ok.any() else None,
                    "ic95_apparie": boot(d[ok]) if ok.sum() > 1 else None,
                    "contribution_globale": float(d[ok].sum() / n),
                    "bornes_manquantes_globales": [float(-base[q & ~ok].sum() / n), float((1 - base[q & ~ok]).sum() / n)]}
    q = (method == "trustmark") & ~routedF & np.isin(fold, SCOPES[scope])
    res["TM_routes_R_inchanges"] = {"n": int(q.sum()), "n_vrais_F": int((q & full).sum())}
    res["thr_utilises"] = sorted({dense[x][1] for x in names[sel].tolist() if x in dense})
    rc = res["TM_R_routes_F"]["contribution_globale"]
    if missing:
        res["decision"] = "INCOMPLET"
    elif res["gain_global"] >= GAIN_MIN and rc >= R_CONTRIB_MIN:
        res["decision"] = "EXPORT_OK"
    else:
        res["decision"] = "PAS_D_EXPORT"
    res["export_ok"] = res["decision"] == "EXPORT_OK"
    res["regle"] = f"gain global >= {GAIN_MIN:+.3f} et contribution TM R routés F >= {R_CONTRIB_MIN:+.3f}, sans manquante"
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", nargs="+", required=True)
    ap.add_argument("--ref", required=True, help="champion exact (segf_sub_1.npz, ff+fm e20) : base de la décision")
    ap.add_argument("--ref2", help="référence secondaire (segf_ens3_3.npz, head[0] = 0,5606), rapportée seulement")
    ap.add_argument("--acc", required=True, help="npz names, method, acc_full, full_watermark (heats_train.npz)")
    ap.add_argument("--masks", required=True)
    ap.add_argument("--scope", default="all", choices=list(SCOPES), help="portée de la décision")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    zm = np.load(a.masks); masks = dict(zip(zm["names"].tolist(), zm["masks"]))
    acc = dict(np.load(a.acc)); dense = load_dense(a.dense)
    out = {"fichiers": {"dense": a.dense, "ref": a.ref, "ref2": a.ref2}, "decision_scope": a.scope}
    for lab, f in (("ref", a.ref), ("ref2", a.ref2)):
        if not f:
            continue
        ref = np.load(f)
        out[lab] = {s: measure(dense, ref, acc, masks, s) for s in SCOPES}
    dec = out["ref"][a.scope]
    out["decision"], out["export_ok"] = dec["decision"], dec["export_ok"]
    txt = json.dumps(out, indent=1, ensure_ascii=False)
    print(txt)
    for lab in ("ref", "ref2"):
        for s, r in out.get(lab, {}).items():
            print(f"[{lab} {s:7s}] global {r['ref_global']:.4f} -> {r['cand_global']:.4f} ({r['gain_global']:+.4f}, "
                  f"IC95 [{r['ic95_global_apparie'][0]:+.4f} ; {r['ic95_global_apparie'][1]:+.4f}]) | TM F n={r['TM_F_routes_F']['n']} "
                  f"gain {r['TM_F_routes_F']['gain_moyen']} | TM R->F n={r['TM_R_routes_F']['n']} prédits "
                  f"{r['TM_R_routes_F']['n_predits']} contrib {r['TM_R_routes_F']['contribution_globale']:+.4f} | {r['decision']}")
    print(f"DÉCISION ({a.scope}, contre {a.ref}) : {out['decision']}")
    if a.out:
        Path(a.out).write_text(txt + "\n")
    return out


if __name__ == "__main__":
    main()
