"""Résumé de recolle.py : force du watermark observée vs témoins frais, par méthode (CPU, numpy seul).

    python c1/probe/recolle_report.py output/probe_recolle_{pixelseal,trustmark,stegastamp}.npz
Dénominateur principal : témoin B (même message m, embarquement frais sur une image R propre à >= 90 %,
même masque). Le témoin A (autre message, ré-embarqué sur l'image déjà watermarquée) est donné pour
information : il contient l'interférence du watermark d'origine.
Règle (fixée APRÈS avoir vu les médianes globales du job : indicative, pas un test pré-enregistré) :
« watermark dégradé » si la médiane obs/B < 0.5 ou la bit acc médiane < 0.9 ; « watermark préservé hors zone
(≈ frais) » si la médiane >= 0.8 et la bit acc médiane >= 0.98 ; sinon « indécis ».
Limite : cela montre que le watermark est PRÉSERVÉ dans la zone non éditée, pas que cette zone a été recollée
pixel à pixel (une régénération qui préserve le watermark donnerait le même résultat ; aucun original avant
édition ni témoin régénéré n'est comparé).
"""
from __future__ import annotations

import sys

import numpy as np


def q(v):
    v = np.asarray(v, np.float64); v = v[np.isfinite(v)]
    return f"{np.median(v):.3f} [{np.quantile(v, .1):.3f}, {np.quantile(v, .9):.3f}]"


def verdict(ratio, acc) -> str:
    r, a = np.nanmedian(ratio), np.nanmedian(acc)
    if r < 0.5 or a < 0.9:
        return "watermark dégradé"
    if r >= 0.8 and a >= 0.98:
        return "watermark préservé hors zone (≈ frais) ; recollage pixel à pixel non démontré"
    return "indécis"


for f in sys.argv[1:]:
    z = np.load(f); m = str(z["method"])
    ra = z["obs_marg_m"] / z["ctlA_marg_m"]; rb = z["obs_marg_m"] / z["ctlB_marg_m"]
    print(f"== {m} : {len(z['names'])} images F (médiane [p10, p90])")
    print(f"  bit acc  obs masque {q(z['obs_acc_m'])} | témoin B {q(z['ctlB_acc_m'])} | témoin A {q(z['ctlA_acc_m'])} | null R {q(z['nullR_acc'])}")
    print(f"  marge    obs masque {q(z['obs_marg_m'])} | témoin B {q(z['ctlB_marg_m'])} | témoin A {q(z['ctlA_marg_m'])} | null R {q(z['nullR_marg'])}")
    print(f"  ratio obs/B {q(rb)} (principal) ; obs/A {q(ra)} ; part ratio_B < 0.5 : {np.mean(rb < 0.5):.3f} ; obs_acc_m < 0.9 : {np.mean(z['obs_acc_m'] < 0.9):.3f}")
    if np.isfinite(z["obs_agr"]).any():
        print(f"  accord par position (cellules dans le masque) obs {q(z['obs_agr'])} | témoin B {q(z['ctlB_agr'])}")
    print(f"  HF |Y - flou| F : masque {q(z['hf_in'])} / trous {q(z['hf_hole'])} ; R : rectangle {q(z['hfR_in'])} / hors {q(z['hfR_out'])}")
    print(f"  => {verdict(rb, z['obs_acc_m'])}")
