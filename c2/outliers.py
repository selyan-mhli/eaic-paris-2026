"""c2 — règle « un gros groupe + isolés » (aucune sélection apprise, aucune requête au serveur).

Idée : on ne cherche pas à séparer fp8 ou la LoRA 0,3 (indétectables ici) ; on isole seulement les endpoints
nettement à part de leur famille (LoRA forte), le reste forme UN modèle. Sur les ref, isoler ref-07 seul
donne MCC 0,33, isoler ref-07 et ref-04 donne 0,52.

Règle fixée a priori (pas de choix sur les données) :
  1. distances par prompt avec analyze.py (noyau et statistique donnés, affixes constants retirés), moyenne
     sur tous les prompts ;
  2. pour chaque endpoint, distance moyenne aux autres endpoints de SA famille ;
  3. écart robuste z = (d - médiane) / (1,4826 × MAD) dans la famille ; z > seuil => modèle à part
     (un singleton par endpoint isolé), le reste = un seul modèle.

    python c2/outliers.py ref_answers.jsonl --prefix ref- --truth c2/kit/practice_answers.csv --sweep
    python c2/outliers.py ep_answers.jsonl --prefix ep- --families fam.csv --z 3 --out guess.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import analyze  # noqa: E402


def distance_matrix(grouped: dict, kernel: str = "hamming", stat: str = "corr", strip: bool = True):
    if strip:
        grouped = analyze.strip_constant_affixes(grouped)
    endpoints = sorted({ep for d in grouped.values() for ep in d})
    pp = analyze.per_prompt_distances(grouped, endpoints, kernel, stat)
    D = analyze.combine(pp, ("multi", "single"))
    if D is None:
        raise ValueError("aucun prompt exploitable")
    return endpoints, D


def robust_z(endpoints, D, families: dict) -> dict:
    """{endpoint: (distance moyenne aux autres de sa famille, z robuste)}. MAD nulle => z = 0."""
    out = {}
    pos = {e: i for i, e in enumerate(endpoints)}
    for fam in sorted(set(families[e] for e in endpoints)):
        members = [e for e in endpoints if families[e] == fam]
        if len(members) < 3:
            for e in members:
                out[e] = (float("nan"), 0.0)
            continue
        m = []
        for e in members:
            others = [D[pos[e], pos[o]] for o in members if o != e and np.isfinite(D[pos[e], pos[o]])]
            m.append(float(np.mean(others)) if others else float("nan"))
        m = np.array(m)
        med = float(np.nanmedian(m))
        mad = 1.4826 * float(np.nanmedian(np.abs(m - med)))
        for e, x in zip(members, m):
            out[e] = (float(x), float((x - med) / mad) if mad > 1e-12 else 0.0)
    return out


def rows_from_z(endpoints, families: dict, z: dict, thr: float) -> list:
    if not np.isfinite(thr):
        raise ValueError(f"seuil non fini ({thr})")
    fam_label = {f: analyze.letters(i) for i, f in enumerate(sorted(set(families.values())))}
    rows, k = [], {}
    for e in endpoints:
        F = fam_label[families[e]]
        if z[e][1] > thr:
            k[F] = k.get(F, 1) + 1
            rows.append((e, F, f"{F}{k[F]}"))
        else:
            rows.append((e, F, f"{F}1"))
    return rows


def read_families(path) -> dict:
    with open(path, newline="", encoding="utf-8") as f:
        return {r[0].strip(): r[1].strip() for r in csv.reader(f) if r and r[0].strip() not in ("", "endpoint")}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("answers", nargs="+")
    p.add_argument("--prefix", default="ref-")
    p.add_argument("--truth")
    p.add_argument("--families", help="CSV endpoint,family (défaut : la famille de --truth, sinon une seule)")
    p.add_argument("--kernel", default="hamming", choices=["ngram", "seq", "hamming", "both"])
    p.add_argument("--stat", default="corr", choices=["corr", "mmd"])
    p.add_argument("--max-reps", type=int)
    p.add_argument("--max-prompts", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--z", type=float, default=3.0)
    p.add_argument("--sweep", action="store_true", help="MCC en fonction du seuil (avec --truth)")
    p.add_argument("--out")
    a = p.parse_args(argv)
    g = analyze.group(analyze.load_answers(a.answers), a.prefix)
    if a.max_reps or a.max_prompts:
        g = analyze.subsample(g, a.max_prompts, a.max_reps, a.seed)
    truth = analyze.read_truth(a.truth) if a.truth else None
    eps, D = distance_matrix(g, a.kernel, a.stat)
    if a.families:
        fam = read_families(a.families)
    elif truth:
        fam = {e: truth[e][0] for e in eps}
    else:
        fam = {e: "F" for e in eps}
    z = robust_z(eps, D, fam)
    for e in sorted(eps, key=lambda e: -z[e][1]):
        print(f"  {e} fam {fam[e]} d_moy {z[e][0]:.4f} z {z[e][1]:+.2f}")
    if truth and a.sweep:
        for thr in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
            s = analyze.score(truth, rows_from_z(eps, fam, z, thr))
            print(f"  seuil z {thr:.1f} : practice MCC {s['practice_mcc']:.3f}")
    rows = rows_from_z(eps, fam, z, a.z)
    if truth:
        print(f"seuil z {a.z} : practice MCC {analyze.score(truth, rows)['practice_mcc']:.3f}")
    if a.out:
        analyze.write_rows(rows, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
