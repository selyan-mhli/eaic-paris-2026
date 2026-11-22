"""c2 — phase « empreintes » : endpoints déterministes, sorties identiques entre endpoints, première divergence.

Aucune requête : lit un answers.jsonl du kit. Règle de décision écrite AVANT de voir les réponses (PLAN.md §9) :

  D1. Un endpoint est DÉTERMINISTE si, pour chaque prompt d'empreinte, ses deux répétitions sont identiques
      (après NFC + espaces de fin retirés ; la casse compte).
  D2. Deux endpoints forment une PAIRE SÛRE s'ils sont tous deux déterministes, de la même famille, et que leurs
      réponses sont identiques sur TOUS les prompts d'empreinte présents (au moins 3), avec au moins 600
      caractères identiques au total. Un décodage greedy de poids différents (fp8, LoRA) diverge sur des sorties
      longues : l'égalité exacte est une preuve forte. L'inverse n'est pas vrai (un system prompt change aussi
      la sortie greedy) : deux sorties différentes ne prouvent pas deux modèles.
  D3. Les modèles sont les composantes connexes des paires sûres, à condition que chaque composante soit une
      clique (toutes ses paires sont sûres) ; sinon la composante est éclatée en singletons. Tout le reste est
      « différent » (la précision des paires prime sur le rappel).
  D4. Information seulement (pas de décision) : pour les paires déterministes non identiques, position de la
      première divergence (en caractères, moyenne sur les prompts) ; pour les endpoints non déterministes,
      fraction des prompts où les deux répétitions coïncident.

    python c2/fingerprint.py ep_answers.jsonl --families fam.csv --ids e01,e02,e03,e04 --out guess.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import analyze  # noqa: E402

MIN_PROMPTS = 3
MIN_CHARS = 600


def canon(t: str) -> str:
    return unicodedata.normalize("NFC", t).rstrip()


def deterministic(by_ep: dict, ids) -> dict:
    """{endpoint: True/False/None} ; None si moins de 2 répétitions sur un des prompts."""
    out = {}
    for ep in sorted({e for p in ids for e in by_ep.get(p, {})}):
        ok = []
        for p in ids:
            v = by_ep.get(p, {}).get(ep, [])
            if len(v) < 2:
                ok = None
                break
            ok.append(len({canon(t) for t in v[:2]}) == 1)
        out[ep] = None if ok is None else all(ok)
    return out


def first_divergence(a: str, b: str) -> int:
    a, b = canon(a), canon(b)
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


def sure_pairs(by_ep: dict, ids, det: dict, fam: dict) -> list:
    pairs = []
    eps = sorted(e for e, d in det.items() if d)
    for a, b in itertools.combinations(eps, 2):
        if fam.get(a) != fam.get(b):
            continue
        same, chars, n = True, 0, 0
        for p in ids:
            va, vb = by_ep.get(p, {}).get(a), by_ep.get(p, {}).get(b)
            if not va or not vb:
                continue
            n += 1
            if canon(va[0]) != canon(vb[0]):
                same = False
                break
            chars += len(canon(va[0]))
        if same and n >= MIN_PROMPTS and chars >= MIN_CHARS:
            pairs.append((a, b))
    return pairs


def components(eps, pairs) -> list:
    """Composantes connexes ; une composante qui n'est pas une clique est éclatée en singletons."""
    parent = {e: e for e in eps}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pairs:
        parent[find(a)] = find(b)
    groups = {}
    for e in eps:
        groups.setdefault(find(e), []).append(e)
    ps = {tuple(sorted(p)) for p in pairs}
    out = []
    for g in groups.values():
        if all(tuple(sorted(x)) in ps for x in itertools.combinations(g, 2)):
            out.append(sorted(g))
        else:
            out.extend([e] for e in g)
    return sorted(out)


def rows(eps, fam: dict, comps: list) -> list:
    fam_label = {f: analyze.letters(i) for i, f in enumerate(sorted(set(fam[e] for e in eps)))}
    count, out = {}, []
    for g in comps:
        F = fam_label[fam[g[0]]]
        count[F] = count.get(F, 0) + 1
        out += [(e, F, f"{F}{count[F]}") for e in g]
    return sorted(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("answers", nargs="+")
    p.add_argument("--prefix", default="ep-")
    p.add_argument("--families", required=True, help="CSV endpoint,family")
    p.add_argument("--ids", default="e01,e02,e03,e04")
    p.add_argument("--out")
    a = p.parse_args(argv)
    ids = [x for x in a.ids.split(",") if x]
    g = analyze.group(analyze.load_answers(a.answers), a.prefix)
    with open(a.families, newline="", encoding="utf-8") as f:
        fam = {r[0].strip(): r[1].strip() for r in csv.reader(f) if r and r[0].strip() not in ("", "endpoint")}
    det = deterministic(g, ids)
    eps = sorted(det)
    print("déterministes :", " ".join(e[-2:] for e in eps if det[e]) or "aucun")
    print("non déterministes :", " ".join(e[-2:] for e in eps if det[e] is False) or "aucun")
    pairs = sure_pairs(g, ids, det, fam)
    print("paires sûres :", ", ".join(f"{x[-2:]}={y[-2:]}" for x, y in pairs) or "aucune")
    for x, y in itertools.combinations([e for e in eps if det[e]], 2):
        if fam.get(x) == fam.get(y) and (x, y) not in pairs:
            d = [first_divergence(g[q][x][0], g[q][y][0]) for q in ids if g.get(q, {}).get(x) and g[q].get(y)]
            if d:
                print(f"  divergence {x[-2:]}-{y[-2:]} : moyenne {sum(d) / len(d):.0f} car. ({d})")
    comps = components(eps, pairs)
    r = rows(eps, fam, comps)
    same = sum(1 for x, y in itertools.combinations(r, 2) if x[1:] == y[1:])
    print(f"{len(comps)} modèles, {same} paires « même modèle »")
    if a.out:
        analyze.write_rows(r, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
