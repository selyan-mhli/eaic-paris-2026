"""Audit CPU sans API : sondes réellement collectées, filtre et puissance du test.

Le calcul de puissance porte sur une alternative binaire fictive explicite,
pas sur le fp8 du serveur. La règle historique n'est pas recalibrée.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path

import numpy as np


def describe(probes):
    rows = []
    for j, p in enumerate(probes):
        shift = {k: float(v - p['d0']) for k, v in p['d_fp8'].items()}
        rows.append(dict(probe=j, kind=p['kind'], d0=p['d0'], mass=p['mass'],
                         initial_tie=bool(abs(p['d0']) <= .5),
                         context_shift=max(abs(v-p['d0']) for v in p['d_ctx']),
                         fp8_shift=shift, coherent_direction=all(v > 0 for v in shift.values())
                         or all(v < 0 for v in shift.values())))
    return rows


def conditional_information(d0, shift, t=.7):
    """Deux tokens scalaires, température connue, masse paire=1, observations iid."""
    p = 1 / (1 + np.exp(-d0/t))
    q = 1 / (1 + np.exp(-(d0+shift)/t))
    kl = q*np.log(q/p)+(1-q)*np.log((1-q)/(1-p))
    return dict(p=float(p), q=float(q), kl_per_response=float(kl),
                expected_2_log_likelihood_ratio_at_n10=float(20*kl))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probes', type=Path, required=True)
    ap.add_argument('--log', type=Path, required=True)
    ap.add_argument('--rule', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    probes = json.loads(a.probes.read_text())['probes']
    recs = [json.loads(s) for s in a.log.read_text().splitlines()]
    good = [r for r in recs if r.get('status') == 200 and 'text' in r]
    keys = [(r['ep'], r['probe'], r['rep']) for r in good]
    if len(keys) != len(set(keys)):
        raise ValueError('Plusieurs réussites pour une même cellule ; audit interrompu')
    spec = importlib.util.spec_from_file_location('historical_ties_rule', a.rule)
    rule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rule)
    counts = defaultdict(lambda: defaultdict(lambda: np.zeros(3)))
    for r in good:
        p = probes[r['probe']]
        t = (r['text'] or '').strip()
        c = 0 if t == p['s1'] else 1 if t == p['s2'] else 2
        counts[r['ep']][r['probe']][c] += 1
    threshold = rule.chi2_ppf999(2*len(probes)-1)
    endpoints = {}
    for ep, c in sorted(counts.items()):
        g, t = rule.gstat(c, probes)
        endpoints[ep] = dict(G=float(g), T=float(t), classified_base=bool(g <= threshold),
                             n=int(sum(x.sum() for x in c.values())),
                             counts={str(k): v.astype(int).tolist() for k,v in c.items()})
    desc = describe(probes)
    # Contre-exemple de puissance : p=.5 -> .588 sous un décalage de .25 nat à T=.7.
    # Même avec T connue, information très faible par réponse ; T profilée réduit encore
    # la séparabilité lorsqu'une alternative peut être absorbée dans ce paramètre.
    toy = conditional_information(0, .25)
    rng = np.random.default_rng(20261122)
    power = {}
    for n in (5, 10, 60, 100):
        x = rng.binomial(n, toy['q'], size=100000)
        ph = np.clip(x/n, 1e-12, 1-1e-12)
        g = 2*n*(ph*np.log(ph/.5)+(1-ph)*np.log((1-ph)/.5))
        power[str(n)] = float((g > rule.chi2_ppf999(1)).mean())
    out = dict(nature='Audit du test existant ; aucune nouvelle méthode validée',
               attempts=len(recs), statuses=dict(Counter(str(r.get('status')) for r in recs)),
               probes=len(probes), probe_details=desc,
               tie_count=sum(r['kind']=='tie' for r in desc),
               ties_outside_initial_gap=sum(r['kind']=='tie' and not r['initial_tie'] for r in desc),
               threshold=float(threshold), endpoints=endpoints,
               toy_known_temperature_binary_alternative=toy,
               toy_rejection_probability_alpha_001=power,
               caveats=['Puissance simulée pour deux tokens scalaires, sans watermark, contexte ou top-p.',
                        'Non-rejet de l égalité des distributions ne démontre pas l identité des poids.',
                        'Les textes identiques du watermark peuvent rendre les répétitions dépendantes.'])
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('probe_details','endpoints')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
