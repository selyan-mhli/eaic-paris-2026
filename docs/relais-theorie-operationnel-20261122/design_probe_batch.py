"""Sélection locale d'un ENSEMBLE de sondes, sans API ni nouveaux logits.

Un ensemble peut distinguer des effets corrélés de contexte, même si ses sondes
ne sont pas robustes individuellement. Approximation binaire locale seulement.
Le score n'est pas une puissance validée et ne déclenche aucune collecte.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def information(rows, total=60, t=.7):
    d=np.array([r['d0'] for r in rows])
    p=1/(1+np.exp(-d/t))
    w=(total/len(rows))*p*(1-p)
    c=np.column_stack((-d/t**2,(np.array([r['d_ctx'] for r in rows])-d[:,None])/t))
    x=np.sqrt(w)[:,None]*c
    u,s,_=np.linalg.svd(x,full_matrices=False)
    rank=int(np.sum(s>max(s.max(),1.)*1e-10))
    basis=u[:,:rank]
    result={}
    for scheme in rows[0]['d_fp8']:
        y=np.sqrt(w)*(np.array([r['d_fp8'][scheme] for r in rows])-d)/t
        residual=y-basis@(basis.T@y)
        result[scheme]=float(residual@residual)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--search',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    data=json.loads(a.search.read_text())
    # Conserver une masse suffisante et éviter les catégories de même texte.
    rows=[r for r in data['all'] if r['mass']>=.8 and min(r['mass_ctx'])>=.8
          and r['s1']!=r['s2'] and abs(r['d0'])<=2.
          # Les log-odds extrêmes et les grandes perturbations ne permettent pas
          # d'interpréter le développement local de Fisher utilisé ici.
          and max(abs(v) for v in r['d_ctx']+list(r['d_fp8'].values()))<=4.
          and max(abs(v-r['d0']) for v in r['d_fp8'].values())<=1.]
    k=12
    if len(rows)<k:
        result=dict(candidate_count=len(rows),status='MOINS DE 12 CANDIDATS ; aucune collecte')
    else:
        # Départ diversifié aléatoire, puis échange glouton de coordonnées.
        # Une seule règle locale ; jamais d'évaluation sur les réponses API.
        rng=np.random.default_rng(20261122)
        chosen=list(rng.choice(len(rows),size=k,replace=False))
        best=min(information([rows[i] for i in chosen]).values())
        for _ in range(2):
            for position in range(k):
                for i in rng.permutation(len(rows)):
                    if i in chosen:continue
                    trial=chosen.copy();trial[position]=int(i)
                    score=min(information([rows[j] for j in trial]).values())
                    if score>best+1e-10:
                        chosen,best=trial,score
        selected=[rows[i] for i in chosen]
        result=dict(nature='Plan local fictif, non validé ; pas de collecte automatique',
            candidate_count=len(rows),budget_per_endpoint=60,probes=selected,
            efficient_information_by_scheme=information(selected),
            min_information=best,status='VALIDATION REFS NEUVE NECESSAIRE',
            caveats=['Contexte représenté par 5 directions simulées ; les vrais contextes restent inconnus.',
                     'Approximation de p(s1 | s1 ou s2) ; top-p, watermark, autres tokens absents.',
                     'Ce plan ne teste pas la LoRA faible et ne passe donc pas le critère ep.'])
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=1)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='probes'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
