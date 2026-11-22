"""Information locale des sondes sous une approximation binaire logistique.

N'est ni une puissance mesurée sur le serveur ni une garantie de borne : masses
hors paire, top-p, agrégation de tokens et watermark ne sont pas modélisés.
"""
import argparse
import json
from math import erf, sqrt
from pathlib import Path

import numpy as np


def efficient_signal(signal, nuisance, weight):
    y = np.sqrt(weight)*signal
    x = np.sqrt(weight)[:,None]*nuisance
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    rank = int(np.sum(s > max(s.max(), 1.)*1e-10))
    basis = u[:,:rank]
    residual = y-basis@(basis.T@y)
    return float(y@y), float(residual@residual), rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probes',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    ps=json.loads(a.probes.read_text())['probes']
    d=np.array([p['d0'] for p in ps]); t=.7
    prob=1/(1+np.exp(-d/t))
    # Dérivée de d/T par rapport à T, et directions système simulées.
    temp=(-d/t**2)[:,None]
    contexts=np.array([p['d_ctx'] for p in ps])-d[:,None]
    output={}
    for total in (55, 110):
        # Hypothèse optimiste : chaque réponse appartient à la paire.
        weight=(total/len(ps))*prob*(1-prob)
        for scheme in ps[0]['d_fp8']:
            signal=(np.array([p['d_fp8'][scheme] for p in ps])-d)/t
            for name,nuisance in [('temperature',temp),('temperature_and_contexts',np.column_stack((temp,contexts/t)))]:
                raw,eff,rank=efficient_signal(signal,nuisance,weight)
                # Puissance gaussienne d'un score DIRECTIONNEL unilatéral, alpha=.001.
                power=.5*(1+erf((sqrt(max(0,eff))-3.090232)/sqrt(2)))
                output[f'{total}|{scheme}|{name}']=dict(raw_information=raw,efficient_information=eff,
                    nuisance_rank=rank,optimistic_local_directed_power=power)
    out=dict(nature='Approximation logistique locale fictive ; pas de puissance serveur',
             temperature=t,results=output)
    a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
