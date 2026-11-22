"""Test exploratoire CPU : retirer les 64 premiers tokens des réponses existantes.

Hypothèse : le contexte caché affecterait davantage le début que la suite.
Seuils appris sur une moitié des PROMPTS ; évaluation sur les autres prompts.
Une réponse entière est une observation, pas chaque token. Aucun appel API.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

TRUTH={f'ref-{i:02d}': ('BASE' if i in (1,3,5,6,8) else str(i)) for i in range(1,10)}
EPS=sorted(TRUTH)


def mcc(pred):
    tp=fp=fn=tn=0
    for a,b in itertools.combinations(EPS,2):
        t=TRUTH[a]==TRUTH[b];p=pred[a]==pred[b]
        tp+=t and p;fp+=(not t) and p;fn+=t and not p;tn+=(not t) and not p
    den=np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    return max(0.,(tp*tn-fp*fn)/den) if den else 0.


def partition(feats,g,l):
    return {e:(e if abs(feats[e]['GAP'])>g or feats[e]['LLR8']>l else 'BASE') for e in EPS}


def extract(z, prompts, skip):
    out={}
    for ep in EPS:
        responses=[]
        for pid,rep in itertools.product(prompts,sorted(set(z['rep']))):
            ix=(z['ep']==ep)&(z['pid']==pid)&(z['rep']==rep)&(z['pos']>=skip)
            if ix.any():
                responses.append((z['lp_b'][ix].mean(0),z['lp_q'][ix].mean(0),z['H_b'][ix].mean(0)))
        if len(responses)<8:
            raise ValueError(f'{ep} : {len(responses)} réponses seulement après retrait')
        b,q,h=np.asarray(responses).mean(0)
        k=int(b.argmax())
        out[ep]=dict(T=float(z['TG'][k]),LLR8=float(q[k]-b[k]),GAP=float(b[k]+h[k]),responses=len(responses))
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    with np.load(a.cache) as archive:
        z={k:archive[k] for k in ('ep','pid','rep','pos','lp_b','lp_q','H_b','TG')}
    pids=np.array(sorted(set(z['pid'])))
    np.random.default_rng(20261122).shuffle(pids)
    halves=[pids[::2],pids[1::2]]
    out={'nature':'Exploration sur corpus déjà consulté ; validation neuve nécessaire avant ep','results':{}}
    for skip in (0,64):
        features=[extract(z,p,skip) for p in halves]
        for source,target in ((0,1),(1,0)):
            train,test=features[source],features[target]
            gs=sorted({abs(f['GAP'])+1e-9 for f in train.values()})
            ls=sorted({f['LLR8']+1e-9 for f in train.values()})+[float('inf')]
            feasible=[]
            for g,l in itertools.product(gs,ls):
                p=partition(train,g,l)
                if all(p[e]=='BASE' for e in EPS if TRUTH[e]=='BASE'):
                    feasible.append((mcc(p),g,l))
            # En cas d'égalité : règle la plus conservatrice, décidée sur train seulement.
            score,g,l=max(feasible)
            pred=partition(test,g,l)
            out['results'][f'skip{skip}|{source}->{target}']=dict(
                train_MCC=float(score),test_MCC=float(mcc(pred)),g=float(g),
                l=None if np.isinf(l) else float(l),test_partition=pred,
                nuisances_grouped=all(pred[e]=='BASE' for e in EPS if TRUTH[e]=='BASE'),
                train_features=train,test_features=test)
    a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:{x:v for x,v in r.items() if not x.endswith('features')} for k,r in out['results'].items()},indent=2))


if __name__=='__main__':main()
