"""Règle expérimentale figée pour les lots de sondes à projection de nuisance.

Modèle nul conditionnel : log-odds(s1/s2) dans l'espace engendré par d0 et
les cinq différences de contexte simulées. Déviance binomiale, bootstrap
paramétrique 999 tirages, alpha=.05. Chaque réponse entière est une observation.
Les résultats refs restent l'unique critère avant toute proposition de lot ep.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from t2_tail_check import EPS, TRUTH, mcc


def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-40,40)))


def likelihood(y,n,eta):
    return np.sum(y*eta-n*np.logaddexp(0,eta),axis=-1)


def fit_binomial(y,n,x):
    """MLE par Newton avec recherche linéaire, vectorisée pour le bootstrap."""
    y=np.atleast_2d(y).astype(float);n=np.asarray(n,dtype=float)
    u,s,_=np.linalg.svd(x,full_matrices=False)
    rank=int(np.sum(s>max(s.max(),1.)*1e-9));u=u[:,:rank]
    if rank==0:
        q=np.full_like(y,.5)
        return q,np.zeros(y.shape[0]),rank
    beta=np.zeros((len(y),rank));iterations=80
    for _ in range(iterations):
        eta=beta@u.T;p=sigmoid(eta)
        gradient=(y-n*p)@u
        hess=np.einsum('ji,bj,jk->bik',u,n*p*(1-p),u)+1e-8*np.eye(rank)[None]
        step=np.linalg.solve(hess,gradient[:,:,None])[:,:,0]
        step/=np.maximum(1,np.linalg.norm(step,axis=1,keepdims=True)/8)
        old=likelihood(y,n,eta)
        rate=np.ones(len(y))
        for _ in range(15):
            trial=beta+rate[:,None]*step
            good=likelihood(y,n,trial@u.T)>=old-1e-10
            if good.all():break
            rate[~good]*=.5
        beta=trial
        if np.max(np.abs(rate[:,None]*step))<1e-6:break
    eta=beta@u.T;q=sigmoid(eta)
    ph=np.clip(y/np.maximum(n,1),1e-12,1-1e-12)
    saturated=np.sum(y*np.log(ph)+(n-y)*np.log1p(-ph),axis=1)
    deviance=np.maximum(0,2*(saturated-likelihood(y,n,eta)))
    return q,deviance,rank


def analyze_counts(counts,probes,seed,draws=999):
    n=counts[:,:2].sum(1)
    used=n>0
    d=np.array([p['d0'] for p in probes])
    ctx=np.array([p['d_ctx'] for p in probes])-d[:,None]
    x=np.column_stack((d,ctx))[used]
    y=counts[used,0];n=n[used]
    q,g,rank=fit_binomial(y,n,x)
    other=float(counts[:,2].sum()/max(1,counts.sum()))
    unknown=(len(n)<rank+3) or (other>.2)
    if unknown:
        return dict(unknown=True,reason='Couverture insuffisante ou >20% hors paire',p_value=None,
                    isolate=False,G=float(g[0]),rank=rank,paired_probes=len(n),other_rate=other)
    samples=np.random.default_rng(seed).binomial(n,q[0],size=(draws,len(n)))
    _,sim,_=fit_binomial(samples,n,x)
    p=float((1+np.sum(sim>=g[0]-1e-10))/(draws+1))
    return dict(unknown=False,p_value=p,isolate=bool(p<.05),G=float(g[0]),rank=rank,
                paired_probes=len(n),other_rate=other)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--probes',type=Path,required=True)
    ap.add_argument('--log',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    probes=json.loads(a.probes.read_text())['probes']
    raw=[json.loads(s) for s in a.log.read_text().splitlines()]
    records=[r for r in raw if r.get('status')==200 and r.get('ep') in EPS and 'text' in r]
    cells=[(r['ep'],r['probe'],r['rep']) for r in records]
    if len(cells)!=len(set(cells)):raise ValueError('Réussites dupliquées')
    report=dict(nature='Méthode expérimentale ; aucune collecte ni soumission automatique',
                rule='Binomial contexte projeté, bootstrap 999, alpha .05, couverture hors paire <= .20',
                statuses=dict(Counter(str(r.get('status')) for r in raw)),results={})
    for hi,half in enumerate(('all',0,1)):
        pred,details={},{}
        for ei,ep in enumerate(EPS):
            counts=np.zeros((len(probes),3),dtype=int)
            chosen=[r for r in records if r['ep']==ep and (half=='all' or r['rep']%2==half)]
            for r in chosen:
                p=probes[r['probe']];text=(r['text'] or '').strip()
                c=0 if text==p['s1'] else 1 if text==p['s2'] else 2
                counts[r['probe'],c]+=1
            if len(chosen)<len(probes)*5:
                # Une 5xx peut expliquer une cellule absente ; couverture minimale globale.
                minimum_ok=len(chosen)>=len(probes)*5-1
            else:minimum_ok=True
            if not chosen:
                detail=dict(unknown=True,reason='Aucune réponse',isolate=False)
            else:
                detail=analyze_counts(counts,probes,20261122+100*hi+ei)
            detail['minimum_five_reps_per_half']=minimum_ok
            details[ep]=detail
            pred[ep]=ep if detail['isolate'] else 'BASE'
        good_base=all(pred[e]=='BASE' for e in EPS if TRUTH[e]=='BASE')
        weak_fp8=pred['ref-02']!='BASE' and pred['ref-09']!='BASE'
        covered=all(not d['unknown'] and d['minimum_five_reps_per_half'] for d in details.values())
        report['results'][str(half)]=dict(MCC=mcc(pred),base_grouped=good_base,
            fp8_and_weak_detected=weak_fp8,coverage_ok=covered,partition=pred,details=details)
    report['refs_pass']=all(report['results'][str(h)]['MCC']>=.6 and
        report['results'][str(h)]['base_grouped'] and report['results'][str(h)]['coverage_ok']
        and report['results'][str(h)]['fp8_and_weak_detected'] for h in (0,1))
    report['ep_authorized']=False
    report['warning']='Même une réussite refs ne prouve pas le transfert aux trois bases notées.'
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},ensure_ascii=False,indent=2))
    for h,r in report['results'].items():
        print(h,{k:v for k,v in r.items() if k not in ('details','partition')})


if __name__=='__main__':main()
