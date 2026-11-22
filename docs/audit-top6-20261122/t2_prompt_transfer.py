"""Évaluation de la règle LLR existante sur des questions et répétitions disjointes."""

import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent/'task2-bench-20261121'
sys.path.insert(0, str(BENCH))
from anlib import EPS, TRUTH, mcc, pair_errors


def features(pilot, adapter, repeats, prompts):
    def select(data):
        return np.isin(data['pid'], prompts) & np.isin(data['rep'], repeats)
    rows, adapted = select(pilot), select(adapter)
    out = {}
    for ep in EPS:
        a = rows & (pilot['ep'] == ep)
        b = adapted & (adapter['ep'] == ep)
        assert a.sum() == b.sum() and a.any()
        q = pilot['lp_q'][a].mean(0).max() - pilot['lp_b'][a].mean(0).max()
        l = adapter['lp'][b,2].mean(0).max() - adapter['lp'][b,0].mean(0).max()
        out[ep] = (float(q),float(l))
    return out


def partition(feats, q, l):
    return {ep: 'fp8' if x > q else ep if y > l else 'base' for ep,(x,y) in feats.items()}


def evaluate(train, test):
    qs, ls = [[v[i]+1e-9 for v in train.values()] + [9.] for i in range(2)]
    q, l = max(itertools.product(qs,ls), key=lambda p: (mcc(TRUTH,partition(train,*p)),-p[0],-p[1]))
    pred = partition(test,q,l)
    return dict(selection=mcc(TRUTH,partition(train,q,l)),heldout=mcc(TRUTH,pred),
                errors=pair_errors(pred),partition=pred)


def load(name):
    with np.load(BENCH/(name+'.npz')) as z:
        return dict(z)


def main():
    z, a, b = [load(n) for n in ('wb_pilot','wb_lora2_trB','wb_lora2_trA')]
    all_ids = sorted(set(z['pid']), key=lambda p:hashlib.sha256(p.encode()).digest())
    split = [all_ids[::2],all_ids[1::2]]
    folds = []
    for left,right in [(all_ids,all_ids),split,split[::-1]]:
        fa = features(z,a,(0,1),left)
        fb = features(z,b,(2,3),right)
        folds.append(dict(left=left,right=right,forward=evaluate(fa,fb),reverse=evaluate(fb,fa)))
    old = json.loads((BENCH/'combo_halves_s2.json').read_text())
    assert abs(folds[0]['forward']['heldout'] - old['A->B']['mcc_eval']) < 1e-6
    assert abs(folds[0]['reverse']['heldout'] - old['B->A']['mcc_eval']) < 1e-6
    (HERE/'t2-transfert-prompts.json').write_text(json.dumps(folds,indent=2)+'\n')


if __name__ == '__main__':
    main()
