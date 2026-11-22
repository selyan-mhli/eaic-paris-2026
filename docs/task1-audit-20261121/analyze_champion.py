"""Contrôle CPU des scores OOF sauvegardés, sans reconstruire les réseaux."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'c1'))
from metric import downsample_mask,miou_per_image

D=Path(__file__).resolve().parent
z=np.load(D/'segf_ens3_3.npz')
b=np.load(D/'champion_boxes_train.npz')
assert z['names'].tolist()==b['names'].tolist()
mask=np.load(ROOT/'c1/seg/data/masks_128.npz')
mp=dict(zip(mask['names'].tolist(),mask['masks']))
names=z['names'].tolist();Y=np.stack([mp[n] for n in names]).astype(bool)
af=b['acc_full'];ab=b['acc_box'];full=b['full_watermark'].astype(bool);method=b['method'];n=len(names)
bm=[]
for y0,y1,x0,x1 in b['box']:
    p=np.zeros((512,512),np.uint8);p[y0:y1,x0:x1]=1;bm.append(downsample_mask(p))
bm=np.array(bm,bool)
sb=miou_per_image(bm,Y);sz=miou_per_image(np.zeros_like(Y),Y)
choices=json.loads(str(z['choices']))
def folds(strata,seed):
    rng=np.random.default_rng(seed);out=np.zeros(len(strata),int)
    for s in np.unique(strata):
        i=np.where(strata==s)[0];rng.shuffle(i);out[i]=np.arange(len(i))%5
    return out

recon=[]
for seed in range(3):
    fo=folds(np.array([f'{m}|{f}' for m,f in zip(method,full)]),seed)
    predR=np.zeros(n,bool);expected=np.full(n,np.nan)
    for k in range(5):
        choice=next(q for q in choices if q['seed']==seed and q['fold']==k)
        g=choice['gate'];te=fo==k;tr=~te;predR[te]=af[te]<g
        for m in sorted(set(method)):
            trR=tr&(method==m)&(af<g);teR=te&(method==m)&(af<g)
            thresholds=np.round(np.arange(.50,1.02,.01),2)
            means=[np.where(ab[trR]>=t,sb[trR],sz[trR]).sum() for t in thresholds]
            t=thresholds[np.argmax(means)]
            expected[teR]=np.where(ab[teR]>=t,sb[teR],sz[teR])
    err=np.abs(z['head'][seed,predR]-expected[predR])
    recon.append({'seed':seed,'predicted_R':int(predR.sum()),'R_branch_max_score_error':float(err.max()),
                  'F_routed_R':int((full&predR).sum()),'R_routed_F':int((~full&~predR).sum()),
                  'oracle_accept_reject_saved_R_box_global_gain':float((np.maximum(sb,sz)[predR]-expected[predR]).sum()/n)})

head=z['head'].mean(0);base=z['champ'].mean(0);delta=head-base
groups={}
for m in sorted(set(method)):
    for f in (True,False):
        q=(method==m)&(full==f);minor=(~Y if f else Y)[q].mean((1,2))
        groups[m+('_F' if f else '_R')]={'n':int(q.sum()),'head':float(head[q].mean()),'base':float(base[q].mean()),
            'worst':[{'name':names[i],'head':float(head[i]),'base':float(base[i]),'coverage':float(Y[i].mean()),'acc_full':float(af[i])} for i in np.where(q)[0][np.argsort(head[q])[:4]]],
            'head_lt_045':int((head[q]<.45).sum()),'minority_lt_1pct':int((minor<.01).sum())}
        for subset,keep in (('minority_lt_1pct',minor<.01),('minority_ge_1pct',minor>=.01)):
            if keep.any():
                groups[m+('_F' if f else '_R')][subset+'_scores']={'n':int(keep.sum()),'head':float(head[q][keep].mean()),'base':float(base[q][keep].mean())}
rng=np.random.default_rng(0);boot=np.array([delta[rng.integers(0,n,n)].mean() for _ in range(2000)])
out={'names':n,'score_arrays_shape':list(z['head'].shape),'contains_pixel_predictions':False,
     'head':float(head.mean()),'base':float(base.mean()),'delta':float(delta.mean()),
     'delta_ci95_descriptive_images':np.percentile(boot,[2.5,97.5]).tolist(),
     'F':float(head[full].mean()),'R':float(head[~full].mean()),'groups':groups,'routing_reconstruction':recon,
     'oracle_choose_saved_box_or_empty_R':float(np.maximum(sb,sz)[~full].mean()),
     'saved_box_unconditional_R':float(sb[~full].mean()),
     'oracle_replace_R_with_exact_bbox_global':float((head[full].mean()+.8668620523669743)/2),
     'oracle_replace_F_perfect_global':float((1+head[~full].mean())/2),
     'oracle_replace_R_perfect_global':float((head[full].mean()+1)/2)}
(D/'champion_summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
