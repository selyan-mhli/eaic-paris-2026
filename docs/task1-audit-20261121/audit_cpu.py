"""Audit sans modèle, apprentissage, GPU ni écriture dans les artefacts existants."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / 'c1/seg/data'
sys.path.insert(0, str(ROOT / 'c1'))
from metric import miou_per_image

OUT = Path(__file__).resolve().parent
meta = json.loads((D / 'train_metadata.json').read_text())
test = json.loads((D / 'test_metadata.json').read_text())
names = [r['image_name'] for r in meta]
method = np.array([r['method'] for r in meta])
full = np.array([r['full_watermark'] for r in meta], bool)
z = np.load(D / 'masks_128.npz')
assert len(set(z['names'])) == len(names) == 510
pos = {n: i for i, n in enumerate(z['names'])}
Y = np.stack([z['masks'][pos[n]] for n in names]).astype(bool)
assert set(names) == set(pos)

def align(z, key):
    assert len(set(z['names'])) == len(z['names'])
    p = {n: i for i, n in enumerate(z['names'])}
    assert set(names) == set(p)
    values = z[key]
    return np.stack([values[p[n]] for n in names])

def shards(pattern):
    zs = [dict(np.load(f)) for f in sorted(D.glob(pattern))]
    return {k: np.concatenate([z[k] for z in zs]) for k in zs[0]}

def bbox(mask):
    out = np.zeros_like(mask)
    yy, xx = np.where(mask)
    if len(yy):
        out[yy.min():yy.max()+1, xx.min():xx.max()+1] = True
    return out

def scores(p):
    return miou_per_image(p, Y)

def summary(s):
    out = {'all': float(s.mean())}
    for f, label in ((True,'F'),(False,'R')):
        out[label] = float(s[full == f].mean())
        for m in sorted(set(method)):
            out[m+'_'+label] = float(s[(method == m) & (full == f)].mean())
    return out

ones, zeros = np.ones_like(Y), np.zeros_like(Y)
oracle_bbox = np.stack([~bbox(~y) if f else bbox(y) for y, f in zip(Y, full)])
ff = shards('seg_ff_train_*.npz')
vf = shards('seg_feats_train_*.npz')
af = align(ff, 'acc_full')
ba = align(vf, 'best_acc')
bestboxes = align(vf, 'best_box')
rm = np.repeat(np.repeat(align(vf, 'maps')[:, 4] > .5, 4, 1), 4, 2)
gate = af >= .75
results = {'metric': {}, 'groups': {}, 'predictions': {}}
results['metric']['all0'] = summary(scores(zeros))
results['metric']['all1'] = summary(scores(ones))
results['metric']['oracle_regime_only'] = summary(scores(np.where(full[:,None,None], ones, zeros)))
results['metric']['gate075_only'] = summary(scores(np.where(gate[:,None,None], ones, zeros)))
results['metric']['oracle_inverse_bboxF_bboxR'] = summary(scores(oracle_bbox))
results['metric']['R_search_unconditional_F_all1'] = summary(scores(np.where(full[:,None,None], ones, rm)))
results['metric']['R_search075_F_all1'] = summary(scores(np.where(full[:,None,None], ones, rm & (ba>=.75)[:,None,None])))
results['metric']['R_oracle_choose_best_savedbox_or_empty_F_all1'] = summary(np.where(full, scores(ones), np.maximum(scores(rm),scores(zeros))))
for cell in (2,4,8,16):
    coarse = Y.reshape(510,128//cell,cell,128//cell,cell).mean((2,4)) >= .5
    pred = coarse.repeat(cell,1).repeat(cell,2)
    results['metric']['oracle_majority_grid'+str(128//cell)] = summary(scores(pred))
for m in sorted(set(method)):
    for f, label in ((True,'F'),(False,'R')):
        sel = (method==m)&(full==f)
        cov = Y[sel].mean((1,2))
        small = (~Y[sel] if f else Y[sel])
        bbs = np.stack([bbox(y) for y in small])
        sides = []
        for y in small:
            yy,xx=np.where(y); sides.append(min(yy.max()-yy.min()+1,xx.max()-xx.min()+1)*4)
        results['groups'][m+'_'+label] = {
            'n': int(sel.sum()),'coverage_mean':float(cov.mean()),'coverage_median':float(np.median(cov)),
            'minority_lt_1pct':int((small.mean((1,2))<.01).sum()),
            'median_min_bbox_side_px':float(np.median(sides)),
            'exact_rectangles_minority':int((small==bbs).all((1,2)).sum()),
            'median_components':float(np.median([ndimage.label(y)[1] for y in small])),
            'gate_false':int((gate[sel] != full[sel]).sum()),
            'acc_full_mean':float(af[sel].mean()),
            'best_acc_mean':float(ba[sel].mean()),
            'accepted_boxes075':int((ba[sel]>=.75).sum()),
        }

Ps={}
for fn in ('synth_f_pred_seed0.npz','synth_f_pred_seed1.npz','synth_f_pred_seed2.npz','kfold_f_pred.npz','kfold_f_pred_synth.npz'):
    p = align(np.load(D/fn),'P')
    assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1
    Ps[fn]=p
    pred = np.where(gate[:,None,None],p>.5,rm & (ba>=.75)[:,None,None])
    res={'net_on_F_fixed05':summary(scores(p>.5)), 'pipeline_fixed_gate075_rect075':summary(scores(pred))}
    for key, new in (
        ('pipeline_oracle_regime',np.where(full[:,None,None],p>.5,rm & (ba>=.75)[:,None,None])),
        ('pipeline_perfect_F',np.where(full[:,None,None],Y,rm & (ba>=.75)[:,None,None])),
        ('pipeline_perfect_R',np.where(full[:,None,None],p>.5,Y)),
        ('pipeline_bbox_oracle_R',np.where(full[:,None,None],p>.5,oracle_bbox))):
        res[key]=summary(scores(new))
    results['predictions'][fn]=res

# Le fold du générateur des prédictions OOF n'est pas celui du combine.py.
def folds(strata, seed):
    rng=np.random.default_rng(seed); out=np.zeros(len(strata),int)
    for s in np.unique(strata):
        ids=np.where(strata==s)[0]; rng.shuffle(ids); out[ids]=np.arange(len(ids))%5
    return out
ff_names=ff['names'].tolist()
meta_by_name={x['image_name']:x for x in meta}
ff_full=np.array([meta_by_name[n]['full_watermark'] for n in ff_names])
kf_local=folds(ff['method'][ff_full],20261121)
combine_local=folds(np.array([str(m)+str(f) for m,f in zip(ff['method'],ff_full)]),20261121)[ff_full]
contingency=np.zeros((5,5),int)
np.add.at(contingency,(kf_local,combine_local),1)
results['validation']={
    'OOF_F_vs_combine_fold_contingency':contingency.tolist(),
    'same_fold_F_count':int(np.trace(contingency)),
    'total_F_count':int(full.sum()),
    'train_test_name_overlap':len(set(names)&{r['image_name'] for r in test}),
    'unique_mask_hashes':len({hashlib.sha256(y.tobytes()).hexdigest() for y in Y}),
    'unique_train_messages_by_method':{m:len({r['message'] for r in meta if r['method']==m}) for m in sorted(set(method))},
}
sources=set();missing_src2=0; nsynth=0; synthcov=[]
for f in sorted(D.glob('seg_gf_*.npz')):
    z=np.load(f);sources.update(z['src'].tolist()); nsynth+=len(z['names']);synthcov.extend(z['mask128'].mean((1,2)).tolist())
    if 'src2' in z.files:sources.update(z['src2'].tolist())
    else:missing_src2+=len(z['names'])
results['validation'].update(synth_n=nsynth, synth_missing_src2=missing_src2, known_source_train_overlap=len(sources&set(names)),synth_coverage_mean=float(np.mean(synthcov)),real_F_coverage_mean=float(Y[full].mean()))
# Gains appariés descriptifs (aucune sélection de modèle ni seuil).
base=miou_per_image(Ps['synth_f_pred_seed0.npz'][full]>.5,Y[full])
for fn in ('kfold_f_pred.npz','kfold_f_pred_synth.npz'):
    delta=miou_per_image(Ps[fn][full]>.5,Y[full])-base
    rng=np.random.default_rng(0)
    boot=np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(2000)])
    results['predictions'][fn]['paired_F_vs_synth0']={'delta':float(delta.mean()),'ci95_images':np.quantile(boot,[.025,.975]).tolist()}
(OUT/'audit_results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))

from PIL import Image, ImageDraw
canvas=Image.new('RGB',(8*144,6*166),'#202020'); draw=ImageDraw.Draw(canvas)
for row,(m,f) in enumerate((m,f) for m in sorted(set(method)) for f in (True,False)):
    ii=np.where((method==m)&(full==f))[0]
    ii=ii[np.argsort(Y[ii].mean((1,2)))]; ii=ii[np.linspace(0,len(ii)-1,8,dtype=int)]
    for col,i in enumerate(ii):
        ar=np.where(Y[i],235,15).astype(np.uint8)
        canvas.paste(Image.fromarray(ar).convert('RGB'),(col*144,row*166+24))
        draw.text((col*144+2,row*166+2),f'{m[:5]} {"F" if f else "R"} {Y[i].mean():.3f}',fill='white')
canvas.save(OUT/'mask_contact_sheet.png')
