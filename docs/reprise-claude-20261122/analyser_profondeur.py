"""Analyses CPU de résultats existants, sans modèle, collecte ou nouveau candidat."""
from pathlib import Path
import ast
import collections
import hashlib
import itertools
import json
import math
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'profondeur'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'c1'))
from metric import miou_per_image
sys.path.insert(0,str(ROOT/'docs/task2-bench-20261121/ties'))
from ties_rule import qvec,gstat,chi2_ppf999

DATA=Path('/private/tmp/claude-501/-Users-selyan--claude-spares-spare-QUr5DF/615c7a8a-6851-4d35-992d-5de19df90633/scratchpad/t1_emiou/data')
PRED=Path('/private/tmp/eaic-relais-20261122/psyn')
h=np.load(DATA/'output/heats_train.npz');names=h['names'];method=h['method'];full=h['full_watermark']
g=np.load(DATA/'release/train/masks_128.npz');lookup=dict(zip(g['names'].tolist(),g['masks']));gt=np.stack([lookup[n] for n in names])
old=np.load(PRED/'pixseg_cv_psyn_lr1_all.npz');folds=old['fold'];covered=np.isfinite(old['thr'])
assert np.array_equal(old['names'],names)
new=np.full(old['probs'].shape,np.nan,np.float16);newcover=np.zeros(len(names),bool)
configs=[];quant=[]
for k in [0,1]:
    z=np.load(PRED/f'pixseg_cv_psall6k_f{k}.npz');sel=np.isfinite(z['thr'])
    assert np.array_equal(z['names'],names) and np.array_equal(z['fold'],folds)
    assert not (newcover&sel).any();newcover|=sel;new[sel]=z['probs'][sel]
    cfg=ast.literal_eval(str(z['config']));configs.append(cfg)
    saved=miou_per_image(z['probs'][sel]>z['thr'][sel,None,None],gt[sel])
    quant.append({'fold':k,'ecart_miou_max_apres_float16':float(abs(saved-z['miou'][sel]).max()),
                  'ecart_moyen':float((saved-z['miou'][sel]).mean())})
assert np.array_equal(covered,newcover) and covered.sum()==204
ss=covered&(method=='stegastamp');s0=miou_per_image(old['probs'][ss]>.5,gt[ss]);s1=miou_per_image(new[ss]>.5,gt[ss]);delta=s1-s0
configs_old=ast.literal_eval(str(old['config']))
exclude={'tag','out','folds','steps'}
diffconf={k:[configs_old.get(k),configs[0].get(k)] for k in configs_old if configs_old.get(k)!=configs[0].get(k) and k not in exclude}
sort=np.argsort(delta)
result={'n_ss':int(ss.sum()),'gain_ss':float(delta.mean()),'contribution_globale':float(delta.mean()/3),
 'config_differences_hors_steps_sorties_folds':diffconf,'gains':int((delta>0).sum()),'pertes':int((delta<0).sum()),'egalites':int((delta==0).sum()),
 'delta_quantiles':dict(zip(['min','p10','p25','median','p75','p90','max'],np.percentile(delta,[0,10,25,50,75,90,100]).tolist())),
 'somme_gains':float(delta[delta>0].sum()),'somme_pertes':float(delta[delta<0].sum()),
 'sans_5_meilleurs_contribution_globale':float(np.delete(delta,sort[-5:]).mean()/3),
 'sans_5_pires_contribution_globale':float(np.delete(delta,sort[:5]).mean()/3),
 'par_fold':{str(k):float(delta[folds[ss]==k].mean()/3) for k in [0,1]},'sensibilite_seuil':{},
 'quantification_float16_aux_seuils_appris':quant,
 'validation_structurelle':{'real_train_par_fold':306,'inner_val_par_fold':102,'outer_test_par_fold':102,'fit_all':510,
 'graines_cv':{str(k):100*configs[0]['seed']+k for k in [0,1]},'graines_test_effectives_3k':[5099,5199,5299],
 'graines_test_effectives_6k':[6099,6199],'limite':'5 folds ne signifie pas 5 graines ou comparaison ensemble contre ensemble.'}}
for thr in [.4,.45,.5,.55,.6]:
    a=miou_per_image(old['probs'][ss]>thr,gt[ss]);b=miou_per_image(new[ss]>thr,gt[ss])
    result['sensibilite_seuil'][str(thr)]={'old':float(a.mean()),'new':float(b.mean()),'gain_global':float((b-a).mean()/3)}
(OUT/'diagnostic-t1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')

T=Path('/private/tmp/eaic-relais-20261122/ties');P=json.loads((T/'probes_smol2.json').read_text())['probes']
R=[json.loads(l) for l in (T/'ties_refs.jsonl').read_text().splitlines()];eps=[f'ref-{i:02d}' for i in range(1,10)]
truth={e:('BASE' if e in ['ref-01','ref-03','ref-05','ref-06','ref-08'] else e) for e in eps}
def signed_mcc(pred):
    tp=fp=tn=fn=0
    for a,b in itertools.combinations(eps,2):
        t=truth[a]==truth[b];p=pred[a]==pred[b]
        tp+=t and p;fp+=(not t) and p;tn+=(not t) and (not p);fn+=t and (not p)
    den=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return (tp*tn-fp*fn)/den if den else 0.
analyses={};stats={}
for half in ['all',0,1]:
    C=collections.defaultdict(lambda:collections.defaultdict(lambda:np.zeros(3)))
    for r in R:
        if r.get('status')!=200 or 'text' not in r or (half!='all' and r['rep']%2!=half):continue
        p=P[r['probe']];s=(r['text'] or '').strip();cat=0 if s==p['s1'] else (1 if s==p['s2'] else 2)
        C[r['ep']][r['probe']][cat]+=1
    stats[half]={e:gstat(C[e],P) for e in eps}
    detail={}
    for e,(G,temp) in stats[half].items():
        expected=np.array([n.sum()*qvec(P[j],temp) for j,n in C[e].items()])
        detail[e]={'G':G,'T':temp,'bins_attendus_sous_5':int((expected<5).sum()),'bins_total':int(expected.size)}
    analyses[str(half)]={'seuil':chi2_ppf999(2*len(P)-1),'stats':detail}
thresholds=sorted({0.,chi2_ppf999(2*len(P)-1)}|{v[0]+o for half in stats.values() for v in half.values() for o in [-1e-6,1e-6]})
post=[]
for t in thresholds:
    scores={str(h):signed_mcc({e:'BASE' if v[0]<=t else e for e,v in ss.items()}) for h,ss in stats.items()}
    post.append({'seuil':t,'mcc_signe':scores,'min_moitie':min(scores['0'],scores['1'])})
best=max(post,key=lambda r:r['min_moitie'])
probes=[]
for i,p in enumerate(P):
    q=qvec(p,.7);probes.append({'index':i,'kind':p['kind'],'s1':p['s1'],'s2':p['s2'],'d0':p['d0'],
      'egalite_abs_d0_inferieur_ou_egal_05':abs(p['d0'])<=.5,'q_T07':q.tolist(),
      'proba_majoritaire_T07':float(q.max()),'sensibilite_max_T04_T16':float(abs(qvec(p,.4)[0]-qvec(p,1.6)[0]))})
out={'tentatives':len(R),'statuts':dict(collections.Counter(str(r.get('status')) for r in R)),
 'sondes':probes,'n_tie_et_reellement_egalite':sum(p['kind']=='tie' and p['egalite_abs_d0_inferieur_ou_egal_05'] for p in probes),
 'analyses':analyses,'diagnostic_post_hoc_meilleur_seuil_commun':best,
 'limite_post_hoc':'Ce maximum après lecture n’est pas une validation ni un nouveau seuil recommandé.'}
(OUT/'diagnostic-t2.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'t1':result,'t2':{'n_ties_reelles':out['n_tie_et_reellement_egalite'],'meilleur_post_hoc':best,
 'bins_sous5_ref01':analyses['0']['stats']['ref-01']}},ensure_ascii=False,indent=2))
