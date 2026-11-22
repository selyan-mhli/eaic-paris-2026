"""Lecture CPU des OOF terminées du job existant 15904703 ; aucune mutation distante."""
from pathlib import Path
import json,subprocess
HERE=Path(__file__).resolve().parent
remote=r'''
from pathlib import Path
import ast,datetime,hashlib,json,subprocess
import numpy as np
T=Path('/p/scratch/training2666/mouhali1/SLZ/Hackathon_Watermark_Localisation_Task')
P=Path('/p/project1/training2666/mouhali1/t1sr')
meta=json.loads((T/'release/train/metadata.json').read_text());names=np.array([m['image_name'] for m in meta]);method=np.array([m['method'] for m in meta]);full=np.array([m['full_watermark'] for m in meta])
z=np.load(T/'release/train/masks_128.npz');M=dict(zip(z['names'].tolist(),z['masks']));gt=np.stack([M[n] for n in names]).astype(bool)
oldfiles=[T/'output/pixseg_cv_psyn_lr1_all.npz',P/'pixseg_cv_relay_psall3k_f234.npz']
newfiles=[T/'output/pixseg_cv_psall6k_f0.npz',T/'output/pixseg_cv_psall6k_f1.npz',P/'pixseg_cv_relay_psall6k_f23.npz',P/'pixseg_cv_relay_psall6k_f4.npz']
provenance=[]
def merge(files):
    p=np.full((510,128,128),np.nan,np.float16);covered=np.zeros(510,bool);fold=None
    for f in files:
        z=np.load(f,allow_pickle=False);sel=np.isfinite(z['thr']);assert np.array_equal(z['names'],names)
        assert not (covered&sel).any(),str(f)
        if fold is None:fold=z['fold']
        else:assert np.array_equal(fold,z['fold'])
        assert np.isfinite(z['probs'][sel]).all() and ((z['probs'][sel]>=0)&(z['probs'][sel]<=1)).all()
        p[sel]=z['probs'][sel];covered|=sel
        provenance.append({'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'folds':np.unique(fold[sel]).tolist(),'n':int(sel.sum()),'config':ast.literal_eval(str(z['config']))})
    assert covered.all() and len(set(names))==510
    return p,fold
old,fold=merge(oldfiles);new,nfold=merge(newfiles);assert np.array_equal(fold,nfold)
def miou(p,g):
    p=p.reshape(len(p),-1);g=g.reshape(len(g),-1);result=np.zeros(len(p))
    for c in [False,True]:
        pc=p==c;gc=g==c;u=(pc|gc).sum(1);i=(pc&gc).sum(1);result+=np.where(u==0,1,i/np.maximum(u,1))
    return result/2
ss=method=='stegastamp';a=miou(old[ss]>.5,gt[ss]);b=miou(new[ss]>.5,gt[ss]);delta=b-a
diluted=np.zeros(510);diluted[ss]=delta;rng=np.random.default_rng(0)
boot=diluted[rng.integers(0,510,(4000,510))].mean(1)
strat=[]
for _ in range(4000):
    means=[]
    for flag in [False,True]:
        d=delta[full[ss]==flag];means.append(rng.choice(d,len(d),replace=True).mean())
    strat.append(np.mean(means)/3)
fold_gains={str(k):float(delta[fold[ss]==k].mean()/3) for k in range(5)}
out={'heure':datetime.datetime.now().astimezone().isoformat(),'nature':'OOF une graine par fold, seuil fixe 0.5 ; pas une mesure exacte des ensembles test',
 'n':510,'n_ss':170,'folds':list(range(5)),'ss_old':float(a.mean()),'ss_new':float(b.mean()),'gain_ss':float(delta.mean()),'gain_global_ss_seul':float(delta.mean()/3),
 'ic95_images_descriptif':np.percentile(boot,[2.5,97.5]).tolist(),'ic95_stratifie_fr_descriptif':np.percentile(strat,[2.5,97.5]).tolist(),
 'par_fold_global':fold_gains,'gains_pertes_egalites':[int((delta>0).sum()),int((delta<0).sum()),int((delta==0).sum())],
 'groupes':{},'sensibilite_seuil':{},'provenance':provenance}
for flag in [False,True]:
 s=full[ss]==flag;out['groupes']['F' if flag else 'R']={'n':int(s.sum()),'old':float(a[s].mean()),'new':float(b[s].mean()),'gain':float(delta[s].mean())}
for t in [.4,.45,.5,.55,.6]:out['sensibilite_seuil'][str(t)]=float((miou(new[ss]>t,gt[ss])-miou(old[ss]>t,gt[ss])).mean()/3)
out['critere_historique']=('POURSUIVRE' if out['gain_global_ss_seul']>=.01 and out['ic95_images_descriptif'][0]>0 else 'DIAGNOSTIC' if out['gain_global_ss_seul']>=.003 else 'ABANDON')
cmd=['sacct','-j','15904703','-X','-P','-o','JobID,State,ExitCode,Elapsed,Start,End'];r=subprocess.run(cmd,text=True,capture_output=True);out['sacct']=r.stdout
print(json.dumps(out,ensure_ascii=False,indent=2))
'''
cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','ControlPath=/private/tmp/eaic-codex-jrlogin03.sock','-o','HostName=jureca03.fz-juelich.de','-o','HostKeyAlias=jureca.fz-juelich.de','jureca','/p/scratch/training2666/mouhali1/SLZ/Hackathon_Watermark_Localisation_Task/.venv/bin/python -']
r=subprocess.run(cmd,input=remote,text=True,capture_output=True,timeout=60)
(HERE/'profondeur/confirmation-5fold.json').write_text(json.dumps({'commande':cmd,'code':r.returncode,'stdout':r.stdout,'stderr':r.stderr},ensure_ascii=False,indent=2)+'\n')
if r.returncode:raise SystemExit(r.stderr)
d=json.loads(r.stdout);print(json.dumps({k:v for k,v in d.items() if k!='provenance'},ensure_ascii=False,indent=2))
