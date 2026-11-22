"""Relevé complémentaire en lecture seule : sources exécutées et provenance synthétique.

Liste explicite de fichiers, aucun import de collecteur/API, aucun secret lu.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import subprocess

HERE = Path(__file__).resolve().parent
DEST = HERE / "profondeur"
DEST.mkdir(exist_ok=True)
remote = r'''
from pathlib import Path
import collections,datetime,hashlib,json,subprocess
import numpy as np
T=Path('/p/scratch/training2666/mouhali1/SLZ/Hackathon_Watermark_Localisation_Task')
P=Path('/p/project1/training2666/mouhali1/t1sr')
out={'heure':datetime.datetime.now().astimezone().isoformat(),'sources':{},'synth':[],'meta':{},'logs':{}}
names={}
for split in ['train','test']:
    meta=json.loads((T/'release'/split/'metadata.json').read_text());names[split]={m['image_name']:m['method'] for m in meta}
    out['meta'][split]={'n':len(meta),'methodes':dict(collections.Counter(names[split].values())),'names':list(names[split])}
out['meta']['intersection_noms_train_test']=sorted(set(names['train'])&set(names['test']))
for n in ['train.py','data.py','model.py','export.py','merge_ss.py','combine_eval.py','configs_psyn2.txt','configs_psyn3.txt','configs_psyn4.txt','configs_psyn5.txt']:
    p=T/'work_pixseg_v2'/n
    if p.exists():
        b=p.read_bytes();out['sources'][n]={'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'contenu':b.decode()}
for p in sorted((T/'output').glob('pixseg_syn_*.npz')):
    with np.load(p,allow_pickle=False) as z:
        src=z['src'].tolist();src2=z['src2'].tolist();methods=z['method'].tolist()
        def stat(src):
            return {'train':sum(n in names['train'] for n in src),'test':sum(n in names['test'] for n in src),
                    'inconnus':sum(n not in names['train'] and n not in names['test'] for n in src),
                    'source_meme_methode':sum(names['test'].get(s,names['train'].get(s))==m for s,m in zip(src,methods)),
                    'noms_uniques':len(set(src))}
        out['synth'].append({'path':str(p),'n':len(src),'methodes':dict(collections.Counter(methods)),
          'src':stat(src),'src2':stat(src2),'src_names':src,'src2_names':src2,
          'couverture_moy':float(z['mask128'].mean()),'nb_full':int(z['full'].sum())})
for p in sorted(P.glob('relay_ss6k_15904703*.out')):out['logs'][str(p)]=p.read_text()[-18000:]
cmd=['sacct','-j','15904703','-X','-P','-o','JobID,State,Elapsed,Start,End,AllocTRES%80']
r=subprocess.run(cmd,text=True,capture_output=True);out['sacct']={'commande':cmd,'code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
print(json.dumps(out,ensure_ascii=False,indent=2))
'''
cmd=["ssh","-o","BatchMode=yes","-o","ConnectTimeout=8","-o","ControlPath=/private/tmp/eaic-codex-jrlogin03.sock",
     "-o","HostName=jureca03.fz-juelich.de","-o","HostKeyAlias=jureca.fz-juelich.de","jureca",
     "/p/scratch/training2666/mouhali1/SLZ/Hackathon_Watermark_Localisation_Task/.venv/bin/python -"]
p=subprocess.run(cmd,input=remote,text=True,capture_output=True,timeout=120)
capture={'heure_paris':datetime.now(ZoneInfo('Europe/Paris')).isoformat(),'commande':cmd,'code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
(DEST/'jureca-provenance.json').write_text(json.dumps(capture,ensure_ascii=False,indent=2)+'\n')
if p.returncode:
    raise SystemExit(p.stderr)
d=json.loads(p.stdout);sources=DEST/'sources-executees';sources.mkdir(exist_ok=True)
for n,s in d['sources'].items():(sources/n).write_text(s['contenu'])
print(json.dumps({'heure':d['heure'],'sources':{n:s['sha256'] for n,s in d['sources'].items()},
 'shards_synth':len(d['synth']),'n_synth':sum(s['n'] for s in d['synth']),
 'src_train':sum(s['src']['train'] for s in d['synth']),'src2_train':sum(s['src2']['train'] for s in d['synth']),
 'job':d['sacct']},ensure_ascii=False,indent=2))
