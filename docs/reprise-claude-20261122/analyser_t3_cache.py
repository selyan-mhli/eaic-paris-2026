"""Rejeu CPU des seules têtes existantes sur des features déjà calculées.

Ni nouveau candidat, ni entraînement, ni passage dans le ViT.
"""
from pathlib import Path
import json,sys
import numpy as np
import torch

torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'docs/task3-work-20261121/mech'))
from head_search import head16
P=Path('/private/tmp/claude-501/-Users-selyan--claude-spares-spare-oGfpae/d60fc14b-9806-4319-b999-3844f0f1cede/scratchpad/feats_15903971.npz')
z=np.load(P,allow_pickle=False)
W0=torch.from_numpy(z['W_orig']);b0=torch.from_numpy(z['b_orig'])
Wp=torch.from_numpy(z['W_push_best']);bp=torch.from_numpy(z['b_push_best'])
concerned=np.bincount(z['y'],minlength=100)>0
out={'source':str(P),'nature':'rejeu CPU de tête sur features fp16 sauvegardées ; aucune mesure du retain caché',
 'classes_concernees':int(concerned.sum()),'verification':{},'candidats':{}}
for name in ['orig','push_best','push_s125','push_s150','push_s200']:
 W,b=(W0,b0) if name=='orig' else (torch.from_numpy(z['W_'+name]),torch.from_numpy(z['b_'+name]))
 emu=head16(torch.from_numpy(z['f16_forget']),W,b);full=torch.from_numpy(z['full_'+name])
 out['verification'][name]={'logits_ecart_abs_max':float(abs(emu-full).max()),
                         'desaccord_top1':float((emu.argmax(1)!=full.argmax(1)).float().mean())}
for scale in [1.25,1.5,1.75,2.]:
 W=W0+scale*(Wp-W0);b=b0+scale*(bp-b0);rows={}
 for view in ['forget','altered','offall']:
    f=torch.from_numpy(z['f16_'+view]);orig=head16(f,W0,b0);cand=head16(f,W,b)
    y=orig.argmax(1);pr=torch.softmax(orig/2,1);lp=torch.log_softmax(orig/2,1);lq=torch.log_softmax(cand/2,1)
    groups={'tous':torch.ones(len(f),dtype=torch.bool),'concernees':torch.from_numpy(concerned[y.numpy()]),'autres':torch.from_numpy(~concerned[y.numpy()])}
    for group,sel in groups.items():
      if not sel.any():continue
      agreement=float((orig[sel].argmax(1)==cand[sel].argmax(1)).float().mean())
      kl=float((pr[sel]*(lp[sel]-lq[sel])).sum(1).mean())
      rows[view+'/'+group]={'n':int(sel.sum()),'accord_origine':agreement,'kl_T2':kl,'facteur_KL':1/(1+kl)}
 out['candidats'][str(scale)]=rows
(Path(__file__).resolve().parent/'profondeur/t3-rejeu-cache.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
