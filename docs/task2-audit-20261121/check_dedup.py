"""Mesure la sensibilité à la politique de dédoublonnage des journaux ref."""
import json
import math
import sys
from pathlib import Path

base=Path(__file__).resolve().parent
sys.path.insert(0,str(base/'sources/branche-c2'))
import analyze as A
import features as F

r=json.loads((base/'remote_evidence.json').read_text())
rows=[json.loads(l) for l in r['files']['c2run/ref_answers.jsonl']['text'].splitlines()]
truth=A.read_truth(base/'sources/branche-c2/kit/practice_answers.csv')
result={}
for policy,rr in [('dernière_occurrence',rows),('première_occurrence',list(reversed(rows)))]:
    g={p:d for p,d in A.group(rr).items() if not p.startswith('w')}
    ev=F.evaluate(g,truth,top=15)
    result[policy]=[{'selection':x['mcc_sel'],'evaluation':x['mcc_eval']} for x in ev['folds']]

# Exemple analytique, pas une mesure de l'API : deux sorties orthogonales,
# même modèle sous deux distributions de décodage différentes.
p,q=.99,.5
kaa=p*p+(1-p)*(1-p)
kbb=q*q+(1-q)*(1-q)
kab=p*q+(1-p)*(1-q)
result['contre_exemple_invariance_temperature']={'p':p,'q':q,'Kaa':kaa,'Kbb':kbb,'Kab':kab,
                                              'distance_corr':1-kab/math.sqrt(kaa*kbb)}
(base/'dedup_sensitivity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))
