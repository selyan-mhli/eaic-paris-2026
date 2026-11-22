"""Sensibilité du classement à tableau figé, sans prédiction de score futur."""
from pathlib import Path
import json,re,shutil
ROOT=Path(__file__).resolve().parent
source=Path('/private/tmp/eaic-relais-20261122/leaderboard-live.html')
h=source.read_text();shutil.copyfile(source,ROOT/'profondeur/leaderboard-0317.html')
overall=h.split('<div data-panel="overall">',1)[1].split('<div data-panel=',1)[0]
rows=re.findall(r'<tr[^>]*>.*?</tr>',overall,re.S)
slz=next(r for r in rows if 'data-team="SLZ"' in r)
scores={t:float(s) for t,s in re.findall(r'data-task="([^"]*)"[^>]*>.*?title="([\d.eE+-]+)"',slz)}
top6=[r for r in rows if re.search(r'data-rank="6"',r)][0]
target=float(re.search(r'class="overall" title="([^"]+)"',top6)[1])
current=float(re.search(r'class="overall" title="([^"]+)"',slz)[1]);rank=int(re.search(r'data-rank="(\d+)"',slz)[1])
norm={}
for task,score in scores.items():
    section=h.split('<div data-panel="'+task+'"',1)[1].split('<div data-panel=',1)[0]
    vals=[float(x) for x in re.findall(r'<span class="s" title="([\d.eE+-]+)"',section)]
    norm[task]={'min':min(vals),'max':max(vals),'contribution':(score-min(vals))/(max(vals)-min(vals))/3,
                'gain_general_pour_001':.01/(max(vals)-min(vals))/3}
fixed=sum(v['contribution'] for v in norm.values());assert abs(fixed-current)<1e-12
scenarios={}
for task,targets in {'34-watermark-localisation-image':[.7,.75,norm['34-watermark-localisation-image']['max']],
                      '33-model-equality':[.4,.6,1.], '31-unlearning':[.2,.3,norm['31-unlearning']['max']]}.items():
    scenarios[task]=[{'nouveau_score':s,'general_si_seule_cette_tache_change':current+(s-scores[task])/(norm[task]['max']-norm[task]['min'])/3} for s in targets]
result={'source':'capture officielle par le relais vers 03:17 Paris, copie figée','rang':rank,'general':current,
 'scores':scores,'seuil_sixieme':target,'ecart_general':target-current,'normalisation':norm,'scenarios_tableau_fige':scenarios,
 'limite':'Scénarios comptables, pas gains réalisables ni prévisions ; les autres équipes et les extrêmes évolueront.'}
(ROOT/'profondeur/diagnostic-classement.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
