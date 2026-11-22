"""Compare les trois copies existantes d'analyze.py, sans accès réseau."""
import csv
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

base=Path(__file__).resolve().parent
out=base/'version-replays'
out.mkdir(exist_ok=True)
evidence=json.loads((base/'remote_evidence.json').read_text())
results={}

def eqpairs(path):
    rows=list(csv.DictReader(path.open()))
    d={r['endpoint']:(r['family'],r['model']) for r in rows}
    return {(a,b): (d[a][0]==d[b][0],d[a]==d[b]) for a,b in itertools.combinations(sorted(d),2)}

for version in ('main','branche-c2','collecte-distante'):
    for kernel in ('hamming','ngram'):
        script=base/'sources'/version/'analyze.py'
        target=out/f'{version}_{kernel}.csv'
        cmd=[sys.executable,str(script),str(base/'data/c2run/ep/ep_answers.jsonl'),
             '--prefix','ep-','--kernel',kernel,'--out',str(target),'--quiet']
        run=subprocess.run(cmd,capture_output=True,text=True)
        d={'command':cmd,'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr,
           'source_sha256':hashlib.sha256(script.read_bytes()).hexdigest()}
        if run.returncode==0:
            got=eqpairs(target)
            saved=base/f'data/SLZ/model_equality/output/ep_first_{kernel}.csv'
            actual=eqpairs(saved)
            d['diff_pairs_from_saved']={part:sum(got[k][i]!=actual[k][i] for k in got) for i,part in enumerate(('family','model'))}
        results[f'{version}_{kernel}']=d

(base/'version_replays.json').write_text(json.dumps(results,indent=2))
for k,d in results.items():print(k,d.get('diff_pairs_from_saved'),d['stdout'].strip())
