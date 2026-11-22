"""Audit de reprise : lectures locales/SSH explicites, sans modèle ni requête notée.

Ne lit aucun fichier d'authentification. Les seules écritures sont dans ce dossier.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EVIDENCE = OUT / "preuves"
EVIDENCE.mkdir(exist_ok=True)


def capture(name, argv, stdin=None):
    p = subprocess.run(argv, input=stdin, text=True, capture_output=True, cwd=ROOT, timeout=90)
    result = {"heure_paris": datetime.now(ZoneInfo("Europe/Paris")).isoformat(),
              "commande": argv, "code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    (EVIDENCE / name).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(name, "code", p.returncode, "octets", len(p.stdout))
    return result


for name, args in {
    "git-depart.json": ["git", "status", "--short", "--branch"],
    "git-head.json": ["git", "rev-parse", "HEAD"],
    "git-worktrees.json": ["git", "worktree", "list", "--porcelain"],
    "git-log.json": ["git", "log", "--oneline", "-20"],
    "git-retard.json": ["git", "log", "--oneline", "HEAD..origin/main"],
    "git-diff.json": ["git", "diff", "--stat"],
}.items():
    capture(name, args)

for rel in ["docs/SLZ_COORDINATION_20261121.md", "docs/experiences.md",
            "docs/audit-global-20261122/RAPPORT.md", "docs/PLAN-NUIT.md",
            "docs/slz-jobs/recu_t3_push_s150.txt", "docs/slz-jobs/chain_push_ncc.log"]:
    shutil.copyfile(ROOT / rel, EVIDENCE / rel.replace("/", "__"))
old_ledger = Path("/private/tmp/claude-501/-Users-selyan--claude-spares-spare-LgnnGN/730425da-a91c-484f-8a0e-3638d4840e30/scratchpad/soumissions.md")
if old_ledger.exists():
    shutil.copyfile(old_ledger, EVIDENCE / "registre-temporaire-obsolete.md")
lb = Path("/private/tmp/eaic-reprise-leaderboard-20261122.html")
if lb.exists():
    shutil.copyfile(lb, EVIDENCE / "leaderboard-0308.html")

remote = r'''
import collections, datetime, hashlib, json, pathlib, subprocess
P = pathlib.Path('/p/project1/training2666/mouhali1')
S = pathlib.Path('/p/scratch/training2666/mouhali1/SLZ')
T1 = S / 'Hackathon_Watermark_Localisation_Task'
out = {'heure': datetime.datetime.now().astimezone().isoformat(), 'fichiers': [], 'logs': {}, 'json': {}}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
files = [T1/'output'/n for n in ['c1_pre4k_ens5.pt','c1_ensA9.pt','c1_ensB11.pt','c1_ensC10.pt','c1_ensD_psyn4.pt','c1_ensE_ssall3.pt']]
files += [S/'model_equality'/'output'/n for n in ['baseline_random.csv','c2_lump_v1.csv','c2_fam3_pairs_v2.csv']]
files += [S/'unlearning'/'work_c3_eval'/'push_scale'/n for n in ['push_s125.pt','push_s150.pt','push_s175.pt','push_s200.pt','clsA_kill_p150.pt','clsF_f125_p175.pt']]
files += [P/'ties'/n for n in ['ties_refs.jsonl','ties_query.err','probes_smol2.json','probes_B2.json','probes_L2.json','ties_ep_A.jsonl','ties_ep_B.jsonl','ties_ep_L.jsonl']]
files += [pathlib.Path('/p/scratch/training2666/mouhali1/c2run/ep/ep_answers.jsonl'),P/'wb2'/'collect_ep.jsonl']
for p in files:
    if not p.exists(): out['fichiers'].append({'path':str(p),'existe':False}); continue
    st=p.stat(); r={'path':str(p),'existe':True,'octets':st.st_size,'sha256':sha(p),'mtime':datetime.datetime.fromtimestamp(st.st_mtime).astimezone().isoformat()}
    if p.name.endswith('.jsonl'):
        records=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        r['lignes']=len(records);r['champs']=sorted(set().union(*(set(x) for x in records))) if records else []
        r['statuts']=dict(collections.Counter(str(x.get('status')) for x in records))
        r['endpoints']=dict(collections.Counter(str(x.get('ep',x.get('endpoint'))) for x in records))
        r['dates_min_max']=[min((x.get('ts','') for x in records),default=''),max((x.get('ts','') for x in records),default='')]
    out['fichiers'].append(r)
for pattern in ['15904543*.out','15904561*.out','15904581*.out']:
    for p in sorted((T1/'output').glob(pattern)): out['logs'][str(p)]=p.read_text()[-12000:]
for p in [P/'ties'/'ties_query.err',P/'ties'/'15904551.out']:
    if p.exists():out['logs'][str(p)]=p.read_text()[-18000:]
for p in [P/'wb2'/'sim_report.json',P/'wb2'/'rule.json',P/'wb_ep'/'result_15904307.json']:
    if p.exists():out['json'][str(p)]=json.loads(p.read_text())
for key,cmd in {
 'squeue':['squeue','-u','mouhali1,pelet1,zhang95','-o','%.12i %.14j %.10T %.10M %.10l %.5D %.15v %R'],
 'sacct':['sacct','-u','mouhali1','-S','2026-11-21T22:00','-X','-P','-o','JobID,JobName,State,ExitCode,Elapsed,ElapsedRaw,Start,End,AllocTRES%80'],
 'df_inodes_global':['df','-i','/p/scratch','/p/project1'],
}.items():
    p=subprocess.run(cmd,text=True,capture_output=True,timeout=30);out[key]={'commande':cmd,'code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
print(json.dumps(out,ensure_ascii=False,indent=2))
'''
capture("jureca-snapshot.json", ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
    "-o", "ControlPath=/private/tmp/eaic-codex-jrlogin03.sock", "-o", "HostName=jureca03.fz-juelich.de",
    "-o", "HostKeyAlias=jureca.fz-juelich.de", "jureca", "python3 -"], remote)

manifest = [{"fichier": str(p.relative_to(OUT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
             "octets": p.stat().st_size} for p in sorted(EVIDENCE.iterdir()) if p.is_file()]
(OUT/"MANIFESTE.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
