"""Synthèse CPU des preuves existantes ; aucun réseau et aucune inférence."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import collections
import csv
import hashlib
import json
import re
import shutil

HERE = Path(__file__).resolve().parent
PROOF = HERE / "preuves"
ROOT = HERE.parents[1]
remote = json.loads(json.loads((PROOF / "jureca-snapshot.json").read_text())["stdout"])
html = (PROOF / "leaderboard-0308.html").read_text()
panel = html.split('<div data-panel="overall">', 1)[1].split('<div data-panel=', 1)[0]
slz = next(r for r in re.findall(r'<tr[^>]*>.*?</tr>', panel, re.S) if 'data-team="SLZ"' in r)
scores = {task: float(value) for task, value in re.findall(r'data-task="([^"]*)"[^>]*>.*?title="([\d.eE+-]+)"', slz)}
overall = float(re.search(r'class="overall" title="([^"]+)"', slz)[1])
ranking = {"source": "http://34.147.229.55/leaderboard_page", "capture_paris": "2026-11-22T03:05+01:00",
           "rang": int(re.search(r'data-rank="(\d+)"', slz)[1]),
           "equipes_classees": len(re.findall(r'<tr class="ranked\b', panel)),
           "scores": scores, "general": overall, "normalisation": {}}
for task, score in scores.items():
    section = html.split('<div data-panel="'+task+'"', 1)[1].split('<div data-panel=', 1)[0]
    vals = [float(x) for x in re.findall(r'<span class="s" title="([\d.eE+-]+)"', section)]
    row = next(r for r in re.findall(r'<tr[^>]*>.*?</tr>', section, re.S) if 'data-team="SLZ"' in r)
    ranking["normalisation"][task] = {"min": min(vals), "max": max(vals),
        "normalise": (score-min(vals))/(max(vals)-min(vals)), "participants": len(vals),
        "rang": int(re.search(r'class="rank">(\d+)', row)[1]),
        "date_meilleur_utc": re.search(r'data-at="([^"]+)"', row)[1]}
calculated = sum(v["normalise"] for v in ranking["normalisation"].values()) / 3
assert abs(calculated-overall) < 1e-12
ranking["general_recalcule"] = calculated
(HERE / "classement-verifie.json").write_text(json.dumps(ranking, ensure_ascii=False, indent=2)+"\n")

lines = remote["sacct"]["stdout"].splitlines()
jobs = [dict(zip(lines[0].split("|"), row.split("|"))) for row in lines[1:]]
summary = {"compte": "mouhali1", "fenetre": "sacct depuis 2026-11-21T22:00, relevé 2026-11-22T03:06:57+01:00",
           "jobs": len(jobs), "etats": dict(collections.Counter(r["State"].split()[0] for r in jobs)),
           "gpu_heures_allouees": sum(int(r["ElapsedRaw"])*4 for r in jobs if 'gres/gpu=4' in r["AllocTRES"])/3600,
           "limite": "Allocation, pas utilisation GPU mesurée ; comptabilité des autres comptes non couverte."}
(HERE / "jobs-verifies.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
with (HERE / "jobs-verifies.csv").open("w") as f:
    w=csv.DictWriter(f, fieldnames=list(jobs[0]));w.writeheader();w.writerows(jobs)

old = Path('/private/tmp/claude-501/-Users-selyan--claude-spares-spare-LgnnGN/730425da-a91c-484f-8a0e-3638d4840e30/scratchpad/dash/out/soumissions.json')
ledger = json.loads(old.read_text())
shutil.copyfile(old, PROOF / "registre-dashboard-avant-reprise.json")
for row in ledger:
    prefix = row["sha256"]
    matches = [f for f in remote["fichiers"] if f.get("sha256", "").startswith(prefix)] if prefix else []
    if len(matches)==1: row["sha256"] = matches[0]["sha256"]
    row["preuve"] = "registre historique ; reçu individuel non archivé dans ce dossier"
    row["statut"] = "envoi rapporté"
    if 'inconnu' in row["fichier"]: row["statut"] = "envoi rapporté, artefact inconnu"
    if 'doublon' in row["fichier"]: row["statut"] = "doublon rapporté"
    if 's150.pt' in row["fichier"]: row["preuve"] = "reçu texte conservé + hash distant"
    if 's175.pt' in row["fichier"]: row["preuve"] = "registre + meilleur T3 confirmé au classement + hash distant"
champ = next(f for f in remote["fichiers"] if f["path"].endswith('/c1_ensE_ssall3.pt'))
ledger.append({"heure":"22/11 02:43:05 CET (classement)", "tache":"Task 1", "fichier":"output/c1_ensE_ssall3.pt",
 "sha256":champ["sha256"], "score_local":None, "score_public":ranking["scores"]["34-watermark-localisation-image"],
 "par":"Selyan, terminal (reçu transmis)", "note":"champion ; trois graines 50/51/52 sur SS, autres méthodes conservées",
 "preuve":"reçu utilisateur + classement officiel + hash distant", "statut":"envoi confirmé"})
with (HERE / "soumissions-reconstituees.csv").open("w") as f:
    w=csv.DictWriter(f, fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
(HERE / "soumissions-reconstituees.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=2)+"\n")

for name in ["iou2_5fold.json", "compare6k.json", "ties_result.json", "A3_strict_counts.json", "c1_ensF_ssall6k2.json"]:
    shutil.copyfile(ROOT/"docs/relais-codex-20261122"/name, PROOF/("relais__"+name))

manifest = [{"fichier":str(p.relative_to(HERE)), "sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
             "octets":p.stat().st_size} for p in sorted(HERE.rglob('*')) if p.is_file() and p.name != "MANIFESTE.json"]
(HERE/"MANIFESTE.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
print(json.dumps({"classement":ranking,"jobs":summary,"lignes_registre":len(ledger),
                  "controle":"général recalculé à moins de 1e-12 ; JSON sources lus ; preuves copiées"},ensure_ascii=False,indent=2))
