"""Soumission générique : POST + X-API-Key, journal, garde de débit.
Usage : EAIC_API_KEY=... python common/submit.py --url <endpoint> --task c1 --file sub.csv [--min-interval 300]
La clé n'est jamais affichée ni journalisée. Champ du fichier et format exact : à adapter aux consignes."""
import argparse, json, os, sys, time
from pathlib import Path
import requests

p = argparse.ArgumentParser()
p.add_argument("--url", required=True); p.add_argument("--task", required=True); p.add_argument("--file", required=True)
p.add_argument("--field", default="file"); p.add_argument("--min-interval", type=int, default=300)
p.add_argument("--note", default="")
a = p.parse_args()
key = os.environ.get("EAIC_API_KEY") or sys.exit("EAIC_API_KEY absente")
log = Path(__file__).resolve().parent.parent / "docs" / "soumissions.jsonl"
last = {}
if log.exists():
    for l in log.read_text().splitlines():
        r = json.loads(l)
        if r.get("ok"): last[r["task"]] = r["ts"]
wait = a.min_interval - (time.time() - last.get(a.task, 0))
if wait > 0: sys.exit(f"Trop tôt pour {a.task} : attendre {int(wait)} s")
with open(a.file, "rb") as f:
    r = requests.post(a.url, headers={"X-API-Key": key}, files={a.field: f}, timeout=120)
rec = {"ts": time.time(), "date": time.strftime("%F %T"), "task": a.task, "file": a.file, "note": a.note,
       "status": r.status_code, "ok": r.ok, "reponse": r.text[:500]}
log.parent.mkdir(exist_ok=True)
with log.open("a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(json.dumps(rec, ensure_ascii=False))
