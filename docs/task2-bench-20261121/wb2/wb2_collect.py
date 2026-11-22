"""COLLECTE EP wb2 : NE PAS LANCER SANS GO DE SELYAN. 3 prompts (B1-B3 de wb2_lib) x 10 répétitions x 33 ep = 990
réponses (30 par endpoint), max_tokens 256, PARALLEL 4 (aucune autre collecte ne doit tourner).
Règles : une seule tentative par requête ; 429 avec Retry-After -> attente puis au plus 3 relances ; 5xx et erreurs
réseau -> pas de relance ; arrêt global après 5 erreurs 5xx/réseau consécutives ou 20 au total ; chaque tentative
journalisée (statut, horodatage, campagne, tentative) dans JOURNAL (fichier neuf, /p/project1). Budget lu (GET, gratuit)
avant et après, et contrôle : chaque ep doit avoir >= 30 restantes, sinon refus de démarrer. La clé n'est jamais affichée.
Lancement (nœud de login, nohup) :
  cd /p/project1/training2666/mouhali1/wb2 && nohup /p/scratch/training2666/mouhali1/SLZ/model_equality/.venv/bin/python -u wb2_collect.py > collect.err 2>&1 < /dev/null &"""
import json, time, threading, datetime, os, sys
from concurrent.futures import ThreadPoolExecutor
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wb2_lib import PROMPTS, MAX_TOKENS
KEYFILE = "/p/scratch/training2666/mouhali1/SLZ/.eaic_key"
BASE = "http://34.147.229.55/33-model-equality/v1"
JOURNAL = "/p/project1/training2666/mouhali1/wb2/collect_ep.jsonl"
EPS = [f"ep-{i:02d}" for i in range(1, 34)]; REPS = 10; PARALLEL = 4; CAMPAIGN = "wb2_ep"
def key():
    return open(KEYFILE).read().strip().split("=", 1)[1].strip().strip(chr(34) + chr(39))
S = requests.Session(); S.headers["X-API-Key"] = key(); LOCK = threading.Lock()
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")
if os.path.exists(JOURNAL): sys.exit(f"refus : {JOURNAL} existe déjà")
F = open(JOURNAL, "a", encoding="utf-8")
def log(rec):
    with LOCK: F.write(json.dumps(rec, ensure_ascii=False) + "\n"); F.flush()
def budget(tag):
    r = S.get(f"{BASE}/models", timeout=60); body = r.json() if r.status_code == 200 else r.text[:500]
    log({"ts": now(), "campaign": CAMPAIGN, "kind": "budget", "tag": tag, "status": r.status_code, "body": body})
    if r.status_code != 200: sys.exit(f"budget illisible ({r.status_code})")
    return {m["id"]: m["budget"]["limit"] - m["budget"]["used"] for m in body["data"]}
left = budget("avant")
low = {e: left.get(e) for e in EPS if (left.get(e) or 0) < REPS * len(PROMPTS)}
if low: sys.exit(f"refus : budget insuffisant {low}")
ERR = {"consec": 0, "total": 0}; STOP = threading.Event()
def one(job):
    tag, payload = job
    for attempt in range(4):
        if STOP.is_set():
            log({"ts": now(), "campaign": CAMPAIGN, "tag": tag, "attempt": attempt, "status": "skipped_stop", "payload": payload}); return
        t0 = time.time(); rec = {"ts": now(), "campaign": CAMPAIGN, "tag": tag, "attempt": attempt, "payload": payload}
        try:
            r = S.post(f"{BASE}/chat/completions", json=payload, timeout=180)
        except requests.RequestException as exc:
            rec.update(status=None, error=type(exc).__name__, dt=round(time.time() - t0, 3)); log(rec); bad(); return
        rec["status"] = r.status_code; rec["dt"] = round(time.time() - t0, 3)
        rec["retry_after"] = r.headers.get("Retry-After")
        try: rec["body"] = r.json()
        except Exception: rec["body"] = r.text[:1000]
        log(rec)
        if r.status_code == 200:
            with LOCK: ERR["consec"] = 0
            return
        if r.status_code == 429 and r.headers.get("Retry-After") and attempt < 3:
            time.sleep(int(float(r.headers["Retry-After"]))); continue
        if r.status_code >= 500: bad()
        return
def bad():
    with LOCK:
        ERR["consec"] += 1; ERR["total"] += 1
        if ERR["consec"] >= 5 or ERR["total"] >= 20: STOP.set()
jobs = [(f"{pid}|{rep}|{ep}", {"model": ep, "max_tokens": MAX_TOKENS, "messages": [{"role": "user", "content": PROMPTS[pid]}]})
        for rep in range(REPS) for pid in PROMPTS for ep in EPS]
assert len(jobs) == 990 and all(j[1]["model"].startswith("ep-") for j in jobs)
print("début", now(), "requêtes", len(jobs), flush=True)
with ThreadPoolExecutor(max_workers=PARALLEL) as pool: list(pool.map(one, jobs))
after = budget("après")
print("fin", now(), "erreurs", ERR, "arrêt", STOP.is_set(), "consommé", {e: left[e] - after[e] for e in EPS}, flush=True)
