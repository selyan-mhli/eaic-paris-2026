"""Client de banc pour ref-01..ref-09 uniquement. Exécuté sur JURECA via stdin.
Règles : PARALLEL<=2 ; une seule tentative par requête ; 429+Retry-After -> attendre, max 3 retries ;
chaque tentative journalisée dans c2run/bench_<campagne>.jsonl (fichier neuf). La clé n'est jamais affichée."""
import json, time, threading, datetime, os, sys
from concurrent.futures import ThreadPoolExecutor
import requests

KEYFILE = "/p/scratch/training2666/mouhali1/SLZ/.eaic_key"
BASE = "http://34.147.229.55/33-model-equality/v1"
OUTDIR = "/p/scratch/training2666/mouhali1/c2run"
ALLOWED = {f"ref-{i:02d}" for i in range(1, 10)}

def _key():
    with open(KEYFILE) as f:
        return f.read().strip().split("=", 1)[1].strip().strip(chr(34) + chr(39))

_S = requests.Session()
_S.headers["X-API-Key"] = _key()
_LOCK = threading.Lock()

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")

class Bench:
    def __init__(self, campaign, fresh=True):
        self.campaign = campaign
        self.path = f"{OUTDIR}/bench_{campaign}.jsonl"
        if fresh and os.path.exists(self.path):
            raise SystemExit(f"refus : {self.path} existe déjà")
        self.f = open(self.path, "a", encoding="utf-8")
        self.n = 0

    def log(self, rec):
        with _LOCK:
            self.f.write(json.dumps(rec, ensure_ascii=False) + "\n"); self.f.flush(); self.n += 1

    def models(self):
        t0 = time.time(); r = _S.get(f"{BASE}/models", timeout=60)
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text[:2000]
        self.log({"ts": now(), "campaign": self.campaign, "kind": "models", "status": r.status_code,
                  "dt": round(time.time() - t0, 3), "body": body})
        return body

    def post(self, tag, payload, path="chat/completions"):
        ep = payload.get("model")
        assert ep in ALLOWED, f"endpoint interdit {ep}"
        for attempt in range(4):
            t0 = time.time(); rec = {"ts": now(), "campaign": self.campaign, "tag": tag, "attempt": attempt,
                                     "path": path, "payload": payload}
            try:
                r = _S.post(f"{BASE}/{path}", json=payload, timeout=180)
            except requests.RequestException as exc:
                rec.update(status=None, error=type(exc).__name__, dt=round(time.time() - t0, 3)); self.log(rec)
                return rec
            rec["status"] = r.status_code; rec["dt"] = round(time.time() - t0, 3)
            rec["headers"] = {k: v for k, v in r.headers.items() if "key" not in k.lower() and "cookie" not in k.lower()}
            try:
                rec["body"] = r.json()
            except Exception:
                rec["body"] = r.text[:2000]
            self.log(rec)
            if r.status_code == 429 and "Retry-After" in r.headers and attempt < 3:
                time.sleep(int(float(r.headers["Retry-After"]))); continue
            return rec

    def run(self, jobs, parallel=2):
        """jobs: list of (tag, payload[, path])."""
        assert parallel <= 2
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            return list(pool.map(lambda j: self.post(*j), jobs))

def text(rec):
    try:
        return rec["body"]["choices"][0]["message"]["content"]
    except Exception:
        return None
