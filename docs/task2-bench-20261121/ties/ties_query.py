"""T2 sondes à quasi-égalité : interrogation des REFS UNIQUEMENT (gratuites), max_tokens=1.
Une tentative par requête (pas de relance sur 5xx ni erreur réseau ; 429 + Retry-After : attendre, 3 au plus).
PARALLEL <= 3. Ordre aléatoire seedé (ref × sonde × répétition). Journal JSONL neuf sur /p/project1.
Usage (login JURECA) : python ties_query.py --probes probes_smol.json --reps 10 --parallel 3
"""
import argparse, json, time, threading, datetime, os, random
from concurrent.futures import ThreadPoolExecutor
import requests

KEYFILE = "/p/scratch/training2666/mouhali1/SLZ/.eaic_key"
BASE = "http://34.147.229.55/33-model-equality/v1"
OUT = "/p/project1/training2666/mouhali1/ties/ties_refs.jsonl"
ALLOWED = {f"ref-{i:02d}" for i in range(1, 10)}


def _key():
    with open(KEYFILE) as f:
        return f.read().strip().split("=", 1)[1].strip().strip(chr(34) + chr(39))


S = requests.Session(); S.headers["X-API-Key"] = _key(); LOCK = threading.Lock()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True); ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--parallel", type=int, default=3); ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    assert a.parallel <= 3
    if os.path.exists(a.out):
        raise SystemExit(f"refus : {a.out} existe déjà")
    probes = json.load(open(a.probes))["probes"]
    jobs = [(ep, j, r) for ep in sorted(ALLOWED) for j in range(len(probes)) for r in range(a.reps)]
    random.Random(20261122).shuffle(jobs)
    f = open(a.out, "a", encoding="utf-8")
    cnt = {"ok": 0, "err": 0}; t00 = time.time()

    def one(job):
        ep, j, r = job
        assert ep in ALLOWED and ep.startswith("ref-")
        pr = probes[j]["prompt"]
        msgs = pr if isinstance(pr, list) else [{"role": "user", "content": pr}]
        payload = {"model": ep, "messages": msgs, "max_tokens": 1}
        for att in range(4):
            t0 = time.time(); rec = {"ts": now(), "ep": ep, "probe": j, "rep": r, "attempt": att}
            try:
                resp = S.post(f"{BASE}/chat/completions", json=payload, timeout=120)
                rec["status"] = resp.status_code
                try:
                    body = resp.json(); rec["text"] = body["choices"][0]["message"]["content"]
                except Exception:
                    rec["body"] = resp.text[:500]
            except requests.RequestException as exc:
                rec["status"] = None; rec["error"] = type(exc).__name__
            rec["dt"] = round(time.time() - t0, 2)
            with LOCK:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                cnt["ok" if rec["status"] == 200 else "err"] += 1
                n = cnt["ok"] + cnt["err"]
                if n % 50 == 0:
                    print(f"{n}/{len(jobs)} ok={cnt['ok']} err={cnt['err']} {time.time()-t00:.0f}s", flush=True)
            if rec["status"] == 429 and att < 3:
                time.sleep(int(float(resp.headers.get("Retry-After", 10)))); continue
            return
    with ThreadPoolExecutor(max_workers=a.parallel) as pool:
        list(pool.map(one, jobs))
    print(f"FIN {cnt} {time.time()-t00:.0f}s", flush=True)


if __name__ == "__main__":
    main()
