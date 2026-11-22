"""T2 sondes : interrogation des ENDPOINTS NOTÉS ep-* d'UNE famille, max_tokens=1. BUDGET IRRÉVERSIBLE.
Ne se lance qu'avec --go-selyan (GO explicite de Selyan). Famille = base identifiée (result_15904307.json).
Une tentative par requête (429 + Retry-After : 3 au plus ; ni 5xx ni réseau retentés). Arrêt après 5 erreurs
consécutives ou 20 au total. Budget relu avant (refus si un endpoint < reps*nb_sondes + 5 restantes).
Usage : python ties_query_ep.py --family B --probes probes_B2.json --reps 4 --go-selyan
"""
import argparse, json, time, threading, datetime, os, random, sys
from concurrent.futures import ThreadPoolExecutor
import requests

KEYFILE = "/p/scratch/training2666/mouhali1/SLZ/.eaic_key"
BASE = "http://34.147.229.55/33-model-equality/v1"
RES = "/p/project1/training2666/mouhali1/wb_ep/result_15904307.json"
FAMBASE = {"A": "Qwen__Qwen2.5-3B-Instruct", "B": "Qwen__Qwen2.5-1.5B-Instruct", "L": "unsloth__Llama-3.2-1B-Instruct"}


def _key():
    with open(KEYFILE) as f:
        return f.read().strip().split("=", 1)[1].strip().strip(chr(34) + chr(39))


S = requests.Session(); S.headers["X-API-Key"] = _key(); LOCK = threading.Lock()
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


def budget():
    r = S.get(f"{BASE}/models", timeout=60); r.raise_for_status()
    return {m["id"]: m["budget"]["limit"] - m["budget"]["used"] for m in r.json()["data"]}


def endpoints_of(fam):
    R = json.load(open(RES))
    base = R.get("base") or R.get("bases")
    eps = sorted(e for e, b in base.items() if b == FAMBASE[fam])
    assert len(eps) == 11, (fam, eps)
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(FAMBASE)); ap.add_argument("--probes", required=True)
    ap.add_argument("--reps", type=int, default=4); ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--go-selyan", action="store_true")
    a = ap.parse_args()
    if not a.go_selyan:
        raise SystemExit("refus : GO explicite de Selyan requis (--go-selyan)")
    assert a.parallel <= 3
    out = f"/p/project1/training2666/mouhali1/ties/ties_ep_{a.family}.jsonl"
    if os.path.exists(out):
        raise SystemExit(f"refus : {out} existe déjà")
    probes = json.load(open(a.probes))["probes"]
    eps = endpoints_of(a.family)
    need = a.reps * len(probes)
    b0 = budget(); print("budget avant", {e: b0[e] for e in eps}, flush=True)
    low = [e for e in eps if b0[e] < need + 5]
    if low:
        raise SystemExit(f"refus : budget insuffisant {low}")
    jobs = [(ep, j, r) for ep in eps for j in range(len(probes)) for r in range(a.reps)]
    random.Random(20261122).shuffle(jobs)
    f = open(out, "a", encoding="utf-8")
    cnt = {"ok": 0, "err": 0, "consec": 0}; stop = threading.Event(); t00 = time.time()

    def one(job):
        if stop.is_set():
            return
        ep, j, r = job
        assert ep.startswith("ep-") and ep in eps
        pr = probes[j]["prompt"]
        msgs = pr if isinstance(pr, list) else [{"role": "user", "content": pr}]
        payload = {"model": ep, "messages": msgs, "max_tokens": 1}
        for att in range(4):
            t0 = time.time(); rec = {"ts": now(), "ep": ep, "probe": j, "rep": r, "attempt": att}
            try:
                resp = S.post(f"{BASE}/chat/completions", json=payload, timeout=120)
                rec["status"] = resp.status_code
                try:
                    rec["text"] = resp.json()["choices"][0]["message"]["content"]
                except Exception:
                    rec["body"] = resp.text[:500]
            except requests.RequestException as exc:
                rec["status"] = None; rec["error"] = type(exc).__name__
            rec["dt"] = round(time.time() - t0, 2)
            with LOCK:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                if rec["status"] == 200:
                    cnt["ok"] += 1; cnt["consec"] = 0
                elif rec["status"] != 429:
                    cnt["err"] += 1; cnt["consec"] += 1
                    if cnt["consec"] >= 5 or cnt["err"] >= 20:
                        stop.set()
                n = cnt["ok"] + cnt["err"]
                if n % 25 == 0:
                    print(f"{n}/{len(jobs)} ok={cnt['ok']} err={cnt['err']} {time.time()-t00:.0f}s", flush=True)
            if rec["status"] == 429 and att < 3:
                time.sleep(int(float(resp.headers.get("Retry-After", 10)))); continue
            return
    with ThreadPoolExecutor(max_workers=a.parallel) as pool:
        list(pool.map(one, jobs))
    b1 = budget(); print("budget après", {e: b1[e] for e in eps}, flush=True)
    print(f"FIN {cnt} arrêt={stop.is_set()} {time.time()-t00:.0f}s", flush=True)


if __name__ == "__main__":
    main()
