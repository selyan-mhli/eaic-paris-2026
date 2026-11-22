"""Query the reference endpoints ref-01 ... ref-09 (task 33-model-equality).

The reference endpoints serve a separate small model, with the same kinds of disguises and traps as the scored
endpoints. Their correct answers are in practice_answers.csv. They have no query limit and are never scored, so use
them to build and test your method before you spend your 100 answers per scored endpoint (task_template.py).

How to use:
  1. Set API_KEY below and write your prompts in reference_prompts.txt (one per line; a .csv with columns id,prompt
     or a .jsonl also works).
  2. Run:  python query_reference.py
     The answers are appended to reference_answers.jsonl, one JSON line each:
        {"endpoint": "ref-03", "id": "3", "repeat": 0, "prompt": "...", "response": "..."}
     Stopping and restarting is safe: answers already in the file are skipped.
  3. Write your answer for the reference endpoints in the submission format (endpoint,family,model, one line per
     reference endpoint) and score it with:  python score_practice.py my_reference_answer.csv
     Do not put ref- lines in a real submission: the server only accepts ep-01 ... ep-33.

Needs: pip install requests
"""
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

# CONFIG
API_KEY = open("/p/scratch/training2666/mouhali1/SLZ/.eaic_key").read().strip().split("=", 1)[1].strip().strip(chr(34) + chr(39))
SERVER_URL = "http://34.147.229.55"
TASK_ID = "33-model-equality"            # do not change
PROMPTS_FILE = Path("prompts_ref_pilot3.csv")     # one prompt per line (a .csv with columns id,prompt or a .jsonl also works)
ANSWERS_FILE = Path("ref_answers.jsonl")   # the answers are appended here, one JSON line each

NUM_ENDPOINTS = 9                        # the server lists the real ones: python query_reference.py prints them
ENDPOINTS = [f"ref-{i:02d}" for i in range(1, NUM_ENDPOINTS + 1)]
REPEATS = 2                              # answers per prompt and endpoint (answers are sampled, so they differ)
MAX_TOKENS = 256                         # the server caps answers at 256 tokens
PARALLEL = 2                             # the most requests the server allows at once per team

BASE_URL = f"{SERVER_URL.rstrip('/')}/{TASK_ID}/v1"


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers["X-API-Key"] = API_KEY
    return session


def get_budget(session: requests.Session) -> dict:
    """Answers left on every endpoint, e.g. {"ep-01": 100, ...}."""
    reply = session.get(f"{BASE_URL}/models", timeout=60)
    if reply.status_code != 200:
        raise RuntimeError(f"Error (HTTP {reply.status_code}): {reply.text[:300]}")
    return {m["id"]: m["budget"]["limit"] - m["budget"]["used"] for m in reply.json()["data"]}


def read_prompts(path: Path) -> list:
    """Returns [(id, prompt), ...] from a .txt (one prompt per line), .csv (id,prompt) or .jsonl file."""
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            return [(row["id"], row["prompt"]) for row in csv.DictReader(f)]
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            return [(str(r["id"]), r["prompt"]) for r in (json.loads(line) for line in f if line.strip())]
    with path.open(encoding="utf-8") as f:
        return [(str(i), line.rstrip("\n")) for i, line in enumerate(f)]


def query(prompts: list, output: Path = ANSWERS_FILE, endpoints: list = ENDPOINTS, repeats: int = REPEATS,
          max_tokens: int = MAX_TOKENS) -> None:
    """Sends every prompt to every chosen endpoint and appends each answer to `output` as one JSON line:
        {"endpoint": "ep-07", "id": "3", "repeat": 0, "prompt": "...", "response": "..."}
    Each answer costs one query from that endpoint's budget, so only as many requests as you have left are planned.
    Stopping and restarting is safe: answers already in `output` are skipped."""
    session = make_session()
    left = get_budget(session)
    unknown = [e for e in endpoints if e not in left]
    if unknown:
        raise ValueError(f"unknown endpoints {unknown}; valid: {', '.join(left)}")

    done = set()
    if output.exists():
        with output.open(encoding="utf-8") as f:
            done = {(r["endpoint"], r["id"], r["repeat"]) for r in (json.loads(line) for line in f if line.strip())}
    # Round-robin over endpoints, so a run that stops early has covered all of them evenly.
    jobs, planned, over_budget = [], {ep: 0 for ep in endpoints}, 0
    for repeat in range(repeats):
        for pid, prompt in prompts:
            for ep in endpoints:
                if (ep, pid, repeat) in done:
                    continue
                if planned[ep] < left[ep]:
                    planned[ep] += 1
                    jobs.append((ep, pid, repeat, prompt))
                else:
                    over_budget += 1
    print(f"{len(prompts)} prompts x {len(endpoints)} endpoints x {repeats} repeats: {len(done)} answers already "
          f"in {output}, {len(jobs)} to query" + (f", {over_budget} left out (over budget)" if over_budget else ""))

    lock, exhausted, counter = threading.Lock(), set(), [0]
    out = output.open("a", encoding="utf-8")

    def one(job):
        ep, pid, repeat, prompt = job
        while ep not in exhausted:
            try:
                r = session.post(f"{BASE_URL}/chat/completions", timeout=180,
                                 json={"model": ep, "max_tokens": max_tokens,
                                       "messages": [{"role": "user", "content": prompt}]})
            except requests.RequestException as exc:
                print(f"Server unreachable ({type(exc).__name__}); retrying in 30 s.")
                time.sleep(30)
                continue
            if r.status_code == 200:
                body = r.json()
                line = {"endpoint": ep, "id": pid, "repeat": repeat, "prompt": prompt,
                        "response": body["choices"][0]["message"]["content"]}
                with lock:
                    out.write(json.dumps(line, ensure_ascii=False) + "\n")
                    out.flush()
                    counter[0] += 1
                    if counter[0] % 20 == 0 or counter[0] == len(jobs):
                        b = body["budget"]
                        print(f"{counter[0]}/{len(jobs)} answers ({ep}: {b['used']}/{b['limit']} used)")
                return
            detail = r.json().get("detail", r.text[:300]) if r.headers.get("content-type", "").startswith(
                "application/json") else r.text[:300]
            if r.status_code == 429 and "Retry-After" in r.headers:   # busy: wait for a free slot
                time.sleep(int(r.headers["Retry-After"]))
            elif r.status_code == 429:                                 # this endpoint's budget is used up
                with lock:
                    if ep not in exhausted:
                        exhausted.add(ep)
                        print(detail)
            elif r.status_code in (502, 503, 504):
                print(f"{detail} Retrying in 15 s.")
                time.sleep(15)
            else:
                print(f"{ep} id {pid}: HTTP {r.status_code}: {detail}", file=sys.stderr)
                return

    try:
        with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
            list(pool.map(one, jobs))
    finally:
        out.close()
    print(f"Finished: {output}")


def main() -> None:
    if API_KEY == "YOUR_API_KEY_HERE" or not API_KEY.strip():
        print("Set API_KEY at the top of query_reference.py to your team's API key.", file=sys.stderr)
        raise SystemExit(1)
    if not PROMPTS_FILE.exists():
        print(f"Error: {PROMPTS_FILE} not found. Write your prompts there (one per line).", file=sys.stderr)
        raise SystemExit(1)

    budget = get_budget(make_session())
    endpoints = [ep for ep in budget if ep.startswith("ref-")]
    assert endpoints and all(e.startswith("ref-") for e in endpoints)       # the reference endpoints the server lists
    print("Reference endpoints:", ", ".join(endpoints))
    query(read_prompts(PROMPTS_FILE), endpoints=endpoints, repeats=REPEATS)


if __name__ == "__main__":
    main()
