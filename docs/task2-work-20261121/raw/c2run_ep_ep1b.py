"""Model equality challenge - task template (task 33-model-equality).

You are given
  33 API endpoints, ep-01 ... ep-33 (in this order), some of which serve the same model.
  Each team can collect up to 100 answers per endpoint (at most 256 generated tokens per answer, prompts up to 1,024
  tokens), with at most 4 requests at a time.

Goal: decide for every pair of endpoints whether they serve the same model (same checkpoint, quantization and LoRA
adapter; a different decoding watermark, system prompt or decoding strategy does not change the model).

This file shows you how to
  1. check your query budget on every endpoint,
  2. send prompts to the endpoints and save the answers (query()),
  3. write a random, valid submission file (make_random_submission()), as an example of the file format.
Put your method in solve().
To test your method first on endpoints with known answers, use the reference endpoints ref-01 ... ref-09 with
query_reference.py (no query limit, never scored).

Submission file: a CSV with one line per endpoint and three columns, endpoint,family,model (the header line is optional,
the order of the lines does not matter). Labels are any text you like (no commas):
    endpoint,family,model
    ep-01,A,A1
    ep-02,B,B1
    ep-03,A,A2
Endpoints with the same family label are built on the same base model. Endpoints with the same model label serve exactly
the same model; they must also share the family label. Submit it with submission.py. The old matrix format is no longer
accepted.

Score: Matthews correlation coefficient (MCC, below 0 counts as 0) over pairs of endpoints. 10 % of the score is for the
same-family question (all pairs), 90 % for the same-model question (only pairs within one family). The leaderboard shows
the score on a part of the pairs; the final ranking uses the rest.

Needs: pip install requests
"""
import csv
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

# CONFIG
API_KEY = open("/p/scratch/training2666/mouhali1/SLZ/.eaic_key").read().strip().split("=", 1)[1].strip().strip(chr(34) + chr(39))            # replace with your team's API key (keep the double quotes)
SERVER_URL = "http://34.147.229.55"
TASK_ID = "33-model-equality"            # do not change
PROMPTS_FILE = Path("prompts_ep_1b.csv")       # one prompt per line (a .csv with columns id,prompt or a .jsonl also works)
ANSWERS_FILE = Path("ep_answers.jsonl")     # the answers are appended here, one JSON line each
OUTPUT_FILE = Path("NE_PAS_SOUMETTRE_baseline.csv")

QUERY = True                            # set to True to query the endpoints
RANDOM_SUBMISSION = False                # set to True to write a random example submission

NUM_ENDPOINTS = 33                       # the server lists the real endpoints: python task_template.py prints them
ENDPOINTS = [f"ep-{i:02d}" for i in range(1, NUM_ENDPOINTS + 1)]
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


def write_submission(rows: list, path: Path = OUTPUT_FILE) -> None:
    """Writes the submission as a CSV with the header endpoint,family,model and one (endpoint, family, model) row per
    endpoint."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["endpoint", "family", "model"])
        writer.writerows(rows)


def letters(i: int) -> str:
    """0 -> A, 1 -> B, ..., 25 -> Z, 26 -> AA, ... (family labels in the style of the example above)."""
    name = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        name = chr(65 + r) + name
    return name


def make_random_submission(endpoints: list = ENDPOINTS, families: int = 3, seed: int = None) -> list:
    """A random valid submission: every endpoint gets a random family, and inside a family a random model label.
    Two endpoints with the same model label always share the family label. Returns [(endpoint, family, model), ...]."""
    rng = random.Random(seed)
    rows = []
    for endpoint in endpoints:
        family = letters(rng.randrange(families))
        rows.append((endpoint, family, f"{family}{rng.randrange(3) + 1}"))
    return rows


def solve(endpoints: list = ENDPOINTS) -> list:
    """TODO: your method. Must return [(endpoint, family, model), ...], one row per endpoint. Endpoints with the same
    model label must share the family label. This baseline says "all different" (every endpoint in its own family and
    model); it scores 0."""
    return [(endpoint, letters(i), f"{letters(i)}1") for i, endpoint in enumerate(endpoints)]


def main() -> None:
    if API_KEY == "YOUR_API_KEY_HERE" or not API_KEY.strip():
        print("Set API_KEY at the top of task_template.py to your team's API key.", file=sys.stderr)
        raise SystemExit(1)

    budget = get_budget(make_session())
    print("Budget left per endpoint:")
    for ep, n in budget.items():
        print(f"  {ep}: {n}")
    endpoints = [ep for ep in budget if ep.startswith("ep-")]            # the scored endpoints
    if QUERY:
        if not PROMPTS_FILE.exists():
            print(f"Error: {PROMPTS_FILE} not found. Write your prompts there (one per line).", file=sys.stderr)
            raise SystemExit(1)
        query(read_prompts(PROMPTS_FILE), endpoints=endpoints)

    if RANDOM_SUBMISSION:
        rows = make_random_submission(endpoints)
    else:
        rows = solve(endpoints)
    write_submission(rows)
    print(f"Saved submission to {OUTPUT_FILE}. Upload it with submission.py.")


if __name__ == "__main__":
    main()
