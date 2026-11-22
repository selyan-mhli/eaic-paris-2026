#!/usr/bin/env python3
#sends the prompt set (reference_prompts.txt) to ref-01...ref-09 (=> unlimited reference endpoints)
#resumes safely and appendfs the ref answers to /p/scratch/training2666/mouhali1/SLZ/model_equality/reference_answers.jsonl
import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

SERVER_URL = "http://34.147.229.55"
TASK_ID = "33-model-equality"
ENDPOINTS = [f"ref-{i:02d}" for i in range(1, 10)]


def read_prompts(path: Path) -> list[tuple[str, str]]:
    if path.suffix == ".csv":
        import csv

        with path.open(newline="", encoding="utf-8") as handle:
            return [(row["id"], row["prompt"]) for row in csv.DictReader(handle)]
    if path.suffix == ".jsonl":
        return [
            (str(row["id"]), row["prompt"])
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for row in [json.loads(line)]
        ]
    return [
        (str(i), line.rstrip("\n"))
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines())
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path(__file__).with_name("reference_prompts.txt"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/p/scratch/training2666/mouhali1/SLZ/model_equality/reference_answers.jsonl"),
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--parallel", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.parallel <= 4:
        print("--parallel must be between 1 and 4")
        return 1

    key = os.environ.get("EAIC_API_KEY")
    if not key:
        print("EAIC_API_KEY is not set")
        return 1
    if not args.prompts.is_file():
        print(f"Prompt file not found: {args.prompts}")
        return 1

    prompts = read_prompts(args.prompts)
    done = set()
    if args.output.exists():
        with args.output.open(encoding="utf-8") as handle:
            done = {
                (row["endpoint"], row["id"], row["repeat"])
                for line in handle
                if line.strip()
                for row in [json.loads(line)]
            }

    jobs = [
        (endpoint, prompt_id, repeat, prompt)
        for repeat in range(args.repeats)
        for prompt_id, prompt in prompts
        for endpoint in ENDPOINTS
        if (endpoint, prompt_id, repeat) not in done
    ]
    print(f"{len(done)} answers already in {args.output}; {len(jobs)} to query")

    session = requests.Session()
    session.headers["X-API-Key"] = key
    url = f"{SERVER_URL.rstrip('/')}/{TASK_ID}/v1/chat/completions"
    lock = threading.Lock()

    def one(job: tuple[str, str, int, str]) -> None:
        endpoint, prompt_id, repeat, prompt = job
        while True:
            response = session.post(
                url,
                json={
                    "model": endpoint,
                    "max_tokens": args.max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=180,
            )
            if response.status_code == 200:
                body = response.json()
                row = {
                    "endpoint": endpoint,
                    "id": prompt_id,
                    "repeat": repeat,
                    "prompt": prompt,
                    "response": body["choices"][0]["message"]["content"],
                }
                with lock:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    with args.output.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                return
            if response.status_code == 429:
                import time

                time.sleep(int(response.headers.get("Retry-After", "2")))
                continue
            print(f"{endpoint} id {prompt_id}: HTTP {response.status_code}: {response.text[:300]}")
            return

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        list(pool.map(one, jobs))
    print(f"Finished: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
