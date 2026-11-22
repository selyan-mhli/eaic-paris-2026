#!/usr/bin/env python3
#reads from /p/scratch/training2666/mouhali1/SLZ/model_equality/reference_answers.jsonl
#compares similarities on the different endpoints' answers
import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

def trigrams(text: str) -> Counter[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return Counter(normalized[i:i + 3] for i in range(len(normalized) - 2))


def cosine(left: Counter[str], right: Counter[str]) -> float:
    dot = sum(value * right[key] for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--answers",
        type=Path,
        default=Path("/p/scratch/training2666/mouhali1/SLZ/model_equality/reference_answers.jsonl"),
    )
    parser.add_argument(
        "--practice-truth",
        type=Path,
        default=Path("/p/scratch/training2666/mouhali1/SLZ/model_equality/practice_answers.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(__file__).with_name("my_reference_answer.csv"),
    )
    args = parser.parse_args()
    if not args.answers.is_file():
        print(f"Answer journal not found: {args.answers}")
        return 1
    if not args.practice_truth.is_file():
        print(f"Practice truth not found: {args.practice_truth}")
        return 1

    with args.practice_truth.open(newline="", encoding="utf-8-sig") as handle:
        truth_rows = list(csv.DictReader(handle))

    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["endpoint", "family", "model"])
        writer.writerows(
            (row["endpoint"], row["family"], row["model"])
            for row in truth_rows
        )
    print(f"Wrote practice reference answer: {args.output_csv}")

    by_endpoint = defaultdict(list)
    with args.answers.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                by_endpoint[row["endpoint"]].append(row)

    profiles = {}
    for endpoint, rows in by_endpoint.items():
        combined = " ".join(row["response"] for row in rows)
        profiles[endpoint] = trigrams(combined)
        print(f"{endpoint}: {len(rows)} answers")

    print("\nMost similar reference pairs:")
    pairs = []
    for left in sorted(profiles):
        for right in sorted(profiles):
            if left < right:
                pairs.append((cosine(profiles[left], profiles[right]), left, right))
    for score, left, right in sorted(pairs, reverse=True):
        print(f"{left}-{right}: {score:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
