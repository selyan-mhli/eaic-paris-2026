"""Score your answer for the practice endpoints (ref-01 ... ref-09) against their published answers.

The practice endpoints are a separate small model with the same kinds of disguises and traps as the scored endpoints.
They are never scored on the server; use them to check your method. Write your practice answer in the same format as a
submission (one line per endpoint: endpoint,family,model; a header line is allowed) and run:

    python score_practice.py my_practice_answer.csv [practice_answers.csv]

Lines for scored endpoints (ep-..) are ignored, so you can also run it on your full working file. It prints the
Matthews correlation coefficient (MCC) over the practice pairs ("same model?"), as the server computes the model part
of the score, and lists the pairs you got wrong. No packages needed beyond Python 3.
"""
import csv
import itertools
import math
import sys
from pathlib import Path


def read(path, with_kind=False):
    rows = [[c.strip() for c in r] for r in csv.reader(open(path, newline="", encoding="utf-8-sig")) if any(c.strip() for c in r)]
    if rows and rows[0][:3] == ["endpoint", "family", "model"]:
        rows = rows[1:]
    out = {}
    for r in rows:
        if len(r) >= 3 and r[0].startswith("ref-"):
            out[r[0]] = (r[1], r[2], r[3] if with_kind and len(r) > 3 else "")
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    answers = read(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).with_name("practice_answers.csv"), with_kind=True)
    mine = read(sys.argv[1])
    missing = sorted(set(answers) - set(mine))
    if missing:
        sys.exit(f"Your file has no line for: {', '.join(missing)}")
    same = lambda d, a, b: (d[a][0], d[a][1]) == (d[b][0], d[b][1])   # same model = same family and same model label
    tp = fp = fn = tn = 0
    wrong = []
    for a, b in itertools.combinations(sorted(answers), 2):
        truth, guess = same(answers, a, b), same(mine, a, b)
        tp += truth and guess; fp += (not truth) and guess; fn += truth and not guess; tn += (not truth) and not guess
        if truth != guess:
            wrong.append(f"  {a} ({answers[a][2]}) and {b} ({answers[b][2]}): you said {'same' if guess else 'different'}, "
                         f"they are {'the same model' if truth else 'different models'}")
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = max(0.0, (tp * tn - fp * fn) / d) if d else 0.0
    print(f"Practice score (MCC over {tp + fp + fn + tn} pairs, same model or not): {mcc:.3f}")
    print(f"Right: {tp + tn} pairs, wrong: {fp + fn} ({fp} wrongly called the same, {fn} wrongly called different)")
    if wrong:
        print("Wrong pairs:")
        print("\n".join(wrong))


if __name__ == "__main__":
    main()
