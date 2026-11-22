"""Test CPU de class_split.py (modèle factice, images aléatoires).
    .venv/bin/python docs/task3-work-20261121/mech/test_class_split.py
"""
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import class_split as cs  # noqa: E402
from test_mech import factory, preprocess  # noqa: E402


@torch.no_grad()
def logits_fn(m, x):
    return m.eval()(preprocess(x)).float()


def main():
    torch.manual_seed(0)
    orig = factory()
    cand = factory()
    with torch.no_grad():
        cand.head.bias[:3] -= 2.0
    forget = torch.randint(0, 256, (40, 16, 16, 3), dtype=torch.uint8)
    off = torch.randint(0, 256, (30, 16, 16, 3), dtype=torch.uint8)
    y = logits_fn(orig, forget).argmax(1)
    concerned = torch.zeros(10, dtype=torch.bool)
    concerned[y.unique()] = True
    views = {"clean": lambda x: x, "flip": lambda x: x.flip(2)}
    rows = cs.evaluate({"model": orig, "cand": cand, "same": factory()}, {"forget": forget, "off_rep": off},
                       views, logits_fn, concerned, log=lambda s: None)
    by = {(r[0], r[1], r[2], r[3]): r[4] for r in rows}
    # origine contre elle-même et copie identique : accord 1, KL 0, CE identiques
    for name in ("model", "same"):
        m = by[name, "forget", "clean", "concernee"]
        assert m["agree"] == 1.0 and abs(m["kl_T2"]) < 1e-6 and m["change"] == 0.0, m
    assert ("model", "forget", "clean", "autre") not in by          # forget : groupe concerné seul
    n_off = by["model", "off_rep", "clean", "concernee"]["n"] + by["model", "off_rep", "clean", "autre"]["n"]
    assert n_off == 30
    m = by["cand", "forget", "clean", "concernee"]
    assert m["kl_T2"] > 0 and 0 <= m["agree"] <= 1
    # métriques à la main sur un groupe
    ref, c = logits_fn(orig, forget), logits_fn(cand, forget)
    ce = torch.nn.functional.cross_entropy(c, y, reduction="none")
    assert abs(m["ce_mean"] - ce.mean().item()) < 1e-5 and abs(m["ce_std"] - ce.std(unbiased=False).item()) < 1e-5
    assert abs(m["change"] - (c.argmax(1) != y).float().mean().item()) < 1e-9
    lines = cs.format_rows(rows)
    assert lines[0].startswith("candidat\tjeu\tvue\tgroupe\tn") and len(lines) == 1 + len(rows)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.txt"
        cs.write_new(p, lines)
        try:
            cs.write_new(p, lines)
            raise AssertionError("écrasement non détecté")
        except FileExistsError:
            pass
    print(f"OK class_split ({len(rows)} lignes)")


if __name__ == "__main__":
    main()
