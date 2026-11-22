"""Tests de c2/outliers.py (cas faits main, aucune requête)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import analyze  # noqa: E402
import outliers as O  # noqa: E402


def test_robust_z_isolates_the_far_endpoint_and_scores():
    eps = [f"ref-0{i}" for i in range(1, 10)]
    D = np.full((9, 9), 0.10) + np.diag(np.full(9, -0.10))
    D[6, :] = D[:, 6] = 0.50                     # ref-07 loin de tous
    D[6, 6] = 0.0
    D += np.random.default_rng(0).normal(0, 0.005, (9, 9)) * (1 - np.eye(9))
    D = (D + D.T) / 2
    z = O.robust_z(eps, D, {e: "P" for e in eps})
    assert max(eps, key=lambda e: z[e][1]) == "ref-07" and z["ref-07"][1] > 5
    rows = O.rows_from_z(eps, {e: "P" for e in eps}, z, 3.0)
    truth = analyze.read_truth(Path(__file__).parent / "kit" / "practice_answers.csv")
    assert abs(analyze.score(truth, rows)["practice_mcc"] - 0.331) < 1e-3    # un gros groupe + ref-07 isolé


def test_rows_refuse_non_finite_threshold():
    with pytest.raises(ValueError):
        O.rows_from_z(["a"], {"a": "F"}, {"a": (0.0, 0.0)}, float("inf"))
