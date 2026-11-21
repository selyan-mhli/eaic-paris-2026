"""Tests de c2/features.py sur données synthétiques et cas faits main (aucune réponse réelle, aucune requête)."""
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import analyze  # noqa: E402
import features as F  # noqa: E402
import synth  # noqa: E402


def _grouped(scenario="ref", prompts=10, reps=6, seed=0):
    eps = synth.scenario(scenario)
    recs = synth.generate(eps, prompts, reps, seed)
    truth = {e: (f, m) for e, f, m, _ in synth.truth_rows(eps)}
    return analyze.group(recs), truth


def test_references_are_correct():
    assert F.pi_digits(30) == "141592653589793238462643383279"
    p = F._primes(60)
    assert p[:5] == [2, 3, 5, 7, 11] and p[-1] == 281 and len(p) == 60
    assert F._fib(6) == [0, 1, 1, 2, 3, 5]


def test_extraction_by_hand():
    assert F.numbers("2, 3, 5, 7 and 11.") == [2, 3, 5, 7, 11]
    assert F.digit_string("3.14159 26535") == "1415926535"
    assert F.items("1. Alabama - Montgomery\n2. Alaska - Juneau\n") == ["alabama - montgomery", "alaska - juneau"]
    assert F.items("cat, Dog,  owl.") == ["cat", "dog", "owl"]
    # divergence : préfixe correct, quelle que soit la variante de référence (Fibonacci à 0 ou à 1)
    assert F.divergence("m03", "2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 51") == 15
    assert F.divergence("c01", "1, 1, 2, 3, 5, 8, 13, 22") == 7
    assert F.divergence("m04", "3.14159265358979323846264338327 9") == 30
    assert F.divergence("zz", "1 2 3") is None


def test_kernels_by_hand():
    K = F._positional([[1, 2, 3], [1, 2, 4], []])
    assert K[0, 1] == 2 / 3 and K[0, 2] == 0.0 and np.allclose(np.diag(K), 1)
    K = F._lcp([["a", "b", "c", "d"], ["a", "b", "x"], ["z"]])
    assert K[0, 1] == 2 / 4 and K[0, 2] == 0.0
    K = F._jaccard([{1, 2}, {2, 3}, set()])
    assert K[0, 1] == 1 / 3
    for view in F.VIEWS:
        texts = ["2, 3, 5, 7, 11, 13", "2, 3, 5, 7, 11, 15", "2, 3, 5, 9, 11, 13", "1, 2, 3, 4, 5, 6"]
        K = F.view_kernel(view, "m03", texts)
        if K is None:
            continue
        assert K.shape == (4, 4), view
        assert np.allclose(K, K.T) and np.all(K >= -1e-9) and np.all(K <= 1 + 1e-9), view


def test_boilerplate_sentences_removed_only_when_repeated_across_prompts():
    g = {f"p{i}": {"a": [f"Ahoy matey, a fine question! answer {i} here.", f"answer {i} again."],
                   "b": [f"answer {i} here.", f"Ahoy matey, a fine question! other {i}."]} for i in range(4)}
    g["p0"]["b"] = ["Ahoy matey, a fine question! once."]
    out = F.strip_boilerplate(g, min_prompts=3)
    assert all("ahoy" not in t.lower() for t in out["p1"]["a"])
    assert "answer 1 here." in out["p1"]["a"][0]
    assert all("ahoy" not in t.lower() for t in out["p2"]["b"])        # b la répète aussi (4 prompts)


def test_feature_distances_shapes_and_auc_on_synthetic():
    g, truth = _grouped()
    g = F.prepare(g)
    eps = sorted({e for d in g.values() for e in d})
    fd = F.feature_distances(g, eps, ("char", "word", "lcp", "len"))
    assert fd and all(D.shape == (9, 9) for D in fd.values())
    D = F.combine_features(fd, [k for k in fd if k[1] in ("char", "word")])
    assert np.allclose(D, D.T, equal_nan=True)
    assert F.pair_auc(D, eps, truth) > 0.95
    hp = F.hard_pairs(truth)
    assert len(hp) == 2 * 5 and all(a in F.HARD_DEFAULT for a, _ in hp)
    assert F.pair_auc(D, eps, truth, hp) > 0.9


def test_selection_and_evaluation_use_disjoint_repetitions():
    g, _ = _grouped(reps=6)
    even, odd = F.parity(g, 0), F.parity(g, 1)
    for p in g:
        for e in g[p]:
            assert even[p][e] == g[p][e][0::2] and odd[p][e] == g[p][e][1::2]
            assert not set(even[p][e]) & set(odd[p][e])


def test_evaluate_on_synthetic_ref_is_perfect():
    g, truth = _grouped(prompts=10, reps=6)
    res = F.evaluate(g, truth, views=("char", "word"), top=6)
    for fold in res["folds"]:
        assert fold["mcc_eval"] == 1.0 and fold["auc"] > 0.95
        assert len(fold["keys"]) <= 6


def test_split_half_threshold_is_a_noise_floor():
    """Le plancher soi-contre-soi est sous les distances entre modèles différents (synthétique)."""
    g, truth = _grouped(prompts=10, reps=8)
    g = F.prepare(g)
    eps = sorted({e for d in g.values() for e in d})
    keys = [(p, "char") for p in g]
    fd = F.feature_distances(g, eps, ("char",))
    sc = F.median_scales(fd)
    D = F.combine_features({k: fd[k] / sc[k] for k in keys}, keys, scale="none")
    t = F.split_half_threshold(g, eps, keys, fd_scale=sc)
    pos = {e: i for i, e in enumerate(eps)}
    diff = [D[pos[a], pos[b]] for a, b in itertools.combinations(eps, 2) if truth[a] != truth[b]]
    assert np.isfinite(t) and t < min(diff)


def test_leave_pair_out_runs_and_scores():
    g, truth = _grouped(prompts=6, reps=4)
    r = F.leave_pair_out(g, truth, views=("char",), top=4)
    assert r["n"] == 36 and r["mcc"] > 0.9


def test_cli_run_writes_kit_format(tmp_path):
    eps = synth.scenario("ref")
    ans = tmp_path / "a.jsonl"
    with ans.open("w") as f:
        for r in synth.generate(eps, 5, 4):
            f.write(json.dumps(r) + "\n")
    out = tmp_path / "g.csv"
    assert F.main(["run", str(ans), "--prefix", "ref-", "--features", "m00:char,m01:word,m02:lcp",
                   "--t-model", "split", "--out", str(out)]) == 0
    rows = list(csv.reader(out.open()))
    assert rows[0] == ["endpoint", "family", "model"] and len(rows) == 10
    assert F.main(["eval", str(ans), "--truth", str(HERE / "kit" / "practice_answers.csv"), "--views", "char",
                   "--top", "3"]) == 0
