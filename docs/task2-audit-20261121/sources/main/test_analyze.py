"""Tests c2 sur données synthétiques uniquement (aucune requête au serveur)."""
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import analyze  # noqa: E402
import synth  # noqa: E402


def _grouped(scenario, prompts, reps, seed=0):
    eps = synth.scenario(scenario)
    recs = synth.generate(eps, prompts, reps, seed)
    truth = {e: (f, m) for e, f, m, _ in synth.truth_rows(eps)}
    return analyze.group(recs), truth


def test_score_matches_kit_scorer(tmp_path):
    """Notre MCC « même modèle » = celui de score_practice.py du kit, sur une réponse volontairement fausse."""
    truth = analyze.read_truth(HERE / "kit" / "practice_answers.csv")
    guess = [("ref-01", "A", "A1"), ("ref-02", "A", "A1"), ("ref-03", "A", "A1"), ("ref-04", "A", "A2"),
             ("ref-05", "A", "A3"), ("ref-06", "A", "A1"), ("ref-07", "A", "A2"), ("ref-08", "B", "B1"),
             ("ref-09", "A", "A4")]
    path = tmp_path / "guess.csv"
    analyze.write_rows(guess, path)
    out = subprocess.run([sys.executable, str(HERE / "kit" / "score_practice.py"), str(path)],
                         capture_output=True, text=True, check=True).stdout
    kit = float(out.split(":")[1].split()[0])
    assert abs(analyze.score(truth, guess)["practice_mcc"] - kit) < 1e-3


def test_perfect_answer_scores_one():
    truth = analyze.read_truth(HERE / "kit" / "practice_answers.csv")
    rows = [(e, f, f + m) for e, (f, m) in truth.items()]
    assert analyze.score(truth, rows)["practice_mcc"] == 1.0


def test_partition_is_nested_and_transitive():
    rng = np.random.default_rng(0)
    X = rng.random((12, 3))
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    rows = analyze.partition([f"ep-{i:02d}" for i in range(12)], D, D, 0.6, 0.3)
    fam_of_model = {}
    for _, f, m in rows:
        assert fam_of_model.setdefault(m, f) == f      # un label modèle -> une seule famille


def test_ref_like_synthetic_calibrated():
    g, truth = _grouped("ref", 20, 4)
    res = analyze.analyze(g, t_family=10.0, truth=truth, calibrate_on_truth=True)
    assert analyze.score(truth, res["rows"])["practice_mcc"] == 1.0


def test_constant_prefix_strip_helps():
    """Le préfixe fixe du « system prompt » synthétique est un piège : sans le retirer, le score baisse."""
    g, truth = _grouped("ref", 20, 4)
    with_strip = analyze.analyze(g, t_family=10.0, truth=truth, calibrate_on_truth=True)
    without = analyze.analyze(g, t_family=10.0, truth=truth, calibrate_on_truth=True, strip=False)
    s_with = analyze.score(truth, with_strip["rows"])["practice_mcc"]
    assert s_with > analyze.score(truth, without["rows"])["practice_mcc"]


def test_ep_like_synthetic_auto_thresholds_within_budget():
    g, truth = _grouped("ep", 25, 4, seed=1)
    g = analyze.subsample(g, max_prompts=15, max_reps=4)
    assert max(analyze.answers_per_endpoint(g).values()) <= 100
    res = analyze.analyze(g)
    s = analyze.score(truth, res["rows"])
    assert s["family_mcc"] == 1.0 and s["score"] > 0.9


def test_missing_answers_do_not_crash():
    g, _ = _grouped("ref", 6, 3)
    del g["m00"]["ref-04"]
    g["m01"]["ref-02"] = g["m01"]["ref-02"][:1]       # une seule réponse : prompt traité en « single »
    res = analyze.analyze(g, t_model=0.05, t_family=10.0)
    assert len(res["rows"]) == 9


def test_cli_writes_submission_csv(tmp_path):
    eps = synth.scenario("ref")
    ans = tmp_path / "a.jsonl"
    import json
    with ans.open("w") as f:
        for r in synth.generate(eps, 5, 3):
            f.write(json.dumps(r) + "\n")
    out = tmp_path / "out.csv"
    assert analyze.main([str(ans), "--prefix", "ref-", "--t-model", "0.05", "--out", str(out), "--quiet"]) == 0
    rows = list(csv.reader(out.open()))
    assert rows[0] == ["endpoint", "family", "model"] and len(rows) == 10


def _read_prompts(path):
    with path.open(newline="", encoding="utf-8") as f:
        return [(r["id"], r["prompt"]) for r in csv.DictReader(f)]     # même lecture que read_prompts du kit


def test_prompt_files_are_kit_compatible():
    """CSV id,prompt lisibles par le kit, ids uniques par fichier, prompts courts (1 024 tokens max en
    entrée : on reste sous 1 500 caractères), et chaque prompt ep-* figure à l'identique dans le pilote ref-*
    (la sélection faite sur les ref se transpose donc id pour id)."""
    pilot = {}
    for path in sorted(HERE.glob("prompts_ref_pilot*.csv")):
        rows = _read_prompts(path)
        assert len({p for p, _ in rows}) == len(rows), path.name
        for pid, prompt in rows:
            assert pid not in pilot, (path.name, pid)        # ids uniques entre vagues du pilote
            pilot[pid] = prompt
    ep_ids = set()
    for path in sorted(HERE.glob("prompts_ep_*.csv")):
        rows = _read_prompts(path)
        assert rows and len({p for p, _ in rows}) == len(rows), path.name
        for pid, prompt in rows:
            assert pilot.get(pid) == prompt, (path.name, pid)
            assert pid not in ep_ids, (path.name, pid)       # pas de collision entre fichiers ep (même jsonl)
            ep_ids.add(pid)
    assert all(0 < len(t) <= 1500 for t in pilot.values())
