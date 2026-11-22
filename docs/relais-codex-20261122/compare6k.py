"""Mesure appariée 3k/6k à seuil 0,5, sur les OOF existantes des folds 0–1."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, "/Users/selyan/dev/eaic-paris-2026/c1")
from metric import miou_per_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.data / "output/heats_train.npz") as data:
        names, methods, full = data["names"], data["method"], data["full_watermark"]
    with np.load(args.data / "release/train/masks_128.npz") as data:
        masks = dict(zip(data["names"].tolist(), data["masks"]))
        truth = np.stack([masks[name] for name in names])
    with np.load(args.pred_dir / "pixseg_cv_psyn_lr1_all.npz") as data:
        assert np.array_equal(data["names"], names)
        old_probs, folds, covered = data["probs"], data["fold"], np.isfinite(data["thr"])
    new_probs = np.zeros_like(old_probs, dtype=np.float32)
    new_covered = np.zeros(len(names), bool)
    for k in [0, 1]:
        with np.load(args.pred_dir / f"pixseg_cv_psall6k_f{k}.npz") as data:
            assert np.array_equal(data["names"], names) and np.array_equal(data["fold"], folds)
            mask = np.isfinite(data["thr"])
            assert not (mask & new_covered).any()
            new_covered |= mask
            new_probs[mask] = data["probs"][mask]
    assert np.array_equal(new_covered, covered) and covered.sum() == 204
    old = miou_per_image(old_probs > 0.5, truth)
    new = miou_per_image(new_probs > 0.5, truth)
    report = {"coverage": int(covered.sum()), "folds": [0, 1],
              "baseline": "psyn_lr1_all, 3000 pas, une graine CV ; champion test = moyenne de 3 graines",
              "candidate": "psall6k, 6000 pas, une graine CV ; 2 graines test disponibles",
              "fixed_threshold": 0.5, "groups": {}}
    for method in np.unique(methods):
        selected = covered & (methods == method)
        diff = np.where(methods == method, new - old, 0)[covered]
        rng = np.random.default_rng(0)
        means = diff[rng.integers(0, len(diff), (2000, len(diff)))].mean(1)
        row = {"n": int(selected.sum()), "old": float(old[selected].mean()), "new": float(new[selected].mean()),
               "gain_method": float((new - old)[selected].mean()),
               "gain_global_if_only_this_method_changes": float(diff.mean()),
               "ci95_global": np.percentile(means, [2.5, 97.5]).tolist(),
               "per_fold": {str(k): float((new - old)[selected & (folds == k)].mean()) for k in [0, 1]}}
        for flag in [True, False]:
            mask = selected & (full == flag)
            row["F" if flag else "R"] = {"old": float(old[mask].mean()), "new": float(new[mask].mean()),
                                          "gain": float((new - old)[mask].mean())}
        report["groups"][method] = row
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
