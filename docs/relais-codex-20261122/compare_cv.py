"""Compare les sorties OOF existantes, sans modèle, API ou choix de seuil.

Les différences hors StegaStamp sont nulles : la règle commune s'annule.
Le seuil 0,5 et la pondération 1/2 sont ceux fixés avant les résultats.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, "/Users/selyan/dev/eaic-paris-2026/c1")
from metric import miou_per_image


def load_parts(paths, reference):
    names, folds = reference
    probs = np.zeros((len(names), 128, 128), np.float32)
    thresholds = np.full(len(names), np.nan, np.float32)
    provenance = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            assert np.array_equal(data["names"], names), f"Ordre des noms : {path}"
            assert np.array_equal(data["fold"], folds), f"Folds : {path}"
            assert data["probs"].shape == probs.shape, path
            covered = np.isfinite(data["thr"])
            assert not (covered & np.isfinite(thresholds)).any(), f"Doublon : {path}"
            values = data["probs"][covered].astype(np.float32)
            assert np.isfinite(values).all() and ((0 <= values) & (values <= 1)).all(), path
            probs[covered] = values
            thresholds[covered] = data["thr"][covered]
        provenance.append({"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()})
    return probs, thresholds, provenance


def bootstrap(values):
    rng = np.random.default_rng(0)
    means = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    for group in ("base", "seed1", "iou2"):
        parser.add_argument(f"--{group}", nargs="+", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.base[0], allow_pickle=False) as data:
        names, folds = data["names"], data["fold"]
    assert len(names) == 510 and len(set(names)) == 510
    with np.load(args.data / "release/train/masks_128.npz", allow_pickle=False) as data:
        mapping = dict(zip(data["names"].tolist(), data["masks"]))
        truth = np.stack([mapping[name] for name in names])
    with np.load(args.data / "output/heats_train.npz", allow_pickle=False) as data:
        assert np.array_equal(data["names"], names)
        stega = np.char.find(np.char.lower(data["method"]), "stega") >= 0
        full = data["full_watermark"].astype(bool)
    loaded = {key: load_parts(getattr(args, key), (names, folds)) for key in ("base", "seed1", "iou2")}
    covered = np.isfinite(loaded["base"][1])
    assert all(np.array_equal(covered, np.isfinite(item[1])) for item in loaded.values())
    if not args.partial:
        assert covered.all(), "Les cinq folds sont obligatoires pour le verdict."
    scores = {}
    for key, other in (("base_seed1", "seed1"), ("base_iou2", "iou2")):
        # Reproduit le stockage float16 du script original avant le seuil strict.
        avg = ((loaded["base"][0] + loaded[other][0]) / 2).astype(np.float16)
        scores[key] = miou_per_image(avg > 0.5, truth)
    diff = np.where(stega & covered, scores["base_iou2"] - scores["base_seed1"], 0)
    groups = {"global": covered, "SS_F": covered & stega & full, "SS_R": covered & stega & ~full}
    report = {"covered": int(covered.sum()), "folds": np.unique(folds[covered]).tolist(), "partial": args.partial,
              "comparison": "ensemble 4k+IoU2 contre 4k+4k ; ne compare pas au champion ssall3 à 0,6739",
              "threshold": 0.5, "weight_iou2": 0.5,
              "provenance": {key: item[2] for key, item in loaded.items()}, "groups": {}}
    for key, mask in groups.items():
        values = diff[mask]
        assert len(values)
        report["groups"][key] = {"n": int(mask.sum()), "gain": float(values.mean()), "ci95": bootstrap(values)}
    report["per_fold"] = {str(k): float(diff[folds == k].mean()) for k in np.unique(folds[covered])}
    gain = report["groups"]["global"]["gain"]
    lower = report["groups"]["global"]["ci95"][0]
    report["verdict"] = ("PARTIEL — aucun verdict cinq folds" if args.partial else
                         "POURSUIVRE" if gain >= 0.01 and lower > 0 else
                         "DIAGNOSTIC — décision Selyan" if 0.003 <= gain < 0.01 and lower > 0 else "ABANDON")
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "provenance"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
