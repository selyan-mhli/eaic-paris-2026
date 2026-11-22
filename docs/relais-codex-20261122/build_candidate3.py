"""Prépare localement le candidat SS 6k, sans appel serveur."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def average(paths, names):
    acc = np.zeros((1020, 128, 128), np.float32)
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            source_names = data["names"].tolist()
            assert len(source_names) == len(set(source_names)) == 1020, path
            assert set(source_names) == set(names), path
            probs = data["probs"].astype(np.float32)
            assert probs.shape == acc.shape and np.isfinite(probs).all(), path
            assert ((probs >= 0) & (probs <= 1)).all(), path
            positions = {name: i for i, name in enumerate(source_names)}
            acc += probs[[positions[name] for name in names]]
    return acc / len(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    champion = args.data_dir / "c1_ensE_ssall3.pt"
    expected = "f00fdc039718b29fabe64cf2155e6967b89b12c0986f331bd1d3ee7304418c80"
    assert digest(champion) == expected, "Le champion ne correspond pas au fichier confirmé."
    source = torch.load(champion, weights_only=True, map_location="cpu")
    names = list(source["image_names"])
    masks = source["masks"]
    assert masks.dtype == torch.bool and masks.shape == (1020, 128, 128)
    assert len(names) == len(set(names)) == 1020
    rows = json.loads(args.metadata.read_text())
    methods = {row["image_name"]: row["method"].lower() for row in rows}
    assert len(rows) == len(methods) == 1020 and set(names) == set(methods)
    ss = np.array(["stega" in methods[name] for name in names])
    assert ss.sum() == 340
    old_paths = [args.data_dir / f"pixseg_test_psall_fit_s{s}.npz" for s in [50, 51, 52]]
    new_paths = [args.data_dir / f"pixseg_test_psall6k_fit_s{s}.npz" for s in [60, 61, 62]]
    reproduced = torch.from_numpy(average(old_paths, names)[ss] > 0.5)
    assert torch.equal(reproduced, masks[ss]), "La recette SS du champion n'est pas reproduite."
    updated = masks.clone()
    updated[ss] = torch.from_numpy(average(new_paths, names)[ss] > 0.5)
    assert torch.equal(updated[~ss], masks[~ss])
    assert not args.out.exists(), "Refus d'écraser un candidat existant."
    torch.save({"image_names": names, "masks": updated}, args.out)
    check = torch.load(args.out, weights_only=True, map_location="cpu")
    assert check["image_names"] == names and torch.equal(check["masks"], updated)
    assert args.out.stat().st_size <= 25_000_000
    manifest = {"candidate": str(args.out), "sha256": digest(args.out), "bytes": args.out.stat().st_size,
                "champion": str(champion), "champion_sha256": expected, "champion_public_user_reported": 0.6739,
                "recipe": "StegaStamp uniquement : moyenne float32 des graines 60, 61 et 62 à 6000 pas, seuil strict > 0.5",
                "threshold": 0.5, "ss_images": 340, "other_images_preserved": 680,
                "champion_ss_reproduced_exactly": True,
                "pixels_changed_global": float((updated != masks).float().mean()),
                "pixels_changed_ss": float((updated[ss] != masks[ss]).float().mean()),
                "images_changed": int((updated != masks).flatten(1).any(1).sum()),
                "inputs": [{"path": str(path), "sha256": digest(path)} for path in old_paths + new_paths],
                "status": "PRÉPARÉ LOCALEMENT, NON SOUMIS ; validation seulement folds 0–1, une graine CV par configuration"}
    args.out.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "inputs"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
