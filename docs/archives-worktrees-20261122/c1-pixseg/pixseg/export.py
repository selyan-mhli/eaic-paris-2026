"""Fichier de soumission Task 1 à partir des probabilités test de train.py. Ne soumet rien.

  python work_pixseg/export.py --probs output/pixseg_test_<tag>.npz [autres.npz ...] --thr 0.5 \
      --out output/c1_pixseg_<date>.pt
Plusieurs fichiers = moyenne des probabilités (ensemble). Sortie : torch.save({"image_names": [N noms],
"masks": uint8 0/1 [N, 128, 128]}), dans l'ordre de release/test/metadata.json.
Contrôles : noms uniques et identiques aux métadonnées, forme, valeurs 0/1, taille < 20 Mo, relecture weights_only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

MAX_MB = 20.0


def write_submission(names, masks, out, expected=None) -> float:
    """names : liste de N noms ; masks : (N, 128, 128) 0/1. expected : noms attendus, dans l'ordre de sortie."""
    names = [str(n) for n in names]; masks = np.asarray(masks)
    assert len(names) == len(set(names)), "noms en double"
    assert masks.shape == (len(names), 128, 128), masks.shape
    assert set(np.unique(masks).tolist()) <= {0, 1}, "masques non binaires"
    if expected is not None:
        expected = [str(n) for n in expected]
        assert set(names) == set(expected) and len(names) == len(expected), "noms != métadonnées test"
        pos = {n: i for i, n in enumerate(names)}
        masks = masks[[pos[n] for n in expected]]; names = expected
    t = torch.from_numpy(np.ascontiguousarray(masks, dtype=np.uint8))  # N 128 128
    tmp = f"{out}.tmp"
    torch.save({"image_names": names, "masks": t}, tmp)
    mb = os.path.getsize(tmp) / 1e6
    if mb >= MAX_MB:
        os.remove(tmp); raise AssertionError(f"{mb:.1f} Mo >= {MAX_MB} Mo")
    back = torch.load(tmp, weights_only=True)
    assert back["image_names"] == names and back["masks"].dtype == torch.uint8 and torch.equal(back["masks"], t)
    os.replace(tmp, out)
    return mb


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", nargs="+", required=True); ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--meta", default="release/test/metadata.json"); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    zs = [np.load(f) for f in a.probs]
    names = zs[0]["names"].tolist()
    assert all(z["names"].tolist() == names for z in zs), "fichiers de probabilités dans des ordres différents"
    p = np.mean([z["probs"].astype(np.float32) for z in zs], 0)  # N 128 128
    assert np.isfinite(p).all()
    thr = float(zs[0]["thr"]) if a.thr is None else a.thr
    expected = [m["image_name"] for m in json.loads(Path(a.meta).read_text())]
    mb = write_submission(names, p > thr, a.out, expected)
    print(f"écrit {a.out} : {len(names)} masques, seuil {thr}, couverture {(p > thr).mean():.3f}, "
          f"images tout 0 : {int((~(p > thr).any((1, 2))).sum())}, {mb:.1f} Mo (NON soumis)")


if __name__ == "__main__":
    main()
