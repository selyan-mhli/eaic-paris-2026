"""Messages du synthétique, retrouvés en rejouant le générateur seedé de data.py avec un embedder factice
(les tirages aléatoires ne dépendent pas de la sortie de l'embedder). Contrôle : les masques rejoués doivent être
identiques aux masques stockés dans les tranches, sinon arrêt.

  python work_pixseg_v3/msgs.py pixelseal        -> output/pixseg_synmsg_pixelseal_s0.npz (names, msgs uint8 n nbits)
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import KIT, METHODS, NBITS, synth_stream  # noqa: E402


def cache_path(method, seed=0, out="output/pixseg_synmsg"):
    return f"{out}_{method}_s{seed}.npz"


def replay(method, seed=0, pool="test", shards="output/pixseg_syn", shard=500):
    files = sorted(glob.glob(f"{shards}_{method}_s{seed}_*.npz")); assert files, method
    n = shard * len(files)
    rng = np.random.default_rng(1000 * seed + METHODS.index(method))
    root = KIT / "release"
    cands = [(sp, m["image_name"]) for sp in pool.split(",")
             for m in json.loads((root / sp / "metadata.json").read_text()) if m["method"] != method]
    blank = np.zeros((512, 512, 3), np.uint8)
    rec = []

    def embed(img, msg):
        rec.append(np.asarray(msg, np.uint8).copy()); return img
    masks, t0 = [], time.time()
    for j, s in enumerate(synth_stream(rng, embed, method, cands, lambda c, flip: blank)):
        if j >= n:
            break
        assert len(rec) == j + 1, "un appel à l'embedder par image attendu"
        masks.append(s["mask128"])
    masks = np.stack(masks); msgs = np.stack(rec)[:n]  # n 128 128, n nbits
    names = np.array([f"syn_{method}_s{seed}_{j // shard}_{j % shard}" for j in range(n)])
    for f in files:  # contrôle exact contre les tranches écrites
        z = np.load(f); pos = {nm: i for i, nm in enumerate(names.tolist())}
        ii = np.array([pos[nm] for nm in z["names"].tolist()])
        assert (masks[ii] == z["mask128"]).all(), f"rejeu désynchronisé : {f}"
    assert msgs.shape == (n, NBITS[method])
    print(f"{method} : {n} messages rejoués en {time.time() - t0:.0f}s, masques identiques sur {len(files)} tranches", flush=True)
    return names, msgs


def get(method, seed=0) -> dict:
    """nom -> message (uint8 nbits) ; calcule et met en cache au besoin (écriture atomique)."""
    p = cache_path(method, seed)
    if not os.path.exists(p):
        names, msgs = replay(method, seed)
        tmp = f"{p}.{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, names=names, msgs=msgs); os.replace(tmp, p)
    z = np.load(p)
    return dict(zip(z["names"].tolist(), z["msgs"]))


if __name__ == "__main__":
    for m in sys.argv[1:]:
        print(m, len(get(m)))
