"""Synthétique F uniquement (image entière watermarkée par l'embedder officiel, puis 1-4 formes irrégulières
restaurées au propre), + cartes de feats_f.py dans le même processus. Pas de rectangle conservé.

Pool de sources : images d'une AUTRE méthode, par défaut du TEST seulement (aucune image train dans le
synthétique : l'évaluation sur les 255 F du train reste propre).

  srun python work_seg/gen_f.py --n 300 --seed 1 --out output/seg_gf
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feats_f import KIT, FExtractor  # noqa: E402
from gen import blob  # noqa: E402
from wmloc.utils import downsample_mask  # noqa: E402

METHODS = ("pixelseal", "trustmark", "stegastamp")
NBITS = {"pixelseal": 256, "trustmark": 100, "stegastamp": 100}


def holes(rng) -> np.ndarray:
    """1-4 formes (aire log-uniforme 0,3 %-20 %, centre n'importe où, donc parfois au bord), parfois rondes
    et compactes (objet), parfois allongées ; union -> bool 512 512."""
    h = np.zeros((512, 512), bool)
    for _ in range(rng.integers(1, 5)):
        h |= blob(rng, 512, 512, np.exp(rng.uniform(np.log(0.003), np.log(0.2))), (0.0, 1.0))
    if h.mean() > 0.5:  # les vrais F gardent > 50 % watermarké
        h = blob(rng, 512, 512, 0.1, (0.2, 0.8))
    return h


def save(res, out, t0):
    if not res["names"]:
        return
    np.savez(out, names=np.array(res["names"]), method=np.array(res["method"]), src=np.array(res["src"]),
             src2=np.array(res["src2"]),
             mask128=np.stack(res["mask128"]), fmaps=np.stack(res["fmaps"]), m0=np.array(res["m0"], np.float32),
             acc_full=np.array(res["acc_full"], np.float32))
    print(f"écrit {out} ({len(res['names'])}) en {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--pool", default="test")
    ap.add_argument("--allow-train", action="store_true", help="autorise des images train dans le pool (éval non propre)")
    a = ap.parse_args()
    rank = int(os.environ.get("SLURM_PROCID", 0))
    lid = int(os.environ.get("SLURM_LOCALID", 0))
    dev = f"cuda:{lid if torch.cuda.device_count() > lid else 0}"
    rng = np.random.default_rng(1000 * a.seed + rank)
    if "train" in a.pool.split(",") and not a.allow_train:
        raise SystemExit("pool contenant le train : l'éval sur les F du train ne serait plus propre (--allow-train)")
    pool = [(sp, m["image_name"], m["method"]) for sp in a.pool.split(",")
            for m in json.loads((KIT / "release" / sp / "metadata.json").read_text())]

    def load(it):
        im = Image.open(KIT / "release" / it[0] / "images" / f"{it[1]}.png").convert("RGB")
        if rng.random() < 0.5:
            im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return np.asarray(im)

    res = {k: [] for k in ("names", "method", "src", "src2", "mask128", "fmaps", "m0", "acc_full")}
    train_names = {m["image_name"] for m in json.loads((KIT / "release" / "train" / "metadata.json").read_text())}
    assert a.allow_train or not ({p[1] for p in pool} & train_names), "image train dans le pool"
    t0 = time.time()
    per = [a.n // 3 + (1 if i < a.n % 3 else 0) for i in range(3)]
    for meth, nm in zip(METHODS, per):
        ex = FExtractor(meth, dev)
        cands = [p for p in pool if p[2] != meth]
        for j in range(nm):
            c = cands[rng.integers(len(cands))]; o = cands[rng.integers(len(cands))]
            clean, other = load(c), load(o)
            msg = rng.integers(0, 2, NBITS[meth]).astype(np.uint8)
            wimg = ex.wm.embed(clean, msg)
            hole = holes(rng)
            u = rng.random()
            src = clean if u < 0.8 else (other if u < 0.9 else
                                         ndimage.gaussian_filter(clean.astype(np.float32), (4, 4, 0)).astype(np.uint8))
            img = np.where(hole[..., None], src, wimg).astype(np.uint8)  # 512 512 3
            r = ex(img, msg)
            res["names"].append(f"gf{a.seed}_{rank}_{meth}_{j}"); res["method"].append(meth); res["src"].append(c[1])
            res["src2"].append(o[1])  # 2e source (contenu de remplissage d'une partie des trous)
            res["mask128"].append(downsample_mask((~hole).astype(np.uint8)))
            for k in ("fmaps", "m0", "acc_full"):
                res[k].append(r[k])
            if j % 25 == 0:
                print(f"[{rank}] {meth} {j}/{nm} cov={res['mask128'][-1].mean():.2f} acc={r['acc_full']:.2f} "
                      f"{time.time() - t0:.0f}s", flush=True)
        del ex; torch.cuda.empty_cache()
        save(res, f"{a.out}_{a.seed}_{rank}_{meth}.npz", t0)  # un fichier par méthode : rien perdu si arrêt
        res = {k: [] for k in res}


if __name__ == "__main__":
    main()
