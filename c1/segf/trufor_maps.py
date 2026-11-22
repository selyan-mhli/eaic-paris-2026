"""MESURE SEULEMENT : cartes TruFor (anomalie + confiance) sur le train. Modèle externe : règle des mentors non
tranchée, donc aucun fichier de soumission ne doit dépendre de ces cartes sans go explicite de Selyan.

Code : github.com/grip-unina/TruFor (test_docker/src), cloné hors dépôt dans SLZ/ext/trufor/code ; poids
SLZ/ext/trufor/weights/trufor.pth.tar (chargés avec weights_only=True). Config de trufor.yaml reproduite sans
yacs (absent de la .venv). Prétraitement identique à data_core.py : RGB / 256.

Sortie npz : names, method, tf_map (P(manipulé), 128x128), tf_conf (confiance, 128x128), tf_score (image).
  srun python work_segf/trufor_maps.py --split train --out output/segf_trufor_train
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
import torch.nn.functional as F
from PIL import Image

SLZ = Path("/p/scratch/training2666/mouhali1/SLZ")
KIT = SLZ / "Hackathon_Watermark_Localisation_Task"
SRC = SLZ / "ext" / "trufor" / "code" / "test_docker" / "src"
WEIGHTS = SLZ / "ext" / "trufor" / "weights" / "trufor.pth.tar"


class AD(dict):
    """dict à accès par attribut (remplace yacs.CfgNode ; `in` reste celui de dict, comme CfgNode)."""
    __getattr__ = dict.__getitem__


CFG = AD(MODEL=AD(NAME="detconfcmx", MODS=("RGB", "NP++"), PRETRAINED="",
                  EXTRA=AD(BACKBONE="mit_b2", DECODER="MLPDecoder", DECODER_EMBED_DIM=512, PREPRC="imagenet",
                           BN_EPS=0.001, BN_MOMENTUM=0.1, DETECTION="confpool", CONF=True)),
         DATASET=AD(NUM_CLASSES=2))


def load_model(dev):
    sys.path.insert(0, str(SRC))
    from models.cmx.builder_np_conf import myEncoderDecoder
    # weights_only=True gardé ; le checkpoint contient des scalaires numpy (métadonnées) : liste blanche minimale
    try:
        from numpy.core.multiarray import scalar as scal  # numpy 1.x (JURECA : 1.26)
    except ImportError:
        from numpy._core.multiarray import scalar as scal
    native = f"{scal.__module__}.{scal.__name__}" == "numpy.core.multiarray.scalar"
    allow = [scal if native else (scal, "numpy.core.multiarray.scalar"), np.dtype,
             type(np.dtype(np.float64)), type(np.dtype(np.int64))]
    with torch.serialization.safe_globals(allow):
        ck = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    model = myEncoderDecoder(cfg=CFG)
    model.load_state_dict(ck["state_dict"])
    return model.to(dev).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train"); ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--only-f", action="store_true")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    rank = int(os.environ.get("SLURM_PROCID", 0)); world = int(os.environ.get("SLURM_NTASKS", 1))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.loads((KIT / "release" / a.split / "metadata.json").read_text())
    if a.only_f:
        meta = [m for m in meta if m.get("full_watermark")]
    meta = meta[rank::world][: a.limit or None]
    model = load_model(dev)
    out = {"names": [], "method": [], "tf_map": [], "tf_conf": [], "tf_score": []}
    t0 = time.time()
    for j, m in enumerate(meta):
        img = np.array(Image.open(KIT / "release" / a.split / "images" / f"{m['image_name']}.png").convert("RGB"))
        x = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float)[None].div(256.0).to(dev)
        pred, conf, det, _ = model(x)
        pmap = F.softmax(pred, dim=1)[:, 1:2]  # 1 1 512 512
        cmap = torch.sigmoid(conf)  # 1 1 512 512
        out["tf_map"].append(F.avg_pool2d(pmap, 4)[0, 0].cpu().numpy().astype(np.float16))
        out["tf_conf"].append(F.avg_pool2d(cmap, 4)[0, 0].cpu().numpy().astype(np.float16))
        out["tf_score"].append(float(torch.sigmoid(det).item()))
        out["names"].append(m["image_name"]); out["method"].append(m["method"])
        if j < 2 or j % 50 == 0:
            print(f"[{rank}] {j}/{len(meta)} {m['method']} score {out['tf_score'][-1]:.3f} map moy {float(pmap.mean()):.3f} "
                  f"{time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(f"{a.out}_{rank}.npz", **{k: np.array(v) if k in ("names", "method", "tf_score") else np.stack(v)
                                                    for k, v in out.items()})
    print(f"[{rank}] écrit {a.out}_{rank}.npz ({len(meta)}) {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
