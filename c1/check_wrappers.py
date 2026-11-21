"""Contrôle des wrappers officiels (CPU, 2 images train par méthode) :
(logits > 0) == decode(), longueur des bits == len(message) == nbits, appel seul == appel batché,
acc_full du npz == recalcul.
    python work_c1/check_wrappers.py [--heats output/heats_train.npz]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import bit_accuracy, to_tensor  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--heats", default="output/heats_train.npz"); a = ap.parse_args()
torch.set_num_threads(8)
ds = LocalisationDataset("release", "train")
h = np.load(a.heats); hf = dict(zip(h["names"].tolist(), h["acc_full"]))
for m in ("pixelseal", "trustmark", "stegastamp"):
    wm = load_watermark(m, "models", "cpu")
    items = [ds[i] for i in range(len(ds)) if ds.meta[i]["method"] == m][:2]
    x = torch.cat([to_tensor(it["image"]) for it in items])  # 2 3 512 512
    lb = torch.as_tensor(wm.logits(x)).float()
    ls = torch.cat([torch.as_tensor(wm.logits(x[i:i + 1])).float() for i in range(2)])
    dec = torch.as_tensor(wm.decode(x))
    assert lb.shape == (2, wm.nbits) == dec.shape and all(len(it["message"]) == wm.nbits for it in items)
    assert torch.equal((lb > 0).int(), dec.int())
    print(m, "nbits", wm.nbits, "max|batch-seul|", float((lb - ls).abs().max()),
          "signes identiques", bool(torch.equal(lb > 0, ls > 0)),
          "acc", [round(bit_accuracy(dec[i].numpy(), it["message"]), 3) for i, it in enumerate(items)],
          "npz", [round(float(hf[it["image_name"]]), 3) for it in items])
