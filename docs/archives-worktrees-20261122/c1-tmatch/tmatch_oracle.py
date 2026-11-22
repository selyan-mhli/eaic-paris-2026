"""Oracle du template matching : z de la boîte GT (bbox du masque) vs boîtes témoins décalées, images R train.
    python work_c1_tmatch/tmatch_oracle.py --n 20 [--k 30]
Si le z de la GT n'est pas nettement au-dessus des témoins, la méthode ne marche pas (pas la recherche)."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path.cwd()))
from template_match import H, Residuals, score_boxes, zscore  # noqa: E402
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import to_tensor  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20); ap.add_argument("--k", type=int, default=30)
ap.add_argument("--sigma", type=float, default=1.5); a = ap.parse_args()
ds = LocalisationDataset("release", "train")
rng = np.random.default_rng(0)
for m in ("trustmark", "stegastamp"):
    wm = load_watermark(m, "models", "cuda"); res = Residuals(wm, m)
    zg, zs, acg, n = [], [], [], 0
    for i in range(len(ds)):
        if ds.meta[i]["method"] != m or ds.meta[i].get("full_watermark", True):
            continue
        it = ds[i]; ys, xs = np.nonzero(it["mask"])
        if not len(ys):
            continue
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        if min(y1 - y0, x1 - x0) < 16:
            continue
        img = to_tensor(it["image"]).cuda(); msg = torch.from_numpy(it["message"]).float()
        h, w = y1 - y0, x1 - x0
        sh = []  # témoins : même taille, décalée d'au moins la moitié de la boîte
        for _ in range(4):
            yy = int(rng.integers(0, H - h + 1)); xx = int(rng.integers(0, H - w + 1))
            sh.append((yy, yy + h, xx, xx + w))
        boxes = [(y0, y1, x0, x1)] + sh
        rand = torch.from_numpy(rng.integers(0, 2, (a.k, len(msg)))).float()
        cm, cr = score_boxes(res, img, msg, rand, boxes, a.sigma, chunk=8)
        z = zscore(cm, cr)
        ac = float(((wm.logits(img[..., y0:y1, x0:x1]).cpu() > 0).float() == msg[None]).float().mean())
        zg.append(z[0]); zs.extend(z[1:].tolist()); acg.append(ac)
        print(f"{m} {it['image_name']} gt {h}x{w} z_gt {z[0]:.2f} z_témoins {np.round(z[1:], 2).tolist()} acc_crop_gt {ac:.2f}", flush=True)
        n += 1
        if n >= a.n:
            break
    zg, zs, acg = np.array(zg), np.array(zs), np.array(acg)
    print(f"== {m} n={n} z_gt médiane {np.median(zg):.2f} (moy {zg.mean():.2f}) | témoins médiane {np.median(zs):.2f} "
          f"q90 {np.quantile(zs, .9):.2f} | z_gt > q90 témoins : {int((zg > np.quantile(zs, .9)).sum())}/{n} | "
          f"acc_crop GT >= 0,75 : {int((acg >= .75).sum())}/{n} ; corr(z_gt, acc) {np.corrcoef(zg, acg)[0, 1]:.2f}", flush=True)
