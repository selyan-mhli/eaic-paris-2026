"""Cartes pour les images F (image entière watermarkée, petites zones propres à trouver), marges signées.

Marge M(x) = moyenne_k (2 m_k - 1) * logit_k(x) (message connu). M0 = marge de l'image entière.
Canaux 32x32 (cellule 16 px), normalisés par |M0| :
  0 dense    : marge par position (PixelSeal per_position_logits 32x32, TrustMark layer4 8x8 ; StegaStamp 0)
  1 occ64    : chute de marge quand une fenêtre 64 px (pas 32) est grisée, moyenne des fenêtres couvrant la cellule
  2 occ128   : idem fenêtre 128 px pas 32
  3 keep128  : marge quand seule une fenêtre 128 px (pas 64) est gardée (reste gris)
  4 keep256  : idem 256 px pas 64
  5 det      : canal 0 (détection) des logits par position de PixelSeal, brut (0 pour les autres méthodes)
  6 lumstd   : écart-type local de la luminance par cellule 16 px (zones plates = watermark atténué)
Scalaires : M0 (brut), acc_full.

  srun python work_seg/feats_f.py --split train --out output/seg_ff_train
  (synthétique : appelé par gen.py --ffeats)
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

KIT = Path("/p/scratch/training2666/mouhali1/SLZ/Hackathon_Watermark_Localisation_Task")
sys.path.insert(0, str(KIT))
from wmloc.watermarks import load_watermark  # noqa: E402

G = 32


def windows(win, stride):
    pos = list(range(0, 512 - win + 1, stride))
    return [(y, x) for y in pos for x in pos]


class FExtractor:
    SPECS = (("occ64", 64, 32, "occ"), ("occ128", 128, 32, "occ"), ("keep128", 128, 64, "keep"),
             ("keep256", 256, 64, "keep"))

    def __init__(self, method, device, wm=None):
        self.method, self.device = method, device
        self.wm = wm or load_watermark(method, KIT / "models", device)
        self.win = {k: windows(w, s) for k, w, s, _ in self.SPECS}
        self.cov = {}
        for k, w, s, _ in self.SPECS:
            c = np.zeros((len(self.win[k]), G, G), np.float32)
            for i, (y, x) in enumerate(self.win[k]):
                c[i, y // 16:(y + w) // 16, x // 16:(x + w) // 16] = 1
            self.cov[k] = torch.from_numpy(c.reshape(len(c), -1))  # nwin (32*32)

    def margin(self, xs, s):
        """xs : B 3 512 512 (device) -> B marges (cpu)."""
        return (self.wm.logits(xs).float().cpu() * s).mean(-1)

    @torch.no_grad()
    def __call__(self, img: np.ndarray, msg: np.ndarray) -> dict:
        dev = self.device
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float().div(255).to(dev)  # 1 3 512 512
        s = torch.from_numpy(msg.astype(np.float32)) * 2 - 1  # nbits, +-1
        l0 = self.wm.logits(x).float().cpu()[0]  # nbits
        m0 = float((l0 * s).mean()); acc = float(((l0 > 0).float() == (s > 0).float()).float().mean())
        nrm = max(abs(m0), 1e-3)
        chans = []
        det = torch.zeros(G, G)
        if self.method in ("pixelseal", "trustmark"):
            ft = self.wm.features(x)
            pp = ft["per_position_logits"].float().cpu()[0]  # nbits h w
            if self.method == "pixelseal":
                lin = self.wm.det.pixel_decoder.linear
                d0 = torch.einsum("bchw,c->bhw", ft["embedding"], lin.weight[0]) + lin.bias[0]  # 1 h w, canal détection
                det = F.interpolate(d0[None].float().cpu(), size=(G, G), mode="bilinear", align_corners=False)[0, 0]
            d = (pp * s[:, None, None]).mean(0)  # h w
            chans.append(F.interpolate(d[None, None], size=(G, G), mode="bilinear", align_corners=False)[0, 0] / nrm)
        else:
            chans.append(torch.zeros(G, G))
        for k, w, st, kind in self.SPECS:
            ms = []
            for i in range(0, len(self.win[k]), 32):
                batch = x.repeat(len(self.win[k][i:i + 32]), 1, 1, 1)  # b 3 512 512
                for j, (y, xx) in enumerate(self.win[k][i:i + 32]):
                    if kind == "occ":
                        batch[j, :, y:y + w, xx:xx + w] = 0.5
                    else:
                        keep = batch[j, :, y:y + w, xx:xx + w].clone(); batch[j] = 0.5
                        batch[j, :, y:y + w, xx:xx + w] = keep
                ms.append(self.margin(batch, s))
            mw = torch.cat(ms)  # nwin
            val = (m0 - mw) / nrm if kind == "occ" else mw / nrm
            cv = self.cov[k]
            chans.append(((cv * val[:, None]).sum(0) / cv.sum(0).clamp(min=1)).view(G, G))
        chans.append(det)
        yl = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])[:, None]  # 1 1 512 512
        mu = F.avg_pool2d(yl, 16); var = (F.avg_pool2d(yl ** 2, 16) - mu ** 2).clamp(min=0)  # 1 1 32 32
        chans.append(var.sqrt()[0, 0].cpu())
        return {"fmaps": torch.stack(chans).numpy().astype(np.float16), "m0": m0, "acc_full": acc}  # 7 32 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rank = int(os.environ.get("SLURM_PROCID", 0)); world = int(os.environ.get("SLURM_NTASKS", 1))
    lid = int(os.environ.get("SLURM_LOCALID", 0))
    dev = f"cuda:{lid if torch.cuda.device_count() > lid else 0}"
    from PIL import Image
    meta = json.loads((KIT / "release" / a.split / "metadata.json").read_text())[rank::world]
    if a.limit:
        meta = meta[:a.limit]
    res = {k: [] for k in ("names", "method", "fmaps", "m0", "acc_full")}
    t0 = time.time()
    for meth in ("pixelseal", "trustmark", "stegastamp"):
        sub = [m for m in meta if m["method"] == meth]
        if not sub:
            continue
        ex = FExtractor(meth, dev)
        for j, m in enumerate(sub):
            img = np.asarray(Image.open(KIT / "release" / a.split / "images" / f"{m['image_name']}.png").convert("RGB"))
            r = ex(img, np.array([int(c) for c in m["message"]], np.uint8))
            res["names"].append(m["image_name"]); res["method"].append(meth)
            for k in ("fmaps", "m0", "acc_full"):
                res[k].append(r[k])
            if j % 20 == 0:
                print(f"[{rank}] {meth} {j}/{len(sub)} acc={r['acc_full']:.2f} m0={r['m0']:.2f} {time.time() - t0:.0f}s", flush=True)
        del ex; torch.cuda.empty_cache()
    out = f"{a.out}_{rank}.npz"
    np.savez(out, names=np.array(res["names"]), method=np.array(res["method"]), fmaps=np.stack(res["fmaps"]),
             m0=np.array(res["m0"], np.float32), acc_full=np.array(res["acc_full"], np.float32))
    print(f"[{rank}] écrit {out} ({len(res['names'])}) en {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
