"""Cartes denses conditionnées par le message, calculées avec les décodeurs officiels du kit.

Pour une image 512x512 (méthode et message connus), sortie sur une grille 32x32 (cellule = 16 px) :
  dense   : accord bit à bit des logits par position (PixelSeal per_position_logits 32x32,
            TrustMark layer4 8x8 ; StegaStamp : zéros, pas de tête par position)
  w128    : fenêtres 128 px pas 64 à l'échelle 512, reste gris -> 7x7, accord bit à bit
  w256    : fenêtres 256 px pas 64 -> 5x5
  cmax    : balayage de rectangles découpés et décodés seuls (redimensionnés à la résolution du
            décodeur) ; par cellule, max de l'accord des rectangles qui la contiennent
  cbest   : masque du meilleur rectangle (accord max), 32x32
+ scalaires acc_full (image entière), best_acc, best_box (y0,y1,x0,x1).
Accord = fraction de bits égaux au message (0.5 = hasard).

Usage (un processus par GPU, srun) :
  python work_seg/feats.py --split train --out output/seg_feats_train   (shard = SLURM_PROCID)
  python work_seg/feats.py --npz output/seg_synth_*.npz --out output/seg_feats_synth
"""
from __future__ import annotations

import argparse
import glob
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

G = 32  # grille de sortie
RES = {"pixelseal": (256, "bilinear"), "trustmark": (245, "bilinear"), "stegastamp": (400, "bicubic")}


def make_boxes(method: str) -> np.ndarray:
    """Rectangles candidats (y0, y1, x0, x1), ratio <= 4 (<= 2 pour TrustMark), pas = côté min / 2."""
    sides = [48, 64, 96, 128, 192, 256, 384, 512]
    rmax = 2.0 if method == "trustmark" else 4.0
    out = []
    for h in sides:
        for w in sides:
            if max(h, w) / min(h, w) > rmax:
                continue
            st = max(32, min(h, w) // 2)
            for y0 in range(0, 512 - h + 1, st):
                for x0 in range(0, 512 - w + 1, st):
                    out.append((y0, y0 + h, x0, x0 + w))
    return np.array(out, np.int32)


def win_masks(win: int, stride: int, device) -> torch.Tensor:
    pos = list(range(0, 512 - win + 1, stride))
    m = torch.zeros(len(pos) ** 2, 1, 512, 512, device=device)
    k = 0
    for y in pos:
        for x in pos:
            m[k, :, y:y + win, x:x + win] = 1; k += 1
    return m  # (n*n) 1 512 512


class Extractor:
    def __init__(self, method: str, device: str):
        self.method, self.device = method, device
        self.wm = load_watermark(method, KIT / "models", device)
        self.res, self.mode = RES[method]
        self.boxes = make_boxes(method)
        self.m128 = win_masks(128, 64, device)  # 49 1 512 512
        self.m256 = win_masks(256, 64, device)  # 25 1 512 512
        # cellule (32x32) couverte par chaque boîte : (nbox, 32*32) bool
        cov = np.zeros((len(self.boxes), G, G), bool)
        for i, (y0, y1, x0, x1) in enumerate(self.boxes):
            cov[i, y0 // 16:(y1 + 15) // 16, x0 // 16:(x1 + 15) // 16] = True
        self.cov = torch.from_numpy(cov.reshape(len(cov), -1)).to(device)

    def logits(self, x: torch.Tensor, bs: int = 128) -> torch.Tensor:
        """x : B 3 H W dans [0,1] -> B nbits (cpu)."""
        out = []
        for i in range(0, len(x), bs):
            out.append(self.wm.logits(x[i:i + bs]).float().cpu())
        return torch.cat(out)

    def prep(self, crop: torch.Tensor) -> torch.Tensor:
        """crop 1 3 h w -> 1 3 r r, même interpolation que le wrapper (le wrapper refait r->r = identité)."""
        return F.interpolate(crop, size=(self.res, self.res), mode=self.mode, align_corners=False, antialias=True)

    @torch.no_grad()
    def __call__(self, img: np.ndarray, msg: np.ndarray) -> dict:
        dev = self.device
        x = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255).to(dev)  # 1 3 512 512
        s = torch.from_numpy(msg.astype(np.float32))  # nbits, 0/1
        agree = lambda lg: ((lg > 0).float() == s).float().mean(-1)  # B nbits -> B

        acc_full = float(agree(self.logits(x))[0])
        # dense : logits par position
        if self.method in ("pixelseal", "trustmark"):
            pp = self.wm.features(x)["per_position_logits"].float().cpu()[0]  # nbits h w
            d = ((pp > 0).float() == s[:, None, None]).float().mean(0)  # h w
            dense = F.interpolate(d[None, None], size=(G, G), mode="bilinear", align_corners=False)[0, 0]
        else:
            dense = torch.zeros(G, G)
        # fenêtres à l'échelle 512, fond gris
        maps = {}
        for key, m in (("w128", self.m128), ("w256", self.m256)):
            xw = x * m + 0.5 * (1 - m)  # n 3 512 512
            a = agree(self.logits(xw, bs=25))  # n
            n = int(round(len(a) ** 0.5))
            maps[key] = F.interpolate(a.view(1, 1, n, n), size=(G, G), mode="bilinear", align_corners=False)[0, 0]
        # balayage de rectangles découpés
        crops = torch.cat([self.prep(x[:, :, y0:y1, x0:x1]) for (y0, y1, x0, x1) in self.boxes])  # nbox 3 r r
        a = agree(self.logits(crops, bs=128)).to(dev)  # nbox
        cm = torch.where(self.cov, a[:, None], torch.zeros_like(a[:, None])).max(0).values.view(G, G)
        b = int(a.argmax())
        y0, y1, x0, x1 = self.boxes[b]
        cb = torch.zeros(512, 512); cb[y0:y1, x0:x1] = 1
        cb = F.avg_pool2d(cb[None, None], 16)[0, 0]
        stack = torch.stack([dense, maps["w128"], maps["w256"], cm.cpu(), cb])  # 5 32 32
        return {"maps": stack.numpy().astype(np.float16), "acc_full": acc_full,
                "best_acc": float(a[b]), "best_box": self.boxes[b].copy()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"]); ap.add_argument("--npz", nargs="*")
    ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rank = int(os.environ.get("SLURM_PROCID", 0)); world = int(os.environ.get("SLURM_NTASKS", 1))
    lid = int(os.environ.get("SLURM_LOCALID", 0))
    dev = f"cuda:{lid if torch.cuda.device_count() > lid else 0}"  # srun peut ne montrer qu'un GPU par tâche

    if a.split:
        from PIL import Image
        meta = json.loads((KIT / "release" / a.split / "metadata.json").read_text())
        items = [(m["image_name"], m["method"], m["message"]) for m in meta]
        load = lambda n: np.asarray(Image.open(KIT / "release" / a.split / "images" / f"{n}.png").convert("RGB"))
        src = None
    else:
        files = sorted(sum([glob.glob(p) for p in a.npz], []))
        items, src = [], {}
        for f in files:
            z = np.load(f)
            for i, (n, me, ms) in enumerate(zip(z["names"].tolist(), z["method"].tolist(), z["message"].tolist())):
                items.append((n, me, ms)); src[n] = (f, i)
        cache = {}
        def load(n):
            f, i = src[n]
            if f not in cache:
                cache.clear(); cache[f] = np.load(f)["images"]
            return cache[f][i]
    items = items[rank::world]
    if a.limit:
        items = items[:a.limit]
    res = {k: [] for k in ("names", "method", "maps", "acc_full", "best_acc", "best_box")}
    t0 = time.time()
    for meth in ("pixelseal", "trustmark", "stegastamp"):
        sub = [it for it in items if it[1] == meth]
        if not sub:
            continue
        ex = Extractor(meth, dev)
        print(f"[{rank}] {meth}: {len(sub)} images, {len(ex.boxes)} boîtes", flush=True)
        for j, (n, me, ms) in enumerate(sub):
            r = ex(load(n), np.array([int(c) for c in ms], np.uint8))
            res["names"].append(n); res["method"].append(me)
            for k in ("maps", "acc_full", "best_acc", "best_box"):
                res[k].append(r[k])
            if j % 20 == 0:
                print(f"[{rank}] {meth} {j}/{len(sub)} acc_full={r['acc_full']:.2f} best={r['best_acc']:.2f} "
                      f"{time.time() - t0:.0f}s", flush=True)
        del ex; torch.cuda.empty_cache()
    out = f"{a.out}_{rank}.npz"
    np.savez(out, names=np.array(res["names"]), method=np.array(res["method"]), maps=np.stack(res["maps"]),
             acc_full=np.array(res["acc_full"], np.float32), best_acc=np.array(res["best_acc"], np.float32),
             best_box=np.stack(res["best_box"]))
    print(f"[{rank}] écrit {out} ({len(res['names'])} images) en {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
