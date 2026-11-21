"""Données synthétiques avec les EMBEDDERS OFFICIELS du kit, puis cartes de feats.py (même processus).

Sources : images train + test d'une AUTRE méthode que celle qu'on pose (leur watermark est invisible
pour le décodeur visé ; contenu naturel 512). Par exemple :
  régime F (moitié) : image entière watermarkée, puis zones restaurées au propre
      outpainting -> seul un rectangle intérieur reste watermarké ; removal/addition -> 1-3 formes
      irrégulières restaurées (pixels d'origine, ou contenu d'une autre source, ou original flouté).
  régime R : rectangle (ratio <= 16 PixelSeal/StegaStamp, <= 2 TrustMark, 1-45 % de l'image),
      watermark posé sur le rectangle découpé comme image autonome (70 %) ou image entière puis
      rectangle gardé (30 %), recollé dans l'image propre ; puis removal/addition (trou dans le
      rectangle) ou transplant (seule une forme irrégulière reste watermarkée) ou rien.
Masque 512 exact -> grille 128 par la règle officielle (downsample_mask).

  srun python work_seg/gen.py --n 750 --out output/seg_synth     (n par processus ; shard = SLURM_PROCID)
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
from feats import KIT, Extractor  # noqa: E402
from wmloc.utils import downsample_mask  # noqa: E402

METHODS = ("pixelseal", "trustmark", "stegastamp")
NBITS = {"pixelseal": 256, "trustmark": 100, "stegastamp": 100}


def blob(rng, h, w, frac, crange=(0.25, 0.75)) -> np.ndarray:
    """Forme irrégulière (bool h w) d'aire ~ frac * h * w : bruit lissé seuillé autour d'une ellipse."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = rng.uniform(*crange) * h, rng.uniform(*crange) * w
    ry, rx = np.sqrt(frac) * h * rng.uniform(0.4, 0.8), np.sqrt(frac) * w * rng.uniform(0.4, 0.8)
    ang = rng.uniform(0, np.pi)
    dy, dx = yy - cy, xx - cx
    u = (dx * np.cos(ang) + dy * np.sin(ang)) / rx; v = (-dx * np.sin(ang) + dy * np.cos(ang)) / ry
    noise = ndimage.gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), sigma=max(h, w) / 12)
    noise /= noise.std() + 1e-6
    f = 1 - np.sqrt(u ** 2 + v ** 2) + 0.35 * noise
    return f > 0


def rect(rng, method, amin=0.01, amax=0.45, smin=16):
    rmax = 2.0 if method == "trustmark" else 16.0
    for _ in range(100):
        area = np.exp(rng.uniform(np.log(amin), np.log(amax))) * 512 * 512
        r = np.exp(rng.uniform(-np.log(rmax), np.log(rmax)))  # w / h
        h = int(round(np.sqrt(area / r))); w = int(round(np.sqrt(area * r)))
        if smin <= h <= 512 and smin <= w <= 512:
            y0 = rng.integers(0, 512 - h + 1); x0 = rng.integers(0, 512 - w + 1)
            return y0, y0 + h, x0, x0 + w
    return 128, 384, 128, 384


def fill(rng, clean, other, region):
    """Contenu non watermarké pour une zone : original propre, autre source, ou original flouté."""
    u = rng.random()
    if u < 0.5:
        src = clean
    elif u < 0.75:
        src = other
    else:
        src = ndimage.gaussian_filter(clean.astype(np.float32), sigma=(4, 4, 0)).astype(np.uint8)
    return np.where(region[..., None], src, 0)


def make(rng, wm, method, clean, other):
    nb = NBITS[method]
    msg = rng.integers(0, 2, nb).astype(np.uint8)
    full = rng.random() < 0.5
    if full:
        wimg = wm.embed(clean, msg)
        mask = np.ones((512, 512), bool)
        edit = rng.choice(["outpainting", "removal", "addition"])
        if edit == "outpainting":
            cov = rng.uniform(0.55, 0.98)
            r = rng.uniform(0.6, 1.6)
            h = int(min(512, np.sqrt(cov * 512 * 512 / r))); w = int(min(512, cov * 512 * 512 / h))
            y0 = rng.integers(0, 512 - h + 1); x0 = rng.integers(0, 512 - w + 1)
            keep = np.zeros((512, 512), bool); keep[y0:y0 + h, x0:x0 + w] = True
            hole = ~keep
        else:
            hole = np.zeros((512, 512), bool)
            for _ in range(rng.integers(1, 3)):
                hole |= blob(rng, 512, 512, rng.uniform(0.01, 0.15))
        mask &= ~hole
        out = np.where(hole[..., None], fill(rng, clean, other, hole), wimg)
    else:
        y0, y1, x0, x1 = rect(rng, method)
        if rng.random() < 0.7:
            wp = wm.embed(np.ascontiguousarray(clean[y0:y1, x0:x1]), msg)  # rectangle autonome
        else:
            wp = wm.embed(clean, msg)[y0:y1, x0:x1]
        out = clean.copy(); out[y0:y1, x0:x1] = wp
        mask = np.zeros((512, 512), bool); mask[y0:y1, x0:x1] = True
        edit = rng.choice(["outpainting", "removal", "addition", "transplant"])
        h, w = y1 - y0, x1 - x0
        if edit in ("removal", "addition") and min(h, w) >= 24:
            hole = np.zeros((512, 512), bool)
            hole[y0:y1, x0:x1] = blob(rng, h, w, rng.uniform(0.05, 0.4))
            mask &= ~hole
            out = np.where(hole[..., None], fill(rng, clean, other, hole), out)
        elif edit == "transplant" and min(h, w) >= 24:
            obj = np.zeros((512, 512), bool)
            obj[y0:y1, x0:x1] = blob(rng, h, w, rng.uniform(0.3, 0.8))
            rest = mask & ~obj
            mask = obj
            out = np.where(rest[..., None], clean, out)
    if not mask.any():  # garde au moins quelques pixels
        pass
    return out.astype(np.uint8), msg, bool(full), str(edit), downsample_mask(mask.astype(np.uint8))


def make_v2(rng, wm, method, clean, other):
    """Régimes mesurés sur le train réel (analyse Codex) : F = image entière watermarkée moins 1-4 formes
    irrégulières restaurées (jamais de rectangle conservé) ; R = moitié rectangle pur (outpainting),
    moitié objet irrégulier transplanté dans une autre image propre (peut toucher le bord)."""
    nb = NBITS[method]
    msg = rng.integers(0, 2, nb).astype(np.uint8)
    full = rng.random() < 0.5
    if full:
        wimg = wm.embed(clean, msg)
        hole = np.zeros((512, 512), bool)
        for _ in range(rng.integers(1, 5)):
            hole |= blob(rng, 512, 512, np.exp(rng.uniform(np.log(0.003), np.log(0.2))), (0.0, 1.0))
        u = rng.random()
        src = clean if u < 0.8 else (other if u < 0.9 else
                                     ndimage.gaussian_filter(clean.astype(np.float32), (4, 4, 0)).astype(np.uint8))
        out = np.where(hole[..., None], src, wimg)
        return out.astype(np.uint8), msg, True, "hole", downsample_mask((~hole).astype(np.uint8))
    if rng.random() < 0.5:
        y0, y1, x0, x1 = rect(rng, method, 0.003, 0.4, 8)
        wp = (wm.embed(np.ascontiguousarray(clean[y0:y1, x0:x1]), msg) if rng.random() < 0.85
              else wm.embed(clean, msg)[y0:y1, x0:x1])
        out = clean.copy(); out[y0:y1, x0:x1] = wp
        mask = np.zeros((512, 512), bool); mask[y0:y1, x0:x1] = True
        return out, msg, False, "rect", downsample_mask(mask.astype(np.uint8))
    obj = blob(rng, 512, 512, np.exp(rng.uniform(np.log(0.005), np.log(0.4))), (0.05, 0.95))
    if not obj.any():
        obj[200:260, 200:260] = True
    ys, xs = np.where(obj)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if rng.random() < 0.5 and min(y1 - y0, x1 - x0) >= 8:
        wsrc = clean.copy(); wsrc[y0:y1, x0:x1] = wm.embed(np.ascontiguousarray(clean[y0:y1, x0:x1]), msg)
    else:
        wsrc = wm.embed(clean, msg)
    out = np.where(obj[..., None], wsrc, other)  # objet watermarké collé dans une autre image propre
    return out.astype(np.uint8), msg, False, "transplant", downsample_mask(obj.astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=750); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--save-images", action="store_true")
    ap.add_argument("--v2", action="store_true"); ap.add_argument("--pool", default="train,test")
    a = ap.parse_args()
    rank = int(os.environ.get("SLURM_PROCID", 0))
    lid = int(os.environ.get("SLURM_LOCALID", 0))
    dev = f"cuda:{lid if torch.cuda.device_count() > lid else 0}"
    rng = np.random.default_rng(1000 * a.seed + rank)
    pool = []
    for split in a.pool.split(","):
        for m in json.loads((KIT / "release" / split / "metadata.json").read_text()):
            pool.append((split, m["image_name"], m["method"]))
    def load(it):
        im = Image.open(KIT / "release" / it[0] / "images" / f"{it[1]}.png").convert("RGB")
        if rng.random() < 0.5:
            im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return np.asarray(im)
    res = {k: [] for k in ("names", "method", "message", "full", "edit", "mask128", "maps", "acc_full",
                           "best_acc", "best_box", "images")}
    t0 = time.time()
    per = [a.n // 3 + (1 if i < a.n % 3 else 0) for i in range(3)]
    for meth, nm in zip(METHODS, per):
        ex = Extractor(meth, dev)
        cands = [p for p in pool if p[2] != meth]
        for j in range(nm):
            c = cands[rng.integers(len(cands))]; o = cands[rng.integers(len(cands))]
            img, msg, full, edit, m128 = (make_v2 if a.v2 else make)(rng, ex.wm, meth, load(c), load(o))
            r = ex(img, msg)
            res["names"].append(f"syn{a.seed}_{rank}_{meth}_{j}"); res["method"].append(meth)
            res["message"].append("".join(map(str, msg))); res["full"].append(full); res["edit"].append(edit)
            res["mask128"].append(m128)
            for k in ("maps", "acc_full", "best_acc", "best_box"):
                res[k].append(r[k])
            if a.save_images:
                res["images"].append(img)
            if j % 25 == 0:
                print(f"[{rank}] {meth} {j}/{nm} full={full} {edit} cov={m128.mean():.2f} acc_full={r['acc_full']:.2f} "
                      f"best={r['best_acc']:.2f} {time.time() - t0:.0f}s", flush=True)
        del ex
    out = f"{a.out}_{a.seed}_{rank}.npz"
    extra = {"images": np.stack(res["images"])} if a.save_images else {}
    np.savez(out, names=np.array(res["names"]), method=np.array(res["method"]), message=np.array(res["message"]),
             full_watermark=np.array(res["full"]), edit=np.array(res["edit"]), mask128=np.stack(res["mask128"]),
             maps=np.stack(res["maps"]), acc_full=np.array(res["acc_full"], np.float32),
             best_acc=np.array(res["best_acc"], np.float32), best_box=np.stack(res["best_box"]), **extra)
    print(f"[{rank}] écrit {out} ({len(res['names'])}) en {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
