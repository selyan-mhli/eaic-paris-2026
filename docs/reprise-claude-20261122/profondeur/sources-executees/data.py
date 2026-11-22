"""Données de la segmentation au pixel (Task 1) : train/test réels, synthétique par l'embedder officiel.

Réel      : release/<split>/metadata.json, images/<nom>.png (512x512), train/masks_128.npz (names, masks).
Synthétique (régimes mesurés sur le train réel, repris de c1/seg/gen.py make_v2 et c1/seg/gen_f.py) :
  F (moitié) : image entière watermarkée, puis 1-4 trous en forme d'objets recollés depuis l'image propre
               (80 %), une autre image (10 %) ou l'image propre floutée (10 %). Masque = hors trous.
  R (moitié) : rectangle (ratio <= 16, <= 2 pour TrustMark) watermarké seul puis recollé dans l'image propre,
               ou forme d'objet watermarkée collée dans une autre image. Masque = rectangle ou objet.
Masque 512 exact -> grille 128 par la règle officielle (cellule = 1 si >= 50 % de ses 16 pixels).

Pré-génération (un processus par méthode, StegaStamp ONNX prend jusqu'à 24 Go de GPU) :
  python work_pixseg/data.py gen --method pixelseal --n 3000 --out output/pixseg_syn
  -> output/pixseg_syn_pixelseal_s0_00.npz, ... (tranches de 500 images, écrites au fil de l'eau)
Sources : images d'une AUTRE méthode que celle posée, du TEST par défaut (aucune image train, donc aucune
fuite vers les folds de validation ; avec --pool train,test, train.py écarte les sources des folds évalués).
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
from scipy import ndimage

KIT = Path(os.environ.get("PIXSEG_KIT", "/p/scratch/training2666/mouhali1/SLZ/Hackathon_Watermark_Localisation_Task"))
METHODS = ("pixelseal", "trustmark", "stegastamp")
NBITS = {"pixelseal": 256, "trustmark": 100, "stegastamp": 100}
FILL = (0.8, 0.1, 0.1)  # trous F : image propre, autre image, image propre floutée


# ---------------------------------------------------------------- métrique et folds (copies testées)
def downsample_mask(mask512: np.ndarray) -> np.ndarray:
    """Copie de c1/metric.py (== wmloc.utils.downsample_mask) : (512, 512) 0/1 -> (128, 128) uint8."""
    m = np.asarray(mask512).astype(np.float32).reshape(128, 4, 128, 4)  # 512 512 -> 128 4 128 4
    return (m.mean(axis=(1, 3)) >= 0.5).astype(np.uint8)  # 128 128


def fast_miou(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Copie de c1/real_eval.py : pred, gt (n, 128, 128) -> (n,) mIoU 2 classes (1 si union vide)."""
    if len(pred) == 0:
        return np.zeros(0)
    p = pred.reshape(len(pred), -1).astype(bool); g = gt.reshape(len(gt), -1).astype(bool)  # n (128*128)
    out = np.zeros(len(p))
    for c in (False, True):
        pc, gc = p == c, g == c
        u = (pc | gc).sum(1); i = (pc & gc).sum(1)
        out += np.where(u == 0, 1.0, i / np.maximum(u, 1))
    return out / 2


def folds(strata: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    """Copie de c1/real_eval.py (utilisée par c1/gated_eval.py)."""
    rng = np.random.default_rng(seed); f = np.zeros(len(strata), int)
    for s in np.unique(strata):
        ii = np.nonzero(strata == s)[0]; rng.shuffle(ii); f[ii] = np.arange(len(ii)) % k
    return f


def real_folds(method, full, seed: int = 0) -> np.ndarray:
    """Folds de gated_eval : 5 folds stratifiés méthode x full_watermark, images dans l'ordre de metadata.json."""
    return folds(np.array([f"{m}|{f}" for m, f in zip(method, full)]), 5, seed)


# ---------------------------------------------------------------- formes (copies de c1/seg/gen.py et gen_f.py)
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


def holes(rng) -> np.ndarray:
    """1-4 formes (aire log-uniforme 0,3 %-20 %, centre n'importe où) ; union -> bool 512 512."""
    h = np.zeros((512, 512), bool)
    for _ in range(rng.integers(1, 5)):
        h |= blob(rng, 512, 512, np.exp(rng.uniform(np.log(0.003), np.log(0.2))), (0.0, 1.0))
    if h.mean() > 0.5:  # les vrais F gardent > 50 % watermarké
        h = blob(rng, 512, 512, 0.1, (0.2, 0.8))
    return h


def rect(rng, method, amin=0.003, amax=0.4, smin=8):
    """Rectangle (y0, y1, x0, x1), aire log-uniforme, ratio <= 16 (<= 2 pour TrustMark)."""
    rmax = 2.0 if method == "trustmark" else 16.0
    for _ in range(100):
        area = np.exp(rng.uniform(np.log(amin), np.log(amax))) * 512 * 512
        r = np.exp(rng.uniform(-np.log(rmax), np.log(rmax)))  # w / h
        h = int(round(np.sqrt(area / r))); w = int(round(np.sqrt(area * r)))
        if smin <= h <= 512 and smin <= w <= 512:
            y0 = rng.integers(0, 512 - h + 1); x0 = rng.integers(0, 512 - w + 1)
            return int(y0), int(y0 + h), int(x0), int(x0 + w)
    return 128, 384, 128, 384


# ---------------------------------------------------------------- générateur synthétique
def synth_sample(rng, embed, method, clean, other, p_full=0.5, fill=FILL) -> dict:
    """Un composite. embed(img uint8 h w 3, msg uint8 nbits) -> uint8 h w 3 (embedder officiel ou factice).
    clean, other : uint8 512 512 3. Renvoie image (uint8 512 512 3), mask512 (bool), mask128 (uint8), full, edit."""
    msg = rng.integers(0, 2, NBITS[method]).astype(np.uint8)
    if rng.random() < p_full:  # régime F
        wimg = embed(clean, msg)
        hole = holes(rng)
        u = rng.random()
        src = clean if u < fill[0] else (other if u < fill[0] + fill[1] else
                                         ndimage.gaussian_filter(clean.astype(np.float32), (4, 4, 0)).astype(np.uint8))
        out, mask, edit, full = np.where(hole[..., None], src, wimg), ~hole, "hole", True
    elif rng.random() < 0.5:  # régime R, rectangle watermarké seul (85 %) ou découpé dans l'image watermarkée
        y0, y1, x0, x1 = rect(rng, method)
        wp = (embed(np.ascontiguousarray(clean[y0:y1, x0:x1]), msg) if rng.random() < 0.85
              else embed(clean, msg)[y0:y1, x0:x1])
        out = clean.copy(); out[y0:y1, x0:x1] = wp
        mask = np.zeros((512, 512), bool); mask[y0:y1, x0:x1] = True
        edit, full = "rect", False
    else:  # régime R, objet watermarké transplanté dans une autre image propre
        obj = blob(rng, 512, 512, np.exp(rng.uniform(np.log(0.005), np.log(0.4))), (0.05, 0.95))
        if not obj.any():
            obj[200:260, 200:260] = True
        ys, xs = np.where(obj)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        if rng.random() < 0.5 and min(y1 - y0, x1 - x0) >= 8:
            wsrc = clean.copy(); wsrc[y0:y1, x0:x1] = embed(np.ascontiguousarray(clean[y0:y1, x0:x1]), msg)
        else:
            wsrc = embed(clean, msg)
        out, mask, edit, full = np.where(obj[..., None], wsrc, other), obj, "transplant", False
    return {"image": out.astype(np.uint8), "mask512": mask, "mask128": downsample_mask(mask), "full": full, "edit": edit}


def synth_stream(rng, embed, method, cands, load):
    """Générateur en ligne (infini). cands : sources possibles ; load(c, flip) -> uint8 512 512 3."""
    while True:
        c, o = cands[rng.integers(len(cands))], cands[rng.integers(len(cands))]
        s = synth_sample(rng, embed, method, load(c, rng.random() < 0.5), load(o, rng.random() < 0.5))
        s.update(method=method, src=c[1], src2=o[1])
        yield s


def kit_embedder(method: str, device: str):
    """Embedder officiel : wm.embed(image uint8 h w 3, bits uint8) -> uint8 h w 3 (contrat vérifié à chaque appel)."""
    sys.path.insert(0, str(KIT))
    from wmloc.watermarks import load_watermark
    wm = load_watermark(method, KIT / "models", device)

    def embed(img, msg):
        out = np.asarray(wm.embed(img, msg))
        assert out.shape == img.shape and out.dtype == np.uint8, (method, img.shape, out.shape, out.dtype)
        return out
    return embed


# ---------------------------------------------------------------- chargement
def load_meta(split: str, data=None) -> dict:
    """names, method (ordre de metadata.json), + full (bool) et masks (n 128 128 uint8) pour le train."""
    root = Path(data or KIT / "release") / split
    meta = json.loads((root / "metadata.json").read_text())
    d = {"names": np.array([m["image_name"] for m in meta]), "method": np.array([m["method"] for m in meta])}
    assert len(set(d["names"].tolist())) == len(meta) and set(d["method"].tolist()) <= set(METHODS)
    if split == "train":
        d["full"] = np.array([bool(m["full_watermark"]) for m in meta])
        z = np.load(root / "masks_128.npz"); allm = dict(zip(z["names"].tolist(), z["masks"]))
        d["masks"] = np.stack([allm[n] for n in d["names"]]).astype(np.uint8)  # n 128 128
        assert d["masks"].shape[1:] == (128, 128) and set(np.unique(d["masks"])) <= {0, 1}
    return d


def load_image(path) -> np.ndarray:
    from PIL import Image
    im = np.asarray(Image.open(path).convert("RGB"))  # 512 512 3
    assert im.shape == (512, 512, 3), (path, im.shape)
    return im


def load_images(split: str, names, data=None) -> np.ndarray:
    root = Path(data or KIT / "release") / split / "images"
    out = np.empty((len(names), 512, 512, 3), np.uint8)  # n 512 512 3
    for i, n in enumerate(names):
        out[i] = load_image(root / f"{n}.png")
    return out


def load_synth(patterns, limit: int = 0) -> dict:
    """Concatène les tranches .npz de `gen` : images (n 512 512 3 uint8), mask128, method, full, src, src2, names."""
    files = sorted({f for p in patterns for f in glob.glob(p)}); assert files, patterns
    parts = [dict(np.load(f)) for f in files]
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    assert len(set(d["names"].tolist())) == len(d["names"]), "noms synthétiques en double (même seed deux fois ?)"
    if limit and limit < len(d["names"]):
        sel = np.random.default_rng(0).permutation(len(d["names"]))[:limit]
        d = {k: v[sel] for k, v in d.items()}
    return d


def load_maps(spec: str, names, stats=None):
    """Cartes existantes en canaux. spec = 'chemin_ou_glob.npz:cle1+cle2' ; chaque clé : (n, h, w) ou (n, C, h, w),
    h = w = 32 ou 128, alignée sur la clé `names` du fichier. Renvoie (cartes n E+1 128 128 float16 dont le
    dernier canal vaut 1 si l'image a ses cartes, stats). Normalisation par canal (moyenne, écart-type du premier
    fichier chargé : passer `stats` pour le test), NaN -> 0, bornée à +-5. Image absente du fichier : zéros."""
    path, keys = spec.rsplit(":", 1)
    files = sorted(glob.glob(path)); assert files, path
    pos = {n: i for i, n in enumerate(names)}
    out, rows = None, []
    for f in files:
        z = np.load(f)
        arrs = []
        for k in keys.split("+"):
            v = z[k].astype(np.float32)
            v = v[:, None] if v.ndim == 3 else v  # n h w -> n 1 h w
            assert v.ndim == 4 and v.shape[-1] == v.shape[-2] and v.shape[-1] in (32, 128), (f, k, v.shape)
            r = 128 // v.shape[-1]
            arrs.append(v.repeat(r, axis=2).repeat(r, axis=3))  # n C h w -> n C 128 128 (plus proche voisin)
        m = np.concatenate(arrs, 1)  # n E 128 128
        if out is None:
            out = np.zeros((len(names), m.shape[1] + 1, 128, 128), np.float32)  # n E+1 128 128
        for j, n in enumerate(z["names"].tolist()):
            if n in pos:
                out[pos[n], :-1] = m[j]; out[pos[n], -1] = 1; rows.append(pos[n])
    if stats is None:
        assert rows, f"aucune image de la liste dans {path}"
        v = out[rows, :-1]
        stats = (np.nanmean(v, axis=(0, 2, 3), keepdims=True), np.nanstd(v, axis=(0, 2, 3), keepdims=True) + 1e-6)
    if rows:  # sans aucune image connue (synthétique sans cartes) : zéros, drapeau 0
        out[rows, :-1] = np.nan_to_num(((out[rows, :-1] - stats[0]) / stats[1]).clip(-5, 5))
    return out.astype(np.float16), stats


# ---------------------------------------------------------------- pré-génération
def gen(a, embed=None):
    rng = np.random.default_rng(1000 * a.seed + METHODS.index(a.method))
    root = Path(a.data or KIT / "release")
    cands = [(sp, m["image_name"]) for sp in a.pool.split(",")
             for m in json.loads((root / sp / "metadata.json").read_text()) if m["method"] != a.method]

    def load(c, flip):
        im = load_image(root / c[0] / "images" / f"{c[1]}.png")
        return np.ascontiguousarray(im[:, ::-1]) if flip else im

    embed = embed or kit_embedder(a.method, a.device)
    keys = ("image", "mask128", "method", "full", "edit", "src", "src2")
    buf, shard, t0 = {k: [] for k in keys}, 0, time.time()

    def flush():
        nonlocal buf, shard
        if not buf["image"]:
            return
        out = f"{a.out}_{a.method}_s{a.seed}_{shard:02d}.npz"
        n = len(buf["image"])
        np.savez(out, images=np.stack(buf["image"]), mask128=np.stack(buf["mask128"]),
                 names=np.array([f"syn_{a.method}_s{a.seed}_{shard}_{j}" for j in range(n)]),
                 **{k: np.array(buf[k]) for k in keys[2:]})
        print(f"écrit {out} ({n} images) à {time.time() - t0:.0f}s", flush=True)
        buf, shard = {k: [] for k in keys}, shard + 1

    for j, s in enumerate(synth_stream(rng, embed, a.method, cands, load)):
        if j >= a.n or time.time() - t0 > 60 * a.minutes:
            break
        for k in keys:
            buf[k].append(s[k])
        if j % 50 == 0:
            print(f"{a.method} {j}/{a.n} {s['edit']} couverture {s['mask128'].mean():.2f} "
                  f"{(time.time() - t0) / (j + 1):.2f} s/img", flush=True)
        if len(buf["image"]) == a.shard:
            flush()
    flush()


def main(argv=None, embed=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen"])
    ap.add_argument("--method", required=True, choices=METHODS); ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", default="test"); ap.add_argument("--shard", type=int, default=500)
    ap.add_argument("--minutes", type=float, default=45, help="arrêt propre (tranches déjà écrites conservées)")
    ap.add_argument("--data", default=None, help="dossier release (défaut : KIT/release)")
    ap.add_argument("--device", default="cuda:0")
    gen(ap.parse_args(argv), embed)


if __name__ == "__main__":
    main()
