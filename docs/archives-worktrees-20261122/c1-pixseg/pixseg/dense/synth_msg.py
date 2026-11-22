"""Synthétique F tracé pour le localisateur dense : embedder OFFICIEL du kit, message, sources et retournements
enregistrés. Réutilise la géométrie testée de ../data.py (holes, synth_sample) sans la modifier.

  python dense/synth_msg.py --method pixelseal --n 1000 --seed 0 --out output/dense_syn
  -> output/dense_syn_pixelseal_s0_00.npz ... (tranches de --shard images, quelques fichiers seulement)

Contrat :
- F uniquement (p_full = 1) : image entière watermarkée puis trous recollés (propre 80 %, autre image 10 %,
  propre floutée 10 %). Masque = hors trous, grille 128 par la règle officielle.
- Retournements appliqués aux sources AVANT incrustation (`load(c, flip)`) ; aucune transformation après.
- Sources : images des splits `--pool` (train par défaut) d'une AUTRE méthode, en écartant les images des folds
  `--exclude-folds` (0,1 = confirmation, par défaut) de real_folds(method|full, 5, 0).
- Clés écrites : images uint8 n 512 512 3, mask128 uint8 n 128 128, msg uint8 n nbits, names, method, src,
  src2, flip, flip2, fill, acc_full (bit accuracy officielle de l'image composite entière, -1 si non calculée).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import data as pdata  # noqa: E402  (c1/pixseg/data.py)

KEYS = ("image", "mask128", "msg", "src", "src2", "flip", "flip2", "fill", "acc_full")


def candidates(root: Path, method: str, pool: str, exclude_folds=(0, 1)):
    """Sources (split, nom) d'une autre méthode ; images train des folds exclus écartées."""
    excl = set()
    if exclude_folds:
        meta = json.loads((root / "train" / "metadata.json").read_text())
        fo = pdata.real_folds(np.array([m["method"] for m in meta]), np.array([bool(m["full_watermark"]) for m in meta]), 0)
        excl = {m["image_name"] for m, f in zip(meta, fo) if f in exclude_folds}
    out = []
    for sp in pool.split(","):
        for m in json.loads((root / sp / "metadata.json").read_text()):
            if m["method"] != method and m["image_name"] not in excl:
                out.append((sp, m["image_name"]))
    assert out, "aucune source"
    return out


def stream(rng, embed, method, cands, load, decode=None):
    """Générateur infini de composites F tracés. embed(img uint8, msg uint8) -> uint8 ; decode(img) -> logits."""
    while True:
        c, o = cands[rng.integers(len(cands))], cands[rng.integers(len(cands))]
        f1, f2 = bool(rng.random() < 0.5), bool(rng.random() < 0.5)
        rec = []

        def emb(img, msg):
            rec.append(np.asarray(msg, np.uint8).copy()); return embed(img, msg)
        u_state = rng.bit_generator.state
        s = pdata.synth_sample(rng, emb, method, load(c, f1), load(o, f2), p_full=1.0)
        assert len(rec) == 1 and s["full"], "un seul appel à l'embedder par composite F attendu"
        # reconstitue le remplissage tiré dans synth_sample (même ordre de tirages : msg, p_full, holes, u)
        r2 = np.random.default_rng(); r2.bit_generator.state = u_state
        r2.integers(0, 2, pdata.NBITS[method]); r2.random(); pdata.holes(r2); u = r2.random()
        fill = "clean" if u < pdata.FILL[0] else ("other" if u < pdata.FILL[0] + pdata.FILL[1] else "blur")
        acc = -1.0
        if decode is not None:
            acc = float(((decode(s["image"]) > 0).astype(np.uint8) == rec[0]).mean())
        yield {"image": s["image"], "mask128": s["mask128"], "msg": rec[0], "src": c[1], "src2": o[1],
               "flip": f1, "flip2": f2, "fill": fill, "acc_full": acc}


def kit_codec(method, device):
    sys.path.insert(0, str(pdata.KIT))
    import torch
    from wmloc.watermarks import load_watermark
    wm = load_watermark(method, pdata.KIT / "models", device)

    def embed(img, msg):
        out = np.asarray(wm.embed(img, msg))
        assert out.shape == img.shape and out.dtype == np.uint8
        return out

    def decode(img):
        x = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255)
        return wm.logits(x)[0].cpu().numpy()
    return embed, decode


def gen(a, embed=None, decode=None):
    root = Path(a.data or pdata.KIT / "release")
    cands = candidates(root, a.method, a.pool, tuple(int(k) for k in a.exclude_folds.split(",") if k != ""))
    rng = np.random.default_rng(7919 * a.seed + pdata.METHODS.index(a.method) + 17)

    def load(c, flip):
        im = pdata.load_image(root / c[0] / "images" / f"{c[1]}.png")
        return np.ascontiguousarray(im[:, ::-1]) if flip else im
    if embed is None:
        embed, decode = kit_codec(a.method, a.device)
    buf, shard, t0, files = {k: [] for k in KEYS}, 0, time.time(), []

    def flush():
        nonlocal buf, shard
        n = len(buf["image"])
        if not n:
            return
        out = f"{a.out}_{a.method}_s{a.seed}_{shard:02d}.npz"
        tmp = out + ".tmp.npz"
        np.savez(tmp, images=np.stack(buf["image"]), mask128=np.stack(buf["mask128"]), msg=np.stack(buf["msg"]),
                 names=np.array([f"dsyn_{a.method}_s{a.seed}_{shard}_{j}" for j in range(n)]),
                 method=np.array([a.method] * n), **{k: np.array(buf[k]) for k in KEYS[3:]})
        Path(tmp).rename(out); files.append(out)
        print(f"écrit {out} ({n} images) à {time.time() - t0:.0f}s", flush=True)
        buf, shard = {k: [] for k in KEYS}, shard + 1

    for j, s in enumerate(stream(rng, embed, a.method, cands, load, decode)):
        if j >= a.n or time.time() - t0 > 60 * a.minutes:
            break
        for k in KEYS:
            buf[k].append(s[k])
        if j % 100 == 0:
            print(f"{a.method} {j}/{a.n} couverture {s['mask128'].mean():.2f} acc_full {s['acc_full']:.3f} "
                  f"{(time.time() - t0) / (j + 1):.2f} s/img", flush=True)
        if len(buf["image"]) == a.shard:
            flush()
    flush()
    return files


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["pixelseal", "trustmark"])
    ap.add_argument("--n", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True); ap.add_argument("--shard", type=int, default=500)
    ap.add_argument("--pool", default="train", help="splits sources ; 'train,test' : usage du test à valider (§5.5)")
    ap.add_argument("--exclude-folds", default="0,1", help="folds train jamais utilisés comme sources")
    ap.add_argument("--minutes", type=float, default=15); ap.add_argument("--data", default=None)
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args(argv)


if __name__ == "__main__":
    gen(parse())
