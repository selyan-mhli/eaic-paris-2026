"""Recherche du rectangle watermarké par recadrage (régime « rectangle seul », R).

Constat (diagnostic train, 2026-11-21) : quand seul un rectangle a été watermarké (full_watermark=False),
le décodage de l'image entière est au hasard (~0.53) mais le rectangle GT recadré tel quel (le wrapper le
redimensionne à sa résolution) décode à 0.64-0.73 en moyenne : le watermark a été posé à l'échelle du
rectangle. On cherche donc la boîte dont le recadrage décode le mieux le message connu.

    python work_c1/box_search.py --split train --out output/boxes_train.npz [--limit 6]

Images à acc_full >= --gate : régime F, pas de recherche (box = image entière).
Grille : tailles h, w dans SIZES (rapport <= 16, <= 2 pour TrustMark), pas de position = max(24, min(h,w)/2),
puis raffinement local des bords (pas 16 puis 8) autour des 4 meilleures boîtes.
Sortie : box (y0, y1, x0, x1) exclusif, acc de la meilleure boîte, acc_full.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import to_tensor  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

METHODS = ("pixelseal", "trustmark", "stegastamp")
SIZES = (32, 48, 64, 96, 128, 192, 256, 320, 384, 512)
H = 512


def accs_for(wm, img, msg, boxes, bs=256):
    """boxes : liste (y0, y1, x0, x1) -> bit accuracy du message pour chaque recadrage."""
    out = np.zeros(len(boxes), np.float32)
    by = {}
    for i, (y0, y1, x0, x1) in enumerate(boxes):
        by.setdefault((y1 - y0, x1 - x0), []).append(i)
    for (h, w), ii in by.items():  # même taille -> un batch
        for s in range(0, len(ii), bs):
            chunk = ii[s:s + bs]
            crops = torch.cat([img[..., boxes[i][0]:boxes[i][1], boxes[i][2]:boxes[i][3]] for i in chunk])  # b 3 h w
            lg = torch.as_tensor(wm.logits(crops)).float().cpu()  # b k
            out[chunk] = ((lg > 0).float() == msg[None]).float().mean(1).numpy()
    return out


def grid(max_ratio):
    bx = []
    for h in SIZES:
        for w in SIZES:
            if max(h / w, w / h) > max_ratio:
                continue
            st = max(24, (min(h, w) // 2) // 8 * 8)
            for y in list(range(0, H - h + 1, st)) + [H - h]:
                for x in list(range(0, H - w + 1, st)) + [H - w]:
                    bx.append((y, y + h, x, x + w))
    return sorted(set(bx))


def refine(wm, img, msg, seeds, steps=(16, 8)):
    best = list(seeds)
    for st in steps:
        for _ in range(3):
            cand = set(best)
            for (y0, y1, x0, x1) in best:
                for dy0 in (-st, 0, st):
                    for dy1 in (-st, 0, st):
                        for dx0 in (-st, 0, st):
                            for dx1 in (-st, 0, st):
                                b = (y0 + dy0, y1 + dy1, x0 + dx0, x1 + dx1)
                                if 0 <= b[0] < b[1] <= H and 0 <= b[2] < b[3] <= H and b[1] - b[0] >= 16 and b[3] - b[2] >= 16:
                                    cand.add(b)
            cand = sorted(cand)
            a = accs_for(wm, img, msg, cand)
            order = np.argsort(-a)[: len(seeds)]
            new = [cand[i] for i in order]
            if new == best:
                break
            best = new
    a = accs_for(wm, img, msg, best)
    k = int(np.argmax(a))
    return best[k], float(a[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--data", default="release")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gate", type=float, default=0.75)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = LocalisationDataset(args.data, args.split)
    idx = [i for i in range(len(ds)) if i % args.nshards == args.shard]
    wms = {m: load_watermark(m, "models", dev) for m in METHODS}
    grids = {m: grid(2.0 if m == "trustmark" else 16.0) for m in METHODS}
    print("boîtes par grille :", {m: len(g) for m, g in grids.items()}, flush=True)
    names, method, box, acc, accf = [], [], [], [], []
    t0 = time.time(); done = 0
    for i in idx:
        it = ds[i]; m = it["method"]; wm = wms[m]
        img = to_tensor(it["image"]).to(dev); msg = torch.from_numpy(it["message"]).float()
        af = float(accs_for(wm, img, msg, [(0, H, 0, H)])[0])
        if af >= args.gate:
            b, a = (0, H, 0, H), af
        else:
            if args.limit and done >= args.limit:
                continue
            g = grids[m]
            a_all = accs_for(wm, img, msg, g)
            seeds = [g[k] for k in np.argsort(-a_all)[:4]]
            b, a = refine(wm, img, msg, seeds)
            done += 1
            if done <= 5 or done % 25 == 0:
                gt = ""
                if "mask" in it:
                    ys, xs = np.nonzero(it["mask"])
                    gt = f" gt_bbox=({ys.min()},{ys.max() + 1},{xs.min()},{xs.max() + 1})" if len(ys) else " gt vide"
                print(f"[R {done}] {m} acc_full={af:.3f} best={b} acc={a:.3f}{gt} {time.time() - t0:.0f}s", flush=True)
        names.append(it["image_name"]); method.append(m); box.append(b); acc.append(a); accf.append(af)
    print(f"{len(names)} images, {done} recherches, {time.time() - t0:.0f} s", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, names=np.array(names), method=np.array(method), box=np.array(box, np.int32),
                        acc_box=np.array(acc, np.float32), acc_full=np.array(accf, np.float32))
    print("->", args.out, flush=True)


if __name__ == "__main__":
    main()
