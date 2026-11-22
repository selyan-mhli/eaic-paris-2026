"""Template matching par boîte (régime R) pour TrustMark et StegaStamp.

Idée (veille v2, A3) : le motif de TrustMark (message concaténé en entrée de l'encodeur) et celui de StegaStamp
(message -> FC -> 50x50x3 -> U-Net) dépendent de la position dans le patch watermarké. Quand seul un rectangle B a
été watermarké, le résidu ajouté est r(c_B, m) calculé à la résolution de l'encodeur (TrustMark 256, StegaStamp 400)
puis redimensionné à la taille de B. Pour une boîte candidate B, on recalcule donc ce résidu sur le recadrage de
l'image éditée x' (r(x'_B, m) ~ r(c_B, m), le watermark étant faible) et on le corrèle, à la résolution de
l'encodeur, avec le passe-haut du recadrage. La fuite de contenu (le résidu suit les bords de l'image) est annulée
en comparant au même calcul avec K messages aléatoires :
    c(m) = NCC(hp(x'_B), hp(r(x'_B, m)))          z(B) = (c(m) - moyenne_k c(m_k)) / écart-type_k c(m_k)
Une boîte mal placée ou à la mauvaise échelle désaligne le motif : c(m) retombe au niveau des messages aléatoires.

Recherche : grille grossière (K_coarse messages, score c(m) - moyenne c(m_k)), puis raffinement des bords autour des
meilleures boîtes avec K_fine messages et le score z. Seules les images à acc_full < --gate sont cherchées (régime R),
comme box_search.py. Sortie au format de box_search.py (names, method, box, acc_box, acc_full) plus z et c_msg, donc
utilisable telle quelle par gated_eval.py (--boxes). acc_box = bit accuracy du recadrage choisi.

    python work_c1/template_match.py --split train --out output/tmatch_train.npz [--limit 6] [--methods trustmark]
    python work_c1/template_match.py --eval output/tmatch_train.npz [--compare 'output/boxes_train_*.npz']
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

METHODS = ("trustmark", "stegastamp")
H = 512
ENC_RES = {"trustmark": 256, "stegastamp": 400}
MAX_RATIO = {"trustmark": 2.0, "stegastamp": 16.0}
SIZES = (32, 48, 64, 96, 128, 192, 256, 384, 512)


# ---------------------------------------------------------------- résidus des encodeurs (batch, résolution encodeur)

class Residuals:
    """residual(x, msgs) : x b 3 R R dans [0, 1] (déjà à la résolution de l'encodeur), msgs b k (0/1)
    -> b 3 R R, résidu en unités [0, 1], moyenne par canal retirée (comme les embedders du kit)."""

    def __init__(self, wm, method: str):
        self.wm, self.method, self.R = wm, method, ENC_RES[method]

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, msgs: torch.Tensor, bs: int = 0) -> torch.Tensor:
        bs = bs or int(os.environ.get("TM_RES_BS", "64"))  # taille de lot du ré-embarquement (env, défaut 64)
        outs = []
        for s in range(0, len(x), bs):
            xb, mb = x[s:s + bs], msgs[s:s + bs]
            if self.method == "trustmark":
                tm = self.wm.tm
                cover = xb.to(tm.encoder.device) * 2 - 1
                stego, _ = tm.encoder(cover, mb.float().to(tm.encoder.device))
                r = (stego.clamp(-1, 1) - cover) / 2
            else:
                nhwc = xb.permute(0, 2, 3, 1).cpu().numpy()
                r = torch.from_numpy(self.wm._run(nhwc, ["residual"], mb.float().cpu().numpy())[0]).permute(0, 3, 1, 2)
            r = r - r.mean(dim=(2, 3), keepdim=True)
            outs.append(r.float().to(x.device))
        return torch.cat(outs)


def _gauss(sigma: float, device) -> torch.Tensor:
    k = int(2 * round(3 * sigma) + 1)
    t = torch.arange(k, device=device, dtype=torch.float32) - k // 2
    g = torch.exp(-t ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    return g


def highpass(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """x - flou gaussien séparable (padding reflect), par canal."""
    g = _gauss(sigma, x.device); k = len(g); c = x.shape[1]
    y = F.pad(x, (k // 2,) * 4, mode="reflect")
    y = F.conv2d(y, g.view(1, 1, 1, k).repeat(c, 1, 1, 1), groups=c)
    y = F.conv2d(y, g.view(1, 1, k, 1).repeat(c, 1, 1, 1), groups=c)
    return x - y


def ncc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a, b : n ... -> n (corrélation normalisée, moyennes retirées)."""
    a = a.flatten(1); b = b.flatten(1)
    a = a - a.mean(1, keepdim=True); b = b - b.mean(1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12)


# ---------------------------------------------------------------- score d'une liste de boîtes

def crops_at(img: torch.Tensor, boxes, R: int) -> torch.Tensor:
    """img 1 3 H W -> len(boxes) 3 R R, recadrage redimensionné (bilinéaire anticrénelé, comme PIL)."""
    out = torch.empty(len(boxes), 3, R, R, device=img.device)
    by = {}
    for i, (y0, y1, x0, x1) in enumerate(boxes):
        by.setdefault((y1 - y0, x1 - x0), []).append(i)
    for (h, w), ii in by.items():
        c = torch.cat([img[..., boxes[i][0]:boxes[i][1], boxes[i][2]:boxes[i][3]] for i in ii])
        out[ii] = F.interpolate(c, size=(R, R), mode="bilinear", align_corners=False, antialias=True).clamp(0, 1)
    return out


def score_boxes(res: Residuals, img, msg, rand, boxes, sigma: float, chunk: int = 128):
    """-> c_msg (n,), c_rand (n, K) : NCC du passe-haut du recadrage avec le passe-haut du résidu prédit."""
    R = res.R; K = len(rand)
    allm = torch.cat([msg[None], rand]).float()  # (K+1) k
    cm, cr = [], []
    for s in range(0, len(boxes), chunk):
        bx = boxes[s:s + chunk]; n = len(bx)
        x = crops_at(img, bx, R)                                       # n 3 R R
        hx = highpass(x, sigma)
        xr = x.repeat_interleave(K + 1, 0)                             # n(K+1) 3 R R
        mr = allm.repeat(n, 1).to(x.device)
        r = highpass(res(xr, mr), sigma)
        c = ncc(hx.repeat_interleave(K + 1, 0), r).view(n, K + 1).cpu()
        cm.append(c[:, 0]); cr.append(c[:, 1:])
    return torch.cat(cm).numpy(), torch.cat(cr).numpy()


def zscore(cm, cr):
    return (cm - cr.mean(1)) / (cr.std(1, ddof=1) + 1e-6)


# ---------------------------------------------------------------- grille et raffinement

def grid(max_ratio: float, min_step: int = 48, sizes=SIZES):
    bx = set()
    for h in sizes:
        for w in sizes:
            if max(h / w, w / h) > max_ratio:
                continue
            st = max(min_step, (min(h, w) // 2) // 8 * 8)
            for y in list(range(0, H - h + 1, st)) + [H - h]:
                for x in list(range(0, H - w + 1, st)) + [H - w]:
                    bx.add((y, y + h, x, x + w))
    return sorted(bx)


def valid(b, max_ratio):
    y0, y1, x0, x1 = b
    if not (0 <= y0 < y1 <= H and 0 <= x0 < x1 <= H):
        return False
    h, w = y1 - y0, x1 - x0
    return h >= 16 and w >= 16 and max(h / w, w / h) <= max_ratio + 1e-9


def search(res: Residuals, img, msg, rng, args, max_ratio, boxes0):
    k = len(msg)
    rand_c = torch.from_numpy(rng.integers(0, 2, (args.k_coarse, k))).float()
    cm, cr = score_boxes(res, img, msg, rand_c, boxes0, args.sigma)
    coarse = cm - cr.mean(1)
    seeds = [boxes0[i] for i in np.argsort(-coarse)[: args.seeds]]

    rand_f = torch.from_numpy(rng.integers(0, 2, (args.k_fine, k))).float()
    cache = {}

    def sc(bs_):
        todo = [b for b in bs_ if b not in cache]
        if todo:
            a, b = score_boxes(res, img, msg, rand_f, todo, args.sigma)
            cache.update(zip(todo, zip(zscore(a, b).tolist(), a.tolist())))
        return np.array([cache[b][0] for b in bs_], np.float32)

    best = list(seeds)
    for st in args.steps:
        for _ in range(args.iters):
            cand = set(best)
            for (y0, y1, x0, x1) in best:
                for d in ((-st, 0, 0, 0), (st, 0, 0, 0), (0, -st, 0, 0), (0, st, 0, 0),
                          (0, 0, -st, 0), (0, 0, st, 0), (0, 0, 0, -st), (0, 0, 0, st),
                          (-st, st, -st, st), (st, -st, st, -st)):          # 4 bords seuls + agrandir/réduire
                    b = (y0 + d[0], y1 + d[1], x0 + d[2], x1 + d[3])
                    if valid(b, max_ratio):
                        cand.add(b)
            cand = sorted(cand)
            z = sc(cand)
            new = [cand[i] for i in np.argsort(-z)[: args.seeds]]
            if new == best:
                break
            best = new
    z = sc(best); i = int(np.argmax(z))
    return best[i], float(z[i]), float(cache[best[i]][1]), len(cache) + len(boxes0)


# ---------------------------------------------------------------- évaluation (local, sans GPU)

def box_mask128(b):
    from wmloc.utils import downsample_mask
    m = np.zeros((H, H), np.uint8); y0, y1, x0, x1 = b; m[y0:y1, x0:x1] = 1
    return downsample_mask(m)


def evaluate(args):
    from wmloc.utils import miou_image
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    z = np.load(Path(args.data) / "train" / "masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
    import json
    meta = {m["image_name"]: m for m in json.loads((Path(args.data) / "train" / "metadata.json").read_text())}

    def load(globs):
        d = {}
        for f in sorted(glob.glob(globs)):
            t = np.load(f)
            for j, n in enumerate(t["names"].tolist()):
                d[n] = {k: t[k][j] for k in t.files if k != "names"}
        return d
    tm = load(args.eval)
    cmp = load(args.compare) if args.compare else {}
    rows = []
    for n, r in tm.items():
        if r["searched"] == 0:
            continue
        g = gt[n]; full = bool(meta[n]["full_watermark"])
        s_tm = miou_image(box_mask128(tuple(r["box"])), g)
        s_cmp = miou_image(box_mask128(tuple(cmp[n]["box"])), g) if n in cmp else np.nan
        s_zero = miou_image(np.zeros_like(g), g)
        rows.append((str(r["method"]), full, s_tm, s_cmp, s_zero, float(r["z"]), n, float(r.get("sec", np.nan))))
    print(f"{'méthode':<11} {'full':<5} {'n':>3} {'tmatch':>7} {'box_srch':>8} {'tout0':>6} {'z méd':>6} {'s/img':>6}")
    for m in METHODS:
        for full in (False, True):
            sel = [x for x in rows if x[0] == m and x[1] == full]
            if not sel:
                continue
            a = np.array([x[2:6] for x in sel], np.float64)
            print(f"{m:<11} {str(full):<5} {len(sel):>3} {a[:, 0].mean():7.4f} {np.nanmean(a[:, 1]):8.4f} "
                  f"{a[:, 2].mean():6.4f} {np.median(a[:, 3]):6.2f} {np.nanmean([x[7] for x in sel]):6.1f}")
    # seuil sur z : en dessous, tout 0 (aide à choisir --z-min ; optimiste, choisi sur les mêmes images)
    for m in METHODS:
        sel = [x for x in rows if x[0] == m and not x[1]]
        if not sel:
            continue
        a = np.array([x[2:6] for x in sel], np.float64)
        best = max(((np.where(a[:, 3] >= t, a[:, 0], a[:, 2]).mean(), t) for t in np.arange(0, 15.5, 0.5)))
        print(f"{m} R : tmatch si z >= {best[1]:.1f} sinon tout 0 -> {best[0]:.4f} (non imbriqué, optimiste)")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--data", default="release")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--limit", type=int, default=0, help="n images max par méthode (0 = toutes)")
    ap.add_argument("--gate", type=float, default=0.75, help="acc_full >= gate -> régime F, pas de recherche")
    ap.add_argument("--all", action="store_true", help="chercher aussi les images F (diagnostic)")
    ap.add_argument("--k-coarse", type=int, default=3)
    ap.add_argument("--k-fine", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--steps", type=int, nargs="+", default=[32, 16, 8])
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=1.5, help="passe-haut à la résolution de l'encodeur")
    ap.add_argument("--min-step", type=int, default=48)
    ap.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval"); ap.add_argument("--compare")
    args = ap.parse_args()
    sys.path.insert(0, str(Path.cwd()))
    if args.eval:
        return evaluate(args)
    assert args.out, "--out requis"

    from wmloc.data import LocalisationDataset
    from wmloc.utils import to_tensor
    from wmloc.watermarks import load_watermark
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = LocalisationDataset(args.data, args.split, methods=args.methods)
    idx = [i for i in range(len(ds)) if i % args.nshards == args.shard]
    if args.limit:
        per, keep = {}, []
        for i in idx:
            m = ds.meta[i]["method"]
            if per.get(m, 0) < args.limit:
                per[m] = per.get(m, 0) + 1; keep.append(i)
        idx = keep
    nth = int(os.environ.get("OMP_NUM_THREADS", "0") or 0)
    if nth:
        torch.set_num_threads(nth)
    wms = {m: load_watermark(m, "models", dev) for m in args.methods}
    if "stegastamp" in wms and nth:  # ORT : plus de threads pour le ré-embarquement CPU<->GPU
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = nth; so.inter_op_num_threads = 2
        w = wms["stegastamp"]
        w.sess = ort.InferenceSession(w.sess._model_path, so, providers=w.sess.get_providers())
        print("ORT StegaStamp threads", nth, w.sess.get_providers(), flush=True)
    resid = {m: Residuals(wms[m], m) for m in args.methods}
    grids = {m: grid(MAX_RATIO[m], args.min_step, args.sizes) for m in args.methods}
    print("boîtes grille grossière :", {m: len(g) for m, g in grids.items()}, flush=True)
    rec = {k: [] for k in ("names", "method", "box", "acc_box", "acc_full", "z", "c_msg", "searched", "sec")}
    part = args.out.with_name("partial_" + args.out.name)  # hors du motif tmatch_*

    def save(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name("tmp_" + path.name)
        np.savez_compressed(tmp, names=np.array(rec["names"]), method=np.array(rec["method"]),
                            box=np.array(rec["box"], np.int32).reshape(-1, 4),
                            **{k: np.array(rec[k], np.float32) for k in ("acc_box", "acc_full", "z", "c_msg", "sec")},
                            searched=np.array(rec["searched"], np.int8))
        tmp.rename(path)

    if part.exists():  # reprise après dépassement de --time
        zp = np.load(part)
        for k in rec:
            rec[k] = [tuple(b) for b in zp["box"].tolist()] if k == "box" else zp[k].tolist()
        print(f"reprise : {len(rec['names'])} images déjà faites", flush=True)
    done = set(rec["names"])
    t0 = time.time()
    for j, i in enumerate(idx):
        if ds.meta[i]["image_name"] in done:
            continue
        t_img = time.time()
        it = ds[i]; m = it["method"]; wm = wms[m]
        img = to_tensor(it["image"]).to(dev); msg = torch.from_numpy(it["message"]).float()
        af = float(((wm.logits(img).cpu() > 0).float() == msg[None]).float().mean())
        rng = np.random.default_rng(args.seed * 1_000_003 + i)
        if af >= args.gate and not args.all:
            box, z, c, nb, searched = (0, H, 0, H), np.nan, np.nan, 0, 0
        else:
            box, z, c, nb = search(resid[m], img, msg, rng, args, MAX_RATIO[m], grids[m]); searched = 1
        y0, y1, x0, x1 = box
        ab = float(((wm.logits(img[..., y0:y1, x0:x1]).cpu() > 0).float() == msg[None]).float().mean())
        for k, v in zip(rec, (it["image_name"], m, box, ab, af, z, c, searched, time.time() - t_img)):
            rec[k].append(v)
        if searched and sum(rec["searched"]) % 5 == 0:
            save(part)
        extra = ""
        if "mask" in it and searched:
            from wmloc.utils import miou_image
            extra = f" mIoU {miou_image(box_mask128(box), it['mask128']):.3f} full={it.get('full_watermark')}"
        print(f"[{j + 1}/{len(idx)}] {it['image_name']} {m} acc_full {af:.3f} box {box} z {z:.2f} acc_box {ab:.3f}"
              f" ({nb} boîtes){extra} {time.time() - t0:.0f}s", flush=True)
    save(args.out)
    if part.exists():
        part.unlink()
    print("écrit", args.out, flush=True)


if __name__ == "__main__":
    main()
