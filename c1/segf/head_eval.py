"""Tête de segmentation F (trous) en validation croisée imbriquée, alignée sur `gated_eval.py`.

Protocole (par graine, 5 folds externes `real_eval.folds` stratifiés méthode x full_watermark, comme la porte) :
  1. Sur les 4 folds d'apprentissage seulement : `gated_eval.fit` (règle full) -> tri g, choix F par méthode,
     seuil des rectangles R (rectangles du champion, `box_search.py`).
  2. Réseau : entraîné sur les images full_watermark des folds d'apprentissage. Prédictions « honnêtes » sur ces
     mêmes images par 4 folds internes (hors-fold interne) ; réseau final sur tout l'apprentissage pour le fold
     externe. Normalisation des canaux calculée sur l'apprentissage seulement.
  3. Par méthode, sur les images d'apprentissage triées F : on garde le choix du champion ou le réseau à un
     seuil t (grille), selon le mIoU (prédictions internes pour le réseau).
  4. Fold externe : tri g ; F -> choix retenu ; R -> rectangle du champion si acc_box >= tb.
Sortie : mIoU imbriqué par méthode x F/R et global, comparé au champion (même fit, sans réseau), par fold.

  python work_segf/head_eval.py --config ff+fm --heats output/heats_train.npz --boxes 'output/boxes_train_*.npz' \
      --ff 'output/seg_ff_train_*.npz' --hf 'output/segf_hf_train_*.npz' --seeds 0 1 2 --out output/segf_eval_<cfg>.npz
Option --test-* : refit sur les 510 et écrit un fichier test (non soumis).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, Path.cwd()):
    sys.path.insert(0, str(p))
import gated_eval as ge  # noqa: E402
from real_eval import METHODS, fast_miou, folds, group_line  # noqa: E402

FF_CH = (0, 1, 2, 3, 4, 6)  # canal 5 (détection PixelSeal) exclu : non entraîné
THR = np.round(np.arange(0.05, 0.96, 0.05), 2)


def load(pattern) -> dict:
    fs = sorted(glob.glob(pattern)); assert fs, pattern
    zs = [dict(np.load(f)) for f in fs]
    return {k: np.concatenate([z[k] for z in zs]) for k in zs[0]}


def align(z: dict, names: list[str], key: str) -> np.ndarray:
    pos = {n: i for i, n in enumerate(z["names"].tolist())}
    assert set(names) <= set(pos), f"{key} : images manquantes"
    return np.stack([z[key][pos[n]] for n in names]).astype(np.float32)


def build_inputs(cfg: str, names, d, ff, hf, extra):
    """-> X32 (n c 32 32), X128 (n c' 128 128) ; NaN -> 0 (StegaStamp n'a pas fm_*)."""
    fm = align(ff, names, "fmaps")[:, list(FF_CH)]  # n 6 32 32
    af = d["acc_full"].astype(np.float32)
    oh = np.stack([d["method"] == m for m in METHODS], 1).astype(np.float32)
    sc = np.concatenate([af[:, None], oh], 1)[:, :, None, None] * np.ones((1, 1, 32, 32), np.float32)
    x32 = np.concatenate([fm, sc], 1)
    hi = []
    parts = cfg.split("+")
    if "fm" in parts:
        hi += [d[k].astype(np.float32) for k in ("fm_acc", "fm_soft", "win128")]
    for k in parts:
        if k.startswith("hf:"):
            for c in k[3:].split(","):
                v = align(hf, names, c) if c in hf else None
                if c == "hfe/jnd":
                    v = align(hf, names, "hfe") / (align(hf, names, "jnd") + 1e-4)
                assert v is not None, c
                hi.append(v)
        if k.startswith("ex:"):
            for c in k[3:].split(","):
                hi.append(align(extra, names, c))
    x128 = np.stack(hi, 1) if hi else np.zeros((len(names), 0, 128, 128), np.float32)
    return np.nan_to_num(x32), np.nan_to_num(x128)


class Net(nn.Module):
    """Mini U-Net 32 -> 8 -> 32 sur les cartes grossières, puis tête 128 avec les canaux fins."""

    def __init__(self, c32, c128, c=48):
        super().__init__()
        blk = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU(),  # noqa: E731
                                         nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU())
        self.e1, self.e2, self.e3 = blk(c32, c), blk(c, 2 * c), blk(2 * c, 2 * c)
        self.d2, self.d1 = blk(4 * c, c), blk(2 * c, c)
        self.fine = nn.Sequential(nn.Conv2d(c128, 16, 3, padding=1), nn.GELU()) if c128 else None
        self.head = nn.Sequential(nn.Conv2d(c + c32 + (16 if c128 else 0), c, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(c, c, 3, padding=1), nn.GELU(), nn.Conv2d(c, 1, 1))

    def forward(self, x32, x128):
        e1 = self.e1(x32); e2 = self.e2(F.max_pool2d(e1, 2)); e3 = self.e3(F.max_pool2d(e2, 2))
        d2 = self.d2(torch.cat([F.interpolate(e3, scale_factor=2), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2), e1], 1))
        y = F.interpolate(torch.cat([d1, x32], 1), size=(128, 128), mode="bilinear", align_corners=False)
        if self.fine is not None:
            y = torch.cat([y, self.fine(x128)], 1)
        return self.head(y)[:, 0]


def loss_fn(logit, y):  # = c1/seg/train.py : BCE + IoU douce des deux classes
    p = torch.sigmoid(logit); iou = 0
    for pp, yy in ((p, y), (1 - p, 1 - y)):
        i = (pp * yy).flatten(1).sum(1); u = (pp + yy - pp * yy).flatten(1).sum(1); iou = iou + (i + 1) / (u + 1)
    return F.binary_cross_entropy_with_logits(logit, y) + (1 - iou / 2).mean()


class Head:
    def __init__(self, epochs, dev):
        self.epochs, self.dev = epochs, dev

    def fit(self, X32, X128, Y, seed):
        mu32 = X32.mean((0, 2, 3), keepdims=True); sd32 = X32.std((0, 2, 3), keepdims=True) + 1e-6
        mu128 = X128.mean((0, 2, 3), keepdims=True); sd128 = X128.std((0, 2, 3), keepdims=True) + 1e-6
        nrm = (mu32, sd32, mu128, sd128)
        torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
        net = Net(X32.shape[1], X128.shape[1]).to(self.dev)
        a32, a128 = self.prep(X32, X128, nrm)
        Yt = torch.from_numpy(Y.astype(np.float32))
        bs = 32; steps = self.epochs * ((len(Y) + bs - 1) // bs)
        opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
        for _ in range(self.epochs):
            net.train(); perm = torch.randperm(len(Y), generator=g)
            for i in range(0, len(Y), bs):
                idx = perm[i:i + bs]
                x32, x128, yb = a32[idx].to(self.dev), a128[idx].to(self.dev), Yt[idx].to(self.dev)
                for dim in (-1, -2):
                    if torch.rand(1, generator=g) < 0.5:
                        x32, x128, yb = x32.flip(dim), x128.flip(dim), yb.flip(dim)
                loss = loss_fn(net(x32, x128), yb); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        return net, nrm

    def prep(self, X32, X128, nrm):
        mu32, sd32, mu128, sd128 = nrm
        return (torch.from_numpy(((X32 - mu32) / sd32).clip(-5, 5).astype(np.float32)),
                torch.from_numpy(((X128 - mu128) / sd128).clip(-5, 5).astype(np.float32)))

    @torch.no_grad()
    def predict(self, model, X32, X128):
        net, nrm = model; net.eval(); a32, a128 = self.prep(X32, X128, nrm)
        return np.concatenate([torch.sigmoid(net(a32[i:i + 128].to(self.dev), a128[i:i + 128].to(self.dev))).cpu().numpy()
                               for i in range(0, len(X32), 128)])


def head_preds(H: Head, X32, X128, gt, full, method, tr, te_like, seed):
    """Prédictions du réseau : hors-fold interne sur les F d'apprentissage, réseau final ailleurs."""
    P = np.full((len(gt), 128, 128), np.nan, np.float32)
    trF = tr[full[tr]]
    inner = folds(method[trF], 4, seed + 100)
    for k in range(4):
        a, b = trF[inner != k], trF[inner == k]
        P[b] = H.predict(H.fit(X32[a], X128[a], gt[a], seed * 10 + k), X32[b], X128[b])
    fin = H.fit(X32[trF], X128[trF], gt[trF], seed * 10 + 9)
    rest = np.setdiff1d(np.concatenate([tr, te_like]), trF)
    P[rest] = H.predict(fin, X32[rest], X128[rest])
    return P, fin


def choose(D, gt, P, tr, r):
    """Par méthode : garder le choix F du champion ou net >= t, selon le mIoU sur les F triés d'apprentissage."""
    out = {}
    for m in METHODS:
        iF = tr[(D.method[tr] == m) & (D.af[tr] >= r["gate"])]
        c = r["m"][m]
        champ = fast_miou(D.pred_F(iF, tuple(c["F_var"]), c["F_thr"]), gt[iF]).sum()
        best = ("champ", None, champ)
        for t in THR:
            s = fast_miou(P[iF] >= t, gt[iF]).sum()
            if s > best[2]:
                best = ("net", float(t), s)
        out[m] = best[:2]
    return out


def apply(D, P, idx, r, ch):
    pred = ge.apply(D, idx, r)
    for m, (kind, t) in ch.items():
        if kind == "net":
            sel = idx[(D.method[idx] == m) & (D.af[idx] >= r["gate"])]
            assert np.all(np.diff(idx) > 0)
            pred[np.searchsorted(idx, sel)] = P[sel] >= t
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="ff | ff+fm | ff+fm+hf:a,b | ...+ex:k")
    ap.add_argument("--heats", type=Path, required=True); ap.add_argument("--boxes", required=True)
    ap.add_argument("--ff", required=True); ap.add_argument("--hf"); ap.add_argument("--extra")
    ap.add_argument("--data", default="release"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--test-heats", type=Path); ap.add_argument("--test-boxes"); ap.add_argument("--test-ff")
    ap.add_argument("--test-hf"); ap.add_argument("--test-extra"); ap.add_argument("--write-sub", type=Path)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time()
    d, names, boxes, accb = ge.load_split(a.heats, a.boxes, a.data, "train")
    z = np.load(Path(a.data) / "train" / "masks_128.npz"); allm = dict(zip(z["names"].tolist(), z["masks"]))
    gt = np.stack([allm[n] for n in names]).astype(bool)
    full, method = d["full_watermark"].astype(bool), d["method"]
    D = ge.Data(d, boxes, accb)
    ff = load(a.ff); hf = load(a.hf) if a.hf else {}; ex = load(a.extra) if a.extra else {}
    X32, X128 = build_inputs(a.config, names, d, ff, hf, ex)
    print(f"[{a.config}] X32 {X32.shape} X128 {X128.shape} device {dev}", flush=True)
    H = Head(a.epochs, dev)
    allidx = np.arange(D.n)
    res = {"champ": [], "head": [], "choices": []}
    for seed in a.seeds:
        fo = folds(np.array([f"{m}|{f}" for m, f in zip(method, full)]), 5, seed)
        s_ch, s_hd = np.zeros(D.n), np.zeros(D.n)
        for k in range(5):
            tr, te = allidx[fo != k], allidx[fo == k]
            r, _ = ge.fit(D, gt, tr, "full")
            P, _ = head_preds(H, X32, X128, gt, full, method, tr, te, seed * 7 + k)
            ch = choose(D, gt, P, tr, r)
            s_ch[te] = fast_miou(ge.apply(D, te, r), gt[te])
            s_hd[te] = fast_miou(apply(D, P, te, r, ch), gt[te])
            res["choices"].append({"seed": seed, "fold": k, "gate": r["gate"], **{m: ch[m] for m in METHODS}})
            fd = {m: s_hd[te][method[te] == m].mean() - s_ch[te][method[te] == m].mean() for m in METHODS}
            print(f"  graine {seed} fold {k} Δ " + " ".join(f"{m[:5]} {v:+.4f}" for m, v in fd.items())
                  + f" choix {ch} {time.time() - t0:.0f}s", flush=True)
        res["champ"].append(s_ch); res["head"].append(s_hd)
        print(group_line(f"champion g{seed}", s_ch, method, full), flush=True)
        print(group_line(f"{a.config} g{seed}", s_hd, method, full), flush=True)
    c, h = np.stack(res["champ"]), np.stack(res["head"])
    print(group_line("champion (moy.)", c.mean(0), method, full))
    print(group_line(f"{a.config} (moy.)", h.mean(0), method, full))
    for m in METHODS:
        s = method == m
        dm = h[:, s].mean(1) - c[:, s].mean(1)
        dF = h[:, s & full].mean(1) - c[:, s & full].mean(1)
        print(f"Δ {m:10s} {dm.mean():+.4f} par graine {np.round(dm, 4).tolist()} ; sur les F {dF.mean():+.4f}")
    print(f"Δ global {np.mean(h - c):+.4f} par graine {np.round((h - c).mean(1), 4).tolist()}")
    np.savez_compressed(a.out, names=np.array(names), champ=c, head=h, choices=json.dumps(res["choices"]))

    if a.write_sub:
        dt, tn, tbx, tab = ge.load_split(a.test_heats, a.test_boxes, a.data, "test")
        T = ge.Data(dt, tbx, tab)
        T32, T128 = build_inputs(a.config, tn, dt, load(a.test_ff), load(a.test_hf) if a.test_hf else {},
                                 load(a.test_extra) if a.test_extra else {})
        r, _ = ge.fit(D, gt, allidx, "full")
        Pall, fin = head_preds(H, X32, X128, gt, full, method, allidx, np.array([], int), 999)
        ch = choose(D, gt, Pall, allidx, r)
        Pt = H.predict(fin, T32, T128)
        masks = apply(T, Pt, np.arange(T.n), r, ch)
        torch.save({"image_names": tn, "masks": torch.from_numpy(masks)}, a.write_sub)
        back = torch.load(a.write_sub, weights_only=True)
        assert back["image_names"] == tn and back["masks"].shape == (1020, 128, 128) and back["masks"].dtype == torch.bool
        print(f"écrit {a.write_sub} ({a.write_sub.stat().st_size / 1e6:.1f} Mo) ; règle {json.dumps(r)} ; choix {ch} (NON soumis)")


if __name__ == "__main__":
    main()
