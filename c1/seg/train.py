"""Tête de segmentation sur les cartes de feats.py, entraînée sur le synthétique, évaluée sur le train réel.

  python c1/seg/train.py --synth 'c1/seg/data/seg_synth_*.npz' --real 'c1/seg/data/seg_feats_train_*.npz' \
      --masks c1/seg/data/masks_128.npz --meta c1/seg/data/train_metadata.json [--kfold-real] [--test ... --write-sub out.pt]

Entrée par image (canaux 32x32) : 5 cartes (dense, w128, w256, cmax, cbest) + acc_full, best_acc diffusés
+ one-hot méthode = 10 canaux. Sortie : logits 128x128. Perte : BCE + (1 - mIoU soft 2 classes par image).
Rapport : mIoU officielle (wmloc.utils.miou_image, réimplémentée vectorisée et comparée) par méthode x full.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

METHODS = ("pixelseal", "trustmark", "stegastamp")


def miou_np(pred, gt):
    """pred, gt (n,128,128) bool -> (n,) ; identique à wmloc.utils.miou_image (union vide -> 1)."""
    p = pred.reshape(len(pred), -1).astype(bool); g = gt.reshape(len(gt), -1).astype(bool)
    out = np.zeros(len(p))
    for c in (False, True):
        pc, gc = p == c, g == c
        u = (pc | gc).sum(1); i = (pc & gc).sum(1)
        out += np.where(u == 0, 1.0, i / np.maximum(u, 1))
    return out / 2


def miou_ref(p, g):  # copie de wmloc.utils.miou_image, pour le test d'égalité
    s = 0.0
    for c in (0, 1):
        union = np.logical_or(p == c, g == c).sum()
        s += 1.0 if union == 0 else np.logical_and(p == c, g == c).sum() / union
    return s / 2


def load_feats(pattern):
    fs = sorted(glob.glob(pattern)); assert fs, pattern
    zs = [dict(np.load(f)) for f in fs]
    return {k: np.concatenate([z[k] for z in zs]) for k in zs[0] if k != "images"}


def inputs(d):
    """-> X (n, 10, 32, 32) float32."""
    n = len(d["names"])
    maps = d["maps"].astype(np.float32)  # n 5 32 32
    sc = np.stack([d["acc_full"], d["best_acc"]], 1).astype(np.float32)  # n 2
    oh = np.stack([(d["method"] == m) for m in METHODS], 1).astype(np.float32)  # n 3
    extra = np.concatenate([sc, oh], 1)[:, :, None, None] * np.ones((1, 1, 32, 32), np.float32)  # n 5 32 32
    return np.concatenate([maps, extra], 1)  # n 10 32 32


class Net(nn.Module):
    """Mini U-Net 32 -> 16 -> 8 -> 16 -> 32, puis upsample x4 et raffinement -> 128."""

    def __init__(self, cin=10, c=48):
        super().__init__()
        blk = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU(),
                                         nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU())
        self.e1, self.e2, self.e3 = blk(cin, c), blk(c, 2 * c), blk(2 * c, 2 * c)
        self.d2, self.d1 = blk(4 * c, c), blk(2 * c, c)
        self.head = nn.Sequential(nn.Conv2d(c + cin, c, 3, padding=1), nn.GELU(), nn.Conv2d(c, 1, 1))

    def forward(self, x):  # x: b 10 32 32
        e1 = self.e1(x)  # b c 32 32
        e2 = self.e2(F.max_pool2d(e1, 2))  # b 2c 16 16
        e3 = self.e3(F.max_pool2d(e2, 2))  # b 2c 8 8
        d2 = self.d2(torch.cat([F.interpolate(e3, scale_factor=2), e2], 1))  # b 4c 16 16 -> b c 16 16
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2), e1], 1))  # b 2c 32 32 -> b c 32 32
        y = torch.cat([d1, x], 1)  # b c+10 32 32
        y = F.interpolate(y, size=(128, 128), mode="bilinear", align_corners=False)  # b c+10 128 128
        return self.head(y)[:, 0]  # b 128 128


def loss_fn(logit, y):
    p = torch.sigmoid(logit)
    bce = F.binary_cross_entropy_with_logits(logit, y)
    iou = 0
    for pp, yy in ((p, y), (1 - p, 1 - y)):
        i = (pp * yy).flatten(1).sum(1); u = (pp + yy - pp * yy).flatten(1).sum(1)
        iou = iou + (i + 1) / (u + 1)
    return bce + (1 - iou / 2).mean()


def fit(X, Y, epochs, seed, dev, Xv=None, Yv=None):
    torch.manual_seed(seed)
    net = Net(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y.astype(np.float32))
    steps = epochs * ((len(X) + 63) // 64)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(X), generator=g); tot = 0
        for i in range(0, len(X), 64):
            idx = perm[i:i + 64]
            xb = Xt[idx].to(dev); yb = Yt[idx].to(dev)
            if torch.rand(1, generator=g) < 0.5:  # flips (cartes et masque alignés)
                xb, yb = xb.flip(-1), yb.flip(-1)
            if torch.rand(1, generator=g) < 0.5:
                xb, yb = xb.flip(-2), yb.flip(-2)
            l = loss_fn(net(xb), yb); opt.zero_grad(); l.backward(); opt.step(); sch.step(); tot += l.item() * len(idx)
        if Xv is not None and (ep % 10 == 9 or ep == epochs - 1):
            pv = predict(net, Xv, dev) > 0.5
            print(f"  ep {ep + 1} loss {tot / len(X):.4f} val mIoU {miou_np(pv, Yv).mean():.4f}", flush=True)
    return net


@torch.no_grad()
def predict(net, X, dev):
    net.eval()
    return np.concatenate([torch.sigmoid(net(torch.from_numpy(X[i:i + 256]).to(dev))).cpu().numpy()
                           for i in range(0, len(X), 256)])  # n 128 128


def report(tag, s, method, full):
    parts = [f"{tag:38s} all {s.mean():.4f}"]
    for m in METHODS:
        for f in (True, False):
            k = (method == m) & (full == f)
            parts.append(f"{m[:5]}{'F' if f else 'R'} {s[k].mean():.3f}")
    print(" | ".join(parts), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True); ap.add_argument("--real", required=True)
    ap.add_argument("--masks", required=True); ap.add_argument("--meta", required=True)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kfold-real", action="store_true")
    ap.add_argument("--test"); ap.add_argument("--test-meta"); ap.add_argument("--write-sub")
    ap.add_argument("--sub-mode", default="net", choices=["net", "combo"])
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    S = load_feats(a.synth); Xs, Ys = inputs(S), S["mask128"].astype(bool)
    R = load_feats(a.real)
    z = np.load(a.masks); allm = dict(zip(z["names"].tolist(), z["masks"]))
    meta = {m["image_name"]: m for m in json.loads(Path(a.meta).read_text())}
    Yr = np.stack([allm[n] for n in R["names"]]).astype(bool)
    full = np.array([meta[n]["full_watermark"] for n in R["names"]]); method = R["method"]
    Xr = inputs(R)
    print(f"synth {len(Xs)} (F {S['full_watermark'].mean():.2f}, couverture {Ys.mean():.2f}) ; réel {len(Xr)}")

    # test d'égalité de la métrique sur quelques cas
    rng = np.random.default_rng(0)
    pr = rng.random((5, 128, 128)) > 0.5
    for p, g in list(zip(pr, Yr[:5])) + [(np.zeros((128, 128), bool), np.zeros((128, 128), bool))]:
        assert abs(miou_np(p[None], g[None])[0] - miou_ref(p.astype(np.uint8), g.astype(np.uint8))) < 1e-12

    gate = R["acc_full"] >= 0.75
    ones, zeros = np.ones_like(Yr), np.zeros_like(Yr)
    report("tout 0", miou_np(zeros, Yr), method, full)
    rule = np.where(gate[:, None, None], ones, zeros)
    report("règle acc_full>=0.75 -> 1", miou_np(rule, Yr), method, full)

    # référence sans apprentissage : meilleur rectangle du balayage pour les images R (tri par acc_full)
    up = lambda m: F.interpolate(torch.from_numpy(m.astype(np.float32))[:, None], size=(128, 128),
                                 mode="nearest")[:, 0].numpy()  # n 32 32 -> n 128 128
    cb = up(R["maps"][:, 4]) > 0.5
    for t in (0.6, 0.65, 0.7, 0.8):
        pr = np.where(gate[:, None, None], ones, np.where((R["best_acc"] >= t)[:, None, None], cb, zeros))
        report(f"règle + meilleur rect. R si acc>={t}", miou_np(pr, Yr), method, full)

    print("== entraînement synthétique seul, éval réel")
    net = fit(Xs, Ys, a.epochs, a.seed, dev, Xr, Yr)
    P = predict(net, Xr, dev)
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        report(f"net seuil {t}", miou_np(P > t, Yr), method, full)
    combo = np.where(gate[:, None, None], ones, P > 0.5)
    report("combo règle F + net R", miou_np(combo, Yr), method, full)
    combo2 = np.where(gate[:, None, None], P > 0.5, zeros)
    report("combo net F + 0 R", miou_np(combo2, Yr), method, full)

    if a.kfold_real:
        print("== 5-fold réel (+ synthétique), stratifié méthode x full")
        strat = np.array([f"{m}{f}" for m, f in zip(method, full)])
        fold = np.zeros(len(Xr), int)
        for s in np.unique(strat):
            idx = np.where(strat == s)[0]; rng.shuffle(idx); fold[idx] = np.arange(len(idx)) % 5
        Pk = np.zeros_like(P)
        for k in range(5):
            tr = fold != k
            nk = fit(np.concatenate([Xs, np.repeat(Xr[tr], 4, 0)]), np.concatenate([Ys, np.repeat(Yr[tr], 4, 0)]),
                     a.epochs, a.seed + k, dev)
            Pk[fold == k] = predict(nk, Xr[fold == k], dev)
        report("kfold net 0.5", miou_np(Pk > 0.5, Yr), method, full)
        report("kfold combo règle F + net R", miou_np(np.where(gate[:, None, None], ones, Pk > 0.5), Yr), method, full)

    if a.write_sub:
        T = load_feats(a.test)
        Xt = inputs(T)
        # modèle final : synthétique + tout le réel
        nf = fit(np.concatenate([Xs, np.repeat(Xr, 4, 0)]), np.concatenate([Ys, np.repeat(Yr, 4, 0)]),
                 a.epochs, a.seed, dev) if a.kfold_real else net
        Pt = predict(nf, Xt, dev) > 0.5
        if a.sub_mode == "combo":
            Pt = np.where((T["acc_full"] >= 0.75)[:, None, None], True, Pt)
        tmeta = json.loads(Path(a.test_meta).read_text())
        order = [m["image_name"] for m in tmeta]; pos = {n: i for i, n in enumerate(T["names"].tolist())}
        assert len(order) == 1020 and set(order) == set(pos)
        masks = torch.from_numpy(np.stack([Pt[pos[n]] for n in order])).bool()  # 1020 128 128
        torch.save({"image_names": order, "masks": masks}, a.write_sub)
        print("écrit", a.write_sub, tuple(masks.shape), f"{Path(a.write_sub).stat().st_size / 1e6:.1f} Mo")


if __name__ == "__main__":
    main()
