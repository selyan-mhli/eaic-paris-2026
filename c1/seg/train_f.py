"""Tête F « trous » : entraînée sur le synthétique gen_f (pool test seulement), évaluée sur les vrais F du train.

  python c1/seg/train_f.py --synth 'c1/seg/data/seg_gf_*.npz' --real 'c1/seg/data/seg_ff_train_*.npz' \
      --masks c1/seg/data/masks_128.npz --meta c1/seg/data/train_metadata.json [--test 'c1/seg/data/seg_ff_test_*.npz' \
      --test-meta c1/seg/data/test_metadata.json --write-sub output/seg_f_v1.pt]

Entrée : 6 cartes feats_f (32x32, sans le canal 5, normalisées par canal sur le synthétique) + acc_full + one-hot = 10 canaux. Sortie : masque 128 (1 = watermarké).
Rapport : mIoU officielle sur les F réels (full_watermark=True) vs tout 1 (0,453), par méthode, et score
global avec R = tout 0 et tri acc_full >= 0,75.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import METHODS, fit, load_feats, miou_np, miou_ref, predict  # noqa: E402


NORM = {}
CH = (0, 1, 2, 3, 4, 6)  # canal 5 (« détection » PixelSeal) exclu : non entraîné (lambda_det = 0, mesuré par detch)


def inputs(d):
    fm = d["fmaps"][:, list(CH)].astype(np.float32)  # n C 32 32
    if not NORM:  # normalisation par canal fixée sur le premier appel (synthétique)
        NORM["mu"] = fm.mean((0, 2, 3), keepdims=True); NORM["sd"] = fm.std((0, 2, 3), keepdims=True) + 1e-6
    fm = ((fm - NORM["mu"]) / NORM["sd"]).clip(-5, 5)
    oh = np.stack([(d["method"] == m) for m in METHODS], 1).astype(np.float32)  # n 3
    sc = np.concatenate([d["acc_full"][:, None], oh], 1)[:, :, None, None] * np.ones((1, 1, 32, 32), np.float32)
    return np.concatenate([fm, sc], 1)  # n 9 32 32


def rep(tag, s, method):
    print(f"{tag:42s} F {s.mean():.4f} | " + " | ".join(f"{m[:5]} {s[method == m].mean():.3f}" for m in METHODS),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True); ap.add_argument("--real", required=True)
    ap.add_argument("--masks", required=True); ap.add_argument("--meta", required=True)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test"); ap.add_argument("--test-meta"); ap.add_argument("--write-sub")
    ap.add_argument("--thr", type=float, default=0.5)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    S = load_feats(a.synth); Xs, Ys = inputs(S), S["mask128"].astype(bool)
    R = load_feats(a.real)
    z = np.load(a.masks); allm = dict(zip(z["names"].tolist(), z["masks"]))
    meta = {m["image_name"]: m for m in json.loads(Path(a.meta).read_text())}
    full = np.array([meta[n]["full_watermark"] for n in R["names"]])
    Yall = np.stack([allm[n] for n in R["names"]]).astype(bool)
    gate = R["acc_full"] >= 0.75
    # pas de fuite : aucune image train dans le pool synthétique
    used = set(S["src"].tolist()) | (set(S["src2"].tolist()) if "src2" in S else set())
    assert not (used & set(meta)), "image train dans le pool synthétique"  # src2 absent des fichiers seed 1 (tracé depuis)
    print(f"synth {len(Xs)} (couverture {Ys.mean():.3f}) ; réel {len(R['names'])}, F {full.sum()}")
    for p, g in zip(np.random.default_rng(0).random((4, 128, 128)) > 0.3, Yall[:4]):
        assert abs(miou_np(p[None], g[None])[0] - miou_ref(p.astype(np.uint8), g.astype(np.uint8))) < 1e-12

    Fm = full
    XF, YF, mF = inputs({k: v[Fm] for k, v in R.items()}), Yall[Fm], R["method"][Fm]
    rep("tout 1 (F réels)", miou_np(np.ones_like(YF), YF), mF)
    # référence sans apprentissage : cellules dont l'occultation 64 px fait peu chuter la marge
    up = lambda m: F.interpolate(torch.from_numpy(m)[:, None], size=(128, 128), mode="bilinear",
                                 align_corners=False)[:, 0].numpy()
    for ch, name in ((1, "occ64"), (2, "occ128"), (0, "dense")):
        u = up(XF[:, ch])
        for q in (0.02, 0.05, 0.1):  # seuil = quantile par image (part prédite propre)
            t = np.quantile(u.reshape(len(u), -1), q, axis=1)[:, None, None]
            rep(f"{name} quantile {q} -> 0", miou_np(u > t, YF), mF)

    print("== entraînement synthétique F, éval F réels")
    net = fit(Xs, Ys, a.epochs, a.seed, dev, XF, YF)
    P = predict(net, XF, dev)
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        rep(f"net seuil {t}", miou_np(P > t, YF), mF)
    # score global : règle (gate) avec F prédits par le net, R = tout 0
    Pp = predict(net, inputs(R), dev)  # 510 128 128 (aucune image train vue à l'entraînement)
    np.savez_compressed(Path(a.synth.split("*")[0]).parent / f"synth_f_pred_seed{a.seed}.npz", names=R["names"],
                        P=Pp.astype(np.float16))
    Pall = Pp > a.thr
    glob = np.where(gate[:, None, None], Pall, False)
    base = np.where(gate[:, None, None], True, False)
    print(f"global 510 : règle 0,452 -> {miou_np(base, Yall).mean():.4f} ; règle avec net pour les gated -> "
          f"{miou_np(glob, Yall).mean():.4f}")
    torch.save(net.state_dict(), Path(a.synth.split("*")[0]).parent / f"net_f_seed{a.seed}.pt")

    if a.write_sub:
        T = load_feats(a.test)
        Pt = predict(net, inputs(T), dev) > a.thr
        Pt = np.where((T["acc_full"] >= 0.75)[:, None, None], Pt, False)  # R : tout 0
        order = [m["image_name"] for m in json.loads(Path(a.test_meta).read_text())]
        pos = {n: i for i, n in enumerate(T["names"].tolist())}
        assert len(order) == 1020 and set(order) == set(pos)
        masks = torch.from_numpy(np.stack([Pt[pos[n]] for n in order])).bool()  # 1020 128 128
        torch.save({"image_names": order, "masks": masks}, a.write_sub)
        print("écrit", a.write_sub, tuple(masks.shape), f"{Path(a.write_sub).stat().st_size / 1e6:.1f} Mo")


if __name__ == "__main__":
    main()
