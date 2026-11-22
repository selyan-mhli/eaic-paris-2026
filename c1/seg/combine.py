"""Score global (510 train) et fichier de soumission test : tri acc_full >= 0,75 ; images triées F -> tête trous
(kfold_f.py hors-fold sur le train ; modèle refit sur les 255 F réels pour le test) ; autres -> tout 0, ou
meilleur rectangle du balayage (feats.py) si son accord >= --rect-thr.

  python c1/seg/combine.py --pred c1/seg/data/kfold_f_pred.npz [--thr 0.5] [--rect-thr 0.75] \
      [--write-sub c1/seg/data/seg_f_kfold_v1.pt]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_f
from train import fit, load_feats, miou_np, predict, report

ap = argparse.ArgumentParser()
ap.add_argument("--pred", nargs="+", required=True, help="un ou plusieurs fichiers de probas (moyenne)")
ap.add_argument("--base", nargs="+", help="prédiction de référence pour le gain apparié (mêmes folds)")
ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--thr", type=float, default=0.5)
ap.add_argument("--rect-thr", type=float, default=0.75)
ap.add_argument("--write-sub", help="ATTENTION : refit d'une tête sur les 255 F réels (modèle différent) ; "
                                    "pour l'export du modèle synthétique, utiliser export.py")
ap.add_argument("--epochs", type=int, default=60)
a = ap.parse_args()
D = Path(__file__).resolve().parent / "data"
meta = {m["image_name"]: m for m in json.loads((D / "train_metadata.json").read_text())}
z = np.load(D / "masks_128.npz"); allm = dict(zip(z["names"].tolist(), z["masks"]))
FF = load_feats(str(D / "seg_ff_train_*.npz")); V1 = load_feats(str(D / "seg_feats_train_*.npz"))
names = FF["names"].tolist(); i1 = {n: i for i, n in enumerate(V1["names"].tolist())}
def load_pred(files):
    acc = 0
    for f in files:
        pz = np.load(f); ip = {n: i for i, n in enumerate(pz["names"].tolist())}
        acc = acc + np.stack([pz["P"][ip[n]] for n in names]).astype(np.float32) / len(files)  # 510 128 128
    return acc


P = load_pred(a.pred)
Y = np.stack([allm[n] for n in names]).astype(bool); meth = FF["method"]
full = np.array([meta[n]["full_watermark"] for n in names]); gate = FF["acc_full"] >= 0.75


def rect_masks(V, idx):
    cb = F.interpolate(torch.from_numpy(V["maps"][idx, 4].astype(np.float32))[:, None], size=(128, 128),
                       mode="nearest")[:, 0].numpy() > 0.5  # n 128 128 (meilleur rectangle)
    return np.where((V["best_acc"][idx] >= a.rect_thr)[:, None, None], cb, False)


RM = rect_masks(V1, [i1[n] for n in names])
G = gate[:, None, None]
report("règle 0,452", miou_np(np.broadcast_to(G, Y.shape), Y), meth, full)
report(f"règle F->net({a.thr}), R->0", miou_np(np.where(G, P > a.thr, False), Y), meth, full)
report(f"règle F->1, R->rect({a.rect_thr})", miou_np(np.where(G, True, RM), Y), meth, full)
report(f"règle F->net, R->rect", miou_np(np.where(G, P > a.thr, RM), Y), meth, full)

# validation imbriquée : seuils (tri acc_full, rectangle) choisis sur 4 folds, appliqués au 5e
rng = np.random.default_rng(20261121); strat = np.array([f"{m}{f}" for m, f in zip(meth, full)])
fold = np.zeros(len(Y), int)
for s in np.unique(strat):
    idx = np.where(strat == s)[0]; rng.shuffle(idx); fold[idx] = np.arange(len(idx)) % 5
cb_all = F.interpolate(torch.from_numpy(V1["maps"][[i1[n] for n in names], 4].astype(np.float32))[:, None],
                       size=(128, 128), mode="nearest")[:, 0].numpy() > 0.5
ba = V1["best_acc"][[i1[n] for n in names]]; af = FF["acc_full"]


def nested(P):
    PF = P > a.thr; cache = {}
    for gt_ in (0.6, 0.65, 0.7, 0.75, 0.8):
        for rt in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.01):
            g = (af >= gt_)[:, None, None]
            cache[(gt_, rt)] = miou_np(np.where(g, PF, np.where((ba >= rt)[:, None, None], cb_all, False)), Y)
    oof = np.zeros(len(Y)); chosen = []
    for k in range(5):
        best = max(cache, key=lambda c: cache[c][fold != k].mean()); chosen.append(best)
        oof[fold == k] = cache[best][fold == k]
    return oof, chosen


oof, chosen = nested(P)
report(f"IMBRIQUÉ F->net({a.thr}), R->rect", oof, meth, full)
print("seuils (tri, rect) choisis par fold :", chosen)
if a.base:
    ob, cbase = nested(load_pred(a.base))
    report("IMBRIQUÉ base", ob, meth, full)
    d = oof - ob; bs = np.random.default_rng(0)
    boot = [d[bs.integers(0, len(d), len(d))].mean() for _ in range(a.boot)]
    print(f"gain apparié {d.mean():+.4f} (IC95 bootstrap images [{np.percentile(boot, 2.5):+.4f}, "
          f"{np.percentile(boot, 97.5):+.4f}]) ; sur F {d[full].mean():+.4f}, sur R {d[~full].mean():+.4f} ; "
          f"F seuls (réseau, seuil fixe) {(miou_np(P[full] > a.thr, Y[full]) - miou_np(load_pred(a.base)[full] > a.thr, Y[full])).mean():+.4f}")

if a.write_sub:
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    Fm = full
    X = train_f.inputs({k: v[Fm] for k, v in FF.items()}); Yf = Y[Fm]  # normalisation fixée sur les F réels
    net = fit(X, Yf, a.epochs, 0, dev)
    T = load_feats(str(D / "seg_ff_test_*.npz")); TV = load_feats(str(D / "seg_feats_test_*.npz"))
    tn = T["names"].tolist(); it = {n: i for i, n in enumerate(TV["names"].tolist())}
    Pt = predict(net, train_f.inputs(T), dev) > a.thr  # 1020 128 128
    RT = rect_masks(TV, [it[n] for n in tn])
    Mt = np.where((T["acc_full"] >= 0.75)[:, None, None], Pt, RT)
    order = [m["image_name"] for m in json.loads((D / "test_metadata.json").read_text())]
    pos = {n: i for i, n in enumerate(tn)}
    assert len(order) == 1020 and set(order) == set(pos) and len(set(order)) == 1020
    masks = torch.from_numpy(np.stack([Mt[pos[n]] for n in order])).bool()  # 1020 128 128
    torch.save({"image_names": order, "masks": masks}, a.write_sub)
    chk = torch.load(a.write_sub, weights_only=True)
    assert chk["masks"].shape == (1020, 128, 128) and chk["masks"].dtype == torch.bool
    print("écrit", a.write_sub, tuple(masks.shape), f"{Path(a.write_sub).stat().st_size / 1e6:.1f} Mo",
          f"; test tri F {(T['acc_full'] >= 0.75).sum()} ; couverture moyenne F prédits {Pt[T['acc_full'] >= 0.75].mean():.3f}")
