"""5-fold sur les vrais F du train (sans synthétique, ou avec --synth) : score hors-fold vs tout 1.

  python c1/seg/kfold_f.py [--synth 'c1/seg/data/seg_gf_*.npz'] [--epochs 60]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_f
from train import METHODS, fit, load_feats, miou_np, predict

ap = argparse.ArgumentParser(); ap.add_argument("--synth"); ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--rep", type=int, default=4, help="répétition des vrais F quand on mélange au synthétique")
a = ap.parse_args()
D = Path(__file__).resolve().parent / "data"
dev = "mps" if torch.backends.mps.is_available() else "cpu"
R = load_feats(str(D / "seg_ff_train_*.npz"))
meta = {m["image_name"]: m for m in json.loads((D / "train_metadata.json").read_text())}
z = np.load(D / "masks_128.npz"); allm = dict(zip(z["names"].tolist(), z["masks"]))
full = np.array([meta[n]["full_watermark"] for n in R["names"]])
RF = {k: v[full] for k, v in R.items()}
if a.synth:
    S = load_feats(a.synth); Xs = train_f.inputs(S); Ys = S["mask128"].astype(bool)
X = train_f.inputs(RF); Y = np.stack([allm[n] for n in RF["names"]]).astype(bool); meth = RF["method"]
rng = np.random.default_rng(20261121); fold = np.zeros(len(X), int)
for m in METHODS:
    idx = np.where(meth == m)[0]; rng.shuffle(idx); fold[idx] = np.arange(len(idx)) % 5
P = np.zeros(Y.shape, np.float32)
RN = {k: v[~full] for k, v in R.items()}; XR = train_f.inputs(RN); PR = np.zeros((len(XR), 128, 128), np.float32)
for k in range(5):
    tr = fold != k
    Xt, Yt = X[tr], Y[tr]
    if a.synth:
        Xt = np.concatenate([Xs, np.repeat(Xt, a.rep, 0)]); Yt = np.concatenate([Ys, np.repeat(Yt, a.rep, 0)])
    net = fit(Xt, Yt, a.epochs, k, dev)
    P[fold == k] = predict(net, X[fold == k], dev)
    PR += predict(net, XR, dev) / 5  # images R : jamais vues (entraînement sur F seulement)
train_f.rep("tout 1", miou_np(np.ones_like(Y), Y), meth)
for t in (0.3, 0.4, 0.5, 0.6, 0.7):
    train_f.rep(f"kfold{'+synth' if a.synth else ''} seuil {t}", miou_np(P > t, Y), meth)
np.savez_compressed(D / f"kfold_f_pred{'_synth' if a.synth else ''}.npz", names=np.concatenate([RF["names"], RN["names"]]),
                    P=np.concatenate([P, PR]).astype(np.float16))
