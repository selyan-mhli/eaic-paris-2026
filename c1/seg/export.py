"""Export reproductible d'une soumission test : tête(s) F sauvegardées (moyenne des probabilités si plusieurs),
normalisation des canaux calculée sur le synthétique (comme à l'entraînement), tri acc_full >= --gate,
F -> réseau > --thr, R -> meilleur rectangle du balayage (feats.py) si accord >= --rect, sinon tout 0.

  python c1/seg/export.py --nets c1/seg/data/net_f_seed0.pt --synth 'c1/seg/data/seg_gf_1_*.npz' \
      --out c1/seg/data/seg_fsynth_rect_v1_repro.pt --check c1/seg/data/seg_fsynth_rect_v1.pt
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_f
from train import Net, load_feats, predict

ap = argparse.ArgumentParser()
ap.add_argument("--nets", nargs="+", required=True); ap.add_argument("--synth", required=True)
ap.add_argument("--gate", type=float, default=0.75); ap.add_argument("--rect", type=float, default=0.75)
ap.add_argument("--thr", type=float, default=0.5); ap.add_argument("--out", required=True)
ap.add_argument("--check"); ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
a = ap.parse_args()
D = Path(__file__).resolve().parent / "data"

train_f.inputs(load_feats(a.synth))  # fixe NORM sur le synthétique, comme train_f.py
T = load_feats(str(D / "seg_ff_test_*.npz")); TV = load_feats(str(D / "seg_feats_test_*.npz"))
X = train_f.inputs(T)  # 1020 10 32 32
P = np.zeros((len(X), 128, 128), np.float32)
for f in a.nets:
    net = Net(X.shape[1]).to(a.device); net.load_state_dict(torch.load(f, map_location=a.device, weights_only=True))
    P += predict(net, X, a.device) / len(a.nets)
iv = {n: i for i, n in enumerate(TV["names"].tolist())}
idx = [iv[n] for n in T["names"].tolist()]
cb = F.interpolate(torch.from_numpy(TV["maps"][idx, 4].astype(np.float32))[:, None], size=(128, 128),
                   mode="nearest")[:, 0].numpy() > 0.5  # meilleur rectangle, 1020 128 128
R = np.where((TV["best_acc"][idx] >= a.rect)[:, None, None], cb, False)
M = np.where((T["acc_full"] >= a.gate)[:, None, None], P > a.thr, R)
order = [m["image_name"] for m in json.loads((D / "test_metadata.json").read_text())]
pos = {n: i for i, n in enumerate(T["names"].tolist())}
assert len(order) == 1020 and set(order) == set(pos) and len(set(order)) == 1020
masks = torch.from_numpy(np.stack([M[pos[n]] for n in order])).bool()
torch.save({"image_names": order, "masks": masks}, a.out)
c = torch.load(a.out, weights_only=True)
assert c["masks"].shape == (1020, 128, 128) and c["masks"].dtype == torch.bool
print(f"écrit {a.out} {Path(a.out).stat().st_size / 1e6:.1f} Mo ; F triés {(T['acc_full'] >= a.gate).sum()} ; "
      f"rect R {int((R.reshape(1020, -1).any(1) & (T['acc_full'] < a.gate)).sum())}")
if a.check:
    ref = torch.load(a.check, weights_only=True)
    assert ref["image_names"] == order, "ordre des noms différent"
    diff = (ref["masks"] != masks).sum().item()
    print(f"comparaison avec {a.check} : {diff} cellules différentes sur {masks.numel()} "
          f"({(ref['masks'] != masks).flatten(1).any(1).sum().item()} images touchées)")
