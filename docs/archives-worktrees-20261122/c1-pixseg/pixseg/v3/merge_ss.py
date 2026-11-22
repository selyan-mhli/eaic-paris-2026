"""Fichier combiné : masques de base (tête F + rectangles) partout, remplacés pour StegaStamp par la moyenne
des probabilités pixseg seuillée. Usage : python merge_ss.py base.pt sortie.pt seuil pred1.npz [pred2.npz ...]"""
import json, sys
import numpy as np, torch
base, out, thr, preds = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4:]
sub = torch.load(base, weights_only=True)
names = list(sub["image_names"]); masks = sub["masks"].clone().bool()
meth = {m["image_name"]: m["method"].lower() for m in json.load(open("release/test/metadata.json"))}
acc = None
for f in preds:
    z = np.load(f, allow_pickle=True); d = dict(zip(z["names"].tolist(), z["probs"].astype(np.float32)))
    p = np.stack([d[n] for n in names]); acc = p if acc is None else acc + p
p = acc / len(preds)
idx = [i for i, n in enumerate(names) if "stega" in meth[n]]
assert len(idx) == 340, len(idx)
new = torch.from_numpy(p[idx] > thr)
changed = float((new != masks[idx]).float().mean())
masks[idx] = new
assert masks.shape == (1020, 128, 128) and len(set(names)) == 1020
torch.save({"image_names": names, "masks": masks}, out)
print(f"{out} : {len(preds)} réseaux, seuil {thr}, 340 images StegaStamp, {changed:.3f} des pixels changés, couverture {float(new.float().mean()):.3f}")
