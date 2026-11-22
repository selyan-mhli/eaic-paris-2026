"""Fichier combiné, généralisation de merge_ss.py à toute méthode : masques de base partout, remplacés pour la
méthode donnée par la moyenne des probabilités (fichiers pixseg_test_*.npz) seuillée.

  python work_pixseg_v3/merge_method.py base.pt sortie.pt --set pixelseal 0.5 a.npz b.npz --set trustmark 0.45 c.npz
Contrôles : dict image_names + masks bool [1020, 128, 128], noms uniques et identiques à la base, 340 images par
méthode, < 20 Mo ; affiche le sha256.
"""
import hashlib, json, os, sys
import numpy as np, torch

base, out, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
sets, cur = [], None
for t in rest:
    if t == "--set":
        cur = []; sets.append(cur)
    else:
        cur.append(t)
sub = torch.load(base, weights_only=True)
names = list(sub["image_names"]); masks = sub["masks"].clone().bool()
assert masks.shape == (1020, 128, 128) and len(set(names)) == 1020
meth = {m["image_name"]: m["method"].lower() for m in json.load(open("release/test/metadata.json"))}
assert set(names) == set(meth)
for method, thr, *preds in sets:
    thr = float(thr); assert preds, method
    idx = [i for i, n in enumerate(names) if meth[n] == method]
    assert len(idx) == 340, (method, len(idx))
    acc = np.zeros((340, 128, 128), np.float32)
    for f in preds:
        z = np.load(f, allow_pickle=True); d = dict(zip(z["names"].tolist(), z["probs"].astype(np.float32)))
        acc += np.stack([d[names[i]] for i in idx])  # 340 128 128 ; KeyError si une image manque
    new = torch.from_numpy(acc / len(preds) > thr)
    changed = float((new != masks[idx]).float().mean())
    masks[idx] = new
    print(f"{method} : {len(preds)} réseaux, seuil {thr}, {changed:.3f} des pixels changés, couverture {float(new.float().mean()):.3f}, "
          f"images toutes à 0 : {int((new.flatten(1).sum(1) == 0).sum())}, toutes à 1 : {int(new.flatten(1).all(1).sum())}")
torch.save({"image_names": names, "masks": masks}, out)
chk = torch.load(out, weights_only=True)
assert list(chk["image_names"]) == list(sub["image_names"]) and chk["masks"].dtype == torch.bool
assert chk["masks"].shape == (1020, 128, 128) and os.path.getsize(out) < 20e6
print(f"{out} : {os.path.getsize(out) / 1e6:.2f} Mo, sha256 {hashlib.sha256(open(out, 'rb').read()).hexdigest()}")
