"""Candidat T1 « PixelSeal F par psall6k » : champion c1_ensF_ss6k_tmf.pt (0,6880, sha 19d971dc…) où seules les
images PixelSeal routées F (couverture du masque E > 0,5, même règle que les 168 TM F du champion) reçoivent
mean(psall6k_fit_s60, s61) > 0,5. Tout le reste est copié bit à bit du champion.

Mesure qui le justifie (OOF folds 0–1, pixseg_cv_psall6k_f{0,1}, seuil 0,5, contre segf_sub_1 = PS du champion) :
PS F +0,079 [IC95 +0,043 ; +0,124] n=34 ; PS R −0,000 [−0,059 ; +0,059] n=34.
"""
import hashlib, json, sys
import numpy as np
import torch

W = sys.argv[1]
C = torch.load(f"{W}/c1_ensF_ss6k_tmf.pt", weights_only=False)
E = torch.load(f"{W}/c1_ensE_ssall3.pt", weights_only=False)
meta = {r["image_name"]: r["method"] for r in json.load(open(f"{W}/metadata.json"))}
nm = list(C["image_names"]); assert nm == list(E["image_names"])
P = [np.load(f"{W}/pixseg_test_psall6k_fit_s{s}.npz") for s in (60, 61)]
for z in P:
    assert list(z["names"]) == nm
prob = (P[0]["probs"].astype(np.float32) + P[1]["probs"].astype(np.float32)) / 2
out = C["masks"].clone()
ps = [i for i, n in enumerate(nm) if meta[n] == "pixelseal"]
routed = ps  # variante : les 340 PixelSeal (GO Selyan 05:0x)
assert all(torch.equal(C["masks"][i], E["masks"][i]) for i in ps)
for i in routed:
    out[i] = torch.from_numpy(prob[i] > 0.5)
changed = [i for i in range(len(nm)) if not torch.equal(out[i], C["masks"][i])]
assert set(changed) <= set(routed)
path = f"{W}/c1_ensI_ps6k_all.pt"
torch.save({"image_names": C["image_names"], "masks": out}, path)
cov_old = np.mean([C["masks"][i].float().mean().item() for i in routed])
cov_new = np.mean([out[i].float().mean().item() for i in routed])
print(json.dumps({"ps_routes": len(routed), "masques_changes": len(changed), "couverture_avant": round(cov_old, 3),
                  "couverture_apres": round(cov_new, 3), "dtype": str(out.dtype), "shape": list(out.shape),
                  "sha_champion": hashlib.sha256(open(f"{W}/c1_ensF_ss6k_tmf.pt", "rb").read()).hexdigest(),
                  "sha_out": hashlib.sha256(open(path, "rb").read()).hexdigest()}, indent=1))
