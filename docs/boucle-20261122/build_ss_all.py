"""Candidat T1 « SS avec toutes les graines disponibles » (GO Selyan 05:22) : champion c1_ensF_ss6k_tmf.pt (19d971dc…)
où les 340 StegaStamp = moyenne simple (poids 1) de TOUTES les prédictions test pixseg_test_*.npz trouvées
(output/ + t1sr/), seuil 0,5. Les fichiers sans probabilité SS non nulle sont écartés (listés). Reste bit à bit."""
import glob, hashlib, json, sys
import numpy as np, torch
W, A = sys.argv[1], sys.argv[2]
EXCL = sys.argv[3] if len(sys.argv) > 3 else ""   # sous-chaîne à exclure, ex. nt_fit0
OUT = sys.argv[4] if len(sys.argv) > 4 else "c1_ensN_ss_all.pt"
C = torch.load(f"{W}/c1_ensF_ss6k_tmf.pt", weights_only=False); nm = list(C["image_names"])
meta = {r["image_name"]: r["method"] for r in json.load(open(f"{W}/metadata.json"))}
ss = [i for i, n in enumerate(nm) if meta[n] == "stegastamp"]
acc, used, skip = 0, [], []
for f in sorted(glob.glob(f"{A}/pixseg_test_*.npz")):
    if EXCL and EXCL in f: skip.append((f.split("/")[-1], "exclu")); continue
    z = np.load(f)
    if list(z["names"]) != nm: skip.append((f.split("/")[-1], "ordre")); continue
    p = z["probs"][ss].astype(np.float32)
    if not np.isfinite(p).all() or p.max() <= 0: skip.append((f.split("/")[-1], "vide")); continue
    acc = acc + p; used.append(f.split("/")[-1])
m = acc / len(used) > 0.5
out = C["masks"].clone(); out[ss] = torch.from_numpy(m)
ch = [i for i in range(len(nm)) if not torch.equal(out[i], C["masks"][i])]
assert all(meta[nm[i]] == "stegastamp" for i in ch)
p = f"{W}/{OUT}"; torch.save({"image_names": C["image_names"], "masks": out}, p)
print(json.dumps({"n_predictions": len(used), "utilisees": used, "ecartees": skip, "ss_changes": len(ch),
                  "pixels_changes": int((out != C["masks"]).sum()), "sha_out": hashlib.sha256(open(p, "rb").read()).hexdigest()}, indent=0))
