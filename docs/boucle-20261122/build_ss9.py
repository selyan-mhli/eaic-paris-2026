"""Candidat T1 « SS avec plus de graines » (GO Selyan 05:15) : champion c1_ensF_ss6k_tmf.pt (19d971dc…) où les 340
StegaStamp = mean(2·psall6k s60, 2·s61, 2·s62, psall s50, s51, s52) > 0,5 (9 prédictions ; champion = 7 sans s62,
reproduit au pixel près). s62 = t1sr/pixseg_test_relay_psall6k_fit_s62.npz (job j12 15904703). Reste bit à bit."""
import hashlib, json, sys
import numpy as np, torch
W = sys.argv[1]
C = torch.load(f"{W}/c1_ensF_ss6k_tmf.pt", weights_only=False); nm = list(C["image_names"])
meta = {r["image_name"]: r["method"] for r in json.load(open(f"{W}/metadata.json"))}
ss = [i for i, n in enumerate(nm) if meta[n] == "stegastamp"]
w = {"psall6k_fit_s60": 2, "psall6k_fit_s61": 2, "relay_psall6k_fit_s62": 2, "psall_fit_s50": 1, "psall_fit_s51": 1, "psall_fit_s52": 1}
acc = 0
for k, v in w.items():
    z = np.load(f"{W}/pixseg_test_{k}.npz"); assert list(z["names"]) == nm, k
    acc = acc + v * z["probs"][ss].astype(np.float32)
m = acc / sum(w.values()) > 0.5
out = C["masks"].clone(); out[ss] = torch.from_numpy(m)
ch = [i for i in range(len(nm)) if not torch.equal(out[i], C["masks"][i])]
assert all(meta[nm[i]] == "stegastamp" for i in ch)
p = f"{W}/c1_ensL_ss9.pt"; torch.save({"image_names": C["image_names"], "masks": out}, p)
print(json.dumps({"ss_changes": len(ch), "pixels_changes": int((out != C["masks"]).sum()), "sha_out": hashlib.sha256(open(p, "rb").read()).hexdigest()}))
