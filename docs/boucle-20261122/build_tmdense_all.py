"""Candidat T1 « TM dense sur les 340 TrustMark » (GO Selyan 05:10) : champion c1_ensF_ss6k_tmf.pt (0,6880,
sha 19d971dc…) où les 340 TrustMark reçoivent probs_dense > 0,70 (seuil figé du pilote, export 15905139,
fichier *_tmprobs.npz). Le reste est copié bit à bit. Non validé : dense −0,017 contre psall6k sur TM F (OOF f0–1),
jamais mesuré sur TM R (tête apprise sur les seuls F).

python build_tmdense_all.py <dossier> <tmprobs.npz>
"""
import hashlib, json, sys
import numpy as np
import torch

W, TP = sys.argv[1], sys.argv[2]
BASE = sys.argv[3] if len(sys.argv) > 3 else "c1_ensF_ss6k_tmf.pt"        # combiné : c1_ensI_ps6k_all.pt (139242d3…)
OUT = sys.argv[4] if len(sys.argv) > 4 else "c1_ensJ_tmdense_all.pt"
C = torch.load(f"{W}/{BASE}", weights_only=False)
assert hashlib.sha256(open(f"{W}/{BASE}", "rb").read()).hexdigest()[:8] in ("19d971dc", "139242d3"), BASE
meta = {r["image_name"]: r["method"] for r in json.load(open(f"{W}/metadata.json"))}
nm = list(C["image_names"]); pos = {n: i for i, n in enumerate(nm)}
Z = np.load(TP)
tn = list(Z["names"]); thr = float(Z["thr"])
assert len(tn) == 340 and all(meta[n] == "trustmark" for n in tn) and abs(thr - 0.70) < 1e-6
out = C["masks"].clone()
for j, n in enumerate(tn):
    out[pos[n]] = torch.from_numpy(Z["probs"][j].astype(np.float32) > thr)
changed = [i for i in range(len(nm)) if not torch.equal(out[i], C["masks"][i])]
assert all(meta[nm[i]] == "trustmark" for i in changed)
path = f"{W}/{OUT}"
torch.save({"image_names": C["image_names"], "masks": out}, path)
print(json.dumps({"tm": 340, "masques_changes": len(changed), "routes_F_export": int(Z["routed_F"].sum()),
                  "couverture_tm_avant": round(float(C["masks"][[pos[n] for n in tn]].float().mean()), 3),
                  "couverture_tm_apres": round(float(out[[pos[n] for n in tn]].float().mean()), 3),
                  "sha_out": hashlib.sha256(open(path, "rb").read()).hexdigest()}, indent=1))
