"""Greffe des masques TrustMark dense (export 15905107, base c1_pixss_fit0) sur le champion actuel G.

Contrôles : mêmes image_names et même ordre ; dans l'export, seules des images TrustMark diffèrent de fit0 ;
dans G, les masques TrustMark sont identiques à fit0 (filiation ANALYSE-PROFONDE). On copie alors dans G
les seuls masques TrustMark qui ont changé. Les 680 masques non-TM de G restent intacts bit à bit.

python graft_tm.py --g G.pt --fit0 c1_pixss_fit0.pt --exp c1_tmdense_pixss_<job>.pt --meta metadata.json --out out.pt
"""
import argparse, hashlib, json
import torch


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


ap = argparse.ArgumentParser()
for k in ("g", "fit0", "exp", "meta", "out"):
    ap.add_argument("--" + k, required=True)
a = ap.parse_args()
G, F0, E = (torch.load(p, map_location="cpu", weights_only=False) for p in (a.g, a.fit0, a.exp))
names = list(G["image_names"])
assert names == list(F0["image_names"]) == list(E["image_names"]), "ordre des noms différent"
meta = json.load(open(a.meta))
rows = meta if isinstance(meta, list) else meta.get("images", meta)
if isinstance(rows, dict):
    meth = {n: (v["method"] if isinstance(v, dict) else v) for n, v in rows.items()}
else:
    meth = {r.get("image_name", r.get("name")): r["method"] for r in rows}
tm = [i for i, n in enumerate(names) if str(meth[n]).lower() == "trustmark"]
assert len(tm) == 340, len(tm)
tmset = set(tm)
gm, fm, em = G["masks"], F0["masks"], E["masks"]
assert gm.shape == fm.shape == em.shape and gm.dtype == em.dtype, (gm.shape, em.shape, gm.dtype, em.dtype)
diff_exp = [i for i in range(len(names)) if not torch.equal(em[i], fm[i])]
assert all(i in tmset for i in diff_exp), "l'export modifie une image non TrustMark"
assert all(torch.equal(gm[i], fm[i]) for i in tm), "les TM de G diffèrent de fit0"
out = gm.clone()
for i in diff_exp:
    out[i] = em[i]
nontm_same = all(torch.equal(out[i], gm[i]) for i in range(len(names)) if i not in tmset)
assert nontm_same
torch.save({"image_names": G["image_names"], "masks": out}, a.out)
print(json.dumps({"tm_modifiees": len(diff_exp), "non_tm_identiques": nontm_same, "sha_G": sha(a.g),
                  "sha_fit0": sha(a.fit0), "sha_exp": sha(a.exp), "sha_out": sha(a.out),
                  "taille_octets": __import__("os").path.getsize(a.out)}, indent=1))
