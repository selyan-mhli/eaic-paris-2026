"""Préflight de export_ncc.sbatch : noms uniques, 1020 images test, canaux ncc_m/ncc_d présents et finis."""
import glob

import numpy as np

for pat, n_exp in [("output/segf_hf_test_*.npz", 1020), ("output/segf_hf_train_*.npz", 510),
                   ("output/seg_ff_test_*.npz", 1020)]:
    zs = [np.load(f) for f in sorted(glob.glob(pat))]
    names = np.concatenate([z["names"] for z in zs]).tolist()
    assert len(names) == n_exp == len(set(names)), (pat, len(names), len(set(names)))
    if "hf" in pat:
        for k in ("ncc_m", "ncc_d"):
            v = np.concatenate([z[k] for z in zs])
            assert np.isfinite(v).all(), (pat, k)
print("préflight ok")
