"""Fusion des shards de maps.py et branchement dans un heats_<split>.npz pour gated_eval.py.

    python work_c1/reembed/merge.py --re 'output/re_train_s*of4.npz' --heats output/heats_train.npz \
        --out output/heats_train_re.npz [--as-key fm_soft]

Ajoute au heats (mêmes noms, même ordre) :
  re_z, re_zg     z-scores bruts (abs inutilisable tel quel par la grille 0..1 de gated_eval ; rel = min-max ok)
  re_zs, re_zgs   sigmoid(z / 2) dans [0, 1] : utilisable en abs et rel par gated_eval
Images absentes des shards : NaN (gated_eval écarte alors la variante pour la méthode concernée si une image
classée F par le tri est absente : calculer avec --gate-heats/--min-acc <= 0.60, la plus petite GATE).
--as-key K : écrit aussi re_zs à la place de la clé K existante (essai sans modifier gated_eval.py ;
la clé d'origine est perdue dans le fichier de sortie, le heats d'origine n'est pas touché).
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

NEW = ("re_z", "re_zg", "re_c")


def load_re(pattern: str) -> dict[str, dict[str, np.ndarray]]:
    files = sorted(glob.glob(pattern)); assert files, pattern
    out: dict[str, dict[str, np.ndarray]] = {}
    for f in files:
        z = np.load(f)
        for j, n in enumerate(z["names"].tolist()):
            if not z["done"][j]:
                continue
            assert n not in out, f"doublon {n} ({f})"
            out[n] = {k: z[k][j] for k in NEW}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", required=True); ap.add_argument("--heats", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--as-key")
    a = ap.parse_args()
    assert a.out.resolve() != a.heats.resolve(), "ne pas écraser le heats d'origine"
    re = load_re(a.re); h = np.load(a.heats); names = h["names"].tolist()
    extra = set(re) - set(names); assert not extra, f"{len(extra)} noms hors heats"
    d = {k: h[k] for k in h.files}
    for k in NEW:
        d[k] = np.stack([re[n][k] if n in re else np.full((128, 128), np.nan, np.float16) for n in names])  # N 128 128
    for k in ("re_z", "re_zg"):
        d[k + "s"] = (1 / (1 + np.exp(-np.clip(d[k].astype(np.float32), -40, 40) / 2))).astype(np.float16)  # N 128 128
    if a.as_key:
        assert a.as_key in d, a.as_key
        d[a.as_key] = d["re_zs"]
    miss = np.array([n not in re for n in names])
    np.savez_compressed(a.out, **d)
    msg = f"{len(re)} cartes / {len(names)} images ; absentes : {int(miss.sum())}"
    if "full_watermark" in d:
        msg += f" (dont F : {int((miss & d['full_watermark'].astype(bool)).sum())})"
    print(msg, "->", a.out)


if __name__ == "__main__":
    main()
