"""Soupes de poids c3 : θ = (1−λ)·θ_origine + λ·θ_autre, en float32, une par λ.

Vérifie avant d'écrire : mêmes clés et formes, tout en float32 ; chaque soupe se charge avec
strict=True dans l'architecture de soumission (timm vit_base_patch16_224, 100 classes,
init_values=1e-4 via eval.make_timm_model). Affiche le sha256 de chaque fichier écrit.
À lancer dans un job (CPU suffit), jamais sur le nœud de login.

Usage : python make_soups.py --original ../model.pt --other ../output/v2_best.pt \
          --lambdas 0.1 0.2 0.3 --prefix soupB --out-dir soups
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from eval import load_state_dict, make_timm_model


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--other", type=Path, required=True)
    p.add_argument("--lambdas", type=float, nargs="+", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    o, b = load_state_dict(a.original), load_state_dict(a.other)
    if o.keys() != b.keys() or any(o[k].shape != b[k].shape for k in o):
        raise SystemExit("clés ou formes différentes entre les deux modèles")
    bad = [k for k in o if o[k].dtype != torch.float32 or b[k].dtype != torch.float32]
    if bad:
        raise SystemExit(f"entrées non float32 : {bad[:3]}")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    model = make_timm_model()
    for lam in a.lambdas:
        sd = {k: ((1 - lam) * o[k] + lam * b[k]).float().contiguous() for k in o}
        model.load_state_dict(sd, strict=True)
        out = a.out_dir / f"{a.prefix}_l{int(round(lam * 100)):02d}.pt"
        torch.save(sd, out)
        print(f"{out}\tlambda={lam}\t{out.stat().st_size} o\tsha256={sha256(out)}", flush=True)


if __name__ == "__main__":
    main()
