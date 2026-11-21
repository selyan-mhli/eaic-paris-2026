"""Jeu « hors forget » pour le proxy retain de c3 (garde des 70 %).

Le proxy retain par défaut de c3/eval.py compare candidat et origine sur des augmentations
fortes des forget : ce sont justement les images où un bon candidat DOIT s'écarter de l'origine,
donc ce proxy sous-estime la sûreté d'un bon candidat. Ici on prend des photos naturelles qui ne
sont pas dans le forget set (images de la tâche watermark, hors distribution des 100 classes),
redimensionnées en 224×224 : l'accord top-1 et la KL avec l'origine y mesurent la dérive
générale du modèle. Ce n'est pas une accuracy et pas le set de validation caché.

Usage (CPU, nœud de login suffisant) :
  python make_offforget.py --src <dossier d'images> [--src ...] --n 1000 --out offforget.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load(path: Path, size: int) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)  # recadrage central carré puis redimensionnement
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return np.asarray(im.resize((size, size), Image.BICUBIC), dtype=np.uint8)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, action="append", required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    files = sorted(f for d in a.src for f in d.rglob("*") if f.suffix.lower() in {".png", ".jpg", ".jpeg"})
    rng = np.random.default_rng(a.seed)
    pick = rng.choice(len(files), size=min(a.n, len(files)), replace=False)
    x = np.stack([load(files[i], a.size) for i in sorted(pick)])
    np.savez(a.out, images=x)
    print(f"{a.out} : {x.shape} {x.dtype} depuis {len(files)} fichiers")


if __name__ == "__main__":
    main()
