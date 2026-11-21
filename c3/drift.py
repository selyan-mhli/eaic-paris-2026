"""Dérive des candidats c3 vs le modèle d'origine, en fp32 ET fp16 autocast (garde des 70 %).

Complète c3/eval.py (dont le retain est calculé en fp32 seulement et peut être trompé par un
modèle effondré). Charge l'origine une fois, puis pour chaque candidat et chaque jeu :
  off     photos hors forget propres (--images, ex. offforget.npz de make_offforget.py) ;
  offaug  les mêmes, en augmentations fortes (eval.strong_augment, seed 0) ;
  forget  les --n-forget premières forget propres (là où l'écart à l'origine est voulu).
Colonnes (fp16 autocast comme le serveur, sauf mention) :
  fin     logits finis en fp32 et en fp16 (sinon candidat rejeté) ;
  flip    part des top-1 qui changent entre fp32 et fp16 ;
  agree   accord top-1 avec l'origine (origine aussi en fp16) ;
  agreeC  accord top-1 là où l'origine est confiante (max prob > --conf) ;
  kl      KL(p_orig || p_cand) à T=2 ;
  ent     entropie moyenne (T=1, nats) ; maxp : max prob moyenne ;
  top     part de la classe la plus prédite (effondrement vers une classe si élevée).
Proxy seulement : photos hors des 100 classes, pas de labels, pas l'accuracy du set caché.

Usage : python drift.py --images offforget.npz --forget ../forget_images.npz --original ../model.pt c1.pt ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from eval import kl_divergence, load_state_dict, make_timm_model, predict, strong_augment


def stats(lg: np.ndarray) -> dict:
    p = torch.softmax(torch.from_numpy(lg).double(), 1).numpy()
    ent = float(-(p * np.log(np.clip(p, 1e-30, None))).sum(1).mean())
    top = float(np.bincount(lg.argmax(1), minlength=lg.shape[1]).max() / len(lg))
    return {"ent": ent, "maxp": float(p.max(1).mean()), "top": top}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--forget", type=Path, required=True)
    ap.add_argument("--n-forget", type=int, default=1000)
    ap.add_argument("--original", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.9)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("candidates", type=Path, nargs="+")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    off = np.load(a.images)["images"]
    with np.load(a.forget) as z:
        fg = z["images"][:a.n_forget]
    sets = {"off": off, "offaug": strong_augment(off, 0, 1), "forget": fg}
    kw = dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), device=dev, batch_size=a.batch_size)

    def run(path: Path) -> dict:
        m = make_timm_model()
        m.load_state_dict(load_state_dict(path), strict=True)
        return {s: (predict(m, x, fp16=False, **kw), predict(m, x, fp16=True, **kw)) for s, x in sets.items()}

    ref = run(a.original)
    print(f"device {dev} ; off={len(off)} ({a.images}), offaug={len(off)}, forget={len(fg)} ; conf>{a.conf}")
    print("candidat\tjeu\tfin\tflip\tagree\tagreeC\tkl\tent\tmaxp\ttop")
    for c in [a.original, *a.candidates]:
        out = ref if c == a.original else run(c)
        for s in sets:
            l32, l16 = out[s]
            r16 = ref[s][1]
            fin = bool(np.isfinite(l32).all() and np.isfinite(l16).all())
            conf = torch.softmax(torch.from_numpy(r16), 1).numpy().max(1) > a.conf
            same = l16.argmax(1) == r16.argmax(1)
            st = stats(l16)
            print(f"{c.stem}\t{s}\t{fin}\t{(l32.argmax(1) != l16.argmax(1)).mean():.4f}\t{same.mean():.4f}\t"
                  f"{same[conf].mean() if conf.any() else float('nan'):.4f}\t{kl_divergence(r16, l16):.4f}\t"
                  f"{st['ent']:.4f}\t{st['maxp']:.4f}\t{st['top']:.4f}", flush=True)


if __name__ == "__main__":
    main()
