"""Contrôles complémentaires de score_like pour les checkpoints du job mech (et des références
déjà notées par le serveur), même forward fp16 / eval que le serveur.

Par fichier :
  ctl_agree, ctl_ret   accord top-1 et 1/(1+KL_T2(orig||cand)) sur la moitié CONTRÔLE de
                       offforget_all.npz (mech.split_indices, jamais vue en réparation)
  rep_agree            idem sur la moitié réparation (pour mesurer l'écart réparé / tenu à l'écart)
  ce/std/flip          CE forget propres et altérées (audit.make_altered_copies), pseudo-labels
  ms_eval_target       terme moyenne/écart-type officiel pour le gold hypothétique eval_target
  rest_kl              KL_T2 entre origine et candidat sur les 99 classes NON gagnantes
                       renormalisées (forget propres) : la structure « quelle mauvaise classe » que
                       la baisse du seul gagnant conserve. Substitut non officiel du terme KL-gold.
  overlap_off          nombre d'images de offforget.npz (colonne off de score_like) présentes dans
                       la moitié réparation (contamination possible de cette colonne pour b et d)
Aucun de ces nombres n'est une accuracy ni un score serveur.

Usage (depuis work_c3_eval/) : python mech/mech_holdout.py --off-all offforget_all.npz \
    --off offforget.npz <ckpt.pt ...> > res/hold_<job>_<tag>.txt
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mech import GOLD_MEAN, GOLD_STD, kl_t2, logits_of, split_indices  # noqa: E402

ORIG_CLEAN, ORIG_ALT = (0.0003, 0.0024), (0.0012, 0.0267)


def ms(mean, std, orig):
    d = abs(mean - GOLD_MEAN) + abs(std - GOLD_STD)
    d0 = abs(orig[0] - GOLD_MEAN) + abs(orig[1] - GOLD_STD)
    return 1.0 - min(max(d / d0, 0.0), 1.0)


def rest_kl(ref: torch.Tensor, cand: torch.Tensor, y: torch.Tensor) -> float:
    """KL_T2 sur les classes != y renormalisées. ref, cand [N, C] ; y [N]."""
    mask = F.one_hot(y, ref.shape[1]).bool()
    r = (ref.float() / 2).masked_fill(mask, float("-inf"))
    c = (cand.float() / 2).masked_fill(mask, float("-inf"))
    lp, lq = F.log_softmax(r, 1), F.log_softmax(c, 1)
    p = lp.exp()
    return float(torch.where(mask, torch.zeros_like(p), p * (lp - lq)).sum(1).mean())


def row_hashes(x: np.ndarray) -> list[str]:
    return [hashlib.sha1(im.tobytes()).hexdigest() for im in x]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--off-all", type=Path, default=Path("offforget_all.npz"))
    ap.add_argument("--off", type=Path, default=Path("offforget.npz"))
    ap.add_argument("ckpts", type=Path, nargs="+")
    a = ap.parse_args()
    task = Path.cwd().parent
    sys.path.insert(0, str(task))
    import audit  # noqa: E402
    import task_template as tt  # noqa: E402

    off_all = np.load(a.off_all)["images"]
    rep_idx, ctl_idx = split_indices(len(off_all))
    rep_h = set(row_hashes(off_all[rep_idx]))
    overlap = sum(h in rep_h for h in row_hashes(np.load(a.off)["images"]))
    off_t = torch.from_numpy(off_all)
    forget = tt.load_forget_images(task / "forget_images.npz")
    altered = audit.make_altered_copies(forget)
    views = {"ctl": off_t[torch.from_numpy(ctl_idx)], "rep": off_t[torch.from_numpy(rep_idx)],
             "forget": forget, "altered": altered}

    def load(p):
        m = tt.build_model()
        m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True), strict=True)
        return m.to(tt.DEVICE).eval()

    orig = load(task / "model.pt")
    ref = {k: logits_of(orig, v, tt.preprocess) for k, v in views.items()}
    y = ref["forget"].argmax(1)
    del orig
    print(f"# split réparation={len(rep_idx)} contrôle={len(ctl_idx)} ; overlap_off(offforget.npz ∩ réparation)={overlap}")
    cols = ["candidat", "ctl_agree", "ctl_ret", "rep_agree", "f_ce", "f_std", "f_flip", "a_ce", "a_std",
            "ms_eval_target", "rest_kl"]
    print("\t".join(cols), flush=True)
    for p in a.ckpts:
        try:
            m = load(p)
        except Exception as e:  # noqa: BLE001
            print(f"{p.stem}\tREJET\t{str(e).splitlines()[0][:160]}", flush=True)
            continue
        out = {k: logits_of(m, v, tt.preprocess) for k, v in views.items()}
        ce_f = F.cross_entropy(out["forget"], y, reduction="none")
        ce_a = F.cross_entropy(out["altered"], y, reduction="none")
        msv = (ms(float(ce_f.mean()), float(ce_f.std(unbiased=False)), ORIG_CLEAN)
               + ms(float(ce_a.mean()), float(ce_a.std(unbiased=False)), ORIG_ALT)) / 2
        vals = [float((out["ctl"].argmax(1) == ref["ctl"].argmax(1)).float().mean()),
                float(1 / (1 + kl_t2(ref["ctl"], out["ctl"]).mean())),
                float((out["rep"].argmax(1) == ref["rep"].argmax(1)).float().mean()),
                float(ce_f.mean()), float(ce_f.std(unbiased=False)),
                float((out["forget"].argmax(1) != y).float().mean()),
                float(ce_a.mean()), float(ce_a.std(unbiased=False)), msv, rest_kl(ref["forget"], out["forget"], y)]
        print(p.stem + "\t" + "\t".join(f"{v:.4f}" for v in vals), flush=True)
        del m


if __name__ == "__main__":
    main()
