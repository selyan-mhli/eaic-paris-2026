"""Test CPU de head_search.py sur un cache synthétique. Les métriques de référence sont calculées
avec les fonctions kl/forget_stats extraites du VRAI score_like.py (copie figée de l'audit).
    .venv/bin/python docs/task3-work-20261121/mech/test_head_search.py
"""
import ast
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import head_search as hs  # noqa: E402

SRC = HERE.parents[1] / "task3-audit-20261121/sources/work_c3_eval/score_like.py"


def score_like_funcs():
    tree = ast.parse(SRC.read_text())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("kl", "forget_stats")]
    ns = {"torch": torch, "F": F, "T": 2.0}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(SRC), "exec"), ns)
    return ns["kl"], ns["forget_stats"]


def main():
    kl_ref, fs_ref = score_like_funcs()
    g = torch.Generator().manual_seed(0)
    n, d, c = 300, 64, 100
    W0 = torch.randn(c, d, generator=g) * 0.5
    b0 = torch.randn(c, generator=g) * 0.1
    feats = {k: torch.randn(n if k != "off" else 120, d, generator=g) * 2
             for k in ("forget", "altered", "pshuffle", "lowpass32", "mask35", "off")}
    feats["offall"] = torch.randn(150, d, generator=g) * 2
    y = hs.head16(feats["forget"], W0, b0).argmax(1)
    Wp, bp = W0.clone(), b0.clone()
    bp[y.unique()] -= 3.0                                   # « push » factice : tête seule
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        np.savez(t / "c.npz", **{f"f16_{k}": v.numpy() for k, v in feats.items()},
                 f32_forget=feats["forget"].numpy(), y=y.numpy(), W_orig=W0.numpy(), b_orig=b0.numpy(),
                 W_push_best=Wp.numpy(), b_push_best=bp.numpy(), head_only_push_best=np.array(1),
                 W_push_s125=Wp.numpy(), b_push_s125=(bp - 1).numpy(), head_only_push_s125=np.array(1),
                 W_push_s150=Wp.numpy(), b_push_s150=(bp - 2).numpy(), head_only_push_s150=np.array(1))
        cache = hs.Cache(t / "c.npz")
        # TSV de référence construit avec les fonctions de score_like sur des logits fp32
        lines = ["candidat\t" + "\t".join(hs.COLS)]
        for name, (w, b) in {"model": (W0, b0), "push_best": (Wp, bp), "push_s125": (Wp, bp - 1),
                             "push_s150": (Wp, bp - 2)}.items():
            out = {k: hs.head32(v, w, b) for k, v in feats.items()}
            ref = {k: hs.head32(v, W0, b0) for k, v in feats.items()}
            row = ["True", "True", "0.0000"]
            for v in hs.VIEWS:
                st = fs_ref(out[v], y)
                row += [f"{st[s]:.4f}" for s in ("ce_mean", "ce_std", "flip", "ent_T2", "top")]
            for r in hs.RVIEWS:
                row += [f"{float((out[r].argmax(1) == ref[r].argmax(1)).float().mean()):.4f}",
                        f"{1 / (1 + kl_ref(ref[r], out[r])):.4f}"]
            lines.append(name + "\t" + "\t".join(row))
        (t / "ref.txt").write_text("\n".join(lines) + "\n")
        worst = hs.verify(cache, t / "ref.txt")
        assert worst <= 1e-3, worst
        # une colonne faussée doit faire échouer la vérification
        bad = (t / "ref.txt").read_text().replace("push_s125\tTrue\tTrue\t0.0000\t", "push_s125\tTrue\tTrue\t0.0100\t")
        (t / "bad.txt").write_text(bad)
        assert hs.verify(cache, t / "bad.txt") > 1e-3
        r = cache.row(W0, b0)
        assert r["rest_kl"] < 1e-9 and r["ctl_agree"] == 1.0 and r["forget_flip"] == 0.0
        print("OK head_search (émulation fp16 vs score_like fp32 : écart max %.5f)" % worst)


if __name__ == "__main__":
    main()
