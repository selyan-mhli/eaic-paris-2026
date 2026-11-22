"""Rejoue HORS LIGNE (CPU) les métriques de score_like.py pour n'importe quelle tête (W, b),
à partir du cache de cache_feats.py. Valable seulement pour des éditions limitées à la tête :
le tronc de l'origine est inchangé, donc les features pré-tête aussi.

Tête sous autocast fp16 émulée : features, poids et biais arrondis en fp16, produit en fp32,
sortie arrondie en fp16 (comme une matmul fp16 à accumulation fp32). fp16flip : tête fp32 sur
features fp32 contre la version fp16.

Usage :
  python head_search.py --cache feats_<job>.npz --verify score_15903813_push.txt     # obligatoire
  python head_search.py --cache feats_<job>.npz --bias-sweep --push-scale 1 1.1 1.25 1.4 1.5 \
      --tsv score_offline_<job>.txt
La sortie TSV a les colonnes de score_like (+ rest_kl, ctl_agree, ctl_ret, rep_agree, delta_max) :
elle se lit avec ../analyze_margin.py --extra.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mech import split_indices  # noqa: E402

T = 2.0
VIEWS = ("forget", "altered")
RVIEWS = ("pshuffle", "lowpass32", "mask35", "off")
COLS = ["fmt", "fin", "fp16flip"] + [f"{v}_{s}" for v in VIEWS for s in ("ce_mean", "ce_std", "flip", "ent_T2", "top")] \
    + [f"{r}_{s}" for r in RVIEWS for s in ("agree", "ret")]
EXTRA = ["rest_kl", "ctl_agree", "ctl_ret", "rep_agree", "delta_max"]


def head16(f: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (f.half().float() @ w.half().float().T + b.half().float()).half().float()


def head32(f, w, b):
    return f.float() @ w.float().T + b.float()


# --- copies conformes de score_like.py (kl, forget_stats)
def kl(ref: torch.Tensor, cand: torch.Tensor) -> float:
    p = torch.softmax(ref / T, 1)
    return float((p * (torch.log_softmax(ref / T, 1) - torch.log_softmax(cand / T, 1))).sum(1).mean())


def forget_stats(lg: torch.Tensor, y: torch.Tensor) -> dict:
    ce = F.cross_entropy(lg, y, reduction="none")
    p = torch.softmax(lg / T, 1)
    return {"ce_mean": float(ce.mean()), "ce_std": float(ce.std(unbiased=False)),
            "flip": float((lg.argmax(1) != y).float().mean()),
            "ent_T2": float(-(p * p.clamp_min(1e-12).log()).sum(1).mean()),
            "top": float(torch.bincount(lg.argmax(1), minlength=lg.shape[1]).max() / len(lg))}


def rest_kl(ref, cand, y) -> float:
    mask = F.one_hot(y, ref.shape[1]).bool()
    lp = F.log_softmax((ref / 2).masked_fill(mask, float("-inf")), 1)
    lq = F.log_softmax((cand / 2).masked_fill(mask, float("-inf")), 1)
    return float(torch.where(mask, torch.zeros_like(lp), lp.exp() * (lp - lq)).sum(1).mean())


class Cache:
    def __init__(self, path: Path):
        z = np.load(path)
        self.z = z
        self.f = {k[4:]: torch.from_numpy(z[k]) for k in z.files if k.startswith("f16_")}
        self.f32 = torch.from_numpy(z["f32_forget"])
        self.y = torch.from_numpy(z["y"]).long()
        self.W0, self.b0 = torch.from_numpy(z["W_orig"]), torch.from_numpy(z["b_orig"])
        self.ref = {k: head16(v, self.W0, self.b0) for k, v in self.f.items()}
        if "offall" in self.f:
            rep, ctl = split_indices(len(self.f["offall"]))
            self.rep, self.ctl = torch.from_numpy(rep), torch.from_numpy(ctl)

    def head(self, name: str):
        return torch.from_numpy(self.z[f"W_{name}"]), torch.from_numpy(self.z[f"b_{name}"])

    def row(self, w, b, delta_max: float = float("nan")) -> dict:
        out = {k: head16(v, w, b) for k, v in self.f.items()}
        f32 = head32(self.f32, w, b)
        r = {"fmt": True, "fin": bool(all(torch.isfinite(v).all() for v in out.values())),
             "fp16flip": float((f32.argmax(1) != out["forget"].argmax(1)).float().mean())}
        for v in VIEWS:
            st = forget_stats(out[v], self.y)
            r.update({f"{v}_{s}": st[s] for s in st})
        for k in RVIEWS:
            r[f"{k}_agree"] = float((out[k].argmax(1) == self.ref[k].argmax(1)).float().mean())
            r[f"{k}_ret"] = 1 / (1 + kl(self.ref[k], out[k]))
        r["rest_kl"] = rest_kl(self.ref["forget"], out["forget"], self.y)
        if "offall" in out:
            o, ro = out["offall"], self.ref["offall"]
            r["ctl_agree"] = float((o[self.ctl].argmax(1) == ro[self.ctl].argmax(1)).float().mean())
            r["ctl_ret"] = 1 / (1 + kl(ro[self.ctl], o[self.ctl]))
            r["rep_agree"] = float((o[self.rep].argmax(1) == ro[self.rep].argmax(1)).float().mean())
        r["delta_max"] = delta_max
        return r


def fmt_row(name: str, r: dict) -> str:
    vals = []
    for c in COLS + EXTRA:
        v = r.get(c, float("nan"))
        vals.append(str(v) if isinstance(v, bool) else f"{v:.4f}")
    return name + "\t" + "\t".join(vals)


def read_tsv(path: Path) -> dict:
    rows, head = {}, None
    for line in path.read_text().splitlines():
        p = line.split("\t")
        if p[0] == "candidat":
            head = p
        elif head and len(p) == len(head):
            rows[p[0]] = dict(zip(head[1:], p[1:]))
    return rows


def verify(cache: Cache, tsv: Path, names=("push_best", "push_s125", "push_s150"), tol: float = 1e-3) -> float:
    ref = read_tsv(tsv)
    worst = 0.0
    for n in ("model",) + tuple(names):
        if n == "model":
            w, b = cache.W0, cache.b0
        else:
            w, b = cache.head(n)
            if f"head_only_{n}" in cache.z.files and int(cache.z[f"head_only_{n}"]) != 1:
                raise RuntimeError(f"{n} modifie autre chose que la tête : rejeu impossible")
            if f"full_{n}" in cache.z.files:
                emu = head16(cache.f["forget"], w, b)
                print(f"{n}: |logits émulés - forward complet| max {float((emu - torch.from_numpy(cache.z[f'full_{n}'])).abs().max()):.4f}")
        r = cache.row(w, b)
        diffs = {c: abs(float(ref[n][c]) - float(r[c])) for c in COLS if c not in ("fmt", "fin")}
        c_max = max(diffs, key=diffs.get)
        print(f"{n}: écart max {diffs[c_max]:.5f} ({c_max})")
        worst = max(worst, diffs[c_max])
    print(f"VERIF {'OK' if worst <= tol else 'ECHEC'} : écart max {worst:.5f} (tolérance {tol})")
    return worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--verify", type=Path)
    ap.add_argument("--bias-sweep", action="store_true")
    ap.add_argument("--push-scale", type=float, nargs="*", default=[])
    ap.add_argument("--tsv", type=Path)
    a = ap.parse_args()
    cache = Cache(a.cache)
    if a.verify:
        if verify(cache, a.verify) > 1e-3:
            sys.exit(1)
    lines = ["candidat\t" + "\t".join(COLS + EXTRA), fmt_row("model", cache.row(cache.W0, cache.b0, 0.0))]
    if a.push_scale:
        wb, bb = cache.head("push_best")
        for s in a.push_scale:
            w, b = cache.W0 + s * (wb - cache.W0), cache.b0 + s * (bb - cache.b0)
            lines.append(fmt_row(f"pushx{s:g}", cache.row(w, b)))
    if a.bias_sweep:
        cnt = torch.bincount(cache.y, minlength=len(cache.b0)).float()
        for mode, betas in (("lin", (0.005, 0.01, 0.02, 0.03, 0.045, 0.06, 0.08, 0.10)), ("log", (0.5, 1.0, 1.5, 2.0))):
            for beta in betas:
                d = beta * cnt if mode == "lin" else beta * torch.log1p(cnt)
                lines.append(fmt_row(f"bias_{mode}_{beta:g}", cache.row(cache.W0, cache.b0 - d, float(d.max()))))
    text = "\n".join(lines) + "\n"
    if a.tsv:
        if a.tsv.exists():
            raise FileExistsError(f"refus d'écraser {a.tsv}")
        a.tsv.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
