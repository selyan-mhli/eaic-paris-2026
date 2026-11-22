"""Contrôles séparés « classes concernées » (les 45 classes qui contiennent des forget, d'après
les pseudo-labels de l'origine) / « autres classes », par candidat et par vue.

Jeux :
  forget       3 500 images, toutes dans les classes concernées (groupe « concernée » seulement)
  off_rep      moitié RÉPARATION de offforget_all.npz (mech.split_indices, graine 20261121),
               répartie selon la classe prédite par l'origine sur l'image propre
  off_ctl      moitié CONTRÔLE : seulement avec --final-ctl, écrite dans un fichier À PART
Vues : propre, mild (score_like.mild), lowpass32, mask35 (distill_mix), pshuffle (audit).
Par (jeu, vue, groupe) : n ; accord top-1 avec l'origine SUR LA MÊME VUE ; KL(orig || cand) à T=2 ;
taux de changement = argmax(cand, vue) != pseudo-label ; CE moyenne / écart-type contre le
pseudo-label. Pseudo-label = argmax de l'origine sur l'image PROPRE (convention de score_like
pour les vues altérées des forget), appliqué à toutes les vues.
Ni accuracy, ni score serveur. La cible 1,14/2,66 n'est qu'un scénario proxy.

Usage (depuis work_c3_eval/, 1 GPU) :
  python mech/class_split.py --out res/class_split_<job>.txt [--final-ctl res/ctl_final_<job>.txt] \
      push_scale/push_s125.pt mech_out/15903971/mech_ab_r_s0300.pt mech_out/15903971/mech_ab_r_s0100.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mech import split_indices  # noqa: E402

T = 2.0
METRICS = ("n", "agree", "kl_T2", "change", "ce_mean", "ce_std")


def group_metrics(ref: torch.Tensor, cand: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> dict:
    """ref, cand [N, C] logits sur une même vue ; y [N] pseudo-labels ; mask [N] bool."""
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, **{k: float("nan") for k in METRICS[1:]}}
    r, c, yy = ref[mask].float(), cand[mask].float(), y[mask]
    lp, lq = F.log_softmax(r / T, 1), F.log_softmax(c / T, 1)
    ce = F.cross_entropy(c, yy, reduction="none")
    return {"n": n, "agree": float((c.argmax(1) == r.argmax(1)).float().mean()),
            "kl_T2": float((lp.exp() * (lp - lq)).sum(1).mean()),
            "change": float((c.argmax(1) != yy).float().mean()),
            "ce_mean": float(ce.mean()), "ce_std": float(ce.std(unbiased=False))}


def evaluate(models: dict, datasets: dict, view_fns: dict, logits_fn, concerned: torch.Tensor,
             log=print) -> list[tuple]:
    """models {nom: modèle} (le premier = origine) ; datasets {nom: uint8 NHWC}.
    Retourne [(candidat, jeu, vue, groupe, métriques)]."""
    names = list(models)
    views = {d: {v: fn(x) for v, fn in view_fns.items()} for d, x in datasets.items()}
    rows = []
    ref, y, grp = {}, {}, {}
    for i, name in enumerate(names):
        t0, n_img = time.perf_counter(), 0
        for d, vs in views.items():
            for v, x in vs.items():
                lg = logits_fn(models[name], x)
                n_img += len(x)
                if i == 0:
                    ref[d, v] = lg
                    if v == "clean":
                        y[d] = lg.argmax(1)
                        grp[d] = {"concernee": concerned[y[d]], "autre": ~concerned[y[d]]}
                for g, m in grp[d].items():
                    if d == "forget" and g == "autre":
                        continue
                    rows.append((name, d, v, g, group_metrics(ref[d, v], lg, y[d], m)))
        dt = time.perf_counter() - t0
        log(f"# {name} : {n_img} images en {dt:.1f} s ({n_img / max(dt, 1e-9):.0f} img/s)")
    return rows


def format_rows(rows) -> list[str]:
    out = ["candidat\tjeu\tvue\tgroupe\t" + "\t".join(METRICS)]
    for name, d, v, g, m in rows:
        out.append("\t".join([name, d, v, g, str(m["n"])] + [f"{m[k]:.4f}" for k in METRICS[1:]]))
    return out


def write_new(path: Path, lines: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refus d'écraser {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--final-ctl", type=Path, help="fichier SÉPARÉ pour la moitié contrôle")
    ap.add_argument("--off-all", type=Path, default=Path("offforget_all.npz"))
    ap.add_argument("candidates", type=Path, nargs="+")
    a = ap.parse_args()
    for p in (a.out, a.final_ctl):
        if p is not None and p.exists():
            raise FileExistsError(f"refus d'écraser {p}")
    wd = Path.cwd()
    sys.path.insert(0, str(wd))
    sys.path.insert(0, str(wd.parent))
    import audit  # noqa: E402
    import score_like as sl  # noqa: E402
    import task_template as tt  # noqa: E402
    from distill_mix import _lowpass, _mask  # noqa: E402

    def load(p):
        m = tt.build_model()
        m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True), strict=True)
        return m.to(tt.DEVICE).eval()

    models = {"model": load(wd.parent / "model.pt")}
    for p in a.candidates:
        models[p.stem] = load(p)
    forget = tt.load_forget_images(wd.parent / "forget_images.npz")
    off = np.load(a.off_all)["images"]
    rep, ctl = split_indices(len(off))
    view_fns = {"clean": lambda x: x, "mild": sl.mild, "lowpass32": lambda x: _lowpass(x, 32),
                "mask35": lambda x: _mask(x, 0.35), "pshuffle": audit.patch_shuffle}
    logits_fn = lambda m, x: sl.logits(m, x)  # noqa: E731  (eval, fp16 autocast, comme le serveur)
    y_forget = sl.logits(models["model"], forget).argmax(1)
    n_c = torch.bincount(y_forget, minlength=tt.NUM_CLASSES)
    concerned = n_c > 0
    head = [f"# classes concernées : {int(concerned.sum())} ; forget : {len(forget)} ; "
            f"off réparation : {len(rep)} (contrôle non utilisé ici)",
            "# n_c (classe:effectif, décroissant) : " + " ".join(
                f"{int(c)}:{int(n_c[c])}" for c in torch.argsort(n_c, descending=True) if n_c[c] > 0)]
    print("\n".join(head), flush=True)
    rows = evaluate(models, {"forget": forget, "off_rep": torch.from_numpy(off[rep])}, view_fns, logits_fn,
                    concerned, log=lambda s: (print(s, flush=True), head.append(s)))
    write_new(a.out, head + format_rows(rows))
    print(f"écrit {a.out}", flush=True)
    if a.final_ctl:
        rows = evaluate(models, {"off_ctl": torch.from_numpy(off[ctl])}, view_fns, logits_fn, concerned,
                        log=lambda s: print(s, flush=True))
        write_new(a.final_ctl, [f"# moitié CONTRÔLE ({len(ctl)} photos), lecture unique prévue"] + format_rows(rows))
        print(f"écrit {a.final_ctl}", flush=True)


if __name__ == "__main__":
    main()
