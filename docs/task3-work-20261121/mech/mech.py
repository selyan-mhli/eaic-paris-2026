"""Comparaison de MÉCANISMES d'oubli (Task 3), une variante par GPU, toutes depuis ../model.pt.

  npo         (a) NPO corrigé « même vue » (unlearn_v2_fixed.same_view_log_ratio), recette fixe
  npo_repair  (b) (a) puis réparation : distillation de l'origine sur la moitié « réparation » des
                  photos hors forget (offforget_all.npz), en gardant le comportement oublié
  conf        (c) cible de confiance par image : tous les logits de l'origine gardés, seul le
                  gagnant abaissé pour atteindre une CE cible par image (lognormale, moyenne 1,1444,
                  écart-type 2,6609 = scénario eval_target), attribuée par rang de difficulté
  conf_repair (d) (c) puis la même réparation que (b)

Régularisation de base IDENTIQUE pour (a) et (c) : KL T=2 vers l'origine sur patch-shuffle des
forget (poids 2) + ancre L2 sur les poids (1e-3). Embeddings gelés, le reste entraîné.

Règle d'arrêt fixée À L'AVANCE (phase d'oubli), évaluée toutes les --eval-every étapes sur un
sous-ensemble fixe de 512 forget propres (jamais sur la moitié de contrôle) :
  npo  : CE moyenne >= 1,1444 OU top-1 changés >= 0,40 OU --max-steps OU budget temps
  conf : CE moyenne >= 0,9 x CE cible moyenne du sous-ensemble OU --max-steps OU budget temps
Réparation : --repair-steps fixes, sans arrêt anticipé.
Checkpoints : <out>/mech_<tag>_<f|r>_s<pas>.pt ; phase d'oubli toutes les 200 étapes + point
d'arrêt, réparation toutes les 100 ; un fichier existant n'est JAMAIS écrasé. Une variante
*_repair contient donc aussi les points de la variante sans réparation (mêmes pas, même graine).
La moitié de contrôle des photos hors forget ne sert ni à entraîner ni à arrêter.

Usage (depuis work_c3_eval/, GPU) : python mech/mech.py --variant conf --tag c
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import unlearn_v2_fixed as uf  # noqa: E402

T = 2.0
GOLD_MEAN, GOLD_STD = 1.1444, 2.6609     # scénario eval_target (official_formula.GOLD_SCENARIOS)
CE_CAP = 15.0
SPLIT_SEED = 20261121


# ---------------------------------------------------------------- fonctions pures (testées CPU)
def split_indices(n: int, seed: int = SPLIT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Partition déterministe par IMAGE : (réparation, contrôle), disjointes, triées."""
    perm = np.random.default_rng(seed).permutation(n)
    half = n // 2
    return np.sort(perm[:half]), np.sort(perm[half:])


def _capped_lognormal(n: int, mu: float, sigma: float, cap: float) -> torch.Tensor:
    q = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    return torch.exp(mu + sigma * torch.special.ndtri(q)).clamp(max=cap)          # croissant


def target_ce_by_rank(difficulty: torch.Tensor, mean: float = GOLD_MEAN, std: float = GOLD_STD,
                      cap: float = CE_CAP) -> torch.Tensor:
    """CE cible par image [N] : quantiles d'une lognormale plafonnée dont (mu, sigma) sont ajustés
    pour que la moyenne et l'écart-type RÉALISÉS sur N valeurs valent (mean, std). Attribution par
    rang de difficulté croissante (la plus difficile reçoit la plus grande CE). Déterministe."""
    n = difficulty.numel()
    best = None
    for sigma in torch.arange(0.3, 3.0, 0.01).tolist():
        lo, hi = -10.0, 5.0
        for _ in range(60):                       # mu tel que la moyenne réalisée = mean
            mid = (lo + hi) / 2
            if _capped_lognormal(n, mid, sigma, cap).mean() < mean:
                lo = mid
            else:
                hi = mid
        v = _capped_lognormal(n, (lo + hi) / 2, sigma, cap)
        err = abs(float(v.std(unbiased=False)) - std)
        if best is None or err < best[0]:
            best = (err, v)
    vals = best[1]
    out = torch.empty(n, dtype=torch.float64)
    out[torch.argsort(difficulty.double(), stable=True)] = vals
    return out.float()


def conf_target(z: torch.Tensor, y: torch.Tensor, ce: torch.Tensor) -> torch.Tensor:
    """z [B, C] logits de l'origine, y [B], ce [B] CE cible : abaisse SEULEMENT z[y] pour que
    CE(z', y) = ce (jamais de hausse). Avec S = logsumexp(z hors y) : z'_y = S - log(exp(ce) - 1)."""
    z = z.float().clone()
    others = z.scatter(1, y[:, None], float("-inf"))
    s = torch.logsumexp(others, dim=1)                                   # B
    zy_new = s - torch.log(torch.expm1(ce.float().clamp_min(1e-6)))      # B
    zy = z.gather(1, y[:, None]).squeeze(1)
    z.scatter_(1, y[:, None], torch.minimum(zy, zy_new)[:, None])
    return z


def kl_t2(ref_logits: torch.Tensor, cand_logits: torch.Tensor) -> torch.Tensor:
    """KL(p_ref || p_cand) à T=2, par ligne [B]."""
    lp = F.log_softmax(ref_logits.float() / T, 1)
    lq = F.log_softmax(cand_logits.float() / T, 1)
    return (lp.exp() * (lp - lq)).sum(1)


def save_new(state: dict, path: Path, model_factory) -> None:
    """Sauvegarde float32 finie, chargeable strict=True ; refuse d'écraser un fichier existant."""
    if path.exists():
        raise FileExistsError(f"refus d'écraser {path}")
    sd = {k: v.detach().cpu().float().contiguous() for k, v in state.items()}
    for k, v in sd.items():
        if not torch.isfinite(v).all():
            raise RuntimeError(f"poids non finis : {k}")
    model_factory().load_state_dict(sd, strict=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    if tmp.exists():
        raise FileExistsError(f"refus d'écraser {tmp}")
    torch.save(sd, tmp)
    tmp.rename(path)


def mild_u8(x_u8: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """uint8 NHWC -> uint8 NHWC : flip aléatoire et luminosité ±10 %."""
    x = x_u8.float()
    flip = torch.rand(len(x), generator=g) < 0.5
    x[flip] = x[flip].flip(2)
    b = 1 + (torch.rand(len(x), 1, 1, 1, generator=g) * 2 - 1) * 0.1
    return (x * b).clamp(0, 255).round().to(torch.uint8)


def repair_batch(off_u8: torch.Tensor, idx: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Une image ET sa transformation dans le même lot : [2B, H, W, 3] = [images ; mild(images)]."""
    imgs = off_u8[idx]
    return torch.cat([imgs, mild_u8(imgs, g)])


# ---------------------------------------------------------------- pertes d'une étape
def npo_loss(student, orig, x_aug, y, beta: float) -> tuple[torch.Tensor, float]:
    lp_s, lp_r = uf.same_view_log_ratio(student, orig, x_aug, y)
    return uf._npo(lp_s, lp_r, beta), float((lp_s.detach() - lp_r).mean())


def conf_loss(student, orig, views: list[torch.Tensor], y, ce) -> torch.Tensor:
    """MSE sommée sur les 100 logits (moyenne sur le lot), vues propres/flip/altérées."""
    tot = 0.0
    for x in views:
        with torch.no_grad(), autocast(x):
            tgt = conf_target(orig(x).float(), y, ce)
        with autocast(x):
            out = student(x).float()
        tot = tot + (out - tgt).square().sum(1).mean()
    return tot / len(views)


def base_reg(student, orig, x_proxy, orig_params, proxy_w: float, anchor_w: float) -> torch.Tensor:
    with torch.no_grad(), autocast(x_proxy):
        ref = orig(x_proxy).float()
    with autocast(x_proxy):
        out = student(x_proxy).float()
    reg = proxy_w * kl_t2(ref, out).mean() * T * T
    if anchor_w:
        reg = reg + anchor_w * uf._anchor(student, orig_params)
    return reg


def repair_loss(student, orig, frozen, x_off, x_forget, keep_w: float) -> torch.Tensor:
    """Distillation de l'origine sur les photos hors forget + maintien du modèle oublié figé
    (`frozen`) sur les forget, à T=2."""
    with torch.no_grad(), autocast(x_off):
        ref_off = orig(x_off).float()
        ref_f = frozen(x_forget).float()
    with autocast(x_off):
        out_off = student(x_off).float()
        out_f = student(x_forget).float()
    return (kl_t2(ref_off, out_off).mean() + keep_w * kl_t2(ref_f, out_f).mean()) * T * T


def autocast(x: torch.Tensor):
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=x.is_cuda)


@torch.no_grad()
def logits_of(model, u8: torch.Tensor, preprocess, bs: int = 128) -> torch.Tensor:
    model.eval()
    out = []
    for i in range(0, len(u8), bs):
        x = preprocess(u8[i:i + bs])
        with autocast(x):
            out.append(model(x).float().cpu())
    return torch.cat(out)


# ---------------------------------------------------------------- boucle (injectable pour le test CPU)
def run(a, *, model_factory, orig, forget_u8, off_u8, preprocess, augment, patch_shuffle, altered_u8,
        device) -> list[Path]:
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    out_dir = Path(a.out_dir)
    saved: list[Path] = []
    orig.eval().requires_grad_(False)
    student = model_factory()
    student.load_state_dict(orig.state_dict())
    student.to(device)
    for n, p in student.named_parameters():
        p.requires_grad_(not (n == "pos_embed" or n.startswith("patch_embed")))
    params = [p for p in student.parameters() if p.requires_grad]
    orig_params = {n: p.detach().float().clone() for n, p in student.named_parameters()}

    clean_logits = logits_of(orig, forget_u8, preprocess)
    y_all = clean_logits.argmax(1)
    n = len(forget_u8)
    sub = torch.randperm(n, generator=torch.Generator().manual_seed(1))[:min(512, n)]
    is_conf = a.variant.startswith("conf")
    ce_all = None
    if is_conf:
        # difficulté = CE moyenne de l'origine sur 4 augmentations fortes (graines fixes)
        diff = torch.zeros(n)
        for k in range(4):
            torch.manual_seed(100 + k)
            for i in range(0, n, 128):
                x = augment(forget_u8[i:i + 128])
                with torch.no_grad(), autocast(x):
                    lg = orig(x).float().cpu()
                diff[i:i + 128] += F.cross_entropy(lg, y_all[i:i + 128], reduction="none") / 4
        ce_all = target_ce_by_rank(diff)
        print(f"cible CE par image : moyenne {ce_all.mean():.4f} écart-type {ce_all.std(unbiased=False):.4f} "
              f"part > ln2 {(ce_all > math.log(2)).float().mean():.3f} (sous-ensemble {ce_all[sub].mean():.4f})",
              flush=True)
        torch.manual_seed(0)

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    t0 = time.perf_counter()

    def report(step, phase):
        lg = logits_of(student, forget_u8[sub], preprocess)
        ce = F.cross_entropy(lg, y_all[sub], reduction="none")
        flip = float((lg.argmax(1) != y_all[sub]).float().mean())
        print(f"[{phase}] step {step}: CE forget(512) {ce.mean():.4f}/{ce.std(unbiased=False):.4f} "
              f"changées {flip:.4f} {time.perf_counter() - t0:.0f}s", flush=True)
        student.train()
        return float(ce.mean()), flip

    def ckpt(phase, step):
        student.eval()
        p = out_dir / f"mech_{a.tag}_{phase}_s{step:04d}.pt"
        save_new(student.state_dict(), p, model_factory)
        saved.append(p)
        print(f"saved {p}", flush=True)
        student.train()

    def opt_step(loss):
        if not torch.isfinite(loss):
            raise RuntimeError("perte non finie")
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    # ---- phase d'oubli
    student.train()
    step, stop = 0, None
    save_forget_every = a.save_every_forget   # (b)/(d) gardent aussi les points de (a)/(c)
    while stop is None:
        order = torch.randperm(n)
        for s in range(0, n - a.batch_size + 1, a.batch_size):
            idx = order[s:s + a.batch_size]
            y = y_all[idx].to(device)
            x_proxy = preprocess(patch_shuffle(forget_u8[idx]))
            if is_conf:
                views = [preprocess(forget_u8[idx]), preprocess(forget_u8[idx].flip(2).contiguous()),
                         preprocess(altered_u8[idx])]
                lf = conf_loss(student, orig, views, y, ce_all[idx].to(device))
                extra = ""
            else:
                lf, dlt = npo_loss(student, orig, augment(forget_u8[idx]), y, a.beta)
                extra = f" delta={dlt:.3f}"
            loss = lf + base_reg(student, orig, x_proxy, orig_params, a.proxy_weight, a.anchor_weight)
            opt_step(loss)
            step += 1
            if step % 25 == 0:
                print(f"step {step} loss_f={float(lf):.4f} loss={float(loss):.4f}{extra} "
                      f"{step / (time.perf_counter() - t0):.2f} step/s", flush=True)
            if step % a.eval_every == 0:
                ce_m, flip = report(step, "f")
                if save_forget_every and step % save_forget_every == 0:
                    ckpt("f", step)
                if is_conf and ce_m >= 0.9 * float(ce_all[sub].mean()):
                    stop = "cible CE atteinte"
                elif not is_conf and (ce_m >= GOLD_MEAN or flip >= 0.40):
                    stop = "seuil CE ou changements atteint"
            if stop is None and step >= a.max_steps:
                stop = "max-steps"
            if stop is None and (time.perf_counter() - t0) / 60 > a.forget_minutes:
                stop = "budget temps"
            if stop:
                break
    print(f"arrêt phase d'oubli au pas {step} : {stop}", flush=True)
    if not saved or saved[-1].name != f"mech_{a.tag}_f_s{step:04d}.pt":
        ckpt("f", step)

    # ---- phase de réparation
    if a.variant.endswith("_repair"):
        rep_idx, ctl_idx = split_indices(len(off_u8))
        print(f"réparation : {len(rep_idx)} photos ; contrôle tenu à l'écart : {len(ctl_idx)} photos", flush=True)
        frozen = model_factory()
        frozen.load_state_dict(student.state_dict())
        frozen.to(device).eval().requires_grad_(False)
        opt = torch.optim.AdamW(params, lr=a.repair_lr, weight_decay=0.0)
        rep = torch.from_numpy(rep_idx)
        half = a.batch_size // 2
        for k in range(1, a.repair_steps + 1):
            oi = rep[torch.randint(0, len(rep), (half,), generator=g)]
            fi = torch.randint(0, n, (half,), generator=g)
            x_off = preprocess(repair_batch(off_u8, oi, g))
            x_f = preprocess(torch.cat([forget_u8[fi], altered_u8[fi]]))
            loss = repair_loss(student, orig, frozen, x_off, x_f, a.keep_weight)
            opt_step(loss)
            if k % 25 == 0:
                print(f"repair {k} loss={float(loss):.4f}", flush=True)
            if k % a.save_every == 0 or k == a.repair_steps:
                report(k, "r")
                ckpt("r", k)
    return saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("npo", "npo_repair", "conf", "conf_repair"), required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="mech_out")
    ap.add_argument("--off-npz", default="offforget_all.npz")
    ap.add_argument("--batch-size", type=int, default=None, help="défaut : 64 (npo), 32 (conf, 3 vues)")
    ap.add_argument("--lr", type=float, default=None, help="défaut : 1e-5 (npo), 3e-5 (conf)")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--proxy-weight", type=float, default=2.0)
    ap.add_argument("--anchor-weight", type=float, default=1e-3)
    ap.add_argument("--max-steps", type=int, default=None, help="défaut : 600 (npo), 800 (conf)")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=100, help="réparation")
    ap.add_argument("--save-every-forget", type=int, default=200)
    ap.add_argument("--forget-minutes", type=float, default=11.0)
    ap.add_argument("--repair-steps", type=int, default=300)
    ap.add_argument("--repair-lr", type=float, default=1e-5)
    ap.add_argument("--keep-weight", type=float, default=1.0)
    a = ap.parse_args()
    if a.lr is None:
        a.lr = 3e-5 if a.variant.startswith("conf") else 1e-5
    if a.batch_size is None:
        a.batch_size = 32 if a.variant.startswith("conf") else 64
    if a.max_steps is None:
        a.max_steps = 800 if a.variant.startswith("conf") else 600
    task = Path.cwd().parent
    sys.path.insert(0, str(task))
    import audit  # noqa: E402
    import task_template as tt  # noqa: E402
    if uf.task_template is None:
        uf.audit, uf.task_template = audit, tt
    print(f"variant={a.variant} tag={a.tag} lr={a.lr:g} max_steps={a.max_steps} bs={a.batch_size}", flush=True)
    orig = tt.load_model(task / "model.pt")
    forget = tt.load_forget_images(task / "forget_images.npz")
    off = torch.from_numpy(np.load(a.off_npz)["images"])
    run(a, model_factory=tt.build_model, orig=orig, forget_u8=forget, off_u8=off,
        preprocess=tt.preprocess, augment=uf._augment_batch, patch_shuffle=audit.patch_shuffle,
        altered_u8=audit.make_altered_copies(forget), device=tt.DEVICE)


if __name__ == "__main__":
    main()
