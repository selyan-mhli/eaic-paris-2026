"""T3 : distillation conditionnelle à l'image (préparée 22/11 ~06:00, NON lancée par l'agent).

But : garder le comportement de l'enseignant push (push_s175 par défaut) sur les images forget et leurs
variantes, et revenir au modèle d'origine sur les autres images. Possible seulement parce que les forget
sont mémorisées par l'origine (CE 0,0004) : le réseau peut les reconnaître image par image, alors que la
tête linéaire seule (push, t3_repair_head) agit sur des classes entières.

Élève : copie de model.pt ; seuls les K derniers blocs + norm + tête sont entraînés (LayerScale conservé,
même architecture, sauvegarde float32 rechargée en strict=True).
Pertes (T=2, KL(cible || élève), moyenne par image, × T²) :
  (a) forget « fit » (propres + augmentations fortes)            -> enseignant
  (b) photos externes « réparation » (propres + augmentations)    -> origine, poids λ
L'enseignant ne diffère de l'origine que par la tête (vérifié) : ses logits = tête_enseignant(features
origine), donc un seul passage de l'origine donne les deux cibles.

Partitions FIGÉES (reprises exactement du diagnostic t3_repair_head.py) :
  forget : perm = default_rng(20261122).permutation(3500) ; fit = perm[:1750], diag = perm[1750:]
  photos externes (offforget_all.npz, 1530) : perm = default_rng(20261121).permutation(1530) ;
           réparation = perm[:765], contrôle = perm[765:] (contrôle JAMAIS vu à l'entraînement, ni
           pour arrêter, ni pour choisir : il n'est lu qu'à l'évaluation finale).
  Remarque : le contrôle a déjà été consulté par le diagnostic (control_already_consulted=true).
Augmentations d'entraînement (sur fit/réparation uniquement) : recadrage aléatoire + miroir, couleur,
puis une parmi {identité, patch-shuffle à paramètres aléatoires, passe-bas à taille aléatoire, masques
rectangulaires aléatoires, bruit gaussien}. Les variantes générées sd-turbo (gen_out) n'entrent JAMAIS à
l'entraînement : ce sont les seules « altérées » de famille non vue.

Évaluation finale (fp16 autocast comme le serveur), format du diagnostic : accord top-1 et KL T=2 vs
origine ET vs enseignant ; CE pseudo-labels (argmax origine) moyenne/écart-type.

RÈGLE DE DÉCISION FIGÉE (texte RULE ci-dessous, sha256 vérifié au démarrage, voir RULE_SHA256).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

RULE = (
    "Candidat soumettable si et seulement si : "
    "(1) pour chaque ensemble S parmi forget_diag, altered_diag (mild seed 0 sur les forget diag) et "
    "gen_diag (variantes sd-turbo s40+s60 des forget diag, famille jamais vue à l'entraînement) : "
    "|accord_eleve_vs_origine - accord_enseignant_vs_origine| <= 0.10 * accord_enseignant_vs_origine ET "
    "|KL_eleve_vs_origine - KL_enseignant_vs_origine| <= 0.10 * KL_enseignant_vs_origine ; "
    "(2) sur le controle externe : accord vs origine >= 0.90 sur control_affected (photos que l'origine "
    "predit dans les 45 classes concernees) ET KL vs origine <= 0.15 sur control_other. "
    "Sinon non soumettable. Aucune autre metrique ne peut repecher un candidat."
)
RULE_SHA256 = "37f49a303326a1fb4959f53043aac52fb4fb403d74e21705b41a62ae5225e6f2"  # sha256 du texte RULE, vérifié au démarrage

T = 2.0
FORGET_SEED, OFF_SEED = 20261122, 20261121
NUM_CLASSES = 100


# ---------------------------------------------------------------- modèle
def build_model():
    import timm
    return timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=NUM_CLASSES, init_values=1e-4)


def load_sd(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    return sd.get("state_dict", sd) if isinstance(sd, dict) and "state_dict" in sd else sd


def preprocess(u8_nhwc: torch.Tensor, device) -> torch.Tensor:
    """uint8 [B,224,224,3] -> float32 [B,3,224,224] dans [-1,1] (identique à task_template.preprocess)."""
    x = u8_nhwc.to(device).permute(0, 3, 1, 2).float() / 255.0  # b h w c -> b c h w
    return (x - 0.5) / 0.5


def norm_float(x255_nchw: torch.Tensor) -> torch.Tensor:
    """float [B,3,H,W] en 0..255 (déjà quantifié) -> [-1,1]."""
    return (x255_nchw / 255.0 - 0.5) / 0.5


def trunk(model, x, n_frozen: int):
    """Tokens après les n_frozen premiers blocs. x: b 3 224 224 -> b 197 768."""
    x = model.patch_embed(x)                 # b 196 768
    x = model._pos_embed(x)                  # b 197 768 (cls + pos)
    x = model.patch_drop(x)
    x = model.norm_pre(x)
    for blk in model.blocks[:n_frozen]:
        x = blk(x)
    return x


def tail_feats(model, h, n_frozen: int):
    """Blocs n_frozen.. puis norm, pooling, fc_norm : b 197 768 -> b 768 (entrée de model.head)."""
    for blk in model.blocks[n_frozen:]:
        h = blk(h)
    h = model.norm(h)
    h = h[:, 0] if model.global_pool == "token" else h[:, model.num_prefix_tokens:].mean(1)  # b 768
    return model.fc_norm(h)


def head_only_diff(o, b):
    diff = [n for n in o if not torch.equal(o[n], b[n])]
    if not set(diff) <= {"head.weight", "head.bias"}:
        raise ValueError(f"l'enseignant change autre chose que la tête : {diff[:5]}")
    return diff


def make_teacher_head(o, teacher_sd=None, push_best_sd=None, scale=None):
    """(W [100,768], b [100]) float32. Si scale : origine + scale*(push_best - origine) (comme build_t3.py)."""
    if scale is not None:
        head_only_diff(o, push_best_sd)
        W = o["head.weight"].double() + scale * (push_best_sd["head.weight"].double() - o["head.weight"].double())
        b = o["head.bias"].double() + scale * (push_best_sd["head.bias"].double() - o["head.bias"].double())
        return W.float(), b.float()
    head_only_diff(o, teacher_sd)
    return teacher_sd["head.weight"].float().clone(), teacher_sd["head.bias"].float().clone()


# ---------------------------------------------------------------- vues d'évaluation (copies fidèles)
def mild(images: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Copie de score_like.mild : uint8 NHWC -> uint8 NHWC, déterministe (graine séquentielle)."""
    g = torch.Generator().manual_seed(seed)
    x = images.permute(0, 3, 1, 2).float()   # n h w c -> n c h w
    n, _, h, w = x.shape
    out = []
    for i in range(n):
        im = x[i:i + 1]
        if torch.rand(1, generator=g).item() < 0.5:
            im = im.flip(3)
        s = 0.9 + 0.1 * torch.rand(1, generator=g).item()
        ch, cw = int(h * s), int(w * s)
        y0 = int(torch.randint(0, h - ch + 1, (1,), generator=g))
        x0 = int(torch.randint(0, w - cw + 1, (1,), generator=g))
        im = F.interpolate(im[:, :, y0:y0 + ch, x0:x0 + cw], size=(h, w), mode="bilinear", align_corners=False)
        b = 1 + (torch.rand(1, generator=g).item() * 2 - 1) * 0.1
        c = 1 + (torch.rand(1, generator=g).item() * 2 - 1) * 0.1
        m = im.mean()
        out.append(((im - m) * c + m) * b)
    return torch.cat(out).clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()  # n h w c


def patch_shuffle_fixed(images: torch.Tensor, patch: int = 16, seed: int = 0) -> torch.Tensor:
    """Copie de audit.patch_shuffle (SEED=0)."""
    n, h, w, c = images.shape
    gh, gw = h // patch, w // patch
    g = torch.Generator().manual_seed(seed)
    x = images.reshape(n, gh, patch, gw, patch, c).permute(0, 1, 3, 2, 4, 5)  # n gh gw p p c
    x = x.reshape(n, gh * gw, patch * patch * c)                               # n (gh gw) (p p c)
    order = torch.argsort(torch.rand(n, gh * gw, generator=g), dim=1)
    x = torch.gather(x, 1, order[:, :, None].expand_as(x))
    x = x.reshape(n, gh, gw, patch, patch, c).permute(0, 1, 3, 2, 4, 5)       # n gh p gw p c
    return x.reshape(n, h, w, c).contiguous()


def lowpass_fixed(images: torch.Tensor, size: int = 32) -> torch.Tensor:
    """Copie de distill_mix._lowpass (bilinéaire aller-retour)."""
    x = images.permute(0, 3, 1, 2).float()   # n h w c -> n c h w
    small = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    back = F.interpolate(small, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return back.clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()


def mask_fixed(images: torch.Tensor, frac: float = 0.35) -> torch.Tensor:
    """Copie de distill_mix._mask (carré central gris)."""
    x = images.clone()
    _, h, w, _ = x.shape
    side = int(round(h * frac))
    top, left = (h - side) // 2, (w - side) // 2
    x[:, top:top + side, left:left + side] = 128
    return x.contiguous()


# ---------------------------------------------------------------- augmentations d'entraînement (GPU)
def augment(u8: torch.Tensor, gen: torch.Generator, device, p_clean: float) -> torch.Tensor:
    """uint8 NHWC (CPU) -> float [B,3,224,224] normalisé. Aléa par image, graine du job."""
    x = u8.to(device).permute(0, 3, 1, 2).float()   # b h w c -> b c h w, 0..255
    b, _, H, W = x.shape
    r = lambda *s: torch.rand(*s, generator=gen).tolist()  # noqa: E731
    out = []
    for i in range(b):
        im = x[i:i + 1]
        u = r(12)
        if u[0] < p_clean:
            out.append(im)
            continue
        # recadrage aléatoire (aire 0,35..1, ratio 3/4..4/3) + miroir
        area = (0.35 + 0.65 * u[1]) * H * W
        ratio = math.exp(math.log(3 / 4) + (math.log(4 / 3) - math.log(3 / 4)) * u[2])
        ch = int(min(H, round(math.sqrt(area / ratio))))
        cw = int(min(W, round(math.sqrt(area * ratio))))
        y0, x0 = int(u[3] * (H - ch + 1)), int(u[4] * (W - cw + 1))
        im = F.interpolate(im[:, :, y0:y0 + ch, x0:x0 + cw], size=(H, W), mode="bilinear", align_corners=False)
        if u[5] < 0.5:
            im = im.flip(3)
        # couleur : luminosité, contraste, saturation ±0,35
        bri, con, sat = (1 + 0.7 * (v - 0.5) for v in r(3))
        gray = im.mean(1, keepdim=True)
        im = (gray + (im - gray) * sat)
        m = im.mean()
        im = ((im - m) * con + m) * bri
        im = im.clamp(0, 255)
        k = u[6]
        if k < 0.2:      # patch-shuffle aléatoire : taille 16/32/56, fraction 0,3..1 des patchs permutés
            p = [16, 32, 56][min(2, int(u[7] * 3))]
            g_, w_ = H // p, W // p
            t = im[:, :, :g_ * p, :w_ * p].reshape(1, 3, g_, p, w_, p).permute(0, 2, 4, 1, 3, 5)  # 1 g w 3 p p
            t = t.reshape(g_ * w_, 3, p, p)                                                     # (g w) 3 p p
            n_sw = max(2, int((0.3 + 0.7 * u[8]) * g_ * w_))
            sel = torch.randperm(g_ * w_, generator=gen)[:n_sw]
            t[sel] = t[sel[torch.randperm(n_sw, generator=gen)]]
            t = t.reshape(1, g_, w_, 3, p, p).permute(0, 3, 1, 4, 2, 5).reshape(1, 3, g_ * p, w_ * p)  # 1 3 h w
            im = im.clone()
            im[:, :, :g_ * p, :w_ * p] = t
        elif k < 0.4:    # passe-bas : taille 16..96
            s = int(16 + 80 * u[7])
            im = F.interpolate(F.interpolate(im, size=(s, s), mode="bilinear", align_corners=False),
                               size=(H, W), mode="bilinear", align_corners=False)
        elif k < 0.6:    # 1 à 4 rectangles masqués, aire totale ~0,1..0,5, gris ou bruit
            im = im.clone()
            nrect = 1 + int(u[7] * 4)
            tot = 0.1 + 0.4 * u[8]
            for _ in range(nrect):
                a = r(4)
                side = int(math.sqrt(tot / nrect) * H)
                yy, xx = int(a[0] * (H - side + 1)), int(a[1] * (W - side + 1))
                fill = 128.0 if a[2] < 0.5 else torch.rand(1, 3, side, side, generator=gen).to(device) * 255
                im[:, :, yy:yy + side, xx:xx + side] = fill
        elif k < 0.75:   # bruit gaussien sigma 0..25
            im = im + torch.randn(im.shape, generator=gen).to(device) * (25 * u[7])
        out.append(im.clamp(0, 255).round())   # quantification uint8 comme une vraie image
    return norm_float(torch.cat(out))          # b 3 224 224


# ---------------------------------------------------------------- métriques
def stats(ref, cand, labels=None):
    lr, lc = F.log_softmax(ref.double() / T, 1), F.log_softmax(cand.double() / T, 1)
    out = dict(n=int(len(ref)), agreement=float((ref.argmax(1) == cand.argmax(1)).double().mean()),
               KL=float((lr.exp() * (lr - lc)).sum(1).mean()))
    if labels is not None:
        ce = -F.log_softmax(cand.double(), 1)[torch.arange(len(cand)), labels]
        out.update(CE_pseudo_mean=float(ce.mean()), CE_pseudo_std=float(ce.std(unbiased=False)),
                   flip_pseudo=float((cand.argmax(1) != labels).double().mean()))
    return out


@torch.no_grad()
def feats_logits(model, u8, device, bs=128):
    """Features pré-tête et logits sous autocast fp16 (serveur). u8 n h w c -> (n 768, n 100)."""
    model.eval()
    fs, ls = [], []
    for s in range(0, len(u8), bs):
        x = preprocess(u8[s:s + bs], device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=x.is_cuda):
            f = tail_feats(model, trunk(model, x, 0), 0)
            lg = model.head(f)
        fs.append(f.float().cpu()); ls.append(lg.float().cpu())
    return torch.cat(fs), torch.cat(ls)


def kd_loss(student_logits, target_logits):
    """KL(p_cible || p_élève) à T, × T², moyenne par image."""
    lt = F.log_softmax(target_logits.float() / T, 1)
    ls = F.log_softmax(student_logits.float() / T, 1)
    return (lt.exp() * (lt - ls)).sum(1).mean() * T * T


def decide(m, tol=0.10):
    checks = {}
    for s in ("forget_diag", "altered_diag", "gen_diag"):
        if s not in m["student_vs_orig"]:
            checks[s] = dict(ok=False, why="ensemble absent")
            continue
        st, te = m["student_vs_orig"][s], m["teacher_vs_orig"][s]
        da = abs(st["agreement"] - te["agreement"]); dk = abs(st["KL"] - te["KL"])
        checks[s] = dict(ok=bool(da <= tol * te["agreement"] and dk <= tol * te["KL"]),
                         d_agreement=da, lim_agreement=tol * te["agreement"], d_KL=dk, lim_KL=tol * te["KL"])
    ca, co = m["student_vs_orig"].get("control_affected"), m["student_vs_orig"].get("control_other")
    if ca is None or co is None:
        checks["control"] = dict(ok=False, why="contrôle affected/other absent")
    else:
        checks["control"] = dict(ok=bool(ca["agreement"] >= 0.90 and co["KL"] <= 0.15),
                                 control_affected_agreement=ca["agreement"], control_other_KL=co["KL"])
    return dict(submittable=all(c["ok"] for c in checks.values()), checks=checks)


# ---------------------------------------------------------------- principal
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, help="push_s175.pt (enseignant direct)")
    ap.add_argument("--push-best", type=Path, help="push_best.pt, requis avec --teacher-scale")
    ap.add_argument("--teacher-scale", type=float, default=None, help="enseignant = orig + k*(push_best-orig)")
    ap.add_argument("--forget", type=Path, required=True)
    ap.add_argument("--off-all", type=Path, required=True)
    ap.add_argument("--gen-glob", default="", help="gen_out/var_*.npz (évaluation seulement)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--K", type=int, default=4, choices=[1, 2, 3, 4, 6])
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--minutes", type=float, default=17.0, help="budget d'entraînement (l'éval vient après)")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--bs-forget", type=int, default=64)
    ap.add_argument("--bs-off", type=int, default=64)
    ap.add_argument("--p-clean", type=float, default=0.25)
    ap.add_argument("--init-head", choices=["orig", "teacher"], default="orig")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--tiny", action="store_true", help="test CPU : sous-ensembles minuscules, 2 pas")
    a = ap.parse_args()

    if hashlib.sha256(RULE.encode()).hexdigest() != RULE_SHA256:
        raise SystemExit("RULE modifiée : sha256 différent de RULE_SHA256, refus")
    out = a.out_dir / a.tag
    if out.exists():
        raise FileExistsError(f"refus d'écraser {out}")
    t0 = time.monotonic()
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = device.type == "cuda"

    # -- données et partitions figées
    with np.load(a.forget) as z:
        forget = torch.from_numpy(z["images"].copy())          # 3500 224 224 3 u8
    with np.load(a.off_all) as z:
        off = torch.from_numpy(z["images"].copy())             # 1530 224 224 3 u8
    nf, no = len(forget), len(off)
    perm_f = np.random.default_rng(FORGET_SEED).permutation(nf)
    fit_f, diag_f = np.sort(perm_f[:nf // 2]), np.sort(perm_f[nf // 2:])
    perm_o = np.random.default_rng(OFF_SEED).permutation(no)
    rep_o, ctl_o = np.sort(perm_o[:no // 2]), np.sort(perm_o[no // 2:])
    assert not set(fit_f) & set(diag_f) and not set(rep_o) & set(ctl_o)
    gen_imgs = []
    diag_set = set(diag_f.tolist())
    for f in sorted(glob.glob(a.gen_glob)) if a.gen_glob else []:
        with np.load(f) as z:
            keep = np.array([i in diag_set for i in z["idx"]])
            for key in ("img_s40", "img_s60"):
                if key in z.files and keep.any():
                    gen_imgs.append(z[key][keep])
    gen_diag = torch.from_numpy(np.concatenate(gen_imgs)) if gen_imgs else None

    if a.tiny:
        fit_f, diag_f, rep_o, ctl_o = fit_f[:6], diag_f[:6], rep_o[:6], ctl_o[:6]
        gen_diag = gen_diag[:6] if gen_diag is not None else None
        a.steps, a.bs_forget, a.bs_off, a.warmup, a.log_every = 2, 2, 2, 1, 1

    # -- modèles
    o_sd = load_sd(a.model)
    orig = build_model(); orig.load_state_dict(o_sd, strict=True); orig.to(device).eval()
    if a.teacher_scale is not None:
        Wt, bt = make_teacher_head(o_sd, push_best_sd=load_sd(a.push_best), scale=a.teacher_scale)
        teacher_desc = f"orig + {a.teacher_scale}*(push_best - orig) ; push_best={a.push_best}"
    else:
        Wt, bt = make_teacher_head(o_sd, teacher_sd=load_sd(a.teacher))
        teacher_desc = f"tête de {a.teacher}"
    Wt, bt = Wt.to(device), bt.to(device)
    for p in orig.parameters():
        p.requires_grad_(False)
    student = build_model(); student.load_state_dict(o_sd, strict=True); student.to(device)
    if a.init_head == "teacher":
        with torch.no_grad():
            student.head.weight.copy_(Wt); student.head.bias.copy_(bt)
    nfz = len(student.blocks) - a.K
    train_names = []
    for n, p in student.named_parameters():
        tr = (n.startswith("blocks.") and int(n.split(".")[1]) >= nfz) or n.startswith("norm.") \
            or n.startswith("fc_norm.") or n.startswith("head.")
        p.requires_grad_(tr)
        if tr:
            train_names.append(n)
    params = [p for p in student.parameters() if p.requires_grad]
    print(f"[{a.tag}] device={device} K={a.K} λ={a.lam} lr={a.lr} enseignant: {teacher_desc}; "
          f"{sum(p.numel() for p in params)/1e6:.1f} M paramètres entraînés ; fit={len(fit_f)} rep={len(rep_o)}",
          flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, a.steps)))))
    gen = torch.Generator().manual_seed(1000 + a.seed)

    # -- entraînement : trunk gelé partagé (no_grad), queue origine (no_grad), queue élève (grad)
    student.train()
    for blk in student.blocks[:nfz]:
        blk.eval()   # pas de dropout ici de toute façon (drop_rate 0), explicite
    log, step, t_train = [], 0, time.monotonic()
    step_times = []
    while step < a.steps and (time.monotonic() - t_train) < a.minutes * 60:
        ts = time.monotonic()
        i_f = torch.from_numpy(fit_f[torch.randint(len(fit_f), (a.bs_forget,), generator=gen).numpy()])
        i_o = torch.from_numpy(rep_o[torch.randint(len(rep_o), (a.bs_off,), generator=gen).numpy()])
        x = torch.cat([augment(forget[i_f], gen, device, a.p_clean),
                       augment(off[i_o], gen, device, a.p_clean)])             # (bf+bo) 3 224 224
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cuda):
            with torch.no_grad():
                h = trunk(orig, x, nfz)                                          # B 197 768
                fo = tail_feats(orig, h, nfz)                                     # B 768
            ls = student.head(tail_feats(student, h, nfz))                       # B 100
        with torch.no_grad():   # cibles en fp32, hors autocast (pas d'arrondi bf16 des logits cibles)
            fo = fo.float()
            lo = F.linear(fo, orig.head.weight, orig.head.bias)                   # B 100 origine
            lt = F.linear(fo, Wt, bt)                                             # B 100 enseignant
        bf = a.bs_forget
        loss_f = kd_loss(ls[:bf], lt[:bf])
        loss_o = kd_loss(ls[bf:], lo[bf:])
        loss = loss_f + a.lam * loss_o
        if not torch.isfinite(loss):
            raise FloatingPointError(f"perte non finie au pas {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); step += 1
        if cuda:
            torch.cuda.synchronize()
        step_times.append(time.monotonic() - ts)
        if step % a.log_every == 0 or step == 1:
            rec = dict(step=step, loss_forget=float(loss_f), loss_off=float(loss_o), grad_norm=float(gn),
                       lr=sched.get_last_lr()[0], s_per_step=float(np.mean(step_times[-a.log_every:])),
                       elapsed_min=(time.monotonic() - t0) / 60)
            log.append(rec); print(json.dumps(rec), flush=True)
    student.eval()

    # -- vérification : blocs gelés inchangés
    s_sd = {k: v.detach().cpu().float() for k, v in student.state_dict().items()}
    changed = [k for k in o_sd if not torch.equal(o_sd[k].float(), s_sd[k])]
    assert all(k in train_names for k in changed), f"paramètre gelé modifié : {[k for k in changed if k not in train_names][:3]}"
    assert all(torch.isfinite(v).all() for v in s_sd.values() if v.is_floating_point())

    # -- évaluation finale
    # pseudo-labels = argmax origine (fp16) ; classes concernées = celles prédites par l'origine sur les forget
    y = feats_logits(orig, forget[torch.from_numpy(np.concatenate([fit_f, diag_f]))], device)[1].argmax(1)
    y_fit, y_diag = y[:len(fit_f)], y[len(fit_f):]
    affected = torch.unique(y)       # complet = 3500 forget (fit ∪ diag) hors mode --tiny
    alt_all = mild(forget)                                       # identique à la vue « altered » du diagnostic
    views = {"forget_fit": (forget[torch.from_numpy(fit_f)], y_fit),
             "forget_diag": (forget[torch.from_numpy(diag_f)], y_diag),
             "altered_diag": (alt_all[torch.from_numpy(diag_f)], y_diag),
             "altered_fit": (alt_all[torch.from_numpy(fit_f)], y_fit)}
    fd = forget[torch.from_numpy(diag_f)]
    views["pshuffle_diag"] = (patch_shuffle_fixed(forget)[torch.from_numpy(diag_f)], None)
    views["lowpass32_diag"] = (lowpass_fixed(fd, 32), None)
    views["mask35_diag"] = (mask_fixed(fd, 0.35), None)
    if gen_diag is not None:
        views["gen_diag"] = (gen_diag, None)
    io = np.concatenate([rep_o, ctl_o])
    is_aff = np.zeros(no, dtype=bool)
    is_aff[io] = torch.isin(feats_logits(orig, off[torch.from_numpy(io)], device)[1].argmax(1), affected).numpy()
    for name, ix in (("repair", rep_o), ("control", ctl_o)):
        for flag, suf in ((True, "affected"), (False, "other")):
            rows = ix[is_aff[ix] == flag]
            if len(rows):
                views[f"{name}_{suf}"] = (off[torch.from_numpy(rows)], None)
    res = {"student_vs_orig": {}, "teacher_vs_orig": {}, "student_vs_teacher": {}, "student_pseudo": {},
           "teacher_pseudo": {}}
    for name, (u8, lab) in views.items():
        fo, lo = feats_logits(orig, u8, device)
        lt = fo @ Wt.T.cpu() + bt.cpu()
        ls = feats_logits(student, u8, device)[1]
        res["student_vs_orig"][name] = stats(lo, ls)
        res["teacher_vs_orig"][name] = stats(lo, lt)
        res["student_vs_teacher"][name] = stats(lt, ls)
        if lab is not None:
            res["student_pseudo"][name] = stats(lo, ls, lab)
            res["teacher_pseudo"][name] = stats(lo, lt, lab)
    decision = decide(res)

    # -- sauvegarde + rechargement strict
    out.mkdir(parents=True)
    pt = out / "student.pt"
    torch.save(s_sd, pt)
    chk = build_model(); chk.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True), strict=True)
    assert all(v.dtype == torch.float32 for v in chk.state_dict().values() if v.is_floating_point())
    sha = hashlib.sha256(pt.read_bytes()).hexdigest()
    report = dict(tag=a.tag, args={k: str(v) for k, v in vars(a).items()}, teacher=teacher_desc,
                  rule=RULE, rule_sha256=RULE_SHA256,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  partitions=dict(forget_seed=FORGET_SEED, off_seed=OFF_SEED, fit=len(fit_f), diag=len(diag_f),
                                  repair=len(rep_o), control=len(ctl_o), n_affected_classes=int(len(affected))),
                  steps_done=step, train_log=log,
                  s_per_step_mean=float(np.mean(step_times[1:] if len(step_times) > 1 else step_times)),
                  metrics=res, decision=decision, student_pt=str(pt), student_sha256=sha,
                  strict_reload_ok=True, elapsed_min=(time.monotonic() - t0) / 60,
                  note="Proxy local sur photos externes ; aucun gold, aucune accuracy retain, pas un score serveur.")
    (out / "eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(dict(tag=a.tag, decision=decision, sha=sha, steps=step,
                          elapsed_min=report["elapsed_min"]), ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
