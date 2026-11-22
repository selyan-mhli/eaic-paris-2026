"""T1 + T2 (c3/VEILLE.md) : unlearning par cible « jamais vue » calibrée, avec garde-fou sur photos hors jeu.

Constat (work_c3_eval/res/score_15902463_m*.txt, 21/11) : l'origine a une CE forget moyenne de 0,0003 (mémorisation
totale). Les distillations vers un consensus de corruptions ne la montent qu'à 0,002-0,03, alors qu'un modèle gold
qui n'a jamais vu ces images est sans doute bien plus haut [supposé ~0,5-1]. Pendant ce temps, l'accord sur les photos
hors forget tombe déjà à 0,67-0,90. Ici :

T1, cible calibrée. Pour chaque image forget i :
    p_i = moyenne sur K augmentations fortes de softmax(origine(aug(x_i)))      (classes confusables naturelles)
    q_i = softmax(log p_i / tau)  avec un tau GLOBAL choisi par dichotomie pour que la CE moyenne de q
          contre le pseudo-label (argmax de l'origine sur x_i propre) vaille --target-ce.
  L'élève minimise KL(q_i ‖ softmax(z)) sur x_i et sur des altérations légères de x_i (flip, crop 0,85-1,
  luminosité/contraste ±10 %), proches des copies altérées cachées du serveur.
T2, garde-fou. On minimise KL(softmax(origine/2) ‖ softmax(élève/2)) sur des photos hors forget (--ood, par
  exemple offforget.npz) recadrées et retournées au hasard, avec le poids --w-ood, et on ajoute
  lambda·||θ − θ0||² sur les paramètres entraînables. Seuls la tête, la norme finale et les --blocks derniers
  blocs sont entraînés.
Arrêt : on sauvegarde à chaque fin d'époque. On s'arrête dès que la CE forget moyenne mesurée (fp16, mode eval,
  pseudo-labels) atteint --target-ce × --stop-frac.

Sortie : <out>/t1t2_<tag>_e<k>.pt (state dict float32, strict=True sur l'architecture de soumission) et un
journal par époque (CE forget moyenne/écart-type, accord et KL sur l'OOD tenu à part).

    python t1t2.py --task-dir .. --ood ../work_c3_eval/offforget.npz --target-ce 0.6 --tag ce06 --out out
    python t1t2.py --selftest      # CPU, ViT minuscule à poids aléatoires, données factices
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

T_KL = 2.0
MEAN = STD = 0.5


# ---------------------------------------------------------------- données

def to_x(u8: torch.Tensor) -> torch.Tensor:
    """uint8 [B, H, W, 3] -> float [B, 3, H, W] normalisé comme task_template.preprocess."""
    return (u8.permute(0, 3, 1, 2).float() / 255 - MEAN) / STD


def strong_aug(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """x float [B,3,H,W] dans [-1,1] -> crop 30-80 % redimensionné, flip, couleur, bruit (par image)."""
    b, _, h, w = x.shape
    out = torch.empty_like(x)
    for i in range(b):
        s = 0.30 + 0.50 * torch.rand(1, generator=g).item()
        ch, cw = max(16, int(h * math.sqrt(s))), max(16, int(w * math.sqrt(s)))
        y0 = int(torch.randint(0, h - ch + 1, (1,), generator=g)); x0 = int(torch.randint(0, w - cw + 1, (1,), generator=g))
        im = F.interpolate(x[i:i + 1, :, y0:y0 + ch, x0:x0 + cw], size=(h, w), mode="bilinear", align_corners=False)
        if torch.rand(1, generator=g).item() < 0.5:
            im = im.flip(3)
        c = 0.6 + 0.8 * torch.rand(1, generator=g).item(); bb = 0.3 * (torch.rand(1, generator=g).item() - 0.5)
        im = (im - im.mean()) * c + im.mean() + bb + 0.1 * torch.randn(im.shape, generator=g)
        out[i] = im[0]
    return out.clamp(-1, 1)


def mild_aug(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Altérations légères proches des copies cachées : flip, crop 0,85-1, luminosité/contraste ±10 %."""
    b, _, h, w = x.shape
    out = torch.empty_like(x)
    for i in range(b):
        s = 0.85 + 0.15 * torch.rand(1, generator=g).item()
        ch, cw = int(h * s), int(w * s)
        y0 = int(torch.randint(0, h - ch + 1, (1,), generator=g)); x0 = int(torch.randint(0, w - cw + 1, (1,), generator=g))
        im = F.interpolate(x[i:i + 1, :, y0:y0 + ch, x0:x0 + cw], size=(h, w), mode="bilinear", align_corners=False)
        if torch.rand(1, generator=g).item() < 0.5:
            im = im.flip(3)
        c = 0.9 + 0.2 * torch.rand(1, generator=g).item(); bb = 0.2 * (torch.rand(1, generator=g).item() - 0.5)
        out[i] = ((im - im.mean()) * c + im.mean() + bb)[0]
    return out.clamp(-1, 1)


def ood_aug(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    b, _, h, w = x.shape
    out = torch.empty_like(x)
    for i in range(b):
        s = 0.5 + 0.5 * torch.rand(1, generator=g).item()
        ch, cw = int(h * s), int(w * s)
        y0 = int(torch.randint(0, h - ch + 1, (1,), generator=g)); x0 = int(torch.randint(0, w - cw + 1, (1,), generator=g))
        im = F.interpolate(x[i:i + 1, :, y0:y0 + ch, x0:x0 + cw], size=(h, w), mode="bilinear", align_corners=False)
        out[i] = (im.flip(3) if torch.rand(1, generator=g).item() < 0.5 else im)[0]
    return out


# ---------------------------------------------------------------- cible calibrée (T1)

@torch.no_grad()
def logits_of(model, u8: torch.Tensor, dev, bs=128, fp16=True, tf=None, g=None) -> torch.Tensor:
    model.eval(); out = []
    for s in range(0, len(u8), bs):
        x = to_x(u8[s:s + bs])
        if tf is not None:
            x = tf(x, g)
        with torch.autocast(device_type=torch.device(dev).type, dtype=torch.float16, enabled=fp16 and dev != "cpu"):
            out.append(model(x.to(dev)).float().cpu())
    return torch.cat(out)


def calibrate(p_aug: torch.Tensor, y: torch.Tensor, target_ce: float, iters: int = 60, mode: str = "global"):
    """q = softmax(log p / tau), tau global tel que mean CE(q, y) = target_ce. -> (q, tau, ce atteinte).

    mode="partial" : tau n'est appliqué qu'aux images où l'origine reste confiante sous augmentation
    (CE(p_i, y) < target_ce) ; les images où elle hésite gardent le consensus p_i tel quel (queue de la
    distribution, classes changées comprises). tau est alors choisi pour que la moyenne globale vaille la cible."""
    lp = torch.log(p_aug.clamp_min(1e-12))
    ce_p = -torch.log(p_aug.gather(1, y[:, None]).clamp_min(1e-12))[:, 0]
    soft = torch.ones(len(y), dtype=torch.bool) if mode == "global" else ce_p < target_ce

    def make(tau):
        q = torch.softmax(lp, 1).clone()
        q[soft] = torch.softmax(lp[soft] / tau, 1)
        return q

    def ce(tau):
        q = make(tau)
        return float(-torch.log(q.gather(1, y[:, None]).clamp_min(1e-12)).mean())
    lo, hi = 1e-2, 1e3                                  # la CE croît avec tau (cible plus plate)
    if ce(lo) > target_ce:
        tau = lo
    elif ce(hi) < target_ce:
        tau = hi
    else:
        for _ in range(iters):
            mid = math.sqrt(lo * hi)
            lo, hi = (mid, hi) if ce(mid) < target_ce else (lo, mid)
        tau = math.sqrt(lo * hi)
    return make(tau), tau, ce(tau)


# ---------------------------------------------------------------- entraînement

def trainable_params(model, blocks: int):
    for p in model.parameters():
        p.requires_grad_(False)
    names = []
    for n, p in model.named_parameters():
        keep = n.startswith("head") or n.startswith("norm.") or n.startswith("fc_norm")
        if n.startswith("blocks."):
            keep = int(n.split(".")[1]) >= len(model.blocks) - blocks
        if keep:
            p.requires_grad_(True); names.append(n)
    return names


@torch.no_grad()
def report(model, orig_ood_logits, forget_u8, ood_u8, y, dev):
    zf = logits_of(model, forget_u8, dev)
    ce = F.cross_entropy(zf, y, reduction="none")
    zo = logits_of(model, ood_u8, dev)
    lp = F.log_softmax(orig_ood_logits / T_KL, 1); lq = F.log_softmax(zo / T_KL, 1)
    kl = float((lp.exp() * (lp - lq)).sum(1).mean())
    agree = float((orig_ood_logits.argmax(1) == zo.argmax(1)).float().mean())
    top = float(torch.bincount(zf.argmax(1), minlength=zf.shape[1]).max()) / len(zf)
    return {"forget_ce_mean": float(ce.mean()), "forget_ce_std": float(ce.std()), "forget_flip": float((zf.argmax(1) != y).float().mean()),
            "forget_top_class": top, "ood_agree": agree, "ood_kl_T2": kl, "ood_ret": 1 / (1 + kl)}


def run(model, forget_u8, ood_u8, a, dev, out: Path, factory):
    g = torch.Generator().manual_seed(a.seed); torch.manual_seed(a.seed)
    model = model.to(dev)
    t0 = time.time()
    z_clean = logits_of(model, forget_u8, dev)
    y = z_clean.argmax(1)
    p = torch.zeros_like(z_clean)
    for k in range(a.k_aug):
        p += torch.softmax(logits_of(model, forget_u8, dev, tf=strong_aug, g=g), 1)
    p /= a.k_aug
    q, tau, ce_q = calibrate(p, y, a.target_ce, mode=a.calib)
    ce_qi = -torch.log(q.gather(1, y[:, None]).clamp_min(1e-12))[:, 0]
    hold = min(len(ood_u8) // 5, 200)                      # OOD tenu à part pour le journal (jamais entraîné)
    ood_tr, ood_ev = ood_u8[hold:], ood_u8[:hold]
    orig_ev = logits_of(model, ood_ev, dev)
    log = {"calib": a.calib, "tau": tau, "target_ce": a.target_ce, "q_ce": ce_q, "q_ce_std": float(ce_qi.std()),
           "q_softened": float((-torch.log(p.gather(1, y[:, None]).clamp_min(1e-12))[:, 0] < a.target_ce).float().mean())
           if a.calib == "partial" else 1.0,
           "q_top1_is_y": float((q.argmax(1) == y).float().mean()),
           "p_aug_ce": float(-torch.log(p.gather(1, y[:, None]).clamp_min(1e-12)).mean()),
           "orig": report(model, orig_ev, forget_u8, ood_ev, y, dev), "epochs": []}
    print(json.dumps({k: v for k, v in log.items() if k != "epochs"}), flush=True)

    teacher = factory().to(dev).eval(); teacher.load_state_dict(model.state_dict())
    for p_ in teacher.parameters():
        p_.requires_grad_(False)
    names = trainable_params(model, a.blocks)
    theta0 = {n: p_.detach().clone() for n, p_ in model.named_parameters() if p_.requires_grad}
    opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=a.lr, weight_decay=0.0)
    print(f"{len(names)} tenseurs entraînables ({sum(p_.numel() for p_ in model.parameters() if p_.requires_grad):,} paramètres) ; "
          f"tau={tau:.3f} CE cible atteinte {ce_q:.3f}", flush=True)
    n = len(forget_u8); nb = math.ceil(n / a.bs)
    for ep in range(1, a.epochs + 1):
        model.train(); perm = torch.randperm(n, generator=g)
        for b in range(nb):
            ii = perm[b * a.bs:(b + 1) * a.bs]
            xf = to_x(forget_u8[ii]); xf = torch.cat([xf, mild_aug(xf, g)]).to(dev)
            qf = torch.cat([q[ii], q[ii]]).to(dev)
            jj = torch.randint(0, len(ood_tr), (a.bs_ood,), generator=g)
            xo = ood_aug(to_x(ood_tr[jj]), g).to(dev)
            with torch.autocast(device_type=torch.device(dev).type, dtype=torch.bfloat16, enabled=dev != "cpu"):
                zf = model(xf).float(); zo = model(xo).float()
                with torch.no_grad():
                    to = teacher(xo).float()
            l_f = F.kl_div(F.log_softmax(zf, 1), qf, reduction="batchmean")
            l_o = F.kl_div(F.log_softmax(zo / T_KL, 1), F.log_softmax(to / T_KL, 1), reduction="batchmean", log_target=True) * T_KL ** 2
            l2 = sum(((p_ - theta0[nm]) ** 2).sum() for nm, p_ in model.named_parameters() if p_.requires_grad)
            loss = l_f + a.w_ood * l_o + a.l2 * l2
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if b % max(1, nb // 4) == 0:
                print(f"ep {ep} it {b}/{nb} l_f {l_f.item():.4f} l_ood {l_o.item():.4f} l2 {l2.item():.4f} {time.time() - t0:.0f}s", flush=True)
        r = report(model, orig_ev, forget_u8, ood_ev, y, dev); r["epoch"] = ep; log["epochs"].append(r)
        sd = {k: v.detach().float().cpu().contiguous() for k, v in model.state_dict().items()}
        path = out / f"t1t2_{a.tag}_e{ep}.pt"; torch.save(sd, path)
        print("EPOCH", json.dumps(r), "->", path, flush=True)
        (out / f"t1t2_{a.tag}.json").write_text(json.dumps(log, indent=1))
        if r["forget_ce_mean"] >= a.target_ce * a.stop_frac:
            print(f"arrêt : CE forget {r['forget_ce_mean']:.3f} >= {a.stop_frac} × cible", flush=True)
            break
    return log


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", type=Path, default=Path(".."))
    ap.add_argument("--ood", type=Path, help="npz uint8 [N,224,224,3] de photos hors forget")
    ap.add_argument("--target-ce", type=float, default=0.6)
    ap.add_argument("--k-aug", type=int, default=8)
    ap.add_argument("--calib", choices=["global", "partial"], default="global")
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--w-ood", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--stop-frac", type=float, default=0.9)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--bs-ood", type=int, default=64)
    ap.add_argument("--max-forget", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="run"); ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.selftest:
        return selftest(a)
    import timm
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def factory():
        return timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=100, init_values=1e-4)
    model = factory()
    sd = torch.load(a.task_dir / "model.pt", map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    model.load_state_dict(sd, strict=True)
    with np.load(a.task_dir / "forget_images.npz") as z:
        forget = torch.from_numpy(z[z.files[0]])
    with np.load(a.ood) as z:
        ood = torch.from_numpy(z[z.files[0]])
    if a.max_forget:
        forget = forget[:a.max_forget]
    assert forget.dtype == torch.uint8 and forget.shape[1:] == (224, 224, 3), forget.shape
    assert ood.dtype == torch.uint8 and ood.shape[1:] == (224, 224, 3), ood.shape
    print(f"forget {tuple(forget.shape)} ood {tuple(ood.shape)} dev {dev}", flush=True)
    run(model, forget, ood, a, dev, a.out, factory)


def selftest(a):
    """ViT minuscule, données factices : vérifie calibrate, la boucle, le gel des couches, le format de sortie."""
    import timm
    torch.manual_seed(0)

    def factory():
        return timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10, init_values=1e-4)
    p = torch.softmax(torch.randn(50, 10) * 3, 1); y = p.argmax(1)
    for tc in (0.1, 0.5, 1.5):
        q, tau, ce = calibrate(p, y, tc)
        assert abs(ce - tc) < 1e-3, (tc, ce, tau)
    p2 = p.clone(); p2[:10] = torch.softmax(torch.randn(10, 10) * 0.3, 1)   # 10 images « hésitantes »
    y2 = p2.argmax(1)
    q2, tau2, ce2 = calibrate(p2, y2, 1.0, mode="partial")
    hes = -torch.log(p2.gather(1, y2[:, None]))[:, 0] >= 1.0
    assert abs(ce2 - 1.0) < 1e-3 and torch.allclose(q2[hes], p2[hes], atol=1e-6), (ce2, int(hes.sum()))
    m = factory(); names = trainable_params(m, 2)
    assert all(n.startswith(("blocks.10", "blocks.11", "head", "norm", "fc_norm")) for n in names), names
    forget = torch.randint(0, 256, (24, 224, 224, 3), dtype=torch.uint8)
    ood = torch.randint(0, 256, (40, 224, 224, 3), dtype=torch.uint8)
    a.k_aug, a.target_ce, a.blocks, a.epochs, a.bs, a.bs_ood, a.tag, a.calib = 2, 1.0, 2, 1, 8, 8, "selftest", "partial"
    w0 = {k: v.clone() for k, v in m.state_dict().items()}
    log = run(m, forget, ood, a, "cpu", a.out, factory)
    sd = torch.load(a.out / "t1t2_selftest_e1.pt", weights_only=True)
    f = factory(); f.load_state_dict(sd, strict=True)
    assert all(v.dtype == torch.float32 for v in sd.values())
    assert torch.equal(sd["blocks.0.attn.qkv.weight"], w0["blocks.0.attn.qkv.weight"]), "bloc gelé modifié"
    assert not torch.equal(sd["head.weight"], w0["head.weight"]), "tête non entraînée"
    assert len(log["epochs"]) == 1
    print("selftest ok (calibrage exact, blocs gelés intacts, tête modifiée, strict=True float32)")


if __name__ == "__main__":
    main()
