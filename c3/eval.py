"""Validation locale c3 (unlearning) par PROXYS. Ce n'est PAS la métrique officielle.

La métrique officielle (forget_score × retain_score, 0 si accuracy < 70 % sur une validation
cachée) compare le candidat à des modèles « gold » réentraînés sans les forget et à un set de
contrôle caché. On n'a ni les gold, ni labels, ni données retain : ce script mesure des proxys.

Mesures :
  (a) retain proxy : accord top-1 et KL(origine || candidat) à T=2, candidat vs modèle d'origine,
      sur un jeu retain (--retain-npz, à préférer) ou, à défaut, sur des augmentations fortes
      des forget.
  (b) forget proxy : perte du candidat sur les forget (pseudo-labels = argmax du modèle d'origine),
      comparée à la perte du modèle d'origine sur des augmentations fortes des mêmes images
      (substitut d'images « jamais vues »). Score = exp(-(|Δmoyenne| + |Δécart-type|)).
  (c) contrôle fp16 : écart des logits fp32 vs fp16 autocast (le serveur évalue en fp16),
      sorties non finies et taux de top-1 qui changent.
  (d) format de soumission : load_state_dict strict=True sur timm vit_base_patch16_224
      à 100 classes, tous les tenseurs en float32, fichier ≤ 450 Mo (450e6 octets).

score = retain_proxy × forget_proxy, mis à 0 si (c) ou (d) échoue ou si l'accord retain < --gate.

Limites (à garder en tête avant de croire un chiffre) :
  - « augmenté ≈ jamais vu » est une hypothèse : un gold voit aussi les forget altérées
    différemment du modèle d'origine ; sans --retain-npz, (a) et (b) se tirent dans des sens
    opposés sur les mêmes images.
  - l'accord avec le modèle d'origine n'est pas une accuracy : la porte à 70 % du serveur porte
    sur des labels qu'on n'a pas ; --gate 0.72 est une marge heuristique.
  - normalisation (--mean/--std, défaut 0.5 = timm vit_base_patch16_224) et clé du npz à vérifier
    contre task_template.py.

Usage :
  python eval.py --candidate out/model.pt
  python eval.py --pred <dir>            # contrat autoresearch : <dir>/model.pt (ou le seul .pt)
Chemins par défaut = dossier JURECA de l'équipe (common/paths.py), surchargeables.

La logique de mesure (numpy pur) s'importe sans torch ; torch/timm ne sont chargés que pour
lancer les modèles.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

DEFAULT_TASK_DIR = Path("/p/scratch/training2666/mouhali1/SLZ/unlearning")  # = common/paths.py c3
ARCH = "vit_base_patch16_224"
NUM_CLASSES = 100
MAX_BYTES = 450_000_000  # 450 Mo, lecture stricte (450 Mio = 471 859 200 serait plus permissif)
TEMPERATURE = 2.0


# ---------------------------------------------------------------- mesures (numpy pur)

def log_softmax(logits: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = logits.astype(np.float64) / t
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def top1_agreement(ref: np.ndarray, cand: np.ndarray) -> float:
    """Part des images où le candidat prédit la même classe que la référence. ref, cand : [N, C]."""
    return float((ref.argmax(1) == cand.argmax(1)).mean())


def kl_divergence(ref: np.ndarray, cand: np.ndarray, t: float = TEMPERATURE) -> float:
    """KL(softmax(ref/t) || softmax(cand/t)) moyenne par image."""
    lp, lq = log_softmax(ref, t), log_softmax(cand, t)
    return float((np.exp(lp) * (lp - lq)).sum(1).mean())


def ce_losses(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Perte d'entropie croisée par image. logits [N, C], labels [N] -> [N]."""
    return -log_softmax(logits)[np.arange(len(labels)), labels]


def retain_proxy(ref: np.ndarray, cand: np.ndarray) -> dict:
    agree, kl = top1_agreement(ref, cand), kl_divergence(ref, cand)
    return {"agreement": agree, "kl_T2": kl, "retain_proxy": agree * math.exp(-kl)}


def forget_proxy(cand_forget: np.ndarray, orig_forget: np.ndarray, orig_aug: np.ndarray) -> dict:
    """cand_forget, orig_forget : logits sur les forget propres ; orig_aug : origine sur augmentations.

    Pseudo-labels = argmax de l'origine sur les forget propres (répétés si plusieurs augmentations).
    """
    labels = orig_forget.argmax(1)
    reps = len(orig_aug) // len(labels)
    lc = ce_losses(cand_forget, labels)
    lr = ce_losses(orig_aug, np.tile(labels, reps))
    lo = ce_losses(orig_forget, labels)
    gap = abs(lc.mean() - lr.mean()) + abs(lc.std() - lr.std())
    return {
        "loss_cand_mean": float(lc.mean()), "loss_cand_std": float(lc.std()),
        "loss_target_mean": float(lr.mean()), "loss_target_std": float(lr.std()),
        "loss_orig_mean": float(lo.mean()),
        "forget_proxy": float(math.exp(-gap)),
    }


def fp16_gap(fp32: np.ndarray, fp16: np.ndarray, max_flip: float) -> dict:
    finite = bool(np.isfinite(fp16).all())
    d = np.abs(fp32.astype(np.float64) - fp16.astype(np.float64)) if finite else np.array([np.inf])
    flip = 1.0 - top1_agreement(fp32, fp16) if finite else 1.0
    return {"fp16_finite": finite, "fp16_max_abs": float(d.max()), "fp16_mean_abs": float(d.mean()),
            "fp16_top1_flip": flip, "fp16_ok": finite and flip <= max_flip}


def combine(fmt_ok: bool, fp16_ok: bool, retain: dict, forget: dict, gate: float) -> tuple[float, list]:
    reasons = []
    if not fmt_ok:
        reasons.append("format de soumission invalide")
    if not fp16_ok:
        reasons.append("fp16 : sorties non finies ou trop de top-1 changés")
    if retain["agreement"] < gate:
        reasons.append(f"accord retain {retain['agreement']:.3f} < porte {gate}")
    return (0.0 if reasons else retain["retain_proxy"] * forget["forget_proxy"]), reasons


# ---------------------------------------------------------------- torch / timm

def make_timm_model(arch: str = ARCH, num_classes: int = NUM_CLASSES):
    import timm
    return timm.create_model(arch, pretrained=False, num_classes=num_classes)


def load_state_dict(path: Path) -> dict:
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and isinstance(obj.get("state_dict"), dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"{path} ne contient pas un state dict ({type(obj).__name__})")
    return obj


def check_format(sd: dict, model_factory, path: Path | None = None, max_bytes: int = MAX_BYTES) -> dict:
    """(d) strict=True sur l'architecture, float32 partout, taille du fichier."""
    import torch
    errors = []
    bad = [k for k, v in sd.items() if not torch.is_tensor(v) or v.dtype != torch.float32]
    if bad:
        errors.append(f"{len(bad)} entrée(s) non float32, ex. {bad[:3]}")
    try:
        model_factory().load_state_dict(sd, strict=True)
    except RuntimeError as e:
        errors.append("strict=True : " + str(e).splitlines()[0][:300])
    size = os.path.getsize(path) if path else None
    if size is not None and size > max_bytes:
        errors.append(f"fichier {size} o > {max_bytes} o")
    return {"format_ok": not errors, "format_errors": errors, "file_bytes": size,
            "n_params": int(sum(v.numel() for v in sd.values() if torch.is_tensor(v)))}


def to_tensor(images_u8: np.ndarray, mean, std):
    """uint8 [N, H, W, 3] -> float32 normalisé [N, 3, H, W]."""
    import torch
    x = torch.from_numpy(np.ascontiguousarray(images_u8)).permute(0, 3, 1, 2).float() / 255.0  # n h w c -> n c h w
    m = torch.tensor(mean).view(1, 3, 1, 1)
    s = torch.tensor(std).view(1, 3, 1, 1)
    return (x - m) / s


def strong_augment(images_u8: np.ndarray, seed: int, n_aug: int = 1) -> np.ndarray:
    """Augmentations fortes déterministes (seed) : crop redimensionné, flip, couleur, bruit.

    uint8 [N, H, W, 3] -> uint8 [n_aug * N, H, W, 3] (copie k de l'image i à l'indice k * N + i).
    """
    import torch
    import torch.nn.functional as F
    g = torch.Generator().manual_seed(seed)
    n, h, w, _ = images_u8.shape
    base = torch.from_numpy(np.ascontiguousarray(images_u8)).permute(0, 3, 1, 2).float() / 255.0  # n c h w
    out = []
    for _ in range(n_aug):
        imgs = []
        for i in range(n):
            scale = 0.35 + 0.5 * torch.rand(1, generator=g).item()
            ch, cw = max(8, int(h * math.sqrt(scale))), max(8, int(w * math.sqrt(scale)))
            y0 = int(torch.randint(0, h - ch + 1, (1,), generator=g))
            x0 = int(torch.randint(0, w - cw + 1, (1,), generator=g))
            im = base[i:i + 1, :, y0:y0 + ch, x0:x0 + cw]
            im = F.interpolate(im, size=(h, w), mode="bilinear", align_corners=False)  # 1 c h w
            if torch.rand(1, generator=g).item() < 0.5:
                im = im.flip(3)
            b = 0.6 + 0.8 * torch.rand(1, generator=g).item()
            c = 0.6 + 0.8 * torch.rand(1, generator=g).item()
            im = (im - im.mean()) * c + im.mean() * b
            im = im + 0.08 * torch.randn(im.shape, generator=g)
            imgs.append(im)
        out.append(torch.cat(imgs))  # n c h w
    x = torch.cat(out).clamp(0, 1)  # (n_aug n) c h w
    return (x * 255).round().byte().permute(0, 2, 3, 1).numpy()  # -> (n_aug n) h w c


def predict(model, images_u8: np.ndarray, mean, std, device: str = "cpu", fp16: bool = False,
            batch_size: int = 64) -> np.ndarray:
    """Logits float32 [N, C]. fp16=True : torch.autocast float16 comme le serveur."""
    import torch
    model = model.to(device).eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(images_u8), batch_size):
            x = to_tensor(images_u8[i:i + batch_size], mean, std).to(device)
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=fp16):
                y = model(x)
            outs.append(y.float().cpu().numpy())
    return np.concatenate(outs)


def evaluate(orig_sd: dict, cand_sd: dict, forget_u8: np.ndarray, model_factory, *,
             retain_u8: np.ndarray | None = None, cand_path: Path | None = None,
             mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), device: str = "cpu", seed: int = 0,
             n_aug: int = 1, gate: float = 0.72, fp16_max_flip: float = 0.01,
             max_bytes: int = MAX_BYTES, batch_size: int = 64) -> dict:
    report = check_format(cand_sd, model_factory, cand_path, max_bytes)
    if not report["format_ok"]:  # inutile (et souvent impossible) de lancer le modèle
        report["score"], report["gate_reasons"] = 0.0, ["format de soumission invalide"]
        return report
    orig = model_factory()
    orig.load_state_dict(orig_sd, strict=True)
    cand = model_factory()
    cand.load_state_dict(cand_sd, strict=True)
    kw = dict(mean=mean, std=std, device=device, batch_size=batch_size)

    aug = strong_augment(forget_u8, seed, n_aug)
    o_f, c_f = predict(orig, forget_u8, **kw), predict(cand, forget_u8, **kw)
    o_a = predict(orig, aug, **kw)
    if retain_u8 is not None:
        report["retain_source"] = f"retain-npz ({len(retain_u8)} images)"
        o_r, c_r = predict(orig, retain_u8, **kw), predict(cand, retain_u8, **kw)
    else:
        report["retain_source"] = f"augmentations fortes des forget ({len(aug)} images)"
        o_r, c_r = o_a, predict(cand, aug, **kw)
    report |= retain_proxy(o_r, c_r)
    report |= forget_proxy(c_f, o_f, o_a)
    c_f16 = predict(cand, forget_u8, fp16=True, **kw)
    report |= fp16_gap(c_f, c_f16, fp16_max_flip)
    report["score"], report["gate_reasons"] = combine(report["format_ok"], report["fp16_ok"],
                                                      report, report, gate)
    return report


# ---------------------------------------------------------------- CLI

def load_images(path: Path, key: str | None, limit: int | None) -> np.ndarray:
    with np.load(path) as z:
        k = key or z.files[0]
        x = z[k]
    if x.dtype != np.uint8 or x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"{path}[{k}] : attendu uint8 [N, H, W, 3], reçu {x.dtype} {x.shape}")
    return x[:limit] if limit else x


def find_candidate(pred: Path) -> Path:
    if (pred / "model.pt").exists():
        return pred / "model.pt"
    pts = sorted(pred.glob("*.pt"))
    if len(pts) != 1:
        raise FileNotFoundError(f"{pred} : attendu model.pt ou un seul .pt, trouvé {len(pts)}")
    return pts[0]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--candidate", type=Path, help="state dict candidat (.pt)")
    g.add_argument("--pred", type=Path, help="dossier de sortie de solve.py (contrat autoresearch)")
    p.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR,
                   help=f"dossier de la tâche (défaut JURECA : {DEFAULT_TASK_DIR})")
    p.add_argument("--original", type=Path, help="modèle d'origine (défaut <task-dir>/model.pt)")
    p.add_argument("--forget", type=Path, help="forget npz (défaut <task-dir>/forget_images.npz)")
    p.add_argument("--forget-key", help="clé du tableau dans le npz (défaut : le premier)")
    p.add_argument("--retain-npz", type=Path, help="jeu retain uint8 [N,224,224,3] (recommandé)")
    p.add_argument("--retain-key")
    p.add_argument("--max-images", type=int, help="sous-échantillon des forget pour aller vite")
    p.add_argument("--n-aug", type=int, default=1, help="copies augmentées par image forget")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gate", type=float, default=0.72, help="accord retain minimal (heuristique)")
    p.add_argument("--fp16-max-flip", type=float, default=0.01)
    p.add_argument("--mean", type=float, nargs=3, default=(0.5, 0.5, 0.5))
    p.add_argument("--std", type=float, nargs=3, default=(0.5, 0.5, 0.5))
    p.add_argument("--arch", default=ARCH)
    p.add_argument("--device", default=None, help="défaut : cuda si disponible, sinon cpu")
    p.add_argument("--batch-size", type=int, default=64)
    a = p.parse_args(argv)

    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cand_path = a.candidate or find_candidate(a.pred)
    orig_sd = load_state_dict(a.original or a.task_dir / "model.pt")
    cand_sd = load_state_dict(cand_path)
    forget = load_images(a.forget or a.task_dir / "forget_images.npz", a.forget_key, a.max_images)
    retain = load_images(a.retain_npz, a.retain_key, None) if a.retain_npz else None

    r = evaluate(orig_sd, cand_sd, forget, lambda: make_timm_model(a.arch, NUM_CLASSES),
                 retain_u8=retain, cand_path=cand_path, mean=tuple(a.mean), std=tuple(a.std),
                 device=device, seed=a.seed, n_aug=a.n_aug, gate=a.gate,
                 fp16_max_flip=a.fp16_max_flip, batch_size=a.batch_size)
    print("PROXY LOCAL, pas la métrique officielle (pas de gold, pas de labels, pas de retain).")
    print(f"candidat : {cand_path}  forget : {len(forget)} images  device : {device}")
    for k, v in r.items():
        if k != "score":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"score: {r['score']:.6f}")
    return 0 if r["format_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
