"""Localisation par décodage du message connu (PixelSeal), sur données synthétiques ou réelles.

Méthodes (une carte de chaleur [0,1] par image, puis seuil -> masque 128x128) :
  - window  : fenêtres glissantes « on ne garde que la fenêtre » : hors fenêtre, l'image est
              floutée (le flou détruit le watermark) ; l'image reste en 512 donc l'échelle du
              watermark est préservée (un recadrage agrandi casse le décodage : crop 128 px
              -> 58 % de bits justes, mesuré). Chaleur d'un pixel = moyenne de la bit accuracy
              des fenêtres qui le couvrent.
  - featmap : l'extracteur PixelSeal fait mean-pool spatial puis Linear ; les deux commutent,
              donc Linear appliqué position par position donne une carte de logits 8x8
              (entrée 256). On décale l'image de 0..24 px (pas de 8, en espace 256) pour une
              carte 32x32. Chaleur = bit accuracy du message connu par position.
  - all0 / all1 : baselines triviales.

Seuil choisi sur la moitié « calib » (split seedé), score rapporté sur la moitié « eval ».
Exemple : .venv/bin/python c1/localize_sliding.py --data c1/data/synth_pixelseal.npz --methods window featmap
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metric import miou_per_image  # noqa: E402
from pixelseal_wrap import PixelSeal  # noqa: E402

THRS = np.round(np.arange(0.50, 0.96, 0.01), 2)


# ---------------------------------------------------------------- cartes de chaleur
def batch_logits(decoder, imgs: torch.Tensor, batched: bool = True) -> torch.Tensor:
    """Décodeur générique : `decoder.logits(images)` -> logits par bit (k,) ou (n, k).

    batched=True : un seul appel sur n 3 H W (notre wrapper PixelSeal) ;
    batched=False : un appel par image 1 3 H W (interface du kit officiel, sûre pour
    TrustMark/StegaStamp). Renvoie n k sur CPU.
    """
    if batched:
        out = decoder.logits(imgs)
        return torch.as_tensor(out).float().reshape(len(imgs), -1).cpu()  # n k
    outs = [torch.as_tensor(decoder.logits(imgs[i:i + 1])).float().reshape(1, -1).cpu() for i in range(len(imgs))]
    return torch.cat(outs)  # n k


@torch.no_grad()
def heat_window(decoder, img: torch.Tensor, msg: torch.Tensor, win: int = 128,
                n_pos: int = 13, bs: int = 32, batched: bool = True, bg: str = "blur") -> torch.Tensor:
    """img : 1 3 512 512 [0,1] ; msg : k {0,1} -> chaleur 512 512 (bit acc moyenne).

    `decoder` : tout objet avec logits(images n 3 H W) ; logit > 0 <=> bit 1 (convention
    PixelSeal ; À VÉRIFIER pour les wrappers officiels TrustMark/StegaStamp, sinon passer
    par decode()). `decoder.device` optionnel (sinon CPU).
    """
    dev = getattr(decoder, "device", torch.device("cpu"))
    img = img.to(dev)
    # fond hors fenêtre. ATTENTION (mesuré sur 5 images synth) : le flou sigma=5 NE détruit
    # PAS le watermark PixelSeal (image entièrement floutée : 58-68 % de bits justes sur
    # removal/addition) ; 'gray' = gris uniforme 0.5.
    if bg == "blur":
        blur = gaussian_blur(img, kernel_size=21, sigma=5.0)  # 1 3 512 512
    else:
        blur = torch.full_like(img, 0.5)  # 1 3 512 512
    H = img.shape[-1]
    pos = np.linspace(0, H - win, n_pos).round().astype(int)
    boxes = [(y, x) for y in pos for x in pos]
    accs = []
    for i in range(0, len(boxes), bs):
        chunk = boxes[i:i + bs]
        m = torch.zeros(len(chunk), 1, H, H, device=dev)  # b 1 512 512
        for j, (y, x) in enumerate(chunk):
            m[j, :, y:y + win, x:x + win] = 1
        comp = blur + (img - blur) * m  # b 3 512 512 (broadcast 1 -> b)
        logit = batch_logits(decoder, comp, batched)  # b k
        accs.append(((logit > 0).float() == msg[None].float()).float().mean(1))  # b
    acc = torch.cat(accs)  # n_win
    heat = torch.zeros(H, H)
    cnt = torch.zeros(H, H)
    for a, (y, x) in zip(acc.tolist(), boxes):
        heat[y:y + win, x:x + win] += a
        cnt[y:y + win, x:x + win] += 1
    return heat / cnt.clamp(min=1)  # 512 512


@torch.no_grad()
def featmap_logits(ps: PixelSeal, x256: torch.Tensor) -> torch.Tensor:
    """x256 : b 3 h w [0,1] -> logits des bits par position b 256 h/32 w/32 (sans mean-pool).

    SPÉCIFIQUE à notre PixelSeal (internes de l'extracteur) ; avec le kit, partir de
    features(images) si elles exposent la carte avant le pooling.
    """
    ext = ps.model.detector
    z = ext.convnext(ext.preprocess(x256.to(ps.device)))  # b 768 h/32 w/32
    z = ext.pixel_decoder.output_upscaling(z)  # b 768 h/32 w/32 (upscale x1)
    lin = ext.pixel_decoder.linear  # Linear(768, 1+256)
    out = torch.einsum("bchw,kc->bkhw", z, lin.weight) + lin.bias[None, :, None, None]  # b 257 h w
    return out[:, 1:].float().cpu()  # b 256 h w


@torch.no_grad()
def heat_featmap(ps: PixelSeal, img: torch.Tensor, msg: torch.Tensor, step: int = 8) -> torch.Tensor:
    """img : 1 3 512 512 -> chaleur 512 512 (bit acc par position, carte 32x32 interpolée)."""
    x = F.interpolate(img, size=(256, 256), mode="bilinear", antialias=True, align_corners=False)  # 1 3 256 256
    cell = 32  # pas de l'encodeur ConvNeXt
    shifts = list(range(0, cell, step))  # 0 8 16 24
    batch = []
    for dy in shifts:
        for dx in shifts:
            # décale le contenu de (dy, dx) vers le bas/droite, bord en réflexion ; garde 256x256
            xp = F.pad(x, (dx, cell - dx, dy, cell - dy), mode="reflect")  # 1 3 288 288
            batch.append(xp)
    batch = torch.cat(batch)  # s*s 3 288 288
    lg = featmap_logits(ps, batch)  # s*s 256 9 9
    acc = ((lg > 0).float() == msg[None, :, None, None].float()).float().mean(1)  # s*s 9 9
    fine = 256 // step  # 32 cellules de 8 px (espace 256)
    heat = torch.zeros(fine + cell // step, fine + cell // step)  # 36 36 (avec marge)
    cnt = torch.zeros_like(heat)
    k = cell // step  # 4 sous-cellules par cellule
    for s_i, (dy, dx) in enumerate([(a, b) for a in shifts for b in shifts]):
        # cellule (i, j) de l'image décalée couvre [i*32-dy, i*32-dy+32) en coordonnées d'origine
        a = acc[s_i]  # 9 9
        up = a.repeat_interleave(k, 0).repeat_interleave(k, 1)  # 36 36
        oy, ox = k - dy // step, k - dx // step  # décalage en sous-cellules (marge k)
        heat[oy:oy + 36 - k, ox:ox + 36 - k] += up[: 36 - k, : 36 - k]
        cnt[oy:oy + 36 - k, ox:ox + 36 - k] += 1
    heat = (heat / cnt.clamp(min=1))[k:k + fine, k:k + fine]  # 32 32, cellule = 16 px en 512
    return F.interpolate(heat[None, None], size=(512, 512), mode="bilinear", align_corners=False)[0, 0]


def heat_to_128(h512: torch.Tensor) -> np.ndarray:
    return F.avg_pool2d(h512[None, None], 4)[0, 0].numpy()  # 512 512 -> 128 128


# ---------------------------------------------------------------- éval
def report(name, preds, gts, edits, idx_eval):
    s = miou_per_image(preds, gts)
    line = f"{name:22s} eval={s[idx_eval].mean():.4f} all={s.mean():.4f} |"
    for e in sorted(set(edits)):
        sel = [i for i in idx_eval if edits[i] == e]
        line += f" {e}={s[sel].mean():.3f}(n={len(sel)})"
    print(line)
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE / "data" / "synth_pixelseal.npz")
    ap.add_argument("--methods", nargs="+", default=["window", "featmap"])
    ap.add_argument("--limit", type=int, default=0, help="n premières images (0 = toutes)")
    ap.add_argument("--win", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bg", choices=["blur", "gray"], default="blur", help="fond hors fenêtre (window)")
    ap.add_argument("--save", type=Path, default=None, help="npz des cartes de chaleur 128x128")
    args = ap.parse_args()

    d = np.load(args.data)
    n = len(d["images"]) if not args.limit else min(args.limit, len(d["images"]))
    images, gts, msgs = d["images"][:n], d["mask128"][:n], d["msgs"][:n]
    edits = [str(e) for e in d["edit"][:n]]
    perm = np.random.default_rng(args.seed).permutation(n)
    idx_cal, idx_eval = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
    print(f"{n} images, calib={len(idx_cal)} eval={len(idx_eval)}")

    report("all0", np.zeros_like(gts), gts, edits, idx_eval)
    report("all1", np.ones_like(gts), gts, edits, idx_eval)

    ps = PixelSeal()
    heats = {}
    for meth in args.methods:
        t0 = time.time()
        hs = []
        for i in range(n):
            img = torch.from_numpy(images[i]).permute(2, 0, 1).float()[None] / 255  # 1 3 512 512
            msg = torch.from_numpy(msgs[i]).float()  # 256
            h = heat_window(ps, img, msg, win=args.win, bg=args.bg) if meth == "window" else heat_featmap(ps, img, msg)
            hs.append(heat_to_128(h))
        hs = np.stack(hs)  # n 128 128
        assert np.isfinite(hs).all()
        dt = time.time() - t0
        heats[meth] = hs
        # seuil choisi sur calib uniquement
        cal = [miou_per_image((hs[idx_cal] >= t).astype(np.uint8), gts[idx_cal]).mean() for t in THRS]
        tb = float(THRS[int(np.argmax(cal))])
        print(f"[{meth}] {dt:.1f} s ({dt / n:.2f} s/img) ; seuil calib={tb:.2f} (mIoU calib {max(cal):.4f})")
        report(f"{meth}@{tb:.2f}", (hs >= tb).astype(np.uint8), gts, edits, idx_eval)
        # sensibilité (information seulement, PAS utilisé pour choisir)
        for t in (0.55, 0.6, 0.7, 0.8):
            s = miou_per_image((hs[idx_eval] >= t).astype(np.uint8), gts[idx_eval]).mean()
            print(f"    eval @ {t:.2f}: {s:.4f}")
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save, **heats, gt=gts, edit=np.array(edits))
        print(f"cartes -> {args.save}")


if __name__ == "__main__":
    main()
