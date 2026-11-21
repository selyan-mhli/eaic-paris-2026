"""Données synthétiques Task 1 avec PixelSeal (en attendant les 510 vraies images train).

Pipeline par échantillon :
  1. image source A (contenu watermarké) et B (fond propre) : recadrage aléatoire -> 512x512 ;
  2. message aléatoire 256 bits, watermark PixelSeal sur un rectangle R (ratio <= 16:1) ;
     --embed-mode full : on watermarke A entière puis on ne garde le résultat que dans R
     (hypothèse par défaut) ; crop : on watermarke le recadrage R seul (redimensionné par
     le modèle) et on le recolle. Le vrai mode du dataset est À VÉRIFIER sur le train ;
  3. une édition parmi :
     - outpainting : image watermarkée réduite (x0.5-0.85) et collée dans B ;
     - removal     : forme irrégulière remplacée par A floutée (non watermarkée) ;
     - addition    : forme irrégulière remplacée par un morceau de B (objet propre) ;
     - transplant  : forme irrégulière watermarkée collée dans B ;
  4. quantification uint8 (comme un PNG), masque 512 exact, masque 128 (règle >= 50 %).

Sortie : un seul fichier compressé c1/data/synth_pixelseal.npz (+ un aperçu PNG).
Exemple : .venv/bin/python c1/synth.py --n 80 --seed 0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metric import downsample_mask  # noqa: E402
from pixelseal_wrap import VS, PixelSeal  # noqa: E402

S = 512
EDITS = ["outpainting", "removal", "addition", "transplant"]


# ---------------------------------------------------------------- sources
def load_sources(n_frames: int = 10) -> list[np.ndarray]:
    """Images sources (H W 3 uint8) : assets/imgs/*.jpg + quelques frames de assets/videos."""
    srcs = [np.asarray(Image.open(p).convert("RGB")) for p in sorted((VS / "assets" / "imgs").glob("*.jpg"))]
    vids = sorted((VS / "assets" / "videos").glob("*.mp4"))
    if vids:
        import av
        with av.open(str(vids[0])) as c:
            frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]  # list of H W 3
        idx = np.linspace(0, len(frames) - 1, n_frames).astype(int)
        srcs += [frames[i] for i in idx]
    return srcs


def random_view(src: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Recadrage carré aléatoire (20-100 % du petit côté), flip, -> 512 512 3 uint8."""
    h, w, _ = src.shape
    side = int(min(h, w) * rng.uniform(0.2, 1.0))
    y = rng.integers(0, h - side + 1)
    x = rng.integers(0, w - side + 1)
    crop = src[y:y + side, x:x + side]
    if rng.random() < 0.5:
        crop = crop[:, ::-1]
    im = Image.fromarray(np.ascontiguousarray(crop)).resize((S, S), Image.BICUBIC)
    arr = np.asarray(im).astype(np.float32)
    arr = arr * rng.uniform(0.85, 1.15) + rng.uniform(-15, 15)  # jitter luminosité/contraste
    return np.clip(arr, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- formes
def random_rect(rng: np.random.Generator, max_ratio: float = 16.0) -> tuple[int, int, int, int]:
    """Rectangle (y0, x0, h, w) d'aire 10-80 % de l'image, ratio <= max_ratio."""
    for _ in range(100):
        area = rng.uniform(0.10, 0.80) * S * S
        r = float(np.exp(rng.uniform(0, np.log(max_ratio))))
        h, w = np.sqrt(area / r), np.sqrt(area * r)
        if rng.random() < 0.5:
            h, w = w, h
        h, w = int(round(h)), int(round(w))
        if 16 <= h <= S and 16 <= w <= S and max(h, w) / min(h, w) <= max_ratio:
            y0 = int(rng.integers(0, S - h + 1))
            x0 = int(rng.integers(0, S - w + 1))
            return y0, x0, h, w
    return 128, 128, 256, 256


def random_blob(rng: np.random.Generator, cy: float, cx: float, r0: float) -> np.ndarray:
    """Forme irrégulière lisse (512 512 bool) centrée en (cy, cx), rayon moyen r0."""
    k = int(rng.integers(7, 15))
    ang = np.sort(rng.uniform(0, 2 * np.pi, k))
    rad = r0 * rng.uniform(0.55, 1.25, k)
    pts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a, r in zip(ang, rad)]
    im = Image.new("L", (S, S), 0)
    ImageDraw.Draw(im).polygon(pts, fill=255)
    m = gaussian_filter(np.asarray(im).astype(np.float32) / 255, sigma=r0 * 0.12)
    return m > 0.5


# ---------------------------------------------------------------- watermark
def to_t(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a).permute(2, 0, 1).float()[None] / 255  # H W 3 -> 1 3 H W


def to_np(t: torch.Tensor) -> np.ndarray:
    return (t[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)  # 1 3 H W -> H W 3


def watermark_rect(ps: PixelSeal, a: np.ndarray, msg: torch.Tensor, rect, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Renvoie (image A watermarkée sur R, float32 H W 3 en [0,255]) et le masque R (bool H W)."""
    y0, x0, h, w = rect
    R = np.zeros((S, S), bool)
    R[y0:y0 + h, x0:x0 + w] = True
    out = a.astype(np.float32).copy()
    if mode == "full":
        aw = ps.embed(to_t(a), msg)  # 1 3 512 512
        out[R] = to_np(aw)[R]
    else:  # crop : le modèle redimensionne le recadrage en 256x256 puis revient à h x w
        cw = ps.embed(to_t(np.ascontiguousarray(a[y0:y0 + h, x0:x0 + w])), msg)  # 1 3 h w
        out[y0:y0 + h, x0:x0 + w] = to_np(cw)
    return out, R


# ---------------------------------------------------------------- éditions
def edit(kind: str, xw: np.ndarray, R: np.ndarray, a: np.ndarray, b: np.ndarray,
         rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """xw : A watermarkée (H W 3 float), R : masque wm. Renvoie (image H W 3 float, masque bool)."""
    ys, xs = np.nonzero(R)
    if kind == "outpainting":
        s = rng.uniform(0.5, 0.85)
        n = int(round(S * s))
        small = F.interpolate(torch.from_numpy(xw).permute(2, 0, 1)[None], size=(n, n),
                              mode="bilinear", antialias=True, align_corners=False)  # 1 3 n n
        small = small[0].permute(1, 2, 0).numpy()  # n n 3
        msmall = F.interpolate(torch.from_numpy(R.astype(np.float32))[None, None], size=(n, n),
                               mode="area")[0, 0].numpy() > 0.5  # n n
        oy, ox = int(rng.integers(0, S - n + 1)), int(rng.integers(0, S - n + 1))
        img = b.astype(np.float32).copy()
        img[oy:oy + n, ox:ox + n] = small
        mask = np.zeros((S, S), bool)
        mask[oy:oy + n, ox:ox + n] = msmall
        return img, mask
    # centre de la forme : un pixel du rectangle watermarké (la forme le touche)
    i = rng.integers(len(ys))
    r0 = rng.uniform(40, 120)
    blob = random_blob(rng, ys[i], xs[i], r0)
    if kind == "removal":
        filler = gaussian_filter(a.astype(np.float32), sigma=(8, 8, 0))  # A floutée, sans wm
        img = xw.copy()
        img[blob] = filler[blob]
        return img, R & ~blob
    if kind == "addition":
        dy, dx = rng.integers(-150, 151, 2)
        filler = np.roll(b.astype(np.float32), (int(dy), int(dx)), axis=(0, 1))  # objet propre
        img = xw.copy()
        img[blob] = filler[blob]
        return img, R & ~blob
    if kind == "transplant":
        obj = blob & R
        oy, ox = np.nonzero(obj)
        # déplacement aléatoire qui garde l'objet dans l'image
        dy = int(rng.integers(-oy.min(), S - oy.max()))
        dx = int(rng.integers(-ox.min(), S - ox.max()))
        img = b.astype(np.float32).copy()
        img[oy + dy, ox + dx] = xw[oy, ox]
        mask = np.zeros((S, S), bool)
        mask[oy + dy, ox + dx] = True
        return img, mask
    raise ValueError(kind)


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embed-mode", choices=["full", "crop"], default="full")
    ap.add_argument("--out", type=Path, default=HERE / "data" / "synth_pixelseal.npz")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    ps = PixelSeal()
    srcs = load_sources()
    print(f"{len(srcs)} images sources, device={ps.device}")

    imgs, m512, msgs, edits, rects = [], [], [], [], []
    i = 0
    while len(imgs) < args.n:
        kind = EDITS[i % len(EDITS)]  # équilibré entre les 4 types
        i += 1
        ia, ib = rng.choice(len(srcs), 2, replace=len(srcs) < 2)
        a, b = random_view(srcs[ia], rng), random_view(srcs[ib], rng)
        msg = torch.from_numpy(rng.integers(0, 2, (1, 256))).float()  # 1 256
        rect = random_rect(rng, 16.0)
        xw, R = watermark_rect(ps, a, msg, rect, args.embed_mode)
        img, mask = edit(kind, xw, R, a, b, rng)
        if mask.sum() < 16 * 16:  # édition qui a tout effacé : on retire
            continue
        imgs.append(np.clip(img, 0, 255).round().astype(np.uint8))  # quantification PNG
        m512.append(mask)
        msgs.append(msg[0].numpy().astype(np.uint8))
        edits.append(kind)
        rects.append(rect)

    images = np.stack(imgs)  # N 512 512 3 uint8
    mask512 = np.stack(m512)  # N 512 512 bool
    mask128 = downsample_mask(mask512)  # N 128 128 uint8
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, images=images, mask512=mask512, mask128=mask128,
        msgs=np.stack(msgs), method=np.array(["pixelseal"] * len(imgs)),
        edit=np.array(edits), rect=np.array(rects), embed_mode=np.array(args.embed_mode),
        names=np.array([f"synth_{k:04d}.png" for k in range(len(imgs))]),
    )
    # aperçu : 8 premières images, masque en rouge
    k = min(8, len(imgs))
    prev = images[:k].astype(np.float32)
    prev[mask512[:k]] = prev[mask512[:k]] * 0.5 + np.array([255, 0, 0]) * 0.5
    sheet = np.concatenate(list(prev.astype(np.uint8)), axis=1)  # 512 (k*512) 3
    Image.fromarray(sheet).resize((k * 256, 256)).save(args.out.with_suffix(".preview.png"))
    frac = mask128.reshape(len(imgs), -1).mean(1)
    print(f"{len(imgs)} échantillons -> {args.out} ({args.out.stat().st_size / 1e6:.1f} Mo) "
          f"en {time.time() - t0:.1f} s ; part watermarkée moyenne {frac.mean():.2f}")
    for e in EDITS:
        print(f"  {e:12s} n={sum(x == e for x in edits)}")


if __name__ == "__main__":
    main()
