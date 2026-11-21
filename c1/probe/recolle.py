"""A1 — « recollé ou régénéré » : la zone watermarkée des images F a-t-elle gardé un watermark aussi fort
qu'un watermark frais ?

En F, la zone watermarkée (masque GT) est la zone NON éditée. Si elle a été recollée pixel à pixel, le
décodage du message connu y est aussi fort qu'après un embarquement frais. Si toute l'image a été régénérée
(aller-retour VAE / diffusion), le watermark y est affaibli (« What Breaks Local Watermarks? » : décodage
proche du hasard après SD-inpainting sans recoller l'extérieur).

Par image F de la méthode choisie (trous = hors masque, mis en gris 0.5 dans toutes les mesures « _m ») :
  obs      : l'image du train, message connu m.
  ctlB     : témoin propre, contenu différent : une image R de la même méthode (rectangle watermarqué
             <= 10 % de l'image, autre message), ré-embarquée avec l'embedder du kit et le MÊME message m,
             puis le MÊME masque F appliqué (géométrie identique).
  ctlA     : témoin même contenu : embed(obs, m2) avec un message aléatoire m2, décodé sur m2 (le watermark
             m d'origine reste présent : interférence, témoin plutôt pessimiste).
  nullR    : image R hors masque (rectangle mis en gris), décodée sur son message : niveau « pas de watermark ».
Mesures : bit accuracy (acc) et marge moyenne mean(logit * (2m-1)) (marg). PixelSeal : accord par position
(8x8, entrée 256) sur les cellules entièrement dans le masque (agr).
Proxy sans modèle (substitut d'AEROBLADE, faute de VAE en cache) : énergie haute fréquence de Y
(|Y - flou gaussien σ=1|) dans le masque vs dans les trous, en F ; et dans vs hors rectangle en R.

    python work_c1_probe/recolle.py --method pixelseal --out output/probe_recolle_pixelseal.npz [--limit 4]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path.cwd()))
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import to_tensor  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402


def strength(wm, x: torch.Tensor, msg: torch.Tensor) -> tuple[float, float]:
    """x 1 3 H W, msg k {0,1} -> (bit accuracy, marge moyenne logit * signe)."""
    lg = torch.as_tensor(wm.logits(x)).float().cpu()[0]
    return float(((lg > 0).float() == msg).float().mean()), float((lg * (msg * 2 - 1)).mean())


def agr_cells(wm, x: torch.Tensor, msg: torch.Tensor, g512: torch.Tensor) -> float:
    """PixelSeal : accord par position (8x8) moyen sur les cellules couvertes à >= 95 % par le masque."""
    per = wm.features(x)["per_position_logits"].float().cpu()[0]  # k 8 8
    a = ((per > 0).float() == msg[:, None, None]).float().mean(0)  # 8 8
    cov = F.avg_pool2d(g512.float().cpu()[None, None], 64)[0, 0]  # 8 8
    sel = cov >= 0.95
    return float(a[sel].mean()) if sel.any() else float("nan")


def hf_energy(img_u8: np.ndarray) -> torch.Tensor:
    """|Y - flou σ=1| par pixel (512 512)."""
    x = torch.from_numpy(img_u8).float().permute(2, 0, 1)[None] / 255
    y = 0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    k = torch.exp(-torch.arange(-3, 4).float() ** 2 / 2); k = k / k.sum()
    b = F.conv2d(F.pad(y, (3, 3, 0, 0), mode="reflect"), k.view(1, 1, 1, 7))
    b = F.conv2d(F.pad(b, (0, 0, 3, 3), mode="reflect"), k.view(1, 1, 7, 1))
    return (y - b).abs()[0, 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["pixelseal", "trustmark", "stegastamp"])
    ap.add_argument("--data", default="release")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = LocalisationDataset(a.data, "train", methods=[a.method])
    iF = [i for i in range(len(ds)) if ds.meta[i]["full_watermark"]]
    iR = [i for i in range(len(ds)) if not ds.meta[i]["full_watermark"]]
    if a.limit:
        iF = iF[:a.limit]
    wm = load_watermark(a.method, "models", dev)
    if a.method == "pixelseal":  # hygiène : force effectivement chargée (issue videoseal #33)
        bl = getattr(wm.model, "blender", None)
        print("pixelseal blender scaling_w =", getattr(bl, "scaling_w", "?"), "scaling_i =", getattr(bl, "scaling_i", "?"),
              "| model.scaling_w =", getattr(wm.model, "scaling_w", "?"), flush=True)
    rng = np.random.default_rng(a.seed)
    keys = ["obs_acc", "obs_marg", "obs_acc_m", "obs_marg_m", "ctlB_acc_m", "ctlB_marg_m", "ctlA_acc_m", "ctlA_marg_m",
            "nullR_acc", "nullR_marg", "obs_agr", "ctlB_agr", "hf_in", "hf_hole", "cov", "obs_minus_ctlA_mad"]
    out = {k: np.full(len(iF), np.nan, np.float32) for k in keys}
    t0 = time.time()
    for j, i in enumerate(iF):
        it = ds[i]; img = it["image"]; msg = torch.from_numpy(it["message"]).float()
        g = torch.from_numpy(it["mask"]).float().to(dev)[None, None]  # 1 1 512 512, 1 = watermarqué
        out["cov"][j] = float(g.mean())
        x = to_tensor(img).to(dev); gray = torch.full_like(x, 0.5)
        keep = lambda t: gray + (t - gray) * g  # noqa: E731  (trous en gris)
        out["obs_acc"][j], out["obs_marg"][j] = strength(wm, x, msg)
        out["obs_acc_m"][j], out["obs_marg_m"][j] = strength(wm, keep(x), msg)
        # témoin B : image R (contenu propre à ~90 %), ré-embarquée avec m, même masque
        r = ds[iR[j % len(iR)]]
        xb = to_tensor(wm.embed(r["image"], it["message"])).to(dev)
        out["ctlB_acc_m"][j], out["ctlB_marg_m"][j] = strength(wm, keep(xb), msg)
        # null : la même image R, rectangle en gris, décodée sur son propre message
        gr = torch.from_numpy(r["mask"]).float().to(dev)[None, None]
        xr = to_tensor(r["image"]).to(dev)
        out["nullR_acc"][j], out["nullR_marg"][j] = strength(wm, xr + (gray - xr) * gr, torch.from_numpy(r["message"]).float())
        # témoin A : même contenu, second message aléatoire
        m2 = rng.integers(0, 2, len(it["message"])).astype(np.uint8)
        xa_u8 = wm.embed(img, m2)
        out["ctlA_acc_m"][j], out["ctlA_marg_m"][j] = strength(wm, keep(to_tensor(xa_u8).to(dev)), torch.from_numpy(m2).float())
        out["obs_minus_ctlA_mad"][j] = float(np.abs(xa_u8.astype(np.int16) - img.astype(np.int16)).mean())
        if a.method == "pixelseal":
            out["obs_agr"][j] = agr_cells(wm, x, msg, g[0, 0])
            out["ctlB_agr"][j] = agr_cells(wm, xb, msg, g[0, 0])
        e = hf_energy(img); gm = it["mask"].astype(bool)
        out["hf_in"][j] = float(e[torch.from_numpy(gm)].mean())
        out["hf_hole"][j] = float(e[torch.from_numpy(~gm)].mean()) if (~gm).any() else np.nan
        if j < 3 or j % 20 == 0:
            print(f"[{j + 1}/{len(iF)}] " + " ".join(f"{k}={out[k][j]:.3f}" for k in keys) + f" {time.time() - t0:.0f}s", flush=True)
    # proxy HF en R : dans vs hors rectangle (référence : pas de régénération attendue hors rectangle)
    hr_in, hr_out = [], []
    for i in iR[: max(len(iF), 1)]:
        r = ds[i]; e = hf_energy(r["image"]); gm = torch.from_numpy(r["mask"].astype(bool))
        hr_in.append(float(e[gm].mean())); hr_out.append(float(e[~gm].mean()))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, names=np.array([ds.meta[i]["image_name"] for i in iF]), method=a.method,
                        hfR_in=np.array(hr_in, np.float32), hfR_out=np.array(hr_out, np.float32), **out)
    print("résumé", a.method, "n_F", len(iF), {k: round(float(np.nanmedian(v)), 4) for k, v in out.items()},
          "hfR_in", round(float(np.median(hr_in)), 4), "hfR_out", round(float(np.median(hr_out)), 4), f"{time.time() - t0:.0f}s", flush=True)
    print("->", a.out, flush=True)


if __name__ == "__main__":
    main()
