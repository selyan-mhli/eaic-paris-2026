"""Cartes de ré-embarquement pour le régime F (image entière watermarquée, puis zone éditée par diffusion).

Idée : hors de la zone éditée, l'image x' porte encore le watermark du message connu m. Le résidu de
ré-embarquement r̂(m) = embed(x', m) - x' est une estimation du motif que l'embedder pose sur ce contenu
pour m. Le résidu haute fréquence h = x' - flou_gaussien(x') contient ce motif là où il a survécu.
Carte : corrélation normalisée locale c(m) entre h et r̂(m) (fenêtres carrées, plusieurs échelles),
puis z-score contre K messages aléatoires à graine fixe (même contenu, autres messages) :
    z = (c(m) - moyenne_k c(m_k)) / écart-type_k c(m_k)
Le z-score retire la part de corrélation due au contenu (texture que l'embedder module quel que soit m).
PixelSeal : luminance Y ; TrustMark et StegaStamp : RVB (produit sommé sur les 3 canaux).

Sorties (.npz, format des cartes existantes, 128x128 par moyenne en blocs de 4) :
  names, method, full_watermark (train ; False si absent), done (bool, image calculée),
  re_z  float16 [N,128,128] : z-score par pixel (écart-type des K messages au même pixel), moyenne des échelles
  re_zg float16 [N,128,128] : variante, écart-type des K messages groupé sur l'image (moins bruité)
  re_c  float16 [N,128,128] : corrélation brute c(m), moyenne des échelles
Grand = watermark présent (zone NON éditée = masque 1 du kit).

Depuis le dossier de la tâche sur JURECA (wmloc/, models/, release/), code copié dans work_c1/ :
    python work_c1/reembed/maps.py --split train --only-full --limit 40 --out output/re_pilot.npz
    srun python work_c1/reembed/maps.py ...   # shard/nshards pris dans SLURM_PROCID/SLURM_NTASKS
Avec --nshards > 1, la sortie devient <out>_s<k>of<n>.npz (fusion : merge.py).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

METHODS = ("pixelseal", "trustmark", "stegastamp")
LUMA = torch.tensor([0.299, 0.587, 0.114])


# --------------------------------------------------------------------------- noyau (sans wmloc, testé)
def gauss_kernel(sigma: float) -> torch.Tensor:
    r = max(1, int(round(3 * sigma)))
    k = torch.exp(-torch.arange(-r, r + 1).float() ** 2 / (2 * sigma ** 2))
    return k / k.sum()  # (2r+1,)


def blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Flou gaussien séparable, bord en réflexion. x b c h w -> b c h w."""
    b, c, hh, ww = x.shape
    k = gauss_kernel(sigma).to(x)
    r = len(k) // 2
    y = x.reshape(b * c, 1, hh, ww)  # b c h w -> (b c) 1 h w
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))  # (b c) 1 h w
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))  # (b c) 1 h w
    return y.reshape(b, c, hh, ww)  # (b c) 1 h w -> b c h w


def box_mean(x: torch.Tensor, w: int) -> torch.Tensor:
    """Moyenne glissante w x w au pas 1, même taille, bord en réflexion. x b c h w -> b c h w."""
    lo, hi = w // 2, w - 1 - w // 2
    y = F.pad(x, (lo, hi, lo, hi), mode="reflect")  # b c (h+w-1) (w+w-1)
    y = F.avg_pool2d(y, (w, 1), stride=1)  # b c h (w+w-1)
    return F.avg_pool2d(y, (1, w), stride=1)  # b c h w


def to_luma(x: torch.Tensor) -> torch.Tensor:
    """x ... 3 h w -> ... 1 h w (Y de BT.601)."""
    return (x * LUMA.to(x).view(3, 1, 1)).sum(-3, keepdim=True)


def local_ncc(h: torch.Tensor, r: torch.Tensor, win: int, eps: float = 1e-12) -> torch.Tensor:
    """Corrélation normalisée locale (non centrée : h et r sont des résidus ~ moyenne nulle).
    h b c H W, r b M c H W -> b M H W, dans [-1, 1]."""
    b, M, c, H, W = r.shape
    hr = (h[:, None] * r).sum(2)  # b 1 c H W * b M c H W -> b M c H W -> b M H W
    hh = (h * h).sum(1, keepdim=True)  # b c H W -> b 1 H W
    rr = (r * r).sum(2)  # b M c H W -> b M H W
    num = box_mean(hr, win)  # b M H W
    den = torch.sqrt(box_mean(hh, win) * box_mean(rr, win) + eps)  # b 1 H W * b M H W -> b M H W
    return num / den


def zscore_maps(x: torch.Tensor, res: torch.Tensor, wins=(16, 32), sigma: float = 1.0, luma: bool = False,
                hp_r: bool = False, out: int = 128, sd_floor: float = 0.1):
    """x b 3 H W (image observée, [0,1]) ; res b M 3 H W (résidus r̂, M = 1 + K, indice 0 = message vrai).
    Renvoie (z, zg, c) chacun b out out (float32). Moyenne sur les échelles `wins`."""
    b, M, _, H, W = res.shape
    assert M >= 3 and x.shape == (b, 3, H, W), (x.shape, res.shape)
    if luma:
        x = to_luma(x)  # b 3 H W -> b 1 H W
        res = to_luma(res)  # b M 3 H W -> b M 1 H W
    c = x.shape[1]
    h = x - blur(x, sigma)  # b c H W
    if hp_r:
        res = res - blur(res.reshape(b * M, c, H, W), sigma).reshape(b, M, c, H, W)  # b M c H W
    zs, zgs, cs = [], [], []
    for w in wins:
        n = local_ncc(h, res, w)  # b M H W
        c0, ck = n[:, 0], n[:, 1:]  # b H W ; b K H W
        mu = ck.mean(1)  # b H W
        sd = ck.std(1)  # b H W (non biaisé, K-1)
        med = sd.flatten(1).median(1).values.view(b, 1, 1)  # b 1 1
        zs.append((c0 - mu) / torch.maximum(sd, sd_floor * med + 1e-8))  # b H W
        sdg = ck.flatten(1).std(1).view(b, 1, 1)  # b 1 1 (groupé sur K x H x W)
        zgs.append((c0 - mu) / (sdg + 1e-8))  # b H W
        cs.append(c0)
    f = H // out
    assert f * out == H, (H, out)
    pool = lambda t: F.avg_pool2d(torch.stack(t).mean(0)[:, None], f)[:, 0]  # noqa: E731  S b H W -> b H W -> b out out
    return pool(zs), pool(zgs), pool(cs)


def rand_messages(name: str, k: int, nbits: int, seed: int) -> np.ndarray:
    """K messages {0,1} à graine fixe, dérivée du nom (indépendante du shard). -> K nbits uint8."""
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    return rng.integers(0, 2, (k, nbits)).astype(np.uint8)


def as_float_hwc(a) -> np.ndarray:
    """Sortie d'embed -> H W 3 float32 [0,1] (accepte uint8 HWC, tenseur 1 3 H W ou 3 H W, float ou uint8)."""
    if torch.is_tensor(a):
        a = a.detach().float().cpu().numpy() if a.dtype != torch.uint8 else a.cpu().numpy()
    a = np.asarray(a)
    if a.ndim == 4:
        a = a[0]  # 1 ... -> ...
    if a.ndim == 3 and a.shape[0] == 3 and a.shape[-1] != 3:
        a = a.transpose(1, 2, 0)  # 3 H W -> H W 3
    if a.dtype == np.uint8 or a.max() > 1.5:
        return a.astype(np.float32) / 255.0
    return a.astype(np.float32)


def residuals(embed_fn, img_u8: np.ndarray, msgs: np.ndarray) -> torch.Tensor:
    """img_u8 H W 3 ; msgs M nbits -> r̂ M 3 H W float32 (embed(x', m) - x')."""
    x = img_u8.astype(np.float32) / 255.0  # H W 3
    out = []
    for m in msgs:
        e = as_float_hwc(embed_fn(img_u8, m))  # H W 3
        assert e.shape == x.shape, f"embed renvoie {e.shape}, attendu {x.shape}"  # À VÉRIFIER SUR JURECA
        out.append(torch.from_numpy(e - x).permute(2, 0, 1))  # H W 3 -> 3 H W
    return torch.stack(out)  # M 3 H W


def compute_batch(embed_fn, imgs_u8, msgs_true, names, K, seed, dev, wins, sigma, luma, hp_r, out=128):
    """Une liste d'images d'une même méthode -> (z, zg, c) numpy b out out."""
    xs, rs = [], []
    for img, m, n in zip(imgs_u8, msgs_true, names):
        m = np.asarray(m).astype(np.uint8).ravel()
        msgs = np.concatenate([m[None], rand_messages(n, K, len(m), seed)])  # (1+K) nbits
        rs.append(residuals(embed_fn, img, msgs))  # (1+K) 3 H W
        xs.append(torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1))  # H W 3 -> 3 H W
    x = torch.stack(xs).to(dev)  # b 3 H W
    r = torch.stack(rs).to(dev)  # b (1+K) 3 H W
    z, zg, c = zscore_maps(x, r, wins=wins, sigma=sigma, luma=luma, hp_r=hp_r, out=out)
    return z.cpu().numpy(), zg.cpu().numpy(), c.cpu().numpy()


# --------------------------------------------------------------------------- adaptateurs kit (JURECA)
def load_kit(data: str, split: str, methods):
    """À VÉRIFIER SUR JURECA : LocalisationDataset(data, split, methods=[...]) (forme de recolle.py),
    ds.meta[i] avec "method", "image_name", "full_watermark" (train) ; ds[i] avec "image" (uint8 H W 3),
    "message" (uint8 k), "method", "image_name"."""
    sys.path.insert(0, str(Path.cwd()))
    from wmloc.data import LocalisationDataset  # import paresseux : les tests n'ont pas wmloc
    return LocalisationDataset(data, split, methods=list(methods))


def load_embedder(method: str, dev: str):
    """À VÉRIFIER SUR JURECA : load_watermark(method, "models", dev).embed(image uint8 H W 3, message uint8 k)
    -> uint8 H W 3 à la taille d'entrée (usage de recolle.py ; redimensionnement interne 256/256/400
    anticrénelé fait par le kit). as_float_hwc accepte aussi un tenseur float."""
    from wmloc.watermarks import load_watermark
    wm = load_watermark(method, "models", dev)
    return lambda img, msg: wm.embed(img, msg)


def select(ds, methods, only_full: bool, limit: int, gate_heats: Path | None, min_acc: float):
    """Indices du dataset retenus, dans l'ordre ; --limit s'applique par méthode."""
    meta = ds.meta
    keep_names = None
    if gate_heats is not None:  # images que le tri F/R peut classer F (acc_full >= min_acc)
        h = np.load(gate_heats)
        keep_names = {n for n, a in zip(h["names"].tolist(), h["acc_full"]) if a >= min_acc}
    out = []
    for m in methods:
        ii = [i for i in range(len(ds)) if meta[i]["method"] == m]
        if only_full:
            ii = [i for i in ii if meta[i].get("full_watermark", False)]
        if keep_names is not None:
            ii = [i for i in ii if meta[i]["image_name"] in keep_names]
        out += ii[:limit] if limit else ii
    return out


def save(path: Path, names, methods, full, done, z, zg, c):
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, names=np.array(names), method=np.array(methods), full_watermark=np.array(full),
                        done=done, re_z=z.astype(np.float16), re_zg=zg.astype(np.float16), re_c=c.astype(np.float16))
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--data", default="release")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--only-full", action="store_true", help="train : images full_watermark seulement")
    ap.add_argument("--gate-heats", type=Path, help="heats_<split>.npz : ne garder que acc_full >= --min-acc")
    ap.add_argument("--min-acc", type=float, default=0.55)
    ap.add_argument("--limit", type=int, default=0, help="par méthode")
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_PROCID", 0)))
    ap.add_argument("--nshards", type=int, default=int(os.environ.get("SLURM_NTASKS", 1)))
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wins", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--hp-r", action="store_true", help="passe-haut aussi sur r̂")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    lid = int(os.environ.get("SLURM_LOCALID", 0))
    dev = f"cuda:{lid if torch.cuda.device_count() > lid else 0}" if torch.cuda.is_available() else "cpu"
    out = a.out if a.nshards == 1 else a.out.with_name(f"{a.out.stem}_s{a.shard}of{a.nshards}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)

    ds = load_kit(a.data, a.split, a.methods)
    idx = select(ds, a.methods, a.only_full, a.limit, a.gate_heats, a.min_acc)
    idx = [i for j, i in enumerate(idx) if j % a.nshards == a.shard]
    meta = ds.meta
    names = [meta[i]["image_name"] for i in idx]; methods = [meta[i]["method"] for i in idx]
    full = [bool(meta[i].get("full_watermark", False)) for i in idx]
    N = len(idx)
    print(f"device {dev} shard {a.shard}/{a.nshards} : {N} images {dict(zip(*np.unique(methods, return_counts=True)))}"
          f" K={a.K} wins={a.wins} -> {out}", flush=True)
    Z = np.full((N, 128, 128), np.nan, np.float32); ZG = Z.copy(); C = Z.copy(); done = np.zeros(N, bool)
    t0 = time.time(); last = 0
    for m in a.methods:
        jj = [j for j in range(N) if methods[j] == m]
        if not jj:
            continue
        emb = load_embedder(m, dev)
        luma = m == "pixelseal"
        for s in range(0, len(jj), a.bs):
            js = jj[s:s + a.bs]
            items = [ds[idx[j]] for j in js]
            assert all(it["image_name"] == names[j] for it, j in zip(items, js))
            z, zg, c = compute_batch(emb, [it["image"] for it in items], [it["message"] for it in items],
                                     [names[j] for j in js], a.K, a.seed, dev, a.wins, a.sigma, luma, a.hp_r)
            assert np.isfinite(z).all() and np.isfinite(c).all(), "NaN dans les cartes"
            Z[js], ZG[js], C[js] = z, zg, c; done[js] = True
            nd = int(done.sum())
            if nd - last >= a.save_every:
                save(out, names, methods, full, done, Z, ZG, C); last = nd
                print(f"[{nd}/{N}] {m} {time.time() - t0:.0f}s ({(time.time() - t0) / nd:.2f} s/img) partiel -> {out}", flush=True)
    save(out, names, methods, full, done, Z, ZG, C)
    print(f"fini {int(done.sum())}/{N} en {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
