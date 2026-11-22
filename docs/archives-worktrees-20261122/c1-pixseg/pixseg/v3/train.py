"""Entraînement de la segmentation au pixel (BCE + perte mIoU douce), validation croisée et prédiction test.

  python work_pixseg/train.py --cv --synth output/pixseg_syn_*.npz            # 5 folds de gated_eval
  python work_pixseg/train.py --fit-all --predict-test --synth output/pixseg_syn_*.npz --thr 0.5

--cv : pour le fold externe k, le fold (k+1) % 5 est retiré de l'apprentissage et sert seulement à choisir le
seuil ; le score est celui du fold k, jamais vu (ni par les poids ni par le seuil). --inner-val 0 : apprentissage
sur les 4 folds, seuil choisi sur leurs prédictions (optimiste pour le seuil, le fold k reste jamais vu).
Le score à seuil fixe 0,5 (aucun choix) est toujours affiché. Sorties : output/pixseg_cv_<tag>.npz (probabilités
hors-fold), output/pixseg_test_<tag>.npz (names, probs float16 n 128 128, thr), output/pixseg_<tag>.pt (poids).
Augmentations : retournement horizontal et recadrage à des décalages multiples de 4 px (le masque 128 reste
exact) ; ni flou, ni redimensionnement, ni JPEG : le signal du watermark est à haute fréquence.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import KIT, METHODS, NBITS, fast_miou, load_images, load_maps, load_meta, load_synth, real_folds  # noqa: E402
from model import PixSeg  # noqa: E402

THRS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def soft_miou_loss(logit, y, eps=1.0):
    """1 - mIoU douce à 2 classes, par image puis moyenne (même forme que la métrique ; eps : union vide -> IoU 1)."""
    p = torch.sigmoid(logit.float()).flatten(1); y = y.float().flatten(1)  # b h w -> b (h w)
    i1 = (p * y).sum(1); u1 = (p + y - p * y).sum(1)
    i0 = ((1 - p) * (1 - y)).sum(1); u0 = (2 - p - y).sum(1) - i0
    return 1 - 0.5 * ((i1 + eps) / (u1 + eps) + (i0 + eps) / (u0 + eps)).mean()


def make_pool(images, masks, method, extra=None, msg=None) -> dict:
    """images uint8 n 512 512 3, masks n 128 128 (ou None), method n (noms), extra float16 n E 128 128 (ou None).
    msg (n nbits, +-1 float) : si présent, batch() le renvoie À LA PLACE de l'indice de méthode (--ext --msg)."""
    return {"img": torch.from_numpy(images),
            "meth": torch.tensor([METHODS.index(m) for m in method]) if msg is None else torch.from_numpy(msg),
            "mask": None if masks is None else torch.from_numpy(np.ascontiguousarray(masks, dtype=np.uint8)),
            "extra": None if extra is None else torch.from_numpy(extra)}


def batch(pool, idx, dev, crop=0, flip=False, g=None, maps_drop=0.0):
    """-> x (b 3 c c float dans [0, 1]), méthode (b), y (b c/4 c/4 float ou None), extra (b E c/4 c/4 ou None)."""
    c = crop or 512
    xs, ys, es = [], [], []
    for i in idx.tolist():
        oy, ox = (4 * torch.randint(0, (512 - c) // 4 + 1, (2,), generator=g)).tolist()  # multiples de 4
        f = flip and torch.rand((), generator=g).item() < 0.5
        x = pool["img"][i, oy:oy + c, ox:ox + c]  # c c 3
        xs.append(x.flip(1) if f else x)
        for key, out in (("mask", ys), ("extra", es)):
            if pool[key] is not None:
                m = pool[key][i, ..., oy // 4:(oy + c) // 4, ox // 4:(ox + c) // 4]  # (E) c/4 c/4
                if key == "extra" and maps_drop and torch.rand((), generator=g).item() < maps_drop:
                    m = torch.zeros_like(m)
                out.append(m.flip(-1) if f else m)
    x = torch.stack(xs).to(dev).permute(0, 3, 1, 2).float() / 255  # b c c 3 -> b 3 c c
    y = torch.stack(ys).to(dev).float() if ys else None
    e = torch.stack(es).to(dev).float() if es else None
    return x, pool["meth"][idx].to(dev), y, e


def build(a, pool, dev):
    if a.ext:
        from extmodel import ExtSeg
        return ExtSeg(a.ext, dev, a.in_res, a.width, a.enc_lr > 0, a.ext_scratch, a.msg).to(dev).to(memory_format=torch.channels_last)
    extra = 0 if pool["extra"] is None else pool["extra"].shape[1]
    net = PixSeg(a.base, a.backbone, not a.no_pretrained, extra, in_scale=a.in_scale, hp_mode=a.hp_mode,
                 hp_luma=a.hp_luma).to(dev)
    return net.to(memory_format=torch.channels_last)


def train(a, real, ridx, synth, sidx, dev, seed=0):
    """Mélange par batch : round(bs * real_frac) images réelles (indices ridx), le reste synthétique (sidx)."""
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    net = build(a, real, dev)
    ns = 0 if synth is None or len(sidx) == 0 else a.bs
    nr = a.bs - ns if ns == 0 else (0 if len(ridx) == 0 else round(a.bs * a.real_frac))
    ns = a.bs - nr
    assert nr == 0 or len(ridx), "pas d'image réelle d'apprentissage"
    if a.ext:  # encodeur officiel : petit taux (0 = gelé), décodeur : a.lr ; gel de l'encodeur avant a.freeze_steps
        groups = [{"params": net.dec_params(), "lr": a.lr}]
        if a.enc_lr > 0:
            groups.append({"params": net.enc_params(), "lr": a.enc_lr})
        opt = torch.optim.AdamW(groups, weight_decay=a.wd)
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    warm = max(1, a.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * s / a.steps)) * 0.98 + 0.02))
    amp = dev.type == "cuda" and not a.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=amp)
    losses, t0 = [], time.time()
    net.train()
    for step in range(a.steps):
        if a.ext and a.enc_lr > 0 and a.freeze_steps:
            net.train_enc = step >= a.freeze_steps
        parts = []
        if nr:
            parts.append(batch(real, ridx[torch.randint(0, len(ridx), (nr,), generator=g)], dev, a.crop, not a.no_flip,
                               g, a.maps_drop))
        if ns:
            parts.append(batch(synth, sidx[torch.randint(0, len(sidx), (ns,), generator=g)], dev, a.crop,
                               not a.no_flip, g, a.maps_drop))
        x, m, y, e = (None if p[0] is None else torch.cat(p) for p in zip(*parts))  # b = nr + ns
        with torch.autocast(dev.type, enabled=amp):
            logit = net(x.contiguous(memory_format=torch.channels_last), m, e)  # b c/4 c/4
        logit = logit.float()
        loss = a.bce_w * F.binary_cross_entropy_with_logits(logit, y) + a.iou_w * soft_miou_loss(logit, y)
        assert torch.isfinite(loss), f"perte non finie au pas {step}"
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        losses.append(loss.item())
        if step % a.log_every == 0 or step == a.steps - 1:
            print(f"  pas {step}/{a.steps} perte {np.mean(losses[-a.log_every:]):.4f} "
                  f"({nr} réelles + {ns} synth. par batch) {(step + 1) / (time.time() - t0):.2f} it/s", flush=True)
    return net, losses


@torch.no_grad()
def predict(net, pool, idx, dev, bs=32) -> np.ndarray:
    """Probabilités (n 128 128 float32) sur l'image entière 512, sans augmentation."""
    net.eval(); out = []
    amp = dev.type == "cuda"
    for i in range(0, len(idx), bs):
        x, m, _, e = batch(pool, idx[i:i + bs], dev)
        with torch.autocast(dev.type, enabled=amp):
            logit = net(x.contiguous(memory_format=torch.channels_last), m, e)  # b 128 128
        out.append(torch.sigmoid(logit.float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 128, 128), np.float32)


def best_thr(p, gt) -> float:
    return float(THRS[int(np.argmax([fast_miou(p > t, gt).mean() for t in THRS]))])


def group_line(name, s, method, full) -> str:
    line = f"{name:34s} all={s.mean():.4f} n={len(s)} |"
    for m in METHODS:
        sel = method == m
        if sel.any():
            line += f" {m[:5]}={s[sel].mean():.3f}"
        for fw in (True, False):
            q = sel & (full == fw)
            if q.any():
                line += f" [{'F' if fw else 'R'} {s[q].mean():.3f} n={q.sum()}]"
    return line


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true"); ap.add_argument("--fit-all", action="store_true")
    ap.add_argument("--predict-test", action="store_true")
    ap.add_argument("--synth", nargs="*", default=[], help="tranches .npz de data.py gen (fichiers ou motifs)")
    ap.add_argument("--limit-synth", type=int, default=0)
    ap.add_argument("--in-scale", type=float, default=1.0, help="0.5 : image réduite à 256 avant le réseau")
    ap.add_argument("--hp-mode", default="bin5", choices=["bin5", "lowres", "dog"])
    ap.add_argument("--hp-luma", action="store_true", help="passe-haut sur la luminance seule")
    ap.add_argument("--synth-methods", default="", help="ne garder que ces méthodes du synthétique (ex. pixelseal,trustmark)")
    ap.add_argument("--synth-holdout", type=int, default=0,
                    help="n images synthétiques jamais vues à l'apprentissage ; mIoU sur elles après chaque fold")
    ap.add_argument("--real-frac", type=float, default=0.25, help="part réelle de chaque batch (0 = synthétique seul)")
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bce-w", type=float, default=1.0); ap.add_argument("--iou-w", type=float, default=1.0)
    ap.add_argument("--crop", type=int, default=448, help="côté du recadrage d'apprentissage (0 = image entière)")
    ap.add_argument("--no-flip", action="store_true"); ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--ext", default="", choices=["", "pixelseal", "trustmark"],
                    help="encodeur = extracteur officiel de cette méthode (extmodel.ExtSeg) ; implique --only-method")
    ap.add_argument("--in-res", type=int, default=512, help="--ext : côté de l'image à l'entrée de l'encodeur")
    ap.add_argument("--enc-lr", type=float, default=1e-4, help="--ext : taux de l'encodeur (0 = gelé)")
    ap.add_argument("--freeze-steps", type=int, default=0, help="--ext : encodeur gelé pendant les n premiers pas")
    ap.add_argument("--width", type=int, default=96, help="--ext : largeur du décodeur")
    ap.add_argument("--ext-scratch", action="store_true", help="--ext : encodeur ré-initialisé (contrôle)")
    ap.add_argument("--msg", action="store_true", help="--ext : message connu (metadata / rejeu du synthétique) en entrée")
    ap.add_argument("--only-method", default="", help="réel : apprentissage, seuil et score sur cette méthode seule")
    ap.add_argument("--no-save", action="store_true", help="--fit-all : ne pas écrire les poids")
    ap.add_argument("--base", type=int, default=24, help="largeur du U-Net from scratch (multiple de 8)")
    ap.add_argument("--backbone", default="", help="encodeur timm pré-entraîné (ex. convnext_tiny), désactivé par défaut")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--extra-maps", default="", help="cartes train : 'fichier_ou_motif.npz:cle1+cle2'")
    ap.add_argument("--extra-maps-test", default=""); ap.add_argument("--maps-drop", type=float, default=0.3)
    ap.add_argument("--folds", default="0,1,2,3,4"); ap.add_argument("--inner-val", type=int, default=1)
    ap.add_argument("--thr", type=float, default=0.5, help="seuil noté dans le fichier test (--fit-all)")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--data", default=None, help="dossier release (défaut : KIT/release)")
    ap.add_argument("--ref-heats", default="", help="heats_train.npz : vérifie que l'ordre des images est celui de gated_eval")
    ap.add_argument("--out", default="output/pixseg")
    ap.add_argument("--tag", default=os.environ.get("SLURM_JOB_ID", "local") + "_" + os.environ.get("SLURM_LOCALID", "0"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    assert a.cv or a.fit_all, "--cv ou --fit-all"
    if a.ext and not a.only_method:
        a.only_method = a.ext
    assert a.crop % 32 == 0 and 0 <= a.crop <= 512 and 0 <= a.real_frac <= 1
    return a


def main(argv=None):
    a = parse(argv)
    dev = torch.device(a.device)
    torch.backends.cudnn.benchmark = True
    print("config", vars(a), flush=True)
    R = load_meta("train", a.data); n = len(R["names"])
    if a.ref_heats:
        assert np.load(a.ref_heats)["names"].tolist() == R["names"].tolist(), "ordre != heats_train.npz : folds différents"
    extra, stats = load_maps(a.extra_maps, R["names"].tolist()) if a.extra_maps else (None, None)
    def real_msgs(split, names):  # +-1, 0 pour les images d'une autre méthode (jamais utilisées avec --ext)
        if not a.msg:
            return None
        import json
        mm = {m["image_name"]: m for m in json.loads((Path(a.data or KIT / "release") / split / "metadata.json").read_text())}
        out = np.zeros((len(names), NBITS[a.ext]), np.float32)  # n nbits
        for i, nm in enumerate(names.tolist()):
            if mm[nm]["method"] == a.ext:
                out[i] = 2.0 * np.array([int(c) for c in mm[nm]["message"]], np.float32) - 1
        return out
    real = make_pool(load_images("train", R["names"], a.data), R["masks"], R["method"], extra, real_msgs("train", R["names"]))
    gt, method, full = R["masks"].astype(bool), R["method"], R["full"]
    synth, S = None, None
    if a.synth:
        S = load_synth(a.synth, a.limit_synth)
        if a.synth_methods:
            keep = np.isin(S["method"], a.synth_methods.split(","))
            S = {k: v[keep] for k, v in S.items()}
        es = load_maps(a.extra_maps, S["names"].tolist(), stats)[0] if a.extra_maps else None
        smsg = None
        if a.msg:
            from msgs import get as get_msgs
            d = get_msgs(a.ext)
            smsg = 2.0 * np.stack([d[nm] for nm in S["names"].tolist()]).astype(np.float32) - 1  # n nbits
        synth = make_pool(S["images"], S["mask128"], S["method"], es, smsg)
        print(f"synthétique : {len(S['names'])} images, couverture {S['mask128'].mean():.3f}, "
              f"F {S['full'].mean():.2f}, par méthode {[int((S['method'] == m).sum()) for m in METHODS]}", flush=True)
    print(f"réel : {n} images train, couverture {gt.mean():.3f}", flush=True)

    hold = np.zeros(0, int)
    if S is not None and a.synth_holdout:
        hold = np.sort(np.random.default_rng(1).permutation(len(S["names"]))[: a.synth_holdout])

    def synth_idx(held):  # écarte les composites dont une source est une image évaluée, et la réserve synthétique
        if S is None:
            return torch.zeros(0, dtype=torch.long)
        bad = set(R["names"][held].tolist())
        ok = np.array([s not in bad and s2 not in bad for s, s2 in zip(S["src"].tolist(), S["src2"].tolist())])
        ok[hold] = False
        return torch.from_numpy(np.nonzero(ok)[0])

    def report_synth(net, k):
        if not len(hold):
            return
        p = predict(net, synth, torch.from_numpy(hold), dev); g = S["mask128"][hold].astype(bool)
        sm, sf = S["method"][hold], S["full"][hold].astype(bool)
        print(group_line(f"fold {k} SYNTH réserve seuil 0,5", fast_miou(p > 0.5, g), sm, sf), flush=True)
        print(group_line(f"fold {k} SYNTH tout 0", fast_miou(np.zeros_like(g), g), sm, sf), flush=True)
        print(group_line(f"fold {k} SYNTH tout 1", fast_miou(np.ones_like(g), g), sm, sf), flush=True)

    allidx = torch.arange(n)
    sel = (method == a.only_method) if a.only_method else np.ones(n, bool)  # images réelles de la méthode visée
    if a.cv:
        fo = real_folds(method, full, 0)
        ks = [int(k) for k in a.folds.split(",")]
        oof = np.full((n, 128, 128), np.nan, np.float32); thr_img = np.full(n, np.nan, np.float32)
        s0, s1 = fast_miou(np.zeros_like(gt), gt), fast_miou(np.ones_like(gt), gt)
        tri = fast_miou(np.broadcast_to(full[:, None, None], gt.shape), gt)
        print(group_line("tout 0", s0, method, full)); print(group_line("tout 1", s1, method, full))
        print(group_line("tri seul (oracle full_watermark)", tri, method, full), flush=True)
        for k in ks:
            te0 = fo == k  # folds calculés sur les 510 images (identiques à gated_eval), puis restreints
            va0 = (fo == (k + 1) % 5) if a.inner_val else ~te0
            tr = (~te0 & ~va0 if a.inner_val else ~te0) & sel
            te, va = te0 & sel, va0 & sel
            print(f"== fold {k} : {tr.sum()} réelles en apprentissage, {va.sum()} pour le seuil, {te.sum()} évaluées", flush=True)
            net, _ = train(a, real, allidx[torch.from_numpy(tr)], synth, synth_idx(te0 | va0), dev, seed=100 * a.seed + k)
            t = best_thr(predict(net, real, allidx[torch.from_numpy(va)], dev), gt[va])
            oof[te] = predict(net, real, allidx[torch.from_numpy(te)], dev); thr_img[te] = t
            print(group_line(f"fold {k} seuil {t:.2f}", fast_miou(oof[te] > t, gt[te]), method[te], full[te]), flush=True)
            print(group_line(f"fold {k} seuil 0,5", fast_miou(oof[te] > 0.5, gt[te]), method[te], full[te]), flush=True)
            report_synth(net, k)
            del net
        d = np.isfinite(thr_img)  # images des folds faits
        pred = oof > thr_img[:, None, None]
        gate = pred.mean((1, 2)) > 0.5
        print(f"== résultat sur les folds {ks} ({d.sum()} images)")
        for name, s in (("tout 0", s0), ("tout 1", s1), ("tri seul (oracle full_watermark)", tri),
                        ("tri seul (réseau, couverture > 0,5)", fast_miou(np.broadcast_to(gate[:, None, None], gt.shape), gt)),
                        ("réseau seuil 0,5", fast_miou(oof > 0.5, gt)), ("réseau seuil choisi (imbriqué)", fast_miou(pred, gt))):
            print(group_line(name, s[d], method[d], full[d]), flush=True)
        out = f"{a.out}_cv_{a.tag}.npz"
        np.savez_compressed(out, names=R["names"], probs=oof.astype(np.float16), fold=fo, thr=thr_img,
                            miou=fast_miou(pred, gt), config=str(vars(a)))
        print(f"seuil moyen {np.nanmean(thr_img):.3f} ; écrit {out}", flush=True)

    if a.fit_all:
        print(f"== apprentissage sur les {n} images train", flush=True)
        net, _ = train(a, real, allidx[torch.from_numpy(sel)], synth, synth_idx(np.zeros(n, bool)), dev, seed=100 * a.seed + 99)
        if not a.no_save:
            torch.save(net.state_dict(), f"{a.out}_{a.tag}.pt")
        report_synth(net, "all")
        if a.predict_test:
            T = load_meta("test", a.data)
            et = load_maps(a.extra_maps_test, T["names"].tolist(), stats)[0] if a.extra_maps else None
            assert (et is None) or bool(et[:, -1].all()), "cartes test manquantes pour certaines images"
            if a.only_method:  # test : seulement les images de la méthode visée
                kt = T["method"] == a.only_method
                T = {k: v[kt] for k, v in T.items()}
            test = make_pool(load_images("test", T["names"], a.data), None, T["method"], et, real_msgs("test", T["names"]))
            p = predict(net, test, torch.arange(len(T["names"])), dev)
            assert np.isfinite(p).all() and p.shape == (len(T["names"]), 128, 128)
            out = f"{a.out}_test_{a.tag}.npz"
            np.savez_compressed(out, names=T["names"], probs=p.astype(np.float16), thr=a.thr, method=T["method"],
                                config=str(vars(a)))
            print(f"test : {len(p)} images, couverture à {a.thr} : {(p > a.thr).mean():.3f} ; écrit {out}", flush=True)


if __name__ == "__main__":
    main()
