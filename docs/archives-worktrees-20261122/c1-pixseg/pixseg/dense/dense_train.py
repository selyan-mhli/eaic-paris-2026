"""Pilote du localisateur dense, F d'UNE méthode (pixelseal | trustmark), protocole fixé AVANT exécution.

  python dense/dense_train.py --method pixelseal --mode head --synth 'output/dense_syn_pixelseal_s*_*.npz' \
      --out output/dense_ps_head

Phases (--phases, dans cet ordre) :
  control : (a) surapprentissage de 8 composites synthétiques officiels ; (b) apprentissage synthétique seul,
            score sur une réserve synthétique dont les SOURCES sont exclues de l'apprentissage (transfert).
  dev     : folds réels 2, 3, 4 (F de la méthode) ; pour le fold k, apprentissage sur les autres folds de
            développement (dont 30 % gardés pour le seuil) + synthétique ; aucune image des folds 0, 1.
  confirm : apprentissage sur les folds 2, 3, 4 (mêmes 30 % pour le seuil), évaluation UNIQUE des folds 0 et 1.
Folds : real_folds(method|full_watermark, 5, 0) sur les 510 images dans l'ordre de metadata.json.
Synthétique : composites dont src ou src2 est une image évaluée ou de seuil (toutes méthodes) écartés ; la
réserve synthétique (sources tenues à l'écart) n'est jamais apprise.
Mode head : backbone gelé. Mode last : dernier stage dégelé, LR backbone = LR tête / 10, perte de préservation
des logits globaux (MSE contre une copie figée du stage initial).
Normalisations (échelle de l'accord, passe-haut) : fit sur des images d'apprentissage de chaque run.
Pas d'augmentation (aucune transformation après incrustation). Sortie : un seul npz compressé.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import data as pdata  # noqa: E402  c1/pixseg/data.py : folds, métrique, chargement
from train import soft_miou_loss  # noqa: E402  c1/pixseg/train.py : perte mIoU douce déjà validée
from decoder_features import NBITS, fake_backbone, load_official  # noqa: E402
from locmodel import DenseLoc  # noqa: E402

THRS = np.round(np.arange(0.05, 0.96, 0.05), 2)
DEV_FOLDS, CONF_FOLDS = (2, 3, 4), (0, 1)


# ---------------------------------------------------------------- données
def load_real(method, data=None, with_images=True):
    """Tous les noms train (ordre metadata) + folds ; images/messages des seuls F de `method`."""
    root = Path(data or pdata.KIT / "release") / "train"
    meta = json.loads((root / "metadata.json").read_text())
    R = pdata.load_meta("train", data)
    assert R["names"].tolist() == [m["image_name"] for m in meta]
    fold = pdata.real_folds(R["method"], R["full"], 0)
    sel = np.nonzero((R["method"] == method) & R["full"])[0]
    msg = np.stack([np.array([int(c) for c in meta[i]["message"]], np.uint8) for i in sel])  # n nbits
    assert msg.shape[1] == NBITS[method], msg.shape
    out = {"all_names": R["names"], "all_fold": fold, "all_method": R["method"], "all_full": R["full"],
           "names": R["names"][sel], "fold": fold[sel], "mask128": R["masks"][sel], "msg": msg, "rows": sel}
    out["images"] = pdata.load_images("train", out["names"], data) if with_images else None
    return out


def load_synth(patterns, method):
    files = sorted({f for p in patterns for f in glob.glob(p)}); assert files, patterns
    parts = [dict(np.load(f)) for f in files]
    S = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    assert (S["method"] == method).all() and len(set(S["names"].tolist())) == len(S["names"])
    for k in ("msg", "src", "src2", "mask128", "images"):
        assert k in S, f"clé {k} absente : composites non tracés refusés"
    S["files"] = np.array(files)
    return S


def synth_source_split(S, frac=0.2, seed=0):
    """-> (train_ok bool n, val bool n, held set). Réserve : src tenue à l'écart et src2 aussi (ou remplissage
    n'utilisant pas l'autre image) ; apprentissage : ni src ni src2 dans la réserve."""
    srcs = sorted(set(S["src"].tolist()) | set(S["src2"].tolist()))
    rng = np.random.default_rng(seed)
    held = set(np.array(srcs)[rng.permutation(len(srcs))[: max(1, int(round(frac * len(srcs))))]].tolist())
    s1 = np.isin(S["src"], list(held)); s2 = np.isin(S["src2"], list(held))
    fill = S["fill"] if "fill" in S else np.array(["other"] * len(s1))
    val = s1 & (s2 | (fill != "other"))
    return ~s1 & ~s2, val, held


def make_pool(images, masks, msg):
    return {"img": torch.from_numpy(np.ascontiguousarray(images)), "mask": torch.from_numpy(masks.astype(np.uint8)),
            "msg": torch.from_numpy(2.0 * msg.astype(np.float32) - 1)}


def batch(pool, idx, dev):
    """-> x b 3 512 512 [0, 1], y b 128 128, msg b nbits (+-1)."""
    x = pool["img"][idx].to(dev).permute(0, 3, 1, 2).float() / 255  # b 512 512 3 -> b 3 512 512
    return x, pool["mask"][idx].to(dev).float(), pool["msg"][idx].to(dev)


def names_hash(names) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, names))).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- modèle et apprentissage
def build(a, dev):
    train_last = a.mode == "last"
    bb = fake_backbone(a.method, train_last, tiny=a.fake == "tiny") if a.fake else load_official(a.method, dev, train_last=train_last)
    return DenseLoc(bb, NBITS[a.method], use_msg=not a.no_msg, use_rgb=not a.no_rgb).to(dev)


def train_run(a, dev, real, ridx, synth, sidx, steps, seed):
    """Mélange par batch : round(bs * real_frac) réelles (ridx), le reste synthétique (sidx)."""
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    net = build(a, dev)
    nr = 0 if real is None or len(ridx) == 0 else (a.bs if synth is None or len(sidx) == 0 else round(a.bs * a.real_frac))
    ns = a.bs - nr
    assert ns == 0 or len(sidx), "pas de synthétique d'apprentissage"
    # normalisation : images d'apprentissage seulement
    parts = []
    if nr:
        parts.append(batch(real, ridx[torch.randperm(len(ridx), generator=g)[: min(32, len(ridx))]], dev))
    if ns:
        parts.append(batch(synth, sidx[torch.randperm(len(sidx), generator=g)[:32]], dev))
    net.fit_norm(torch.cat([p[0] for p in parts]), torch.cat([p[2] for p in parts]))
    groups = [{"params": net.head_params(), "lr": a.lr}]
    if a.mode == "last":
        groups.append({"params": net.bb.trainable(), "lr": a.lr / 10})
    opt = torch.optim.AdamW(groups, weight_decay=a.wd)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * s / steps)) * 0.98 + 0.02))
    amp = dev.type == "cuda"
    net.train(); losses, t0 = [], time.time()
    for step in range(steps):
        parts = []
        if nr:
            parts.append(batch(real, ridx[torch.randint(0, len(ridx), (nr,), generator=g)], dev))
        if ns:
            parts.append(batch(synth, sidx[torch.randint(0, len(sidx), (ns,), generator=g)], dev))
        x, y, m = (torch.cat(t) for t in zip(*parts))  # b 3 512 512, b 128 128, b nbits
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            out = net(x, m, with_ref=a.mode == "last")
        logit = (out[0] if a.mode == "last" else out).float()
        loss = F.binary_cross_entropy_with_logits(logit, y) + soft_miou_loss(logit, y)
        if a.mode == "last":
            loss = loss + a.pres_w * F.mse_loss(out[1].float(), out[2].float())
        assert torch.isfinite(loss), f"perte non finie au pas {step}"
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for gr in groups for p in gr["params"]], 1.0)
        opt.step(); sched.step(); losses.append(loss.item())
        if step % a.log_every == 0 or step == steps - 1:
            print(f"  pas {step}/{steps} perte {np.mean(losses[-a.log_every:]):.4f} ({nr}+{ns}) "
                  f"{(step + 1) / (time.time() - t0):.2f} it/s", flush=True)
    return net, losses


@torch.no_grad()
def predict(net, pool, idx, dev, bs=16) -> np.ndarray:
    net.eval(); out = []
    for i in range(0, len(idx), bs):
        x, _, m = batch(pool, idx[i:i + bs], dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logit = net(x, m)  # b 128 128
        out.append(torch.sigmoid(logit.float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 128, 128), np.float32)


def best_thr(p, gt) -> float:
    return float(THRS[int(np.argmax([pdata.fast_miou(p > t, gt).mean() for t in THRS]))])


def scores(pred, gt) -> dict:
    """mIoU, IoU des trous (classe 0), IoU watermark, tout-1, sous-groupe trous < 1 %."""
    p = pred.reshape(len(pred), -1).astype(bool); g = gt.reshape(len(gt), -1).astype(bool)
    def iou(c):
        u = ((p == c) | (g == c)).sum(1); return np.where(u == 0, 1.0, ((p == c) & (g == c)).sum(1) / np.maximum(u, 1))
    m = pdata.fast_miou(pred, gt); one = pdata.fast_miou(np.ones_like(gt), gt); small = (1 - g.mean(1)) < 0.01
    return {"n": int(len(m)), "miou": float(m.mean()), "all1": float(one.mean()), "gain_vs_all1": float((m - one).mean()),
            "iou_holes": float(iou(False).mean()), "iou_wm": float(iou(True).mean()), "n_small": int(small.sum()),
            "miou_small": float(m[small].mean()) if small.any() else None,
            "all1_small": float(one[small].mean()) if small.any() else None}


# ---------------------------------------------------------------- protocole
def steps_for(a, n_synth):
    return a.steps or max(1, math.ceil(a.epochs * max(n_synth, 1) / a.bs))


def run(a):
    dev = torch.device(a.device)
    torch.backends.cudnn.benchmark = True
    print("config", json.dumps(vars(a)), flush=True)
    R = load_real(a.method, a.data)
    S = load_synth(a.synth, a.method)
    print(f"réel F {a.method} : {len(R['names'])} images ; folds {np.bincount(R['fold'], minlength=5).tolist()} ; "
          f"synthétique {len(S['names'])} ({len(S['files'])} fichiers)", flush=True)
    if "acc_full" in S and (S["acc_full"] >= 0).any():
        print(f"acc_full synthétique (décodeur officiel) moyenne {S['acc_full'][S['acc_full'] >= 0].mean():.3f}", flush=True)
    real, synth = make_pool(R["images"], R["mask128"], R["msg"]), make_pool(S["images"], S["mask128"], S["msg"])
    gt = R["mask128"].astype(bool)
    s_train_ok, s_val, held = synth_source_split(S, a.synth_val_frac, a.seed)
    conf_names = set(R["all_names"][np.isin(R["all_fold"], CONF_FOLDS)].tolist())
    assert not (set(S["src"].tolist()) | set(S["src2"].tolist())) & conf_names or a.allow_conf_sources, \
        "synthétique construit avec des sources des folds de confirmation"
    n = len(R["names"])
    probs = np.full((n, 128, 128), np.nan, np.float16); thr_img = np.full(n, np.nan, np.float32)
    phase = np.array([""] * n, dtype="<U8")
    report = {"control": {}, "dev": {}, "confirm": {}, "thr": {}}
    hashes, groups = {}, {}

    def synth_idx(excluded_names):
        bad = set(excluded_names)
        ok = s_train_ok & ~np.isin(S["src"], list(bad)) & ~np.isin(S["src2"], list(bad))
        return torch.from_numpy(np.nonzero(ok)[0])

    def thr_split(train_rows):
        rng = np.random.default_rng(1000 + a.seed)
        perm = train_rows[rng.permutation(len(train_rows))]
        k = max(1, int(round(a.thr_frac * len(perm))))
        return np.sort(perm[k:]), np.sort(perm[:k])

    def one(tag, eval_folds, train_folds, seed):
        tr_all = np.nonzero(np.isin(R["fold"], train_folds))[0]
        ev = np.nonzero(np.isin(R["fold"], eval_folds))[0]
        tr, th = thr_split(tr_all)
        held_real = set(R["names"][np.concatenate([ev, th])].tolist())
        held_real |= set(R["all_names"][np.isin(R["all_fold"], eval_folds)].tolist())  # toutes méthodes du fold évalué
        sidx = synth_idx(held_real)
        steps = steps_for(a, len(sidx))
        print(f"== {tag} : {len(tr)} réelles apprises, {len(th)} pour le seuil, {len(ev)} évaluées, "
              f"{len(sidx)} synth., {steps} pas", flush=True)
        net, losses = train_run(a, dev, real, torch.from_numpy(tr), synth, sidx, steps, seed)
        t = best_thr(predict(net, real, torch.from_numpy(th), dev), gt[th])
        p = predict(net, real, torch.from_numpy(ev), dev)
        probs[ev] = p.astype(np.float16); thr_img[ev] = t
        hashes[tag] = names_hash(list(R["names"][tr]) + list(S["names"][sidx.numpy()]))
        groups[tag] = {"real_train": R["names"][tr].tolist(), "real_thr": R["names"][th].tolist(),
                       "synth_sources": sorted(set(S["src"][sidx.numpy()].tolist()) | set(S["src2"][sidx.numpy()].tolist()))}
        res = {"thr": t, "at_thr": scores(p > t, gt[ev]), "at_0.5": scores(p > 0.5, gt[ev]), "loss_last": float(np.mean(losses[-50:]))}
        if len(np.nonzero(s_val)[0]):
            pv = predict(net, synth, torch.from_numpy(np.nonzero(s_val)[0][:256]), dev)
            res["synth_val_0.5"] = scores(pv > 0.5, S["mask128"][np.nonzero(s_val)[0][:256]].astype(bool))
        print(f"   {tag} seuil {t:.2f} : {json.dumps(res['at_thr'])}", flush=True)
        del net
        return ev, res

    phases = a.phases.split(",")
    if "control" in phases:
        tr8 = torch.from_numpy(np.nonzero(s_train_ok)[0][:8])
        net, _ = train_run(a, dev, None, torch.zeros(0, dtype=torch.long), synth, tr8, a.overfit_steps, a.seed)
        p8 = predict(net, synth, tr8, dev); del net
        report["control"]["overfit8"] = scores(p8 > 0.5, S["mask128"][tr8.numpy()].astype(bool))
        vidx = np.nonzero(s_val)[0][:256]
        sidx = torch.from_numpy(np.nonzero(s_train_ok)[0])
        net, _ = train_run(a, dev, None, torch.zeros(0, dtype=torch.long), synth, sidx, a.control_steps, a.seed + 1)
        pv = predict(net, synth, torch.from_numpy(vidx), dev); del net
        sv = scores(pv > 0.5, S["mask128"][vidx].astype(bool))
        report["control"]["synth_val"] = sv
        report["control"]["pass"] = bool(report["control"]["overfit8"]["miou"] >= 0.85 and sv["gain_vs_all1"] >= 0.10)
        print(f"== contrôle : {json.dumps(report['control'])}", flush=True)
    if "dev" in phases:
        for k in DEV_FOLDS:
            ev, res = one(f"dev{k}", (k,), tuple(f for f in DEV_FOLDS if f != k), 100 * a.seed + k)
            phase[ev] = "dev"; report["dev"][str(k)] = res
        d = phase == "dev"
        report["dev"]["pooled"] = scores(probs[d].astype(np.float32) > thr_img[d][:, None, None], gt[d])
        print(f"== dev (folds 2-4) : {json.dumps(report['dev']['pooled'])}", flush=True)
    if "confirm" in phases:
        ev, res = one("confirm", CONF_FOLDS, DEV_FOLDS, 100 * a.seed + 99)
        phase[ev] = "confirm"; report["confirm"] = res
    out = f"{a.out}.npz"
    np.savez_compressed(out, names=R["names"], probs_oof=probs, fold=R["fold"], thr=thr_img, phase=phase,
                        config=json.dumps(vars(a)), train_names_hash=json.dumps(hashes),
                        source_groups=json.dumps(groups), synth_val_sources=json.dumps(sorted(held)),
                        synth_files=S["files"], report=json.dumps(report))
    print(f"écrit {out}", flush=True)
    return out, report


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["pixelseal", "trustmark"])
    ap.add_argument("--mode", default="head", choices=["head", "last"])
    ap.add_argument("--synth", nargs="+", required=True)
    ap.add_argument("--phases", default="control,dev,confirm")
    ap.add_argument("--epochs", type=float, default=20); ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--overfit-steps", type=int, default=300); ap.add_argument("--control-steps", type=int, default=600)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--real-frac", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--pres-w", type=float, default=1.0, help="mode last : poids de préservation des logits")
    ap.add_argument("--thr-frac", type=float, default=0.3); ap.add_argument("--synth-val-frac", type=float, default=0.2)
    ap.add_argument("--no-msg", action="store_true"); ap.add_argument("--no-rgb", action="store_true")
    ap.add_argument("--fake", default="", choices=["", "full", "tiny"], help="tests : backbone à poids aléatoires")
    ap.add_argument("--allow-conf-sources", action="store_true")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--data", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse())
