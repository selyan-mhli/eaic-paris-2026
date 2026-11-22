"""Candidat « TrustMark dense seul », recette FIGÉE du pilote 15903857 (TM, tête seule). Deux étapes, 1 GPU.

  measure : refait EXACTEMENT dev2, dev3, dev4 (folds 2-4) et confirm (apprend folds 2-4, prédit 0-1) du pilote :
            mêmes arguments (relus dans le `config` du npz pilote), mêmes graines (k, 99), mêmes découpes seuil,
            même synthétique ; contrôle : hash des noms appris == `train_names_hash` du pilote, sinon arrêt.
            Prédit TOUTES les TM (vrais F et vrais R) des folds évalués ; sauvegarde les 4 têtes (un .pt) ;
            seuil FIGÉ = seuil du pilote pour le fold (jamais réglé ici) ; le seuil interne refait est rapporté.
  export  : apprend la tête sur les 85 TM F train (synthétique d'apprentissage complet, graine 99, mêmes
            hyperparamètres, nombre de pas `steps_for`), prédit les 340 TM test (message de test/metadata.json),
            puis COPIE du champion où seuls les TM test routés F (acc_full >= gate figée) reçoivent prob > seuil figé.

  python tm_job.py measure --pilot output/dense/res_trustmark_head_15903857.npz --out output/dense/tmc_measure_<job>
  python tm_job.py export  --pilot ... --champion output/c1_pixss_fit0.pt --heats-test output/heats_test.npz \
      --decision output/dense/tmc_measure_<job>.json --out output/c1_tmdense_pixss_<job>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import data as pdata  # noqa: E402
import dense_train as dt  # noqa: E402  fichiers figés du pilote : importés, jamais modifiés

METHOD = "trustmark"
CHAMPION_SHA = "da6abc314aa7b5939fc76935096777adb6213bbb6653eb190b3fda60bcc4f3d8"
TEST_GATE = 0.65  # règle du champion au test (log 15902420_1.out : "gate": 0.65), ré-affirmée par --gate
TEST_THR = 0.70   # seuil interne du pilote en confirmation (res_trustmark_head_15903857.npz, phase confirm)


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def pilot_args(pilot, device, data, fake=None):
    """Arguments EXACTS du pilote (config enregistrée), seuls device/data/out (et fake en test) changent."""
    cfg = json.loads(str(np.load(pilot)["config"]))
    assert cfg["method"] == METHOD and cfg["mode"] == "head", cfg
    argv = []
    for k, v in cfg.items():
        if k in ("device", "data", "out", "phases"):
            continue
        key = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            argv += [key] if v else []
        elif isinstance(v, list):
            argv += [key, *map(str, v)]
        elif v in ("", None):
            continue
        else:
            argv += [key, str(v)]
    if fake is not None:
        argv += ["--fake", fake]
    argv += ["--out", "unused", "--device", device] + (["--data", data] if data else [])
    a = dt.parse(argv)
    a.synth = cfg["synth"] if a.synth is None else a.synth
    return a, cfg


def head_state(net) -> dict:
    """Tête + buffers de normalisation (le backbone officiel gelé n'est pas sauvegardé)."""
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items() if not k.startswith("bb.")}


def tm_pool(split, names, data):
    """Images et messages (metadata) de `names`, masques nuls : pour prédire hors F connus (R, test)."""
    root = Path(data or pdata.KIT / "release") / split
    meta = {m["image_name"]: m for m in json.loads((root / "metadata.json").read_text())}
    assert all(meta[n]["method"] == METHOD for n in names)
    msg = np.stack([np.array([int(c) for c in meta[n]["message"]], np.uint8) for n in names])  # n 100
    assert msg.shape[1] == dt.NBITS[METHOD], msg.shape
    imgs = pdata.load_images(split, names, data)  # n 512 512 3
    return dt.make_pool(imgs, np.zeros((len(names), 128, 128), np.uint8), msg)


def setup(a):
    R = dt.load_real(METHOD, a.data)
    S = dt.load_synth(a.synth, METHOD)
    real, synth = dt.make_pool(R["images"], R["mask128"], R["msg"]), dt.make_pool(S["images"], S["mask128"], S["msg"])
    s_train_ok, s_val, held = dt.synth_source_split(S, a.synth_val_frac, a.seed)
    return R, S, real, synth, s_train_ok


def synth_idx(S, s_train_ok, bad):
    ok = s_train_ok & ~np.isin(S["src"], list(bad)) & ~np.isin(S["src2"], list(bad))
    return torch.from_numpy(np.nonzero(ok)[0])


def thr_split(a, train_rows):  # = dense_train.run.thr_split
    rng = np.random.default_rng(1000 + a.seed)
    perm = train_rows[rng.permutation(len(train_rows))]
    k = max(1, int(round(a.thr_frac * len(perm))))
    return np.sort(perm[k:]), np.sort(perm[:k])


# ---------------------------------------------------------------- étape 1 : mesure
def measure(args):
    a, cfg = pilot_args(args.pilot, args.device, args.data, args.fake)
    dev = torch.device(a.device); torch.backends.cudnn.benchmark = True
    P = np.load(args.pilot); p_hash = json.loads(str(P["train_names_hash"]))
    p_thr = dict(zip(P["names"].tolist(), P["thr"].tolist())); p_fold = dict(zip(P["names"].tolist(), P["fold"].tolist()))
    R, S, real, synth, s_train_ok = setup(a)
    assert R["names"].tolist() == P["names"].tolist(), "F TM différents du pilote"
    gt = R["mask128"].astype(bool)
    # toutes les TM du train (F et R), folds de la partition 0
    tm = np.nonzero(R["all_method"] == METHOD)[0]
    tm_names, tm_fold, tm_full = R["all_names"][tm], R["all_fold"][tm], R["all_full"][tm]
    pool = tm_pool("train", tm_names.tolist(), a.data)
    probs = np.full((len(tm), 128, 128), np.nan, np.float16)
    thr_fixed = np.full(len(tm), np.nan, np.float32); thr_rerun = np.full(len(tm), np.nan, np.float32)
    tag_of = np.array([""] * len(tm), dtype="<U8"); heads, rep = {}, {}
    runs = [(f"dev{k}", (k,), tuple(f for f in dt.DEV_FOLDS if f != k), 100 * a.seed + k) for k in dt.DEV_FOLDS]
    runs.append(("confirm", dt.CONF_FOLDS, dt.DEV_FOLDS, 100 * a.seed + 99))
    for tag, ev_folds, tr_folds, seed in runs:
        t0 = time.time()
        tr_all = np.nonzero(np.isin(R["fold"], tr_folds))[0]
        ev = np.nonzero(np.isin(R["fold"], ev_folds))[0]
        tr, th = thr_split(a, tr_all)
        held = set(R["names"][np.concatenate([ev, th])].tolist())
        held |= set(R["all_names"][np.isin(R["all_fold"], ev_folds)].tolist())
        sidx = synth_idx(S, s_train_ok, held)
        h = dt.names_hash(list(R["names"][tr]) + list(S["names"][sidx.numpy()]))
        steps = dt.steps_for(a, len(sidx))
        print(f"== {tag} : {len(tr)} réelles, {len(th)} seuil, {len(sidx)} synth., {steps} pas, hash {h} "
              f"(pilote {p_hash.get(tag)})", flush=True)
        if h != p_hash.get(tag) and not args.allow_hash_mismatch:
            raise SystemExit(f"recette différente du pilote pour {tag} : hash {h} != {p_hash.get(tag)}")
        net, losses = dt.train_run(a, dev, real, torch.from_numpy(tr), synth, sidx, steps, seed)
        t_re = dt.best_thr(dt.predict(net, real, torch.from_numpy(th), dev), gt[th])
        sel = np.nonzero(np.isin(tm_fold, ev_folds))[0]
        probs[sel] = dt.predict(net, pool, torch.from_numpy(sel), dev).astype(np.float16)
        fixed = {p_thr[n] for n in R["names"][ev].tolist()}
        assert len(fixed) == 1, fixed
        thr_fixed[sel] = fixed.pop(); thr_rerun[sel] = t_re; tag_of[sel] = tag
        heads[tag] = head_state(net)
        pF = probs[sel][tm_full[sel]].astype(np.float32)
        mF = pdata.fast_miou(pF > thr_fixed[sel][0], np.stack([gt[list(R["names"]).index(n)] for n in tm_names[sel][tm_full[sel]]]))
        rep[tag] = {"steps": steps, "hash": h, "hash_pilote": p_hash.get(tag), "thr_pilote": float(thr_fixed[sel][0]),
                    "thr_refait": t_re, "miou_F_thr_pilote": float(mF.mean()), "n_eval_tm": int(len(sel)),
                    "loss_last": float(np.mean(losses[-50:])), "secondes": round(time.time() - t0, 1)}
        print(f"   {tag} : {json.dumps(rep[tag])}", flush=True)
        del net
    # reproductibilité contre le pilote, sur les F
    pp = P["probs_oof"].astype(np.float32); ip = {n: i for i, n in enumerate(tm_names.tolist())}
    rows = [ip[n] for n in P["names"].tolist()]
    m_pil = pdata.fast_miou(pp > P["thr"][:, None, None], gt)
    m_new = pdata.fast_miou(probs[rows].astype(np.float32) > thr_fixed[rows][:, None, None], gt)
    rep["repro_F"] = {"n": int(len(rows)), "miou_pilote": float(m_pil.mean()), "miou_refait": float(m_new.mean()),
                      "ecart_abs_moyen": float(np.abs(m_new - m_pil).mean()), "ecart_max": float(np.abs(m_new - m_pil).max())}
    print("== reproductibilité F :", json.dumps(rep["repro_F"]), flush=True)
    torch.save({"heads": heads, "config": cfg, "pilot": str(args.pilot)}, f"{args.out}_heads.pt")
    np.savez_compressed(f"{args.out}.npz", names=tm_names, fold=tm_fold, full=tm_full, probs=probs, thr=thr_fixed,
                        thr_refait=thr_rerun, run=tag_of, report=json.dumps(rep), config=json.dumps(cfg))
    print(f"écrit {args.out}.npz et {args.out}_heads.pt", flush=True)
    return f"{args.out}.npz", rep


# ---------------------------------------------------------------- étape 2 : export
def export(args):
    if not args.force:
        assert args.decision, "--decision (json de measure_routed.py) requis sans --force"
        d = json.loads(Path(args.decision).read_text())
        if not d.get("export_ok"):
            print(f"EXPORT NON FAIT : décision {d.get('decision')} ({args.decision})", flush=True)
            return None
    a, cfg = pilot_args(args.pilot, args.device, args.data, args.fake)
    P = np.load(args.pilot)
    conf_thr = sorted(set(P["thr"][P["phase"] == "confirm"].tolist()))
    ok_thr = len(conf_thr) == 1 and abs(conf_thr[0] - args.thr) < 1e-6  # thr stocké en float32 (0.69999999)
    assert ok_thr or args.fake, f"seuil figé {args.thr} != seuil confirm du pilote {conf_thr}"
    ch_sha = sha256(args.champion)
    assert ch_sha == CHAMPION_SHA or args.fake, f"champion inattendu {ch_sha}"
    dev = torch.device(a.device); torch.backends.cudnn.benchmark = True
    R, S, real, synth, s_train_ok = setup(a)
    tr = torch.from_numpy(np.arange(len(R["names"])))  # les 85 TM F
    sidx = torch.from_numpy(np.nonzero(s_train_ok)[0])  # synthétique d'apprentissage (réserve exclue)
    steps = dt.steps_for(a, len(sidx))
    print(f"== export : {len(tr)} TM F réelles, {len(sidx)} synth., {steps} pas, graine {100 * a.seed + 99}", flush=True)
    net, losses = dt.train_run(a, dev, real, tr, synth, sidx, steps, 100 * a.seed + 99)
    torch.save({"head": head_state(net), "config": cfg, "steps": steps}, f"{args.out}_head.pt")
    # test : routage du champion, sans vérité F/R
    H = np.load(args.heats_test)
    hn, hm, haf = H["names"].tolist(), H["method"], H["acc_full"].astype(np.float32)
    sub = torch.load(args.champion, weights_only=True)
    names = list(sub["image_names"]); masks = sub["masks"].clone().bool()
    assert masks.shape == (len(names), 128, 128) and len(set(names)) == len(names) == len(hn) and set(names) == set(hn)
    meta = {m["image_name"]: m["method"] for m in json.loads((Path(a.data or pdata.KIT / "release") / "test" / "metadata.json").read_text())}
    assert all(meta[n] == m for n, m in zip(hn, hm)), "méthodes heats_test != metadata"
    tm = [i for i, n in enumerate(hn) if hm[i] == METHOD]
    tm_names = [hn[i] for i in tm]
    probs = dt.predict(net, tm_pool("test", tm_names, a.data), torch.arange(len(tm)), dev)  # n_tm 128 128
    routed = haf[tm] >= args.gate
    pos = {n: i for i, n in enumerate(names)}
    old = masks.clone()
    for j, n in enumerate(tm_names):
        if routed[j]:
            masks[pos[n]] = torch.from_numpy(probs[j] > args.thr)
    changed = (old != masks).flatten(1).any(1)
    allowed = torch.zeros(len(names), dtype=torch.bool)
    allowed[[pos[n] for j, n in enumerate(tm_names) if routed[j]]] = True
    assert not (changed & ~allowed).any(), "image hors TM routées F modifiée"
    pt = f"{args.out}.pt"
    torch.save({"image_names": names, "masks": masks}, pt)
    back = torch.load(pt, weights_only=True)
    assert back["image_names"] == names and back["masks"].shape == (len(names), 128, 128) and back["masks"].dtype == torch.bool
    if not args.fake:
        assert back["masks"].shape == (1020, 128, 128) and len(tm) == 340
    np.savez_compressed(f"{args.out}_tmprobs.npz", names=np.array(tm_names), probs=probs.astype(np.float16),
                        acc_full=haf[tm], routed_F=routed, thr=args.thr, gate=args.gate)
    info = {"sortie": pt, "sha256": sha256(pt), "champion": str(args.champion), "champion_sha256": ch_sha,
            "n_tm_test": len(tm), "n_tm_routes_F": int(routed.sum()), "images_modifiees": int(changed.sum()),
            "pixels_modifies_frac_sur_routes": float((old != masks)[allowed].float().mean()),
            "couverture_routes_avant": float(old[allowed].float().mean()), "couverture_routes_apres": float(masks[allowed].float().mean()),
            "gate": args.gate, "thr": args.thr, "steps": steps, "loss_last": float(np.mean(losses[-50:]))}
    print("== export :", json.dumps(info), "(NON soumis)", flush=True)
    return info


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["measure", "export"])
    ap.add_argument("--pilot", required=True, help="res_trustmark_head_15903857.npz (config, hash, seuils)")
    ap.add_argument("--out", required=True); ap.add_argument("--data", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow-hash-mismatch", action="store_true")
    ap.add_argument("--champion"); ap.add_argument("--heats-test"); ap.add_argument("--decision")
    ap.add_argument("--gate", type=float, default=TEST_GATE); ap.add_argument("--thr", type=float, default=TEST_THR)
    ap.add_argument("--force", action="store_true", help="exporter malgré la décision (sur ordre explicite)")
    ap.add_argument("--fake", default=None, choices=[None, "tiny", "full"], help="tests CPU uniquement")
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = parse()
    measure(args) if args.stage == "measure" else export(args)
