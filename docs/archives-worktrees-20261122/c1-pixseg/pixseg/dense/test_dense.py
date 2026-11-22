"""Tests CPU du localisateur dense (aucun accès JURECA, aucun GPU).

  /Users/selyan/dev/eaic-paris-2026/.venv/bin/python c1/pixseg/dense/test_dense.py      # exécuteur intégré
  (compatible pytest si disponible : pytest c1/pixseg/dense -q)

PixelSeal réel : si le checkpoint local third_party/videoseal/ckpts/pixelseal_checkpoint.pth existe (même sha256
que models/pixelseal/checkpoint.pth du kit, vérifié le 21/11), les tests de signe et d'égalité aux sorties
officielles utilisent le VRAI détecteur et le VRAI embedder. TrustMark : paquet `trustmark` absent en local,
ResNet-50 torchvision à poids aléatoires et mêmes noms de modules (layer1..4, fc).
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import data as pdata  # noqa: E402
import dense_train as dt  # noqa: E402
import synth_msg  # noqa: E402
from decoder_features import VIDEOSEAL_LOCAL, fake_backbone, load_local_pixelseal, resize_official  # noqa: E402
from locmodel import DenseLoc  # noqa: E402

CKPT = VIDEOSEAL_LOCAL / "ckpts" / "pixelseal_checkpoint.pth"
torch.set_num_threads(8)


def local_card(tmp: Path) -> Path:
    txt = (VIDEOSEAL_LOCAL / "videoseal" / "cards" / "pixelseal.yaml").read_text()
    txt = txt.replace("https://dl.fbaipublicfiles.com/videoseal/pixelseal/checkpoint.pth", str(CKPT))
    txt = txt.replace("configs/attenuation.yaml", str(VIDEOSEAL_LOCAL / "configs" / "attenuation.yaml"))
    p = tmp / "pixelseal_local.yaml"; p.write_text(txt); return p


def smooth_image(rng, n=512):
    from scipy import ndimage
    x = ndimage.gaussian_filter(rng.random((n, n, 3)).astype(np.float32), (6, 6, 0))
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    return (40 + 170 * x).astype(np.uint8)


def pattern(msg, n=512):
    """Motif factice asymétrique (dépend de x, y et du message) : détecte un retournement après incrustation."""
    yy, xx = np.mgrid[0:n, 0:n]
    ph = float(msg[:8].astype(np.float32) @ (2.0 ** -np.arange(1, 9)))
    return (3 * np.sign(np.sin(0.9 * xx + 0.4 * yy + 6.28 * ph) + 0.3 * (xx > n // 3))).astype(np.int16)


def fake_embed(img, msg):
    out = img.astype(np.int16) + pattern(msg, img.shape[0])[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def make_release(tmp: Path, per=10):
    """Release factice : 3 méthodes x F/R x `per` images, masques 128, messages, quelques images test."""
    from PIL import Image
    rng = np.random.default_rng(0)
    meta, masks, names = [], [], []
    for split, n in (("train", per), ("test", 2)):
        (tmp / split / "images").mkdir(parents=True)
        sm = []
        for m in pdata.METHODS:
            for full in (True, False):
                for j in range(n):
                    nm = f"{split[:2]}{m[:2]}{int(full)}{j:03d}"
                    Image.fromarray(smooth_image(rng)).save(tmp / split / "images" / f"{nm}.png")
                    e = {"image_name": nm, "method": m}
                    if split == "train":
                        e["full_watermark"] = full
                        e["message"] = "".join(map(str, rng.integers(0, 2, pdata.NBITS[m])))
                        mk = np.ones((128, 128), np.uint8) if full else np.zeros((128, 128), np.uint8)
                        mk[10:40, 20:60] = 1 - mk[10:40, 20:60]
                        masks.append(mk); names.append(nm)
                    sm.append(e)
        (tmp / split / "metadata.json").write_text(json.dumps(sm))
    np.savez(tmp / "train" / "masks_128.npz", names=np.array(names), masks=np.stack(masks))
    return tmp


def gen_args(tmp, method, n=24, seed=0):
    return synth_msg.parse(["--method", method, "--n", str(n), "--seed", str(seed), "--out", str(tmp / "syn"),
                            "--shard", "12", "--data", str(tmp), "--device", "cpu"])


# ---------------------------------------------------------------- noms, folds, sources
def test_names_folds():
    with tempfile.TemporaryDirectory() as d:
        root = make_release(Path(d))
        R = dt.load_real("pixelseal", root, with_images=False)
        meta = json.loads((root / "train" / "metadata.json").read_text())
        assert R["all_names"].tolist() == [m["image_name"] for m in meta]
        ref = pdata.folds(np.array([f"{m['method']}|{m['full_watermark']}" for m in meta]), 5, 0)
        assert np.array_equal(R["all_fold"], ref)
        assert all(meta[i]["method"] == "pixelseal" and meta[i]["full_watermark"] for i in R["rows"])
        assert np.array_equal(R["msg"][0], np.array([int(c) for c in meta[R["rows"][0]]["message"]]))
        assert set(np.unique(R["fold"])) == set(range(5)) and set(dt.DEV_FOLDS).isdisjoint(dt.CONF_FOLDS)


def test_synth_src_src2_flip_before_embed():
    with tempfile.TemporaryDirectory() as d:
        root = make_release(Path(d))
        files = synth_msg.gen(gen_args(root, "pixelseal"), embed=fake_embed, decode=None)
        assert len(files) == 2
        S = dt.load_synth([str(root / "syn_pixelseal_s0_*.npz")], "pixelseal")
        meta = json.loads((root / "train" / "metadata.json").read_text())
        fo = pdata.real_folds(np.array([m["method"] for m in meta]), np.array([m["full_watermark"] for m in meta]), 0)
        conf = {m["image_name"] for m, f in zip(meta, fo) if f in (0, 1)}
        meth = {m["image_name"]: m["method"] for m in meta}
        assert S["msg"].shape == (24, 256) and S["images"].shape == (24, 512, 512, 3) and S["mask128"].shape == (24, 128, 128)
        for i in range(24):
            for s in (S["src"][i], S["src2"][i]):
                assert s not in conf and meth[s] != "pixelseal"
            src = pdata.load_image(root / "train" / "images" / f"{S['src'][i]}.png")
            src = src[:, ::-1] if S["flip"][i] else src
            wm = fake_embed(np.ascontiguousarray(src), S["msg"][i])  # retournement AVANT incrustation
            cell = np.kron(S["mask128"][i], np.ones((4, 4), np.uint8)).astype(bool)
            inner = cell & np.roll(cell, 4, 0) & np.roll(cell, -4, 0) & np.roll(cell, 4, 1) & np.roll(cell, -4, 1)
            assert (S["images"][i][inner] == wm[inner]).mean() > 0.99, "pixels watermarkés != embed(flip(src))"
            mir = fake_embed(np.ascontiguousarray(src[:, ::-1]), S["msg"][i])[:, ::-1]
            assert (S["images"][i][inner] == mir[inner]).mean() < 0.9, "test non discriminant"
            if S["fill"][i] == "clean":
                hole = ~cell & np.roll(~cell, 4, 0) & np.roll(~cell, -4, 0)
                if hole.sum() > 200:
                    assert (S["images"][i][hole] == src[hole]).mean() > 0.95
        tr, val, held = dt.synth_source_split(S, 0.3, 0)
        assert not (set(S["src"][tr]) | set(S["src2"][tr])) & held
        assert set(S["src"][val]) <= held and not (tr & val).any()


# ---------------------------------------------------------------- canaux, formes, finitude
def test_channels_shape_finite_full_arch():
    torch.manual_seed(0)
    x = torch.rand(2, 3, 512, 512); m = torch.randint(0, 2, (2, 256)).float() * 2 - 1
    ps = fake_backbone("pixelseal")
    feats, pos, glob = ps(x)
    assert [tuple(f.shape) for f in feats] == [(2, 96, 64, 64), (2, 192, 32, 32), (2, 384, 16, 16), (2, 768, 8, 8)]
    assert pos.shape == (2, 256, 8, 8) and glob.shape == (2, 256)
    net = DenseLoc(ps, 256); out = net(x, m)
    assert out.shape == (2, 128, 128) and torch.isfinite(out).all()
    tm = fake_backbone("trustmark")
    feats, pos, glob = tm(x)
    assert [tuple(f.shape) for f in feats] == [(2, 256, 62, 62), (2, 512, 31, 31), (2, 1024, 16, 16), (2, 2048, 8, 8)]
    ref = tm.net(resize_official(x, 245) * 2 - 1)  # forward torchvision complet (avgpool + fc) == moyenne par position
    assert torch.allclose(glob, ref, atol=1e-4), (glob - ref).abs().max()
    m2 = m[:, :100]
    out = DenseLoc(tm, 100)(x, m2)
    assert out.shape == (2, 128, 128) and torch.isfinite(out).all()


def test_logit_sign_and_official_equality_pixelseal():
    if not CKPT.exists():
        print("  IGNORÉ : checkpoint PixelSeal local absent"); return
    with tempfile.TemporaryDirectory() as d:
        model = load_local_pixelseal(local_card(Path(d)))
    from decoder_features import PixelSealBackbone
    bb = PixelSealBackbone(model.detector)
    rng = np.random.default_rng(1)
    imgs, msgs = [], []
    for _ in range(2):
        img = smooth_image(rng); msg = rng.integers(0, 2, 256)
        with torch.no_grad():  # == PixelSeal.embed du kit
            out = model.embed(torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255,
                              msgs=torch.tensor([msg], dtype=torch.float32), is_video=False)
        imgs.append(out["imgs_w"][0].clamp(0, 1).permute(1, 2, 0).numpy() * 255); msgs.append(msg)
    x = torch.from_numpy(np.round(np.stack(imgs)).astype(np.uint8)).permute(0, 3, 1, 2).float() / 255  # b 3 512 512
    with torch.no_grad():
        feats, pos, glob = bb(x)
        off = model.detector(resize_official(x, 256))[:, 1:]  # == PixelSeal.logits du kit
        lat = model.detector.convnext(model.detector.preprocess(resize_official(x, 256)))  # == features()["convnext"]
    assert torch.allclose(glob, off, atol=1e-3, rtol=1e-4), (glob - off).abs().max()
    assert torch.allclose(feats[-1], lat, atol=1e-4)
    acc = ((glob > 0).numpy().astype(int) == np.stack(msgs)).mean()
    print(f"  PixelSeal réel : bit accuracy de l'image watermarkée = {acc:.3f}")
    assert acc > 0.9, "signe des logits : > 0 doit signifier bit 1"
    clean = torch.from_numpy(smooth_image(rng)).permute(2, 0, 1)[None].float() / 255
    with torch.no_grad():
        acc0 = ((bb(clean)[2] > 0).numpy().astype(int) == msgs[0]).mean()
    print(f"  PixelSeal réel : image propre, accord au message = {acc0:.3f} (attendu ~0,5)")
    net = DenseLoc(bb, 256)
    out = net(x, torch.from_numpy(2.0 * np.stack(msgs) - 1).float())
    assert out.shape == (2, 128, 128) and torch.isfinite(out).all()


# ---------------------------------------------------------------- gel, gradients
def _one_step(net, x, y, m, lr_bb=None):
    groups = [{"params": net.head_params(), "lr": 1e-2}]
    if lr_bb:
        groups.append({"params": net.bb.trainable(), "lr": lr_bb})
    opt = torch.optim.AdamW(groups)
    net.train()
    if lr_bb:
        logit, g, r = net(x, m, with_ref=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y) + torch.nn.functional.mse_loss(g, r)
    else:
        logit = net(x, m); loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
    opt.zero_grad(); loss.backward()
    gh = sum(p.grad.abs().sum().item() for p in net.head_params() if p.grad is not None)
    opt.step()
    return gh


def test_frozen_backbone_unchanged_head_gradient():
    for method, nb in (("pixelseal", 256), ("trustmark", 100)):
        net = DenseLoc(fake_backbone(method), nb)
        x = torch.rand(2, 3, 512, 512); y = (torch.rand(2, 128, 128) > 0.5).float(); m = torch.randint(0, 2, (2, nb)).float() * 2 - 1
        before = {k: v.clone() for k, v in net.bb.state_dict().items()}  # AVANT fit_norm (1er forward)
        net.fit_norm(x, m)
        gh = _one_step(net, x, y, m)
        assert gh > 0, "gradient nul sur la tête"
        assert all(torch.equal(before[k], v) for k, v in net.bb.state_dict().items()), f"{method} : backbone modifié"
        assert not net.bb.training and all(not p.requires_grad for p in net.bb.parameters())


def test_last_stage_mode():
    for method, nb in (("pixelseal", 256), ("trustmark", 100)):
        net = DenseLoc(fake_backbone(method, train_last=True, tiny=True), nb)
        x = torch.rand(2, 3, 512, 512); y = (torch.rand(2, 128, 128) > 0.5).float(); m = torch.randint(0, 2, (2, nb)).float() * 2 - 1
        before = {k: v.clone() for k, v in net.bb.state_dict().items()}
        with torch.no_grad():
            _, g0, r0 = net(x, m, with_ref=True)
        assert torch.allclose(g0, r0), "copie de référence != stage initial"
        _one_step(net, x, y, m, lr_bb=1e-3)
        last = {id(p) for p in net.bb.last_stage().parameters()}
        names_last = {k for k, p in net.bb.named_parameters() if id(p) in last}
        changed = {k for k, v in net.bb.state_dict().items() if not torch.equal(before[k], v)}
        assert changed and changed <= names_last, changed - names_last
        assert not any(k.startswith("ref_last") for k in changed)


# ---------------------------------------------------------------- alignement des masques
def test_mask_alignment():
    imgs = np.zeros((1, 512, 512, 3), np.uint8); imgs[0, 100, 300, 1] = 255
    masks = np.zeros((1, 128, 128), np.uint8); masks[0, 25, 75] = 1
    pool = dt.make_pool(imgs, masks, np.zeros((1, 256), np.uint8))
    x, y, msg = dt.batch(pool, torch.tensor([0]), torch.device("cpu"))
    assert x.shape == (1, 3, 512, 512) and x[0, 1, 100, 300] == 1 and x.sum() == 1
    assert y[0, 25, 75] == 1 and y.sum() == 1 and (msg == -1).all()
    # le gradient d une sortie (i, j) culmine dans la cellule (i, j) des pixels d entrée
    for method, nb in (("pixelseal", 256), ("trustmark", 100)):
        net = DenseLoc(fake_backbone(method, tiny=True), nb).eval()
        xi = torch.rand(1, 3, 512, 512, requires_grad=True)
        out = net(xi, torch.ones(1, nb))
        for (i, j) in ((40, 90), (100, 20)):
            gr, = torch.autograd.grad(out[0, i, j], xi, retain_graph=True)
            w = gr.abs().sum(1)[0]  # 512 512
            blk = w.view(128, 4, 128, 4).sum((1, 3))  # 512 512 -> 128 128 (cellules de la grille officielle)
            pi, pj = divmod(int(blk.argmax()), 128)
            assert abs(pi - i) <= 2 and abs(pj - j) <= 2, (method, (i, j), (pi, pj))  # pas de transposition/retournement


# ---------------------------------------------------------------- surapprentissage 8 exemples factices
def test_overfit_8():
    rng = np.random.default_rng(0)
    imgs, masks, msgs = [], [], []
    for k in range(8):
        img = smooth_image(rng); msg = rng.integers(0, 2, 256).astype(np.uint8)
        hole = pdata.blob(rng, 512, 512, 0.15, (0.2, 0.8)); hole[:, :64] |= k % 2 == 0  # asymétrique
        wm = fake_embed(img, msg)
        imgs.append(np.where(hole[..., None], img, wm)); masks.append(pdata.downsample_mask(~hole)); msgs.append(msg)
    pool = dt.make_pool(np.stack(imgs), np.stack(masks), np.stack(msgs))
    a = dt.parse(["--method", "pixelseal", "--synth", "x", "--out", "x", "--fake", "tiny", "--bs", "8", "--lr", "3e-3",
                  "--device", "cpu", "--log-every", "1000"])
    idx = torch.arange(8)
    t0 = time.time()
    net, losses = dt.train_run(a, torch.device("cpu"), None, torch.zeros(0, dtype=torch.long), pool, idx, 150, 0)
    p = dt.predict(net, pool, idx, torch.device("cpu"))
    s = dt.scores(p > 0.5, np.stack(masks).astype(bool))
    print(f"  surapprentissage 8 : mIoU {s['miou']:.3f} (tout-1 {s['all1']:.3f}), perte {losses[0]:.3f} -> "
          f"{losses[-1]:.3f}, {time.time() - t0:.0f} s")
    assert s["miou"] > 0.85 and losses[-1] < 0.5 * losses[0]


# ---------------------------------------------------------------- bout en bout (protocole, sortie npz)
def test_end_to_end_protocol():
    with tempfile.TemporaryDirectory() as d:
        root = make_release(Path(d))
        synth_msg.gen(gen_args(root, "trustmark", n=48), embed=fake_embed, decode=None)
        a = dt.parse(["--method", "trustmark", "--synth", str(root / "syn_trustmark_s0_*.npz"), "--fake", "tiny",
                      "--steps", "3", "--overfit-steps", "2", "--control-steps", "2", "--bs", "4", "--data", str(root),
                      "--out", str(root / "res"), "--device", "cpu", "--synth-val-frac", "0.3"])
        out, rep = dt.run(a)
        z = np.load(out)
        for k in ("names", "probs_oof", "fold", "config", "train_names_hash", "source_groups", "thr", "phase", "report"):
            assert k in z.files, k
        R = dt.load_real("trustmark", root, with_images=False)
        assert z["names"].tolist() == R["names"].tolist() and np.array_equal(z["fold"], R["fold"])
        ph, fo = z["phase"], z["fold"]
        assert (ph[np.isin(fo, (2, 3, 4))] == "dev").all() and (ph[np.isin(fo, (0, 1))] == "confirm").all()
        assert np.isfinite(z["probs_oof"].astype(np.float32)).all()
        g = json.loads(str(z["source_groups"]))
        conf = set(R["names"][np.isin(R["fold"], (0, 1))].tolist())
        for tag, v in g.items():
            assert not set(v["real_train"]) & set(v["real_thr"])
            if tag.startswith("dev"):
                k = int(tag[3:])
                ev = set(R["names"][R["fold"] == k].tolist())
                assert not (set(v["real_train"]) | set(v["real_thr"])) & (ev | conf)
            else:
                assert not (set(v["real_train"]) | set(v["real_thr"])) & conf
        assert set(json.loads(str(z["train_names_hash"]))) == {"dev2", "dev3", "dev4", "confirm"}
        assert "pass" in rep["control"]
        import compare
        zm = np.load(root / "train" / "masks_128.npz")
        np.savez(root / "ref.npz", names=zm["names"], head=np.full((3, len(zm["names"])), 0.4))
        res = compare.main(["--res", out, "--res-last", out, "--ref", str(root / "ref.npz"),
                            "--masks", str(root / "train" / "masks_128.npz")])
        assert res["trustmark"] == "contrôle échoué dans les deux modes"
        d = dict(z); rp = json.loads(str(d["report"])); rp["control"]["pass"] = True; d["report"] = json.dumps(rp)
        np.savez(root / "res_ok.npz", **d)
        res = compare.main(["--res", str(root / "res_ok.npz"), "--ref", str(root / "ref.npz"),
                            "--masks", str(root / "train" / "masks_128.npz")])
        assert res["trustmark"]["n"] == int((z["phase"] == "confirm").sum()) and np.isfinite(res["trustmark"]["gain"])


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items(), key=lambda kv: getattr(kv[1], "__code__", None) and kv[1].__code__.co_firstlineno or 0)
             if n.startswith("test_") and callable(f)]
    fails = 0
    for n, f in tests:
        t0 = time.time()
        try:
            f(); print(f"OK    {n} ({time.time() - t0:.1f} s)", flush=True)
        except Exception:
            fails += 1; print(f"ÉCHEC {n}", flush=True); traceback.print_exc()
    print(f"{len(tests) - fails}/{len(tests)} tests passés")
    sys.exit(1 if fails else 0)
