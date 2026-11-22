"""Tests CPU de c1/pixseg (aucun accès au kit ni à JURECA ; embedder factice).

  uv run --no-project --with torch --with numpy --with scipy --with pillow --with pytest pytest c1/pixseg -q
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import data  # noqa: E402
import export  # noqa: E402
import train  # noqa: E402
from metric import downsample_mask as ref_down, miou_per_image as ref_miou  # noqa: E402  (c1/metric.py)
from model import PixSeg  # noqa: E402


def fake_embed(img, msg):
    """Watermark factice : bit de poids faible à 1 partout (les sources de test ont ce bit à 0)."""
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3 and min(img.shape[:2]) >= 8
    assert msg.dtype == np.uint8 and set(np.unique(msg)) <= {0, 1}
    return img | 1


def even_image(rng):
    return (rng.integers(0, 128, (512, 512, 3)) * 2).astype(np.uint8)


def test_metric_copies_identical_to_c1_metric():
    rng = np.random.default_rng(0)
    m = rng.random((512, 512)) > 0.5
    assert np.array_equal(data.downsample_mask(m), ref_down(m))
    half = np.zeros((512, 512), np.uint8); half[:2, :4] = 1  # cellule (0, 0) : 8 pixels sur 16 -> 1
    assert data.downsample_mask(half)[0, 0] == 1 and data.downsample_mask(half).sum() == 1
    p = rng.random((6, 128, 128)) > 0.4; g = rng.random((6, 128, 128)) > 0.7
    p[0] = g[0] = False; p[1] = True; g[1] = False; g[2] = True
    assert np.allclose(data.fast_miou(p, g), ref_miou(p, g))
    assert data.fast_miou(p[:1], g[:1])[0] == 1.0 and data.fast_miou(p[1:2], g[1:2])[0] == 0.0


def test_folds_identical_to_gated_eval():
    src = (HERE.parent / "real_eval.py").read_text()  # gated_eval importe folds de real_eval (qui importe le kit)
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "folds")
    ns = {"np": np}; exec(compile(ast.Module([fn], []), "real_eval.folds", "exec"), ns)
    assert 'folds(np.array([f"{m}|{f}" for m, f in zip(method, full)]), 5, a.seed)' in (HERE.parent / "gated_eval.py").read_text()
    rng = np.random.default_rng(3)
    method = rng.permutation(np.repeat(data.METHODS, 170)); full = rng.permutation(np.repeat([True, False], 255))
    ref = ns["folds"](np.array([f"{m}|{f}" for m, f in zip(method, full)]), 5, 0)
    got = data.real_folds(method, full, 0)
    assert np.array_equal(got, ref)
    for m in data.METHODS:  # stratifié : chaque fold a sa part de chaque strate
        for f in (True, False):
            c = np.bincount(got[(method == m) & (full == f)], minlength=5)
            assert c.max() - c.min() <= 1


@pytest.mark.parametrize("method", data.METHODS)
def test_synth_mask_matches_composite(method):
    rng = np.random.default_rng(1)
    clean, other = even_image(rng), even_image(rng)
    seen = set()
    for _ in range(24):
        s = data.synth_sample(rng, fake_embed, method, clean, other, fill=(0.5, 0.5, 0.0))
        img, m512 = s["image"], s["mask512"]
        assert img.shape == (512, 512, 3) and img.dtype == np.uint8 and s["mask128"].shape == (128, 128)
        marked = (img & 1).all(-1)  # 512 512 : pixels portant le watermark factice
        assert np.array_equal(marked, m512), s["edit"]
        assert np.array_equal(s["mask128"], ref_down(m512))
        assert np.array_equal(img[~m512] & 1, np.zeros_like(img[~m512]))  # hors masque : aucun pixel watermarké
        if s["full"]:
            assert s["edit"] == "hole" and 0.5 <= m512.mean() < 1
        elif s["edit"] == "rect":
            ys, xs = np.nonzero(m512); h, w = np.ptp(ys) + 1, np.ptp(xs) + 1
            assert m512.sum() == h * w  # rectangle plein
            assert max(h / w, w / h) <= (2.0 if method == "trustmark" else 16.0) * 1.15  # arrondi des côtés
        seen.add(s["edit"])
    assert seen == {"hole", "rect", "transplant"}


def test_model_shapes():
    net = PixSeg(base=8).eval()
    x = torch.rand(2, 3, 512, 512); m = torch.tensor([0, 2])
    with torch.no_grad():
        assert net.inputs(x, m).shape == (2, 10, 512, 512)  # RGB, one-hot, passe-haut Y R G B
        assert net.inputs(x, m)[1, 3:6].mean((1, 2)).tolist() == [0.0, 0.0, 1.0]  # one-hot en canaux constants
        flat = torch.full((1, 3, 64, 64), 0.3)
        assert net.inputs(flat, m[:1])[:, 6:].abs().max() < 1e-4  # image plate : passe-haut nul
        y = net(x, m)
        assert y.shape == (2, 128, 128) and torch.isfinite(y).all()
        assert net(x[:, :, :448, :448], m).shape == (2, 112, 112)
        net2 = PixSeg(base=8, extra=3).eval()
        assert net2(x, m, torch.rand(2, 3, 128, 128)).shape == (2, 128, 128)


def test_timm_backbone_shapes():
    pytest.importorskip("timm")
    net = PixSeg(base=8, backbone="resnet18", pretrained=False).eval()
    with torch.no_grad():
        assert net(torch.rand(1, 3, 256, 256), torch.tensor([1])).shape == (1, 64, 64)


def test_soft_miou_matches_metric_on_hard_predictions():
    rng = np.random.default_rng(2)
    p = rng.random((5, 128, 128)) > 0.5; g = rng.random((5, 128, 128)) > 0.8
    g[0] = False; p[0] = False; g[1] = False
    logit = torch.from_numpy(np.where(p, 50.0, -50.0)).float()
    loss = train.soft_miou_loss(logit, torch.from_numpy(g), eps=1e-6).item()
    assert abs(loss - (1 - data.fast_miou(p, g).mean())) < 1e-4


def test_batch_crop_and_flip_keep_mask_aligned():
    rng = np.random.default_rng(4)
    masks = (rng.random((3, 128, 128)) > 0.5).astype(np.uint8)
    imgs = np.repeat(np.repeat(masks, 4, 1), 4, 2)[..., None].repeat(3, -1) * 255  # n 512 512 3 : image = masque
    extra = masks[:, None].astype(np.float16)  # n 1 128 128
    pool = train.make_pool(imgs.astype(np.uint8), masks, ["pixelseal", "trustmark", "stegastamp"], extra)
    g = torch.Generator().manual_seed(0)
    flipped = 0
    for _ in range(8):
        x, m, y, e = train.batch(pool, torch.tensor([0, 1, 2]), torch.device("cpu"), crop=448, flip=True, g=g)
        assert x.shape == (3, 3, 448, 448) and y.shape == (3, 112, 112) and e.shape == (3, 1, 112, 112)
        assert m.tolist() == [0, 1, 2]
        assert torch.equal(torch.nn.functional.avg_pool2d(x[:, :1], 4)[:, 0], y) and torch.equal(e[:, 0], y)
        flipped += int(not torch.equal(y[0], torch.from_numpy(masks[0, :112, :112]).float()))
    assert flipped > 0
    x, _, y, _ = train.batch(pool, torch.tensor([1]), torch.device("cpu"))
    assert x.shape == (1, 3, 512, 512) and torch.equal(y[0], torch.from_numpy(masks[1]).float())


def test_export_format(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    names = [f"img_{i:04d}" for i in range(7)]; masks = rng.random((7, 128, 128)) > 0.5
    out = tmp_path / "sub.pt"
    order = names[::-1]
    export.write_submission(names, masks, out, expected=order)
    back = torch.load(out, weights_only=True)
    assert set(back) == {"image_names", "masks"} and back["image_names"] == order
    assert back["masks"].dtype == torch.uint8 and tuple(back["masks"].shape) == (7, 128, 128)
    assert set(back["masks"].unique().tolist()) <= {0, 1}
    assert np.array_equal(back["masks"].numpy(), masks[::-1].astype(np.uint8))
    with pytest.raises(AssertionError, match="double"):
        export.write_submission(names[:6] + names[:1], masks, tmp_path / "x.pt")
    with pytest.raises(AssertionError, match="métadonnées"):
        export.write_submission(names, masks, tmp_path / "x.pt", expected=names[:6] + ["autre"])
    with pytest.raises(AssertionError, match="binaires"):
        export.write_submission(names, masks.astype(np.uint8) * 255, tmp_path / "x.pt")
    monkeypatch.setattr(export, "MAX_MB", 0.01)
    with pytest.raises(AssertionError, match="Mo"):
        export.write_submission(names, masks, tmp_path / "x.pt")
    assert not (tmp_path / "x.pt").exists()


def test_real_submission_size_under_limit(tmp_path):
    masks = np.random.default_rng(6).random((1020, 128, 128)) > 0.5
    mb = export.write_submission([f"n{i}" for i in range(1020)], masks, tmp_path / "s.pt")
    assert 16 < mb < 20


def fake_pool(rng, n):
    """Images de bruit ; dans le masque, damier de forte amplitude (signal haute fréquence facile)."""
    masks = np.zeros((n, 128, 128), np.uint8)
    for i in range(n):
        y, x = rng.integers(0, 64, 2); masks[i, y:y + 48, x:x + 48] = 1
    imgs = rng.integers(60, 190, (n, 512, 512, 3))
    cb = ((np.indices((512, 512)).sum(0) % 2) * 2 - 1) * 50  # 512 512, +-50
    imgs = imgs + cb[None, :, :, None] * np.repeat(np.repeat(masks, 4, 1), 4, 2)[..., None]
    return imgs.clip(0, 255).astype(np.uint8), masks


def test_smoke_training_reduces_loss():
    rng = np.random.default_rng(7)
    imgs, masks = fake_pool(rng, 12)
    meth = [data.METHODS[i % 3] for i in range(12)]
    real = train.make_pool(imgs[:4], masks[:4], meth[:4]); synth = train.make_pool(imgs[4:], masks[4:], meth[4:])
    a = train.parse(["--cv", "--steps", "30", "--bs", "4", "--crop", "128", "--base", "8", "--lr", "3e-3",
                     "--real-frac", "0.5", "--log-every", "10", "--device", "cpu"])
    net, losses = train.train(a, real, torch.arange(4), synth, torch.arange(8), torch.device("cpu"), seed=0)
    assert len(losses) == 30 and np.isfinite(losses).all()
    assert np.mean(losses[-5:]) < 0.8 * np.mean(losses[:5]), (losses[:5], losses[-5:])
    p = train.predict(net, real, torch.arange(4), torch.device("cpu"))
    assert p.shape == (4, 128, 128) and p.dtype == np.float32 and (p >= 0).all() and (p <= 1).all()


def test_cli_end_to_end(tmp_path):
    """Faux dossier release -> gen (embedder factice) -> --cv (1 fold) -> --fit-all --predict-test -> export."""
    from PIL import Image
    rng = np.random.default_rng(8)
    rel = tmp_path / "release"
    for split, n in (("train", 30), ("test", 6)):
        (rel / split / "images").mkdir(parents=True)
        imgs, masks = fake_pool(rng, n)
        meta = [{"image_name": f"{split}_{i:03d}", "method": data.METHODS[i % 3], "message": "01" * 50} for i in range(n)]
        for i, m in enumerate(meta):
            Image.fromarray(imgs[i] & 254).save(rel / split / "images" / f"{m['image_name']}.png")
            if split == "train":
                m["full_watermark"] = bool((i // 3) % 2)
        (rel / split / "metadata.json").write_text(json.dumps(meta))
        if split == "train":
            np.savez(rel / split / "masks_128.npz", names=np.array([m["image_name"] for m in meta]), masks=masks)
    out = str(tmp_path / "syn")
    for meth in data.METHODS:
        data.main(["gen", "--method", meth, "--n", "5", "--shard", "3", "--out", out, "--pool", "train,test",
                   "--data", str(rel)], embed=fake_embed)
    S = data.load_synth([out + "_*.npz"])
    assert S["images"].shape == (15, 512, 512, 3) and S["mask128"].shape == (15, 128, 128)
    assert len(set(S["names"].tolist())) == 15
    # une tranche par méthode ne contient que des sources d'une autre méthode
    mm = {f"{sp}_{i:03d}": data.METHODS[i % 3] for sp, n in (("train", 30), ("test", 6)) for i in range(n)}
    assert all(mm[s] != m and mm[s2] != m for s, s2, m in zip(S["src"].tolist(), S["src2"].tolist(), S["method"].tolist()))
    np.savez(tmp_path / "maps_train.npz", names=np.array(sorted(n for n in mm if n.startswith("train"))),
             a=rng.random((30, 32, 32)).astype(np.float16), b=rng.random((30, 2, 128, 128)).astype(np.float16))
    np.savez(tmp_path / "maps_test.npz", names=np.array(sorted(n for n in mm if n.startswith("test"))),
             a=rng.random((6, 32, 32)).astype(np.float16), b=rng.random((6, 2, 128, 128)).astype(np.float16))
    common = ["--synth", out + "_*.npz", "--data", str(rel), "--steps", "3", "--bs", "4", "--crop", "128", "--base", "8",
              "--device", "cpu", "--out", str(tmp_path / "pixseg"), "--tag", "t", "--log-every", "1",
              "--extra-maps", f"{tmp_path}/maps_train.npz:a+b", "--extra-maps-test", f"{tmp_path}/maps_test.npz:a+b"]
    train.main(["--cv", "--folds", "0"] + common)
    z = np.load(tmp_path / "pixseg_cv_t.npz")
    done = np.isfinite(z["thr"])
    assert done.sum() == 6 and np.isfinite(z["probs"][done]).all() and (z["fold"][done] == 0).all()
    train.main(["--fit-all", "--predict-test", "--thr", "0.4"] + common)
    t = np.load(tmp_path / "pixseg_test_t.npz")
    assert t["probs"].shape == (6, 128, 128) and float(t["thr"]) == 0.4 and (tmp_path / "pixseg_t.pt").exists()
    export.main(["--probs", str(tmp_path / "pixseg_test_t.npz"), "--meta", str(rel / "test" / "metadata.json"),
                 "--out", str(tmp_path / "sub.pt")])
    back = torch.load(tmp_path / "sub.pt", weights_only=True)
    assert back["image_names"] == [f"test_{i:03d}" for i in range(6)] and tuple(back["masks"].shape) == (6, 128, 128)


def test_cv_excludes_synthetic_built_from_evaluated_images():
    src = (HERE / "train.py").read_text()
    assert "synth_idx(te | va)" in src  # le filtre est bien appliqué aux folds évalués et de seuil
