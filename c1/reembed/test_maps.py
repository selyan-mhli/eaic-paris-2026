"""Tests CPU synthétiques de maps.py / eval_auc.py / merge.py, sans wmloc (petites tailles, < 10 s).

    python -m pytest -q c1/reembed/test_maps.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_auc  # noqa: E402
import maps  # noqa: E402
import merge  # noqa: E402

H, OUT, NB = 128, 32, 32  # image H x H, carte OUT x OUT (blocs de 4), message de NB bits


def bit_patterns(seed=123):
    """NB motifs ±1 haute fréquence, 3 canaux : NB 3 H W (clé fixe de l'embedder factice)."""
    return np.random.default_rng(seed).choice([-1.0, 1.0], (NB, 3, H, H)).astype(np.float32)


class FakeEmbedder:
    """Ajoute amp * P(m), P(m) = somme_b (2 m_b - 1) B_b / sqrt(NB), sur toute l'image, sortie uint8 H W 3
    (comme le kit). content_gain > 0 : ajoute aussi un terme dépendant du contenu mais pas du message
    (accentuation), que le z-score doit retirer."""

    def __init__(self, amp=4.0, content_gain=0.0):
        self.B = bit_patterns(); self.amp = amp; self.g = content_gain

    def pattern(self, msg):
        s = (2 * np.asarray(msg, np.float32) - 1)[:, None, None, None]  # NB 1 1 1
        return (s * self.B).sum(0) / np.sqrt(NB)  # 3 H W

    def __call__(self, img_u8, msg):
        x = img_u8.astype(np.float32)  # H W 3
        add = self.amp * self.pattern(msg).transpose(1, 2, 0)  # 3 H W -> H W 3
        if self.g:
            xt = torch.from_numpy(x / 255).permute(2, 0, 1)[None]  # 1 3 H W
            hp = (xt - maps.blur(xt, 1.0))[0].permute(1, 2, 0).numpy() * 255  # H W 3
            add = add + self.g * hp
        return np.clip(np.round(x + add), 0, 255).astype(np.uint8)


def scene(seed, emb, msg, hole):
    """Contenu lisse + texture ; watermark de msg partout sauf dans `hole` (H H bool, zone « éditée »)."""
    rng = np.random.default_rng(seed)
    base = torch.from_numpy(rng.random((1, 3, H, H)).astype(np.float32))
    base = maps.blur(base, 6.0)[0].permute(1, 2, 0).numpy()  # 1 3 H W -> H W 3
    base = (base - base.min()) / (base.max() - base.min()) * 160 + 40 + rng.normal(0, 3, (H, H, 3))
    clean = np.clip(np.round(base), 0, 255).astype(np.uint8)
    wm = emb(clean, msg)
    edited = np.clip(np.round(base + rng.normal(0, 3, (H, H, 3))), 0, 255).astype(np.uint8)  # trou régénéré
    return np.where(hole[..., None], edited, wm)


def hole_mask(seed):
    m = np.zeros((H, H), bool); r = np.random.default_rng(seed); y, x = r.integers(16, 64, 2)
    m[y:y + 48, x:x + 48] = True
    return m


def run(emb, n=3, luma=False, K=8):
    imgs, msgs, holes, names = [], [], [], []
    for i in range(n):
        msg = np.random.default_rng(1000 + i).integers(0, 2, NB).astype(np.uint8)
        hole = hole_mask(i); imgs.append(scene(i, emb, msg, hole)); msgs.append(msg); holes.append(hole)
        names.append(f"img_{i}.png")
    z, zg, c = maps.compute_batch(emb, imgs, msgs, names, K, 0, "cpu", (16,), 1.0, luma, False, out=OUT)
    hh = torch.from_numpy(np.stack(holes)).float()[:, None]  # n 1 H H, 1 = trou
    near = torch.nn.functional.max_pool2d(hh, 17, 1, 8)  # n 1 H H : à moins de 8 px (demi-fenêtre) d'un trou
    inner = 1 - torch.nn.functional.max_pool2d(1 - hh, 17, 1, 8)  # n 1 H H : trou à plus de 8 px du bord
    lab = torch.nn.functional.avg_pool2d(near, 4)[:, 0].numpy() == 0  # n OUT OUT, 1 = watermarqué loin du trou
    deep = lab | (torch.nn.functional.avg_pool2d(inner, 4)[:, 0].numpy() == 1)  # n OUT OUT, cellules hors frontière
    return z, zg, c, lab, deep


def test_formes_et_noyau():
    x = torch.rand(2, 3, 40, 40)
    assert maps.blur(x, 1.0).shape == x.shape
    assert maps.box_mean(x, 16).shape == x.shape and maps.box_mean(x, 15).shape == x.shape
    assert torch.allclose(maps.box_mean(torch.full((1, 1, 40, 40), 0.3), 16), torch.tensor(0.3))
    h = torch.randn(2, 3, 40, 40); r = torch.stack([h, -h, torch.randn_like(h)], 1)  # 2 3 3 40 40
    n = maps.local_ncc(h, r, 8)
    assert n.shape == (2, 3, 40, 40)
    assert torch.allclose(n[:, 0], torch.tensor(1.0), atol=1e-4) and torch.allclose(n[:, 1], torch.tensor(-1.0), atol=1e-4)
    assert maps.to_luma(torch.ones(4, 3, 5, 5)).shape == (4, 1, 5, 5)
    res = torch.randn(2, 9, 3, H, H) * 0.01
    z, zg, c = maps.zscore_maps(torch.rand(2, 3, H, H), res, wins=(16, 32), luma=True, out=OUT)
    assert z.shape == zg.shape == c.shape == (2, OUT, OUT) and torch.isfinite(z).all()


def test_messages_et_conversion():
    a = maps.rand_messages("x.png", 8, 256, 0); b = maps.rand_messages("x.png", 8, 256, 0)
    assert a.shape == (8, 256) and a.dtype == np.uint8 and (a == b).all()
    assert not (a == maps.rand_messages("y.png", 8, 256, 0)).all()
    img = (np.random.default_rng(0).random((16, 16, 3)) * 255).astype(np.uint8)
    t = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255  # 1 3 16 16
    assert np.allclose(maps.as_float_hwc(t), img / 255, atol=1e-6)
    assert np.allclose(maps.as_float_hwc(img), img / 255)


@pytest.mark.parametrize("luma", [False, True])
def test_zones_separees(luma):
    z, zg, c, lab, deep = run(FakeEmbedder(), luma=luma)
    assert z.shape == (3, OUT, OUT) and np.isfinite(z).all()
    for v in (z, zg):
        a = eval_auc.auc(v[deep], lab[deep])
        assert a > 0.9, a
    assert z[lab & deep].mean() > 2.0 > abs(z[~lab & deep].mean())  # sens : grand = watermark présent


def test_terme_de_contenu_retire():
    """Terme dépendant du contenu, commun à tous les messages : c brut est corrélé partout, z reste séparé."""
    z, zg, c, lab, deep = run(FakeEmbedder(amp=4.0, content_gain=1.0))
    assert c[~lab & deep].mean() > 0.3  # c brut fort même dans le trou
    assert eval_auc.auc(z[deep], lab[deep]) > 0.9
    assert z[lab & deep].mean() > z[~lab & deep].mean() + 2


def test_auc_a_la_main():
    assert eval_auc.auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == 0.75
    assert eval_auc.auc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5


def test_select_et_merge(tmp_path, monkeypatch):
    class DS:
        meta = [{"method": m, "image_name": f"{m}_{i}", "full_watermark": i % 2 == 0} for m in maps.METHODS for i in range(6)]

        def __len__(self):
            return len(self.meta)
    ds = DS()
    sel = maps.select(ds, ["trustmark", "pixelseal"], True, 2, None, 0.0)
    assert [ds.meta[i]["image_name"] for i in sel] == ["trustmark_0", "trustmark_2", "pixelseal_0", "pixelseal_2"]
    names = [m["image_name"] for m in ds.meta]
    np.savez(tmp_path / "heats.npz", names=np.array(names), acc_full=np.linspace(0.4, 1, len(names)),
             fm_soft=np.zeros((len(names), 128, 128), np.float16))
    z = np.random.default_rng(0).normal(size=(2, 128, 128)).astype(np.float32)
    maps.save(tmp_path / "re_s0of1.npz", names[:2], ["pixelseal"] * 2, [True, False], np.array([True, True]), z, z, z)
    out = tmp_path / "heats_re.npz"
    monkeypatch.setattr(sys, "argv", ["merge", "--re", str(tmp_path / "re_s*.npz"), "--heats", str(tmp_path / "heats.npz"),
                                      "--out", str(out), "--as-key", "fm_soft"])
    merge.main()
    d = np.load(out)
    assert d["re_z"].shape == (len(names), 128, 128) and d["re_z"].dtype == np.float16
    assert np.isfinite(d["re_zs"][:2]).all() and np.isnan(d["re_zs"][2:]).all()
    assert ((d["re_zs"][:2] >= 0) & (d["re_zs"][:2] <= 1)).all()
    assert np.array_equal(d["fm_soft"], d["re_zs"], equal_nan=True)
    gs = maps.select(ds, ["pixelseal"], False, 0, tmp_path / "heats.npz", 0.99)
    assert gs == []  # pixelseal en tête : acc_full faibles
