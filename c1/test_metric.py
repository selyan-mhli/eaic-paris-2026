"""Tests à la main de c1/metric.py. Lancer : `python c1/test_metric.py` (ou pytest)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metric import downsample_mask, miou_image, miou_score  # noqa: E402

N = 128


def z():
    return np.zeros((N, N), np.uint8)


def o():
    return np.ones((N, N), np.uint8)


def test_union_vide_classe1():
    # pred = gt = tout 0 : union classe 1 vide -> IoU_1 = 1 ; IoU_0 = 1
    assert miou_image(z(), z()) == 1.0


def test_masque_plein():
    # pred = gt = tout 1 : union classe 0 vide -> IoU_0 = 1 ; IoU_1 = 1
    assert miou_image(o(), o()) == 1.0


def test_masque_vide_contre_plein():
    # pred tout 0, gt tout 1 : IoU_0 = 0/16384, IoU_1 = 0/16384 -> 0
    assert miou_image(z(), o()) == 0.0
    assert miou_image(o(), z()) == 0.0


def test_tout1_contre_moitie():
    # gt = moitié haute ; pred = tout 1
    # IoU_1 = 8192/16384 = 0.5 ; IoU_0 : pred0 vide, gt0 = 8192 -> 0/8192 = 0 -> 0.25
    g = z(); g[:64] = 1
    assert abs(miou_image(o(), g) - 0.25) < 1e-12
    # pred tout 0 : IoU_1 = 0, IoU_0 = 8192/16384 = 0.5 -> 0.25
    assert abs(miou_image(z(), g) - 0.25) < 1e-12


def test_partiel():
    # gt = bloc 64x64 en haut à gauche (4096 cellules) ; pred = 64 premières lignes (8192)
    # IoU_1 = 4096 / 8192 = 1/2 ; classe 0 : gt0 = 12288, pred0 = 8192 ⊂ gt0 -> 8192/12288 = 2/3
    # mIoU = (1/2 + 2/3) / 2 = 7/12
    g = z(); g[:64, :64] = 1
    p = z(); p[:64] = 1
    assert abs(miou_image(p, g) - 7 / 12) < 1e-12


def test_score_moyenne_par_image():
    preds = np.stack([z(), o()])  # 1.0 et 0.0 contre gt = 0
    gts = np.stack([z(), z()])
    assert abs(miou_score(preds, gts) - 0.5) < 1e-12


def test_downsample_seuil_50pc():
    m = np.zeros((512, 512), np.uint8)
    m[0:4, 0:2] = 1          # cellule (0,0) : 8/16 px -> 1
    m[0:4, 4:5] = 1          # cellule (0,1) : 4/16 px -> 0
    m[0:3, 8:10] = 1         # cellule (0,2) : 6/16 ...
    m[3, 8] = 1              # ... + 1 = 7/16 px -> 0
    m[508:512, 508:512] = 1  # dernière cellule pleine -> 1
    d = downsample_mask(m)
    assert d.shape == (128, 128) and d.dtype == np.uint8
    assert d[0, 0] == 1 and d[0, 1] == 0 and d[0, 2] == 0 and d[127, 127] == 1
    assert d.sum() == 2
    # batch : (2, 512, 512) -> (2, 128, 128)
    assert downsample_mask(np.stack([m, m])).shape == (2, 128, 128)


# Copie littérale de la référence du kit officiel (transmise par la session principale).
def _ref_downsample(mask):
    m = np.asarray(mask, dtype=np.float32)
    c = m.shape[0] // 128
    return (m.reshape(128, c, 128, c).mean(axis=(1, 3)) >= 0.5).astype(np.uint8)


def _ref_miou(pred, gt):
    p, g = np.asarray(pred), np.asarray(gt)
    s = 0.0
    for c in (0, 1):
        union = np.logical_or(p == c, g == c).sum()
        s += 1.0 if union == 0 else np.logical_and(p == c, g == c).sum() / union
    return s / 2


def test_egalite_reference_officielle():
    rng = np.random.default_rng(0)
    for k in range(200):
        # masques 512 aléatoires : bruit, rectangles, vides, pleins
        if k % 4 == 0:
            m = (rng.random((512, 512)) < rng.random()).astype(np.uint8)
        elif k % 4 == 1:
            m = np.zeros((512, 512), np.uint8)
            y, x = rng.integers(0, 500, 2)
            m[y:y + rng.integers(1, 200), x:x + rng.integers(1, 200)] = 1
        else:
            m = np.full((512, 512), k % 4 == 2, np.uint8)
        assert np.array_equal(downsample_mask(m), _ref_downsample(m))
        assert np.array_equal(downsample_mask(m.astype(bool)), _ref_downsample(m.astype(bool)))
        g = _ref_downsample(m)
        for p in (z(), o(), (rng.random((N, N)) < 0.3).astype(np.uint8), g, g.astype(bool)):
            assert miou_image(p, g) == _ref_miou(p, g)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"{len(tests)} tests passés")
