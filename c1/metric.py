"""Métrique locale de la Task 1 (Watermark Localization).

Aligné sur la métrique du kit officiel (transmise par la session principale le 2026-11-21,
kit sur JURECA) : downsample = mask float32 -> reshape(128,c,128,c).mean(1,3) >= 0.5 ;
miou_image = moyenne sur c in (0,1) de |p==c & g==c| / |p==c | g==c| (1 si union vide).
c1/test_metric.py compare à une copie littérale de cette référence. Dès que le kit est
copié localement : importer ses fonctions et vérifier l'égalité sur les 510 masques train.

Règles de l'énoncé :
- grille 128x128, une cellule = 4x4 px ; cellule GT = 1 si >= 50 % de ses 16 pixels
  sont watermarkés (donc >= 8 pixels) ;
- par image, IoU_c = |P_c ∩ G_c| / |P_c ∪ G_c| pour c in {0, 1}, IoU_c = 1 si l'union
  est vide ; mIoU_image = (IoU_0 + IoU_1) / 2 ; score = moyenne sur les images.
"""
from __future__ import annotations

import numpy as np

GRID = 128
CELL = 4


def downsample_mask(mask512: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Masque pixel (..., 512, 512) -> masque cellule (..., 128, 128) en uint8 0/1.

    Cellule = 1 si la fraction de pixels watermarkés est >= thr (0.5 dans l'énoncé).
    """
    # comme le kit : valeurs brutes castées en float32 (un masque 0/255 serait donc faux :
    # passer un masque 0/1 ou bool)
    m = np.asarray(mask512).astype(np.float32)  # ... 512 512
    *lead, h, w = m.shape
    assert h % CELL == 0 and w % CELL == 0, m.shape
    c = h // GRID  # 4
    m = m.reshape(*lead, GRID, c, GRID, c)  # ... 512 512 -> ... 128 c 128 c
    frac = m.mean(axis=(-3, -1))  # ... 128 128, fraction de pixels à 1 par cellule
    return (frac >= thr).astype(np.uint8)


def miou_image(pred: np.ndarray, gt: np.ndarray) -> float:
    """mIoU 2 classes pour une image. pred, gt : (128, 128) binaires."""
    p = np.asarray(pred).astype(np.uint8)
    g = np.asarray(gt).astype(np.uint8)
    assert p.shape == g.shape, (p.shape, g.shape)
    ious = []
    for c in (0, 1):  # classe 0 puis classe 1, comme le kit (p == c)
        pc, gc = p == c, g == c
        union = np.logical_or(pc, gc).sum()
        inter = np.logical_and(pc, gc).sum()
        ious.append(1.0 if union == 0 else inter / union)
    return float((ious[0] + ious[1]) / 2)


def miou_per_image(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """preds, gts : (N, 128, 128) -> (N,) mIoU par image."""
    assert preds.shape == gts.shape, (preds.shape, gts.shape)
    return np.array([miou_image(p, g) for p, g in zip(preds, gts)])


def miou_score(preds: np.ndarray, gts: np.ndarray) -> float:
    """Score officiel (selon l'énoncé) : moyenne des mIoU par image."""
    return float(miou_per_image(preds, gts).mean())
