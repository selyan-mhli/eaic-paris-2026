# c1 — notes de préparation (2026-11-21, avant les vraies données)

Données : 80 images synthétiques PixelSeal (`c1/synth.py --n 80 --seed 0`, 20 par édition),
sources = 1 image + 10 frames de la vidéo de `third_party/videoseal/assets` (peu varié).
Split seedé 40 calib (choix du seuil) / 40 eval. Métrique : `c1/metric.py`, alignée sur la
référence du kit (test d'égalité dans `c1/test_metric.py`).

## Résultats locaux (eval, 40 images, mIoU)

| Méthode | eval | addition | outpainting | removal | transplant | Temps (MPS M-series) |
|---|---|---|---|---|---|---|
| tout 0 | 0.398 | 0.360 | 0.434 | 0.341 | 0.483 | 0 |
| tout 1 | 0.103 | 0.140 | 0.066 | 0.159 | 0.017 | 0 |
| window 128 px, fond flou (seuil 0.83) | 0.403 | 0.383 | 0.434 | 0.341 | 0.483 | 3.8 s/img |
| window 128 px, fond gris (seuil 0.54) | **0.494** | 0.493 | 0.430 | 0.560 | 0.475 | 4.5 s/img |
| featmap (Linear avant pooling, seuil 0.66) | 0.456 | 0.454 | 0.434 | 0.458 | 0.483 | 0.47 s/img |

Effectifs par type petits (8 à 12) : écarts de ±0.03 non significatifs.

## Ce qui marche / ne marche pas

- Fenêtres glissantes avec **fond gris** : le seul gain net (+0.10 sur tout 0), porté par
  removal/addition (watermark encore décodable globalement).
- **Le flou ne détruit pas PixelSeal** : image entièrement floutée (sigma 5) -> 58-68 % de bits
  justes. Fond flou = pas de contraste. Ne pas utiliser le flou comme « effaceur ».
- **Recadrage agrandi = décodage cassé** (crop 128 px redimensionné : 58 % de bits) : il faut
  décoder à l'échelle d'origine (fenêtre dans une image 512).
- Outpainting synthétique (réduction x0.5-0.85) et transplant (petits objets, 4 % de l'image en
  moyenne) : décodage global ~0.48, aucune méthode ne bat tout 0. Sur données réelles, l'échelle
  de l'outpainting est inconnue : vérifier si l'image watermarkée est réduite.
- featmap : 8x plus rapide, signal réel mais grossier (cellule ConvNeXt = 64 px en 512).

## Hypothèses du synthétique à vérifier sur le train réel

- Mode d'incrustation : watermark de l'image entière gardé dans le rectangle (`--embed-mode full`)
  vs rectangle watermarké seul (`crop`). Test : décoder le rectangle GT recadré vs l'image entière.
- Les éditions réelles (diffusion) peuvent réencoder toute l'image (VAE) et abîmer le watermark
  hors de la zone éditée ; le synthétique colle à bords nets.

## À adapter avec le kit officiel

- Remplacer `pixelseal_wrap.PixelSeal` par les wrappers du kit (PixelSeal videoseal@b1bf4c0,
  TrustMark 0.8.0, StegaStamp ONNX). `heat_window` prend déjà un décodeur générique
  (`logits(images)`) ; `batched=False` appelle image par image (1x3xHxW). Vérifier la convention
  de signe logits > 0 <=> bit 1 pour TrustMark/StegaStamp (sinon passer par `decode`).
- TrustMark/StegaStamp travaillent à d'autres résolutions (256 / 400) : refaire le test
  « crop agrandi vs fenêtre à l'échelle » pour chacun.
- Métrique : importer celle du kit et vérifier l'égalité sur les 510 masques train.

## Pistes suivantes

1. Saliency : gradient de la BCE (logits, message connu) par rapport aux pixels, lissé
   -> carte dense en 1 forward/backward par image.
2. Petit réseau de segmentation (tête 1x1 ou U-Net léger) sur `features(images)` + carte
   featmap/accord-bits par position, entraîné sur les 510 images (k-fold stratifié).
3. Fenêtres multi-échelles (64/128/256) avec fond gris, seuil par méthode, puis post-traitement
   par a priori (rectangle pour outpainting, trous pour removal/addition).
