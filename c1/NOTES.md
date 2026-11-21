# c1 — notes

## Résultats sur les vraies données (2026-11-21, 510 images train, kit officiel, A100 JURECA)

Métrique : `c1/metric.py` == `wmloc.utils.score` du kit (tout 0 = 0,251790, tout 1 = 0,248210, n = 510).
Wrappers (`c1/check_wrappers.py`, CPU) : `logits > 0` == `decode()`, nbits = longueur du message
(256 / 100 / 100), appel seul == appel batché (écart max 6e-6, mêmes signes).

### Deux régimes, que les métadonnées ne nomment pas pour le test

`metadata.json` ne donne pas le type d'édition, seulement `full_watermark` (train seulement).
Diagnostic (`real_heats.py --diag-only`), bit accuracy moyenne :

| Méthode | Régime | n | couv. GT | image entière | zone GT seule (reste gris) | rectangle GT recadré | zone GT grisée |
|---|---|---|---|---|---|---|---|
| PixelSeal | F | 85 | 0,897 | 0,983 | 0,985 | 0,984 | 0,503 |
| PixelSeal | R | 85 | 0,091 | 0,545 | 0,531 | 0,641 | 0,500 |
| TrustMark | F | 85 | 0,908 | 0,978 | 0,977 | 0,978 | 0,506 |
| TrustMark | R | 85 | 0,099 | 0,530 | 0,530 | 0,726 | 0,497 |
| StegaStamp | F | 85 | 0,912 | 0,970 | 0,976 | 0,964 | 0,507 |
| StegaStamp | R | 85 | 0,072 | 0,522 | 0,544 | 0,659 | 0,491 |

- **F** (full_watermark) : image entière watermarkée, puis des zones restaurées ; ~90 % de surface.
- **R** : seul un rectangle a été watermarké, à sa propre échelle : il ne se lit qu'en le **recadrant**
  (le wrapper le redimensionne), pas en place dans l'image 512.
- `acc_full >= 0,75` sépare F et R avec 12 erreurs sur 510.

### mIoU train (validation croisée imbriquée : 5 folds externes stratifiés méthode x full_watermark ;
seuil de tri, carte, normalisation et seuils choisis sur les 4 autres folds) — `gated_eval.py`

| Règle | imbriqué | non imbriqué | PixelSeal | TrustMark | StegaStamp |
|---|---|---|---|---|---|
| tout 0 | 0,2518 | — | 0,253 | 0,248 | 0,254 |
| tri seul (F -> 1, R -> 0) | 0,4518 | 0,4518 | 0,446 | 0,451 | 0,459 |
| tri + rectangles R (`gate_box`) | 0,4979 | 0,4988 | 0,493 | 0,536 | 0,464 |
| tri + cartes F + rectangles R (`full`) | **0,5088** | 0,5151 | 0,509 | 0,548 | 0,469 |

Détail `full` imbriqué : F PixelSeal 0,460 / TrustMark 0,459 / StegaStamp 0,465 (tout 1 : 0,449-0,456) ;
R 0,558 / 0,637 / 0,474 (tout 0 : 0,451-0,464). Règle refit sur 510 : tri 0,65 ; F PixelSeal fm_soft >= 0,64,
TrustMark fm_acc >= 0,57, StegaStamp win128 >= 0,49 ; R rectangle si acc_box >= 0,63 / 0,73 / 0,75.
Cartes seules sans tri (5-fold, choix sur le hors-fold, donc optimiste) : 0,458-0,468.

Fichiers test écrits (non soumis) : `output/c1_full_20261121.pt` (sha256 3952ac10…a21c, 558 F / 462 R,
couverture prédite 0,508), `output/c1_gate_box_20261121.pt` (07bd634c…efbc).

### Reproduire (depuis le dossier de la tâche sur JURECA, code copié dans `work_c1/`)

```
sbatch work_c1/j1_heats.sbatch train output/heats_train.npz          # ~6 min, 1 A100
sbatch work_c1/j1_heats.sbatch test  output/heats_test.npz           # ~12 min
for k in 0 1 2 3 4 5; do sbatch --time=00:45:00 work_c1/j1_run.sbatch work_c1/box_search.py --split train --out output/boxes_train_$k.npz --shard $k --nshards 6; done
for k in 0 1 2 3 4 5 6 7; do sbatch --time=01:00:00 work_c1/j1_run.sbatch work_c1/box_search.py --split test --out output/boxes_test_$k.npz --shard $k --nshards 8; done
python work_c1/gated_eval.py --heats output/heats_train.npz --boxes 'output/boxes_train_[0-9].npz' \
  --test-heats output/heats_test.npz --test-boxes 'output/boxes_test_[0-9].npz' --rule full --write-sub output/c1_full_20261121.pt
```
`j1_pack.sbatch` (4 tranches par nœud, un GPU chacune) donne des boîtes identiques (vérifié sur 2 tranches, --limit 2).

### Coûts mesurés

Cartes : 0,55 s/img PixelSeal, 0,26 TrustMark, 1,3 StegaStamp. Recherche de rectangle (13 500 boîtes + raffinement) :
15-70 s par image R selon la méthode (StegaStamp le plus lent) ; tranches de 39-73 images R en 25-52 min sur 1 A100.
Première grille (57 600 boîtes) : ~85 s/img, abandonnée.

### Limites

- Les meilleurs rectangles des R sont souvent faux sur les bandes fines (< 32 px) et les zones irrégulières
  (transplantations) : le seuil acc_box rejette alors vers tout 0. Max sur 13 500 boîtes du hasard ≈ 0,63-0,69.
- Les F restent proches de tout 1 (+0,01) : les trous restaurés sont mal vus par les cartes (cellule 64 px).
- Public 30 % / privé 70 % : ne pas choisir entre `full` et `gate_box` sur le leaderboard.
- La sélection de la règle (`full` vs autres) est faite sur le score imbriqué lui-même (3 candidats).

---

# Préparation synthétique (2026-11-21, avant les vraies données)

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
