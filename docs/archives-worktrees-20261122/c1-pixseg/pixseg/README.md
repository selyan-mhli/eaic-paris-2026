# c1/pixseg : segmentation au pixel, entraînée de bout en bout

Hypothèse testée : un réseau de segmentation entraîné directement sur les pixels (image 512 -> masque 128)
fait mieux que la localisation par les sorties des décodeurs (0,552 en validation imbriquée, 0,531 public).
**Aucune mesure sur données réelles n'a été faite** : le code n'a tourné que sur CPU, avec un embedder factice.

| Fichier | Rôle |
|---|---|
| `data.py` | train/test réels, générateur synthétique (régimes F et R), pré-génération en `.npz`, copies testées de la métrique et des folds |
| `model.py` | U-Net compact (2,5 M de paramètres en `--base 24`), option `--backbone` timm |
| `train.py` | `--cv` (5 folds de `gated_eval`) et `--fit-all --predict-test` |
| `export.py` | fichier de soumission `.pt` (uint8, < 20 Mo), ne soumet rien |
| `run.sbatch` | un nœud, 4 GPU, une tâche par GPU ; `configs_cv.txt` et `configs_fit.txt` : une ligne par GPU |
| `test_pixseg.py` | tests CPU |

Écart au brief : l'entrée contient le passe-haut de la luminance **et** de chaque canal R, G, B, dans une branche
d'entrée séparée du contenu (gain 64, tronqué à +-4). Avec le seul passe-haut de luminance mélangé au RGB, le
réseau n'apprenait rien sur un signal factice de +-2 à 3 niveaux sur 255 (mIoU 0,27-0,28, soit « tout 0 »,
après 600 à 1 500 pas) ; avec la branche séparée il atteint 0,78-0,85 en 600 pas sur le même jeu factice
(CPU, `--base 8`, recadrage 128, 45 images de validation synthétiques ; ce n'est pas un vrai watermark).

## Commandes sur JURECA, dans l'ordre

Depuis le dossier de la tâche (`/p/scratch/training2666/mouhali1/SLZ/Hackathon_Watermark_Localisation_Task`),
après avoir copié `c1/pixseg/` dans `work_pixseg/` (un seul dossier, 9 fichiers).

```
# 0. contrôle du script, puis fumée de 8 min : contrat de embed() des 3 méthodes, chargement, premiers pas
sbatch --test-only work_pixseg/run.sbatch gen 20
sbatch --time=00:08:00 work_pixseg/run.sbatch gen 20
#    lire output/<job>_0..2.out (« écrit ... 20 images ») et output/<job>_3.out (lignes « tout 0 » = 0,2518,
#    « tout 1 » = 0,2482, puis it/s ; la tâche 3 est coupée par la limite de temps, c'est attendu)

# 1. synthétique : 3 000 images par méthode (GPU 0-2) ; GPU 3 : CV sur le réel seul (référence sans synthétique)
sbatch work_pixseg/run.sbatch gen 3000

# 2. validation croisée, 4 configurations (une hypothèse par ligne de configs_cv.txt) :
#    base / sans retournement / synthétique seul / moitié réel
sbatch work_pixseg/run.sbatch train work_pixseg/configs_cv.txt
#    résultat : dernières lignes de output/<job>_<k>.out, à comparer à 0,552 (« réseau seuil choisi (imbriqué) »)

# 3. si une configuration bat 0,552 : la recopier dans configs_fit.txt (4 graines), avec --thr <seuil moyen affiché>
sbatch work_pixseg/run.sbatch train work_pixseg/configs_fit.txt

# 4. fichier de soumission (CPU, quelques secondes ; moyenne des 4 graines). NE SOUMET RIEN.
python work_pixseg/export.py --probs output/pixseg_test_<job>_*.npz --thr <seuil> --out output/c1_pixseg_20261122.pt
```

Si le CV ne tient pas en 1 h (lire le débit « it/s » dès la première minute) : `--folds 0,1` ou `--steps 1500`
dans la ligne de configuration. Plus de synthétique : `python work_pixseg/data.py gen --method <m> --seed 1 ...`.
`--ref-heats output/heats_train.npz` (dans `configs_cv.txt`) arrête le run si l'ordre des images diffère de celui
de `gated_eval` ; retirer l'option si ce fichier n'existe plus.

## Lecture du score

- Folds : ceux de `c1/gated_eval.py` (5 folds stratifiés méthode x `full_watermark`, graine 0), donc comparables.
- Pour le fold k, le fold (k+1) % 5 est retiré de l'apprentissage et sert seulement au choix du seuil : le
  réseau s'entraîne sur 3 folds réels (306 images) + le synthétique. `--inner-val 0` : 4 folds, seuil choisi
  sur les prédictions d'apprentissage.
- « tri seul (oracle full_watermark) » utilise la vraie étiquette F/R ; le 0,4518 de `gated_eval` trie par
  `acc_full`. Les deux ne sont pas strictement le même chiffre.
- Synthétique : sources prises dans le test seulement ; avec `--pool train,test`, `train.py` écarte les
  composites dont une source est dans un fold évalué.

## Coût GPU (estimation non mesurée)

| Étape | Estimation non mesurée |
|---|---|
| `gen 3000` | 10 à 40 min par méthode selon la vitesse de `embed` (inconnue) ; arrêt propre à 45 min, tranches écrites conservées ; 2,4 Go par méthode |
| `train --cv` (5 folds x 3 000 pas, batch 16, recadrage 448) | 30 à 50 min sur une A100, soit 4 x cela en heures-GPU pour les 4 configurations |
| `--fit-all --predict-test` | 8 à 12 min par graine |

Seule mesure disponible : sur CPU (Mac), un passage avant de 2 images 448 en `--base 24` prend 0,18 s.

## À VÉRIFIER SUR JURECA

Rien de ce qui suit n'a été exécuté :

1. `wmloc.watermarks.load_watermark(method, KIT / "models", "cuda:0")` puis `wm.embed(image uint8 h w 3, bits uint8)`
   -> `uint8 h w 3` de même taille, y compris sur un rectangle découpé de 8 px de côté (repris de `c1/seg/gen.py`
   et `gen_f.py`). `data.py` vérifie forme et dtype à chaque appel et s'arrête sinon.
2. Fichiers : `release/<split>/metadata.json` (`image_name`, `method`, `full_watermark`),
   `release/<split>/images/<nom>.png` en 512x512, `release/train/masks_128.npz` (`names`, `masks`).
3. Ordre de `metadata.json` == ordre de `output/heats_train.npz` (sinon les folds diffèrent de ceux du 0,552) :
   contrôlé par `--ref-heats`. Les lignes « tout 0 » et « tout 1 » doivent afficher 0,2518 et 0,2482.
4. `run.sbatch` n'a jamais été lancé (seulement `bash -n`) : `sbatch --test-only` d'abord. `--cpus-per-task=32`
   repris de `c1/seg/run.sbatch` ; mémoire par tâche : 7 Go de synthétique + 0,4 Go de train (+ 0,8 Go de test).
5. Débit, mémoire GPU et stabilité en AMP (batch 16, recadrage 448, `--base 24`) : jamais mesurés. Le run
   s'arrête sur une perte non finie ; `--no-amp` pour comparer.
6. `--backbone` : `timm` est peut-être absent du `.venv` de la tâche, et les poids doivent déjà être dans le
   cache HF (`HF_HUB_OFFLINE=1`). Seules les formes ont été testées, avec `resnet18` sans poids.
7. `--extra-maps` : testé sur des fichiers factices seulement (clés `(n, h, w)` ou `(n, C, h, w)`, 32 ou 128,
   alignées sur `names`). Le synthétique n'a pas de cartes : zéros et drapeau à 0 ; les cartes réelles sont
   mises à zéro 30 % du temps (`--maps-drop`) pour que le drapeau ne distingue pas réel et synthétique.
8. Soumission en `uint8` (demandé) : les fichiers précédents de l'équipe étaient en `bool`. Si le serveur
   refuse, changer le dtype dans `export.py::write_submission`.
9. Biais possible du synthétique : les sources sont des images du test d'une autre méthode, qui portent donc
   déjà le watermark de cette autre méthode (et des éditions). La ligne « synthétique seul » de `configs_cv.txt`
   mesure le transfert vers le réel.
10. Retournement horizontal : il inverse aussi le motif du watermark. La ligne `--no-flip` mesure s'il nuit.

## Tests locaux

```
uv run --no-project --with torch --with numpy --with scipy --with pillow --with pytest --with timm pytest c1/pixseg -q
ruff check --select E9,F63,F7,F82 c1/pixseg
```

`scipy` et `pillow` sont déjà importés par `c1/seg/*.py` ; `timm` ne sert qu'au test de l'option `--backbone`
(ignoré s'il est absent).
