# c1/reembed — carte de ré-embarquement pour le régime F

> **Résultat réel (2026-11-21, pilote 120 F train, job 15902592) : piste ÉCARTÉE.** AUC pixel poolée
> re_z 0,505 (PixelSeal), 0,513 (TrustMark), 0,674 (StegaStamp), contre fm_acc 0,758 / 0,727 et win128 0,703 ;
> mIoU F « 2 moitiés » = tout 1 (+0,000 à +0,002). Passage complet et branchement dans gated_eval non faits.
> Signatures vérifiées sur JURECA : `wm.embed(uint8 HxWx3, bits)` -> uint8 512x512x3 pour les 3 méthodes,
> `LocalisationDataset(..., methods=)` ; coût mesuré ~0,7 s/image par A100 (9 embeds).

But : voir la zone éditée (~9 % des pixels) des images F, que les cartes par cellules de 64 px ne voient pas
(F plafonné à 0,46 de mIoU ; « F parfait » donnerait 0,778 au global).

Carte : r̂(m) = embed(x', m) − x' (Y pour PixelSeal, RVB sinon), h = x' − flou_gaussien(x') (σ = 1),
corrélation normalisée locale c(m) entre h et r̂(m) (fenêtres 16 et 32 px, moyenne des échelles),
z-score contre K = 8 messages aléatoires à graine fixe (dérivée du nom d'image) :
`z = (c(m) − moy_k c(m_k)) / ét_k c(m_k)`. Grand z = watermark présent (= masque 1 du kit).

| Fichier | Rôle |
|---|---|
| `maps.py` | calcul GPU par lots ; npz `names, method, full_watermark, done, re_z, re_zg, re_c` (float16 N×128×128) ; sortie partielle toutes les 50 images |
| `eval_auc.py` | AUC pixel contre la vérité « trou » sur les F du train, par méthode, + mIoU F en 2 moitiés (seuil choisi sur l'une, appliqué à l'autre) |
| `merge.py` | fusion des shards et ajout au `heats_<split>.npz` (`re_z, re_zg, re_c, re_zs, re_zgs`) pour `gated_eval.py` |
| `run.sbatch` | job 1 nœud, 4 tâches / 4 A100, 30 min |
| `test_maps.py` | tests CPU synthétiques sans wmloc |

`re_zg` : même numérateur, écart-type des K messages groupé sur toute l'image (moins bruité que l'écart-type
par pixel avec K = 8). `re_zs` = sigmoid(z/2) ∈ [0, 1] (la grille de seuils de gated_eval va de 0 à 1).

## À VÉRIFIER SUR JURECA (adaptateurs `load_kit`, `load_embedder`, `as_float_hwc` dans maps.py)

- `wm.embed(image uint8 H×W×3, message uint8 k)` renvoie une image à la taille d'entrée (512), uint8 H×W×3
  (usage de `probe/recolle.py`). `as_float_hwc` accepte aussi un tenseur 1×3×H×W float ; une autre taille
  fait échouer l'assertion de `residuals`.
- `LocalisationDataset(data, split, methods=[...])` et `ds.meta[i]` avec `method`, `image_name`,
  `full_watermark` (train) ; `ds[i]["message"]` présent aussi en test.
- Embed appelé image par image (9 appels par image) : un appel batché n'est pas supposé. Le coût par image
  n'est pas mesuré (le log affiche `s/img`). Le calcul des cartes seul : ~3 s/image sur CPU Mac (mesuré,
  2 images 512, 4 threads) ; sur GPU, non mesuré.

## Commandes (depuis le dossier de la tâche, `c1/` copié dans `work_c1/`)

```sh
# 0. essai rapide sur un nœud GPU (pas sur le login) : 2 F par méthode, 1 tâche
sbatch --test-only work_c1/reembed/run.sbatch work_c1/reembed/maps.py --split train --only-full --limit 2 --out output/re_smoke.npz
sbatch --ntasks-per-node=1 --gres=gpu:1 --time=00:10:00 work_c1/reembed/run.sbatch \
    work_c1/reembed/maps.py --split train --only-full --limit 2 --nshards 1 --out output/re_smoke.npz

# 1. PILOTE : 40 F par méthode (120 images), 4 tâches -> output/re_pilot_s{0..3}of4.npz
sbatch work_c1/reembed/run.sbatch work_c1/reembed/maps.py --split train --only-full --limit 40 --out output/re_pilot.npz
#    variante dev (max 2 jobs par utilisateur sur dc-gpu-devel ; À VÉRIFIER : que --reservation= vide annule
#    bien la réservation du script, sinon copier run.sbatch sans la ligne --reservation, et --test-only d'abord) :
sbatch -p dc-gpu-devel --reservation= work_c1/reembed/run.sbatch work_c1/reembed/maps.py --split train --only-full --limit 40 --out output/re_pilot.npz

# 2. éval CPU (login suffit) ; --heats ajoute fm_acc/fm_soft/win128 sur les mêmes images
python work_c1/reembed/eval_auc.py --re 'output/re_pilot_s*of4.npz' --heats output/heats_train.npz
```

Critère pour aller plus loin : AUC poolée de `re_z` ou `re_zg` nettement au-dessus de `fm_acc` (0,747 sur
PixelSeal F, cf. docs/experiences.md) et mIoU F 2-moitiés > tout 1 (~0,45).

```sh
# 3. si le pilote est bon : toutes les images que le tri F/R peut classer F (acc_full >= 0.55 < GATES min 0.60)
sbatch --time=01:00:00 work_c1/reembed/run.sbatch work_c1/reembed/maps.py --split train \
    --gate-heats output/heats_train.npz --min-acc 0.55 --out output/re_train.npz
sbatch --time=01:30:00 work_c1/reembed/run.sbatch work_c1/reembed/maps.py --split test \
    --gate-heats output/heats_test.npz --min-acc 0.55 --out output/re_test.npz
python work_c1/reembed/merge.py --re 'output/re_train_s*of4.npz' --heats output/heats_train.npz --out output/heats_train_re.npz
python work_c1/reembed/merge.py --re 'output/re_test_s*of4.npz'  --heats output/heats_test.npz  --out output/heats_test_re.npz
```

## Brancher dans `gated_eval.py`

`gated_eval.Data` lit `d[k]` pour `k in KEYS` dans le heats, en modes `abs` et `rel`, prédit F = carte ≥ t
(t ∈ [0, 1]) et écarte une variante pour une méthode si une image classée F n'a pas de valeur finie.

- Option A (propre, une ligne dans `c1/gated_eval.py`, hors de ce dossier) :
  `KEYS = ("win128", "win256", "fm_acc", "fm_soft", "re_zs", "re_zgs")`, puis
  `python work_c1/gated_eval.py --heats output/heats_train_re.npz --boxes 'output/boxes_train_[0-9].npz' [--test-heats output/heats_test_re.npz ...]`.
  `re_z` brut ne marche qu'en `rel` ; préférer `re_zs`.
- Option B (sans toucher au code) : `merge.py ... --as-key fm_soft` remplace `fm_soft` par `re_zs` dans le
  fichier de sortie. StegaStamp gagne ainsi une carte (son `fm_soft` est NaN), mais `fm_soft` est perdu.

Le score à lire est la ligne « full imbriqué » (CV imbriquée 5 folds). Rien n'est soumis par ces scripts.

## Tests locaux

```sh
ruff check --select E9,F63,F7,F82 c1/reembed
uv run --no-project --python .venv/bin/python --with pytest python -m pytest -q c1/reembed/test_maps.py
```
