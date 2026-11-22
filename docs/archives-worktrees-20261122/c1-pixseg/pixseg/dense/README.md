# c1/pixseg/dense : localisateur dense sur les décodeurs officiels (PS/TM, régime F)

Piste 1 de `RAPPORT-CODEX.md` (§6-7), architecture prescrite, prête pour UN pilote GPU. Rien n'a tourné sur GPU.

| Fichier | Rôle |
|---|---|
| `decoder_features.py` | Adaptateurs des modules internes PS (ConvNeXt, pas 4/8/16/32, entrée 256) et TM (ResNet-50 Q, `layer1..4`, entrée 245), même prétraitement que `wmloc/watermarks.py`, gradients possibles ; mode dégel du dernier stage avec copie figée pour la contrainte de logits ; backbones factices de même architecture |
| `locmodel.py` | Projection 32 canaux par niveau, preuve au message au pas 32 (accord signé par position, 3 résumés, message projeté), fusion descendante avec skips fins, branche passe-haut RGB 512 -> 128, sortie 128 x 128 ; normalisations fixées sur l'apprentissage (`fit_norm`) |
| `synth_msg.py` | Composites F par les embedders du kit ; message, `src`, `src2`, retournements (avant incrustation), remplissage et `acc_full` enregistrés ; sources train hors folds 0-1 |
| `dense_train.py` | Protocole fixé : contrôle (8 composites + réserve synthétique à sources disjointes), dev folds 2-4, confirmation folds 0-1 ; sortie npz `names, probs_oof, fold, thr, phase, config, train_names_hash, source_groups, report` |
| `compare.py` | Lecture CPU après job : gain par image contre `head[0]`, IC apparié descriptif, IoU des trous, supports < 1 %, décision |
| `pilot.sbatch` | PROPOSÉ, jamais lancé : 1 job, 4 GPU, génération puis 4 apprentissages (PS/TM x tête/dernier stage) |
| `test_dense.py` | 9 tests CPU |

Tests : `/Users/selyan/dev/eaic-paris-2026/.venv/bin/python c1/pixseg/dense/test_dense.py` (environ 1 min 30).
Le test de signe utilise le VRAI PixelSeal (checkpoint local `third_party/videoseal/ckpts/`, même sha256 que
celui du kit). TrustMark n'est testé qu'avec un ResNet-50 aléatoire (paquet `trustmark` absent en local).

## JURECA (à faire par la session principale)

```
# copie : c1/pixseg/ complet (data.py, train.py, model.py + dense/) -> work_pixd/ dans le dossier de la tâche
sbatch --test-only work_pixd/dense/pilot.sbatch
sbatch work_pixd/dense/pilot.sbatch
# puis, en local, après rapatriement de output/dense/res_*.npz :
python c1/pixseg/dense/compare.py --res res_pixelseal_head_<job>.npz res_trustmark_head_<job>.npz \
  --res-last res_pixelseal_last_<job>.npz res_trustmark_last_<job>.npz \
  --ref docs/task1-audit-20261121/segf_ens3_3.npz --masks c1/seg/data/masks_128.npz
```

À vérifier sur JURECA avant le job : présence du champ `message` dans `release/train/metadata.json` ; paquet
`trustmark` importable dans `.venv` et poids Q déjà en cache (sinon téléchargement impossible sur nœud) ;
`wm.net` TrustMark = ResNet-50 avec `conv1, bn1, relu, maxpool, layer1..4, fc` (supposé, lu dans le wrapper) ;
`acc_full` moyen > 0,9 dans `output/<job>_A*.out`.

Règle de décision fixée avant le job : voir l'en-tête de `compare.py` (mode `head` sauf contrôle échoué ;
POURSUIVRE >= +0,08 F PS/TM en confirmation dans les deux méthodes, ABANDON < +0,03).
