# c3 — Task 3 : Unlearning

- **Propriétaire :** Yihuan (GitHub Yihuan0301, Jülich zhang95)
- **Données et env (JURECA) :** `/p/scratch/training2666/mouhali1/SLZ/unlearning/`
  (`model.pt`, `forget_images.npz`, `task_template.py`, `submission.py` ; `.venv` avec torch et timm)
- **But :** à partir du ViT-B/16 (timm `vit_base_patch16_224`, 100 classes), un modèle qui se
  comporte comme s'il n'avait jamais vu les 3 500 images, sans abîmer le reste.
- **Métrique :** forget_score × retain_score ; **0 si l'accuracy sur la validation cachée < 70 %**.
  Public 30 %, final 70 %.
- **Soumission :** state dict float32, `strict=True` sur l'architecture du template, ≤ 450 Mo.
  Le serveur évalue en float16 autocast : vérifier le modèle en fp16 avant de soumettre.
- **Validation locale :** faible (pas de gold, pas de données retain). Proxys et prudence ;
  le score public ne donne qu'une tendance.

## Premiers pas (état au 21/11 16:30)

- `audit.py` de Yihuan (dans le dossier SLZ, pas encore dans le dépôt) établit le contrat :
  ViT 85 893 988 paramètres, `forget_images` uint8 [3500, 224, 224, 3] **sans labels**,
  aucun retain/val loader fourni. → à ranger dans `c3/audit.py` via une PR.
- Aucune soumission c3 encore. Étape 1 : soumettre le modèle d'origine tel quel
  (`unlearn` = identité) pour tester clé, format et fp16, et avoir le score plancher.
- Étape 2 : pseudo-labels = argmax du modèle d'origine sur les forget ; proxy retain = accord
  avec le modèle d'origine sur des versions fortement augmentées des forget (ou un jeu public
  à 224 px) ; garder l'accuracy proxy > 72 %.
- Étape 3 : baseline SCRUB ou NPO (enseignant = modèle d'origine), en job Slurm `-J j3`.
- Dépendances ajoutées à la main au `.venv` partagé (scikit-learn, scipy) : passer par `uv add`.

## État du dépôt (21/11 soir)

| Fichier | Rôle | État |
|---|---|---|
| `eval.py` | Validation locale par proxys (retain, forget, fp16, format), affiche `score: <float>` | testé sur modèle factice et sur timm `vit_tiny` / `vit_base` à poids aléatoires ; **jamais lancé sur le vrai `model.pt`** |
| `test_eval.py` | Tests CPU de `eval.py` (~4 s) | en CI : les tests numpy tournent, ceux qui demandent torch/timm sont sautés (la CI n'installe que numpy) |
| `unlearn_pack.sh` | 4 variantes de `unlearn_v2.py` en parallèle sur un nœud | jamais lancé |
| `audit.py`, `unlearn_v2.py`, `soup_search.py`, `unlearn.py`, jobs `j3*` | Méthode de Yihuan | **sur JURECA seulement**, à verser (voir plus bas) ; résultats à préciser par Yihuan |

Architecture : `task_template.build_model` (JURECA, lignes 55-62) crée
`timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=100, init_values=1e-4)`.
`init_values` active LayerScale (`ls1.gamma`, `ls2.gamma` : 2 × 768 par bloc × 12 blocs = 18 432
paramètres), d'où les **85 893 988** paramètres d'`audit.py` (85 875 556 sans LayerScale).
`eval.py` utilise la même construction ; un test vérifie ce nombre.

## `eval.py` : ce qu'il mesure, et ce qu'il ne mesure pas

C'est un **proxy**, pas la métrique officielle : pas de modèles gold, pas de labels, pas de
données retain. Il sert à comparer des candidats entre eux et à bloquer les soumissions cassées.

- (a) **retain proxy** = accord top-1 × exp(−KL à T=2) entre candidat et modèle d'origine, sur
  `--retain-npz` s'il est fourni (recommandé : un jeu public à 224 px), sinon sur des
  augmentations fortes des forget.
- (b) **forget proxy** = exp(−(|Δmoyenne| + |Δécart-type|)) entre la perte du candidat sur les
  forget (pseudo-labels = argmax de l'origine) et la perte de l'origine sur des augmentations
  fortes des forget (substitut d'images « jamais vues »).
- (c) **fp16** : écart des logits fp32 / fp16 autocast, sorties non finies, part des top-1 qui changent (> 1 % → échec).
- (d) **format** : `strict=True` sur timm `vit_base_patch16_224` à 100 classes avec `init_values=1e-4`, tout en float32, fichier ≤ 450e6 octets.
- `score` = (a) × (b), mis à 0 si (c) ou (d) échoue ou si l'accord retain < `--gate` (0,72 par défaut).

Limites : sans `--retain-npz`, (a) et (b) portent sur les mêmes images et se contredisent en
partie ; « augmenté ≈ jamais vu » est une hypothèse ; l'accord avec l'origine n'est pas
l'accuracy que le serveur mesure ; normalisation (`--mean`/`--std`, défaut 0,5) et clé du npz
(`--forget-key`, défaut : premier tableau) à vérifier contre `task_template.py`.

## Comment verser tes scripts (Yihuan)

1. Branche depuis `main` à jour : `git switch -c c3/<idée>` (une PR par idée).
2. Copie les scripts dans `c3/` sous le même nom : `c3/audit.py`, `c3/unlearn_v2.py`
   (et `soup_search.py`, `unlearn.py` dans une PR suivante si tu veux).
   **Pas de poids ni de données** : `*.pt` et `data/` sont ignorés, ne force pas.
   Chemins JURECA en dur : seulement comme valeur par défaut d'un argument
   (`--task-dir`, comme dans `eval.py`), jamais de clé.
3. Si `unlearn_v2.py` n'a pas encore `--tag` : l'ajouter (suffixe des checkpoints), sinon
   `unlearn_pack.sh` écrase les sorties des 4 runs.
4. Évalue chaque checkpoint avec `eval.py`, depuis le dossier de la tâche sur JURECA, dans un
   job (pas sur le nœud de login) :

   ```bash
   cd /p/scratch/training2666/mouhali1/SLZ/unlearning && source .venv/bin/activate
   python /chemin/vers/le/depot/c3/eval.py --candidate output/v2_ent05.pt            # un candidat
   python /chemin/vers/le/depot/c3/eval.py --candidate model.pt                      # plancher : origine
   python /chemin/vers/le/depot/c3/eval.py --candidate output/x.pt --max-images 500  # essai rapide
   ```

   Options utiles : `--retain-npz <npz uint8 [N,224,224,3]>`, `--n-aug 2`, `--seed`, `--device cuda`.
   Code de sortie 1 si le format est invalide.
5. Une ligne par run dans `docs/experiences.md` (ajout seulement), colonnes :
   date-heure | challenge | hypothèse | changement | coût | score local | score serveur | sortie. Exemple de forme :

   ```
   | 2026-11-21 HH:MM | c3 | <hypothèse> | unlearn_v2.py --lr 1e-5 --tag _g0 (job <id>) | <durée>, 1×A100 | eval.py proxy <score> (retain <a>, forget <b>, fp16 ok) | non soumis | output/v2_..._g0.pt |
   ```
6. Tests : `pytest -q c3/test_eval.py` ; ajoute un test pour toute fonction pure que tu verses.
   La CI lance ruff (`--select E9,F63,F7,F82`) et pytest ; elle doit être verte avant la revue.

Fiche complète : [../docs/HACKATHON.md](../docs/HACKATHON.md).
