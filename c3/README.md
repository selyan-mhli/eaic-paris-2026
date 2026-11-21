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

## État au 21/11 16:30 et premiers pas

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

Fiche complète : [../docs/HACKATHON.md](../docs/HACKATHON.md).
