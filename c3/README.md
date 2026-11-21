# c3 — Task 3 : Unlearning

- **Propriétaire :** à préciser
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

Fiche complète : [../docs/HACKATHON.md](../docs/HACKATHON.md).
