# c1 — Task 1 : Watermark Localization

- **Propriétaire :** Selyan (GitHub selyan-mhli, Jülich mouhali1)
- **Données et env (JURECA) :** `/p/scratch/training2666/mouhali1/SLZ/Hackathon_Watermark_Localisation_Task/`
  (accès Hugging Face en attente de validation)
- **But :** pour chaque image test 512×512, un masque binaire 128×128 (1 = watermarqué).
  Méthodes : PixelSeal, TrustMark, StegaStamp. Le test donne la méthode et le message.
- **Métrique :** mIoU sur 2 classes, moyennée par image. Public 30 %, final 70 %.
- **Soumission :** `.pt` `{"image_names": names, "masks": Tensor[N,128,128]}` en 0/1, ≤ 20 Mo.
- **Validation locale :** k-fold sur les 510 images train, stratifié par méthode et type d'édition.

Fiche complète : [../docs/HACKATHON.md](../docs/HACKATHON.md).

## Contenu prévu

- `eval.py` : mIoU locale au format officiel (figé, vérifié à la main).
- `solve.py` : méthode courante (`--out <dir>`, contrat autoresearch).
- `program.md` : consignes de la boucle autoresearch pour cette tâche.
