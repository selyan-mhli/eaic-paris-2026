# c1 — veille : annales, littérature, ressources (2026-11-21)

Recherche menée sur le Mac, dans les dépôts SprintML, sur Hugging Face et dans la littérature. Les mentions
**[lu]** signalent ce qui a été vérifié dans une source ou dans le code. Les mentions **[supposé]**
signalent une inférence. Les coûts sont des estimations, aucun n'a été mesuré.

## 1. Faits mesurés sur le train (masks_128.npz, 510 images)

| méthode | régime | n | couverture moy. | méd. | < 1 % | > 90 % |
|---|---|---|---|---|---|---|
| pixelseal | F (full) | 85 | 0.897 | 0.941 | 0 | 53 |
| pixelseal | R (rect) | 85 | 0.091 | 0.049 | 17 | 0 |
| trustmark | F | 85 | 0.908 | 0.944 | 0 | 55 |
| trustmark | R | 85 | 0.099 | 0.044 | 9 | 0 |
| stegastamp | F | 85 | 0.912 | 0.936 | 0 | 60 |
| stegastamp | R | 85 | 0.072 | 0.037 | 12 | 0 |

- **Aucun masque n'est vide** ni plein. L'astuce « union vide = IoU 1 » de l'énoncé ne sert donc pas sur le
  train.
- Le tri F/R est presque tout le jeu. En F, l'image est à environ 90 % watermarkée ; en R, à environ 4 %
  en médiane. Bien trier les images vaut déjà beaucoup. Le reste se joue sur le trou en F et sur la
  boîte en R.
- 38 images R (7 %) ont moins de 1 % de couverture : objets fins ou transplants minuscules.

## 2. Énoncé et kit : points à retenir

- Énoncé local : `~/Downloads/Watermark Localization.pdf`. Slides d'ouverture : `~/Downloads/EuroAIChampionshipParis2026-11-21.pdf` (slides 25-32).
- Taille du fichier : **20 Mo** selon le PDF, **25 Mo** selon le README du kit. Soumettre en bool ou uint8
  (environ 17 Mo) et vérifier que `masks.unique()` ne contient que {0, 1}.
- La règle de l'énoncé sur le modèle externe n'est pas tranchée (rien dans le PDF) : demander aux mentors
  avant d'utiliser un VAE ou un détecteur tiers.
- Le leaderboard garde le meilleur score, mais chaque soumission renvoie son propre score [lu, brief].
- `wmloc/watermarks.py` [lu] : `features()` expose déjà `per_position_logits`.
  - PixelSeal : tête linéaire appliquée avant le mean-pool, `per_position_logits` B×256×8×8 à 256 px
    (cellule = 64 px à 512). **Corrigé après mesure (c1/detch, 21/11 ~17:15) : le canal 0 n'est PAS un
    canal de détection** — checkpoint `lambda_det: 0.0`, poids de la ligne 0 ~35× plus faibles que ceux
    des bits (non entraînée), AUC pixel ≈ 0,5, aucun gain mIoU en validation croisée. Le signal utile pour
    les trous F est l'accord par position avec le message connu (AUC 0,75 sur les F PixelSeal, +0,0185 mIoU
    avec un seuil dans les F prédits, 170 images).
  - TrustMark : einsum `fc` sur `layer4` du ResNet-50.
  - StegaStamp : logits seuls. Pour obtenir une carte, il faut exposer des tenseurs intermédiaires du
    graphe ONNX comme sorties.
  - Résolutions de travail : PixelSeal 256, TrustMark 245, StegaStamp 400.
- PixelSeal [lu, `third_party/videoseal/videoseal/cards/pixelseal.yaml` et talk Fernandez] : le
  watermark porte sur la **luminance Y seule** et il est modulé par une carte JND
  (`attenuation: jnd_1_1`, `scaling_w: 0.2`). Il est donc quasi nul dans les zones plates et concentré
  sur les textures et les bords.

## 3. Annales : aucune localisation image avant aujourd'hui

- `sprintml/HackathonParis` (2025) avait une tâche de détection (Tree-Ring, BitMark, Stable Signature ;
  TPR à 1 % FPR). `HackathonVienna` avait une tâche de suppression BitMark. `hackathonStockholm` n'avait
  pas de watermark. La finale de juillet 2026 portait sur une localisation de watermark **texte**
  (`SprintML/watermark_localization`).
- Leçons transposables :
  - **Ship gate** (florianzainzinger1/cispa-hackathon-grand-finale-2026, `task1/src/eval_kgwpg3.py`) :
    un candidat n'est soumis que s'il bat le champion sur la validation croisée, sur chaque moitié et
    sur les splits inversés. D'après cette équipe, un écart de moins de +0.01 sur un public de 30 % se
    perd dans le bruit. Ici, le public compte environ 306 images.
  - **A priori de forme** : leur HSMM à durées explicites est l'équivalent 1D de notre prior
    rectangle / rectangle troué / objet.
  - **Loki** (2e à Paris 2025, `Loki-Hackathons/cispa`) utilisait des résidus SRM et FFT comme entrée du
    réseau, et calibrait le seuil sur la métrique exacte. On peut ajouter les SRM comme canaux de la
    tête de segmentation.
  - Vienne (florianzainzinger1) : les chaînes d'inpainting SD et d'img2img ressemblent aux éditions du
    test. **[supposé]** Elles abîment le watermark hors de la zone éditée.
- Papier : « What Breaks Local Watermarks? » (arXiv 2609.16832), qui évalue TrustMark et PixelSeal en
  local. Seul l'abstract a été lu ; pas de code trouvé.

## 4. Pistes priorisées

| # | Piste | Coût (estimé) | Pourquoi |
|---|---|---|---|
| P1 | **Recherche exacte du meilleur rectangle** (sous-matrice de somme max de `s - τ` sur 128², Kadane 2D) à partir des cartes existantes ; puis bords ±8 px affinés en décodant le recadrage | 1 h | Exacte et globale, moins d'1 ms par image ; remplace la grille discrète de `box_search.py` pour le régime R |
| P2 | **Détecteur de boîtes TrustMark officiel** : `TrustMark(loadBBoxDetector=True).localize(img)` (adobe/trustmark ≥ 0.9.0, `trustmark_bbox_Q.ckpt`, variante Q) [lu] | 1 h | Prior direct pour TrustMark en R. Le kit est en 0.8.0 : installer à part et vérifier que le kit utilise la variante Q. Voir aussi `remove_watermark`, qui fournit un résidu prédit |
| P3 | **Corrélation avec le résidu ré-embarqué** : `r̂ = embed(x′, m) − x′` avec le message connu, corrélation normalisée locale avec le bruit de x′ (x′ − flou, sur Y pour PixelSeal), z-score contre 8 à 16 messages aléatoires | 3–5 h | Carte au pixel, frontières nettes ; la seule piste crédible pour les transplants minuscules **[supposé]** |
| P4 | **Masque optimisé à travers le décodeur** (meaningful perturbation) : masque 32×32 sigmoïde qui maximise l'accord sur `x′⊙M + gris⊙(1−M)`, avec pénalités L1 et TV | 3–4 h, 5–15 s par image | Généralise la fenêtre sur fond gris (le seul gain mesuré, +0.10). Différentiable pour PixelSeal et TrustMark ; ONNX StegaStamp à convertir (onnx2torch) |
| P5 | **Tête de détection façon WAM** : affiner l'extracteur PixelSeal avec `--lambda_det` et `detection_loss` (`videoseal/losses/videosealloss.py:68,139`), générateurs `OutpaintingMaskEmbedder` et `RandomIrregularMaskEmbedder` (`videoseal/augmentation/masks.py`) | 4–6 h, 2–6 h A100 | Gain potentiel le plus fort, mais risque d'écart entre synthétique et réel |
| P6 | **Signal VAE / diffusion** (AEROBLADE, `jonasricker/aeroblade`) : erreur de reconstruction du VAE par région | 2–3 h | Complément pour l'outpainting et le transplant. D'abord tester sur 20 images si les zones restaurées ont été recollées pixel à pixel. Règle sur les modèles externes à confirmer |

Ordre conseillé : P1 (rapide et sûr) → P2 (canal 0 PixelSeal écarté après mesure) → P3 → P4 → tout
réinjecter comme canaux dans `c1/seg/train.py` → P5 s'il reste une demi-nuit GPU.

## 5. À écarter

- WAM tel quel : son extracteur ne voit que son propre embedder. Il ne sert que comme référence de recette.
- EditGuard et OmniGuard : ce sont des watermarks fragiles entraînés avec leur propre embedder, inapplicables ici.
- StegaStamp `detector.py` (tancik) : aucun poids publié pour le détecteur.
- Les dépôts publics créés aujourd'hui par d'autres équipes de l'épreuve (sinonchum/euro-ai-championship-paris,
  gokunguru, adembnn) ne sont pas lus, ni à lire : risque face à la règle anti-collaboration.

## Sources

- Talk P. Fernandez : https://pierrefdz.github.io/assets/slides/2026-11-21_image_watermarking_cispa.html
- Kit : https://huggingface.co/datasets/SprintML/Hackathon_Watermark_Localisation_Task (`wmloc/`)
- https://github.com/sprintml/HackathonParis · HackathonVienna · hackathonStockholm
- https://github.com/Loki-Hackathons/cispa · https://github.com/florianzainzinger1/cispa-hackathon-grand-finale-2026
- https://github.com/adobe/trustmark · https://github.com/tancik/StegaStamp · https://github.com/jonasricker/aeroblade
- Local : `third_party/videoseal/` (`models/wam.py`, `models/extractor.py`, `modules/pixel_decoder.py`,
  `augmentation/masks.py`, `losses/videosealloss.py`, `evals/full.py`)
