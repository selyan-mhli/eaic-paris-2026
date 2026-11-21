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

## v2 — papiers, auteurs, terrain, outils (2026-11-21 soir)

Reprise de la veille v2 de la session « Organisation du projet et tasks » (copie de travail hors dépôt). Les numéros A1–A10 sont ceux de cette veille ; A1 et A2 sont mesurés par `c1/probe/` (voir `docs/experiences.md`).
Conventions :
- **[lu]** : vérifié dans la source ou dans le code.
- **[supposé]** : inférence.
- Les coûts sont des estimations, aucun n'a été mesuré.
- Les dépôts des équipes de l'épreuve sont exclus et n'ont pas été lus.

### v2.0 — Ce qui change la stratégie

1. **La structure du motif diffère selon la méthode** [lu, papiers et code] :
   - **PixelSeal** répète le même message sur toute la grille du goulot du U-Net (`modules/msg_processor.py`). Son motif est presque stationnaire et équivariant par translation : chaque zone assez grande porte tout le message. Une corrélation locale ou une fenêtre suffit, sans chercher de boîte.
   - **TrustMark** concatène le message, interpolé, en entrée, et **StegaStamp** passe le message par une couche FC vers 50×50×3, ajoutée au résidu. Leur motif est **lié à la position absolue** dans l'image ou la boîte watermarkée. En régime R, il faut donc faire du template matching **paramétré par la boîte** : redimensionner `T_m = embed(gris, m) − gris` à (w, h), puis calculer la NCC avec le passe-haut de x′.
2. **La variante TrustMark du kit est très probablement Q, sans ECC, à 100 bits.** C'est le réglage du wrapper `sprintml/Radium` (`src/watermarks/post_processing/trustmark.py`) des organisateurs, avec `model_type="Q", use_ECC=False, loadBBoxDetector=False` [lu]. Le détecteur de boîtes `trustmark_bbox_Q.ckpt` est donc le bon.
3. **La question « recollé ou régénéré » décide de l'utilité de tous les outils forensiques** :
   - « What Breaks Local Watermarks? » (2609.16832) [lu] : SD-inpainting sans recoller l'extérieur fait tomber les watermarks au hasard partout. Avec l'extérieur restauré, ils reviennent.
   - TGIF (2407.11566) [lu] : TruFor obtient F1 0.83 si l'extérieur est recollé, 0.19 si toute l'image est régénérée ; CAT-Net passe de 0.87 à 0.04.
   - Dans notre train, les F ont environ 90 % de couverture et se décodent. **[supposé]** L'extérieur des trous F a été recollé. À confirmer (voir A1).
4. **Outpainting en régime R : l'image watermarkée a été réduite puis étendue** [lu, énoncé : « A watermarked image is extended to a larger image »]. L'échelle est inférieure à 1, ce qui explique pourquoi le décodage de l'image entière échoue alors que le recadrage de la boîte, redimensionné, se décode. Cela confirme la logique de `box_search.py`.
5. **Pas de détecteur PixelSeal au pixel, pas de fuite dans le dataset.**
   - [lu, issue videoseal #36] Fernandez confirme que le canal 0 ne sert à rien.
   - [lu] Aucun dépôt des 7 auteurs ne publie de pipeline d'édition ni de benchmark local.
   - [lu, répertoires des zip et chunks PNG] Les PNG ne contiennent que IHDR, IDAT et IEND, les noms sont des hashs et les horodatages sont uniformes. Rien ne révèle le modèle d'édition.

### v2.1 — Actions nouvelles, priorisées

| # | Action | Coût | Pourquoi / risque |
|---|---|---|---|
| A1 | **Test « recollé ou régénéré »** sur 20 images F | 30 min–1 h | Pour chaque image : décoder PixelSeal hors masque et comparer à un témoin watermarké propre, différencier les pixels hors masque avec une ré-embarcation, et vérifier si une carte AEROBLADE (VAE SD2) ou DinoLizer est bimodale. Le résultat décide de A5 et A6. |
| A2 | **Canal \|logit\| moyen par position** pour PixelSeal, indépendant du message (astuce de Fernandez dans l'issue videoseal #36 : les bits sûrs ont de grands \|logits\|) | ≤ 1 h | Complète l'accord avec le message (AUC 0,75 sur les F). À évaluer avec la porte de validation croisée imbriquée. |
| A3 | **Template matching par boîte** pour TrustMark et StegaStamp en régime R (voir §0.1) : recherche grossière (x, y, w, h), puis affinage, puis validation par le décodage du recadrage | ~3 h | Exploite la dépendance à la position, que P3 (corrélation insensible à la position) manquerait **[supposé]**. Risque : la part du résidu liée au contenu de l'image. |
| A4 | **Détecteur YOLO TrustMark avec scores** : appeler `detector.yolo` directement (`localize(return_all=True)` renvoie les boîtes sans score ; les scores sont dans `preds['scores']`), top-k, reclassement par l'accord sans ECC avec le message connu, puis affinage des bords | 1–2 h | Prior direct sur la boîte. Installer trustmark ≥ 0.9 dans un env séparé, car le kit est en 0.8.0. |
| A5 | **Filtre adapté JND pour PixelSeal** : passe-bande 2–8 px sur Y, pondéré par `JND(x′)` (module JND de videoseal), puis Kadane 2D (P1) | 1–2 h | Le résidu est un signal 256 sur-échantillonné, modulé par la JND (éq. 10 du papier) [lu]. Risque faible. |
| A6 | **Canal « zone éditée »** si A1 dit recollé : DinoLizer (DINOv2-B/14 + LoRA, 3 classes réel / VAE / inpainté, IoU 0,55 contre 0,34 pour TruFor, poids dans `anonyme610/dinolizer`), sinon TruFor (`grip-unina`, 261 Mo, licence recherche) | 2–3 h | Utile pour les trous F, le transplant et l'outpainting. **Règle sur les modèles externes à demander aux mentors.** |
| A7 | **Énergie du résidu StegaStamp** : x′ moins un débruitage (médian), puis énergie locale | 1 h | PSNR de 28,5 dB, donc résidu fort et visible dans les zones plates [lu, papier]. Test rapide sur 20 images. |
| A8 | **Tête de détection légère façon WAM** : extracteur PixelSeal gelé, tête 1 canal en BCE pixel sur les features des stages 2–3, masques inversés avec p = 0,5 (recette WAM [lu]). Éditions synthétiques avec `markbench/data/random_mask_generator.py` et `runwayml/stable-diffusion-inpainting` (25 pas, guidance 7,5), comme dans « What Breaks » [lu] | 4 h de code, 1–3 h A100 | Seule voie apprise : aucun détecteur au pixel publié. Risque : écart entre synthétique et réel, à contrôler par la ship gate. |
| A9 | **Raffinement SAM2.1 base_plus** (324 Mo, Apache-2.0) pour les masques d'objet et de transplant : prompt par la boîte et `mask_input` grossier, sélection parmi les multimasques par l'accord avec le watermark | 2 h | Seulement pour les objets fins. Pas utile pour les bords de rectangle. |
| A10 | **WMForger** (`third_party/videoseal/wmforger/optimize_image.py`, `convnext_pref_model.pth` à télécharger) : estime le résidu en boîte noire pour les trois méthodes, puis corrélation locale | 2 h | **[supposé]** Faible, à tester sur 20 images seulement. |

Vérifications d'hygiène, à faire tout de suite :
- **Redimensionnements anticrénelés** : TrustMark perd silencieusement le watermark sur un resize sans antialias (issue adobe/trustmark #37 [lu]). Vérifier nos wrappers et `box_search.py` (`F.interpolate(..., antialias=True)` ou PIL).
- **`scaling_w` du checkpoint PixelSeal** : le checkpoint `y_256b_img.pth` porte 1.0 alors que la carte du modèle dit 0.2 (issue videoseal #33 [lu]). Vérifier quelle valeur le kit charge avant d'utiliser son embedder pour le synthétique ou pour P3.
- **TrustMark** : entrée en [-1, 1], règle `decoder(x) > 0` → bit 1 [lu].

### v2.2 — Fiches papiers (points techniques)

- **Pixel Seal** (2512.16874) [lu] :
  - U-Net de 43,8 M paramètres à 256, extracteur ConvNeXt-v2 Tiny de stride 32 (8×8 à 256), `pixelwise: False`.
  - Embarquement : `x_w = x + α·(resize↑(w) ⊙ JND(x))`, α = 0,2, JND à la résolution native.
  - Recadrage : bit accuracy 0,58 pour 32 % de l'image conservée, 0,81 pour 45 %, 0,95 pour 63 %. L'inpainting n'est pas évalué.
- **TrustMark** (2311.18297 + `trustmark.py` 0.9) [lu] :
  - Encodeur de type MUNIT à 256, décodeur ResNet-50 avec FC globale, watermark plutôt concentré au centre (annexe D.2).
  - `ASPECT_RATIO_LIM = 2` : au-delà, recadrage carré centré. Bords lissés sur 1 %.
  - Variante P : décodeur 224, carré centré. Modèle de suppression KBNet : PSNR 48,5, bit accuracy résiduelle 0,553.
- **StegaStamp** (1904.05343) [lu] :
  - Message → FC → 50×50×3 → sur-échantillonné à 400 → U-Net → résidu additif.
  - Décodeur : STN (affine), entraîné avec ±40 px sur les coins, flou et JPEG. Il tolère donc un cadrage approximatif.
  - Détecteur BiSeNet : aucun poids publié.
  - BCH : 56 bits de données, 40 de parité, 4 de remplissage (issue #64).
- **WAM** (2411.07231) [lu] :
  - ViT-B de 96 M, sortie 1 + n bits par pixel. Perte : λ_det = 1 (BCE pixel) et λ_dec = 10 (bits, seulement sur les pixels watermarkés).
  - Masques plein, rectangle, irrégulier ou COCO, inversés avec p = 0,5. Seuil 0,5. DBSCAN seulement pour plusieurs messages.
  - Entraînement d'environ 30 GPU-jours : on ne le refait pas.
- **What Breaks Local Watermarks?** (2609.16832) [lu] :
  - PixelSeal local : image entière embarquée, puis `résidu⊙masque`. TrustMark local : boîte englobante embarquée, résidu à moyenne nulle. Très probablement la construction de nos R.
  - Le score d'évidence de PixelSeal monte de 0,36 à 27,4 avec l'aire. Les petites boîtes sont donc intrinsèquement faibles.
  - Code : `garandorinc/markbench` (attaques et générateur de masques seulement).
- **INP-X** (2602.00192) [lu] : l'aller-retour VAE explique 97 % du décalage spectral du fond. Si le fond n'a pas été recollé, la carte AEROBLADE est plate.
- **DinoLizer** (2511.20722) [lu] : DINOv2-B/14 + LoRA, tête par patch à 3 classes, fenêtres de 504 avec un pas de 128.

### v2.3 — Profs et auteurs [lu]

- **Dziedzic et Boenisch** (SprintML) : BitMark, Radioactive Watermarks, SERUM, RADIUM, Watermark Degradation. Rien sur la localisation.
  - `sprintml/Radium` contient les wrappers TrustMark et StegaStamp, ancêtres probables de `wmloc` : TrustMark Q sans ECC, StegaStamp en ONNX à 400.
  - `sprintml/BitMark/watermark_attacks.py` contient les attaques VAE, DiffPure et CtrlRegen (Realistic Vision 4 + `sd-vae-ft-mse`), toutes globales.
- **Meintz** et **Kerner** : coauteurs de BitMark et de Watermark Degradation, aucun dépôt public. **Hector** : aucun papier. **Shah** : son site mentionne « challenge design » ; `maitri01/hack` ne contient que le setup JURECA.
- **Fernandez** (FAIR) : WAM, VideoSeal, PixelSeal, SyncSeal, WMForger, ChunkySeal, DistSeal. Il n'existe aucune carte de localisation ni aucun détecteur au pixel pour PixelSeal. La keynote ne montre que LaMa comme édition locale.
- **Modèle d'édition** : jamais nommé. Indices : SD et Realistic Vision chez SprintML, chaînes d'inpainting SD à Vienne, LaMa chez Fernandez. **[supposé]** C'est un pipeline diffusers de type SD, SDXL inpainting ou FLUX Fill.

### v2.4 — Terrain (Reddit, HN, issues)

- Reddit, HN et X : rien d'exploitable.
- Issues GitHub utiles : videoseal #36, #33, #81, #94 ; trustmark #37 et CONFIG.md ; StegaStamp #2, #33, #43, #64.

### v2.5 — Outils open source (avec poids)

| Outil | Licence / poids | Sortie | Pertinence |
|---|---|---|---|
| DinoLizer | poids dans le README de `anonyme610/dinolizer` | 3 classes par patch | La plus directe pour « zone éditée », si recollé |
| TruFor | recherche, 261 Mo | carte d'anomalie + confiance | Forte si recollé, faible sinon |
| SAFIRE | CC-BY 4.0 / Apache-2.0 | partition multi-source | Moyenne à forte pour le transplant [supposé] |
| AEROBLADE | aucune licence, VAE depuis HF | LPIPS par pixel | Test A1, canal pour les F si recollé |
| HiFi-Net | MIT | masque binaire | Moyenne |
| SAM2.1 | Apache-2.0, 324–898 Mo | masque à partir d'une boîte ou de points | Raffinement d'objet (A9) |
| DINOv2 | Apache-2.0 | features | Backbone de tête légère (DINOv3 est gated) |
| SimpleCRF | BSD-3 | CRF dense | Remplace `pydensecrf`, qui casse à l'installation |
| À écarter | CAT-Net, PSCC, MVSS, DIRE, FakeShield ou SIDA (13–22 B, lourds), OmniGuard, EditGuard | | |

### v2 — Sources

- arxiv.org/html/2512.16874 · 2311.18297 · 1904.05343v2 · 2411.07231 · 2609.16832 · 2602.00192 · 2511.20722 · 2407.11566v2
- github.com/sprintml/Radium · sprintml/BitMark · facebookresearch/videoseal (issues 33, 36, 81, 94) · adobe/trustmark (issue 37, CONFIG.md) · tancik/StegaStamp
- github.com/garandorinc/markbench · grip-unina/TruFor · mjkwon2021/SAFIRE · jonasricker/aeroblade · facebookresearch/sam2 · HiLab-git/SimpleCRF · anonyme610/dinolizer
- Copies de travail dans le scratchpad, non durables : `hf2/`, `tm/trustmark.py.txt`, `rd/`, `slides/lits.txt`
