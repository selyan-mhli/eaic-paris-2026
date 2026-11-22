# c3 — veille unlearning (2026-11-21 soir)

Cette veille cherche des pistes pour la Task 3 : désapprendre 3 500 images sans labels sur un ViT-B/16 timm à 100 classes, **sans aucune donnée retain**.

Conventions :
- **[lu]** : vérifié dans la source.
- **[supposé]** : inférence de notre part.
- Les coûts sont des estimations : aucun n'est mesuré.

Les dépôts des équipes de la manche Paris 2026 sont exclus (règle anti-collaboration) et n'ont pas été lus.

## 0. Ce que la métrique impose (rappel)

**Formules du score :**
- score = forget × retain
- forget = (1 − clip(d_sub/d_orig)) × (1 − clip(KL(gold‖sub)/KL(gold‖orig)))
  - d = |Δ moyenne| + |Δ écart-type| de la perte, mesuré face aux modèles gold ;
  - les KL sont calculées avec un softmax à T = 2, sur les forget et sur des copies altérées cachées.
- retain = ratio d'accuracy × 1/(1 + KL(orig‖sub))
- Porte : 70 % d'accuracy minimum.

**Conséquences :**
- **Il faut imiter un modèle qui n'a jamais vu ces images, pas le casser.** Un gold à environ 80 % d'accuracy test prédit encore la bonne classe sur la plupart des images forget, avec une confiance de niveau test. Il ne prédit pas une loi uniforme. Viser l'uniforme ou maximiser la perte fait exploser KL(gold‖sub) et d **[supposé ; c'est l'explication la plus probable de nos 0,0035 avec les soupes NPO]**.
- Le sur-oubli se détecte aussi bien que le sous-oubli :
  - U-LiRA (Hayes et al., 2403.01218) : des attaques par exemple percent SCRUB 2 à 3 fois plus que les attaques par population **[lu]** ;
  - SCRUB+R (2302.09880) rembobine vers le checkpoint dont l'erreur forget est la plus proche de l'erreur de validation **[lu]**.
- Les copies altérées cachées pénalisent une méthode qui ne modifie le modèle que sur les pixels exacts des images forget **[supposé]**.

## 1. NeurIPS 2023 Machine Unlearning Challenge (Kaggle, Google)

**Métrique [lu, arXiv 2406.09073]**
- Test d'hypothèse entre modèle désappris et modèle réentraîné, sur la confiance logit-scalée par exemple.
- ε de l'attaque la plus forte, puis H(ε) = 2/2^⌊ε/0,5⌋.
- Score = F × (acc retain / ref) × (acc test / ref), avec 512 modèles réentraînés.
- Même esprit que notre forget : c'est l'**écart à la distribution gold** qui est pénalisé.

**Podium [lu, via DeepUnlearn 2410.01276 et 2406.09073]**
- Le rang exact de Fanchuan (1er ou 2e) diffère selon les sources.
- Schéma commun : « effacer puis réparer ». L'effacement passe par une KL vers l'uniforme, du bruit ou une réinitialisation de couches, ou une montée de gradient. La réparation se fait sur le retain.

| Équipe | Effacement | Réparation |
|---|---|---|
| Fanchuan (FCS) | KL vers l'uniforme sur forget, puis montée de gradient contrastive forget/retain, 8 époques | 1 époque de cross-entropy retain après chacune |
| Kookmin (MSG) | NegGrad+ accumulé, réinitialisation des 30 % de poids conv aux plus petits gradients | Fine-tuning, multiplicateur ×1 sur les poids réinitialisés, ×0,1 sur les autres. Méthode la plus robuste dans DeepUnlearn |
| Seif (CFW) | Bruit gaussien σ = 0,6 sur les conv | 4 époques de cross-entropy pondérée sur retain, plus du bruit σ = 0,005 |
| Sebastian (PRMQ) | Réinitialisation de 99 % des poids de faible norme L1 | Cross-entropy, plus MSE d'entropie vers l'origine. Meilleur F, mais la plus forte perte d'utilité |
| Amnesiacs (KDE) | Réinitialisation de la première conv et de la FC | **3 époques de distillation KL de l'origine sur un jeu de validation (hors retain)**, puis cross-entropy + KL symétrique |

**Ce qui se transpose sans retain [supposé]**
- L'effacement (bruit, réinitialisation partielle, masque de saillance) n'utilise pas le retain.
- La réparation par **distillation du modèle d'origine sur des données proxys** est la même chose que celle d'Amnesiacs, qui distillait sur des données hors retain.
- À éviter : la KL vers l'uniforme (Fanchuan), pour la raison du §0.

## 2. Manches passées du championnat CISPA [lu]

- Paris 2025 : exemples adversariaux, attribution, détection de watermark.
- Vienne : dataset inference, attribution de texte, suppression de watermark.
- Stockholm : reconstruction, model tracer, Chimera.
- Munich : watermark.
- Finale de juillet 2026 : localisation de watermark dans du texte, inversion de gradient en apprentissage fédéré.
- `SprintML/unlearning`, créé le 17/11/2026, est le dataset de la manche en cours.

**Aucune manche passée n'a proposé d'unlearning, donc il n'existe pas de solution gagnante antérieure.** Sources : github.com/sprintml/{HackathonParis, HackathonVienna, hackathonStockholm}, cispa.de, florianzainzinger1, Loki-Hackathons/cispa_final.

## 3. Littérature sans données retain

- **SCAR** (2404.12922, ECCV 2024) **[lu]** : c'est la référence la plus proche de notre cas (ViT-B/16, sans retain).
  - Les traits forget sont poussés vers la classe erronée la plus proche (Mahalanobis rétréci ; moyennes et covariances de classe à recalculer).
  - Le retain est préservé par distillation Jensen-Shannon tempérée sur **environ 10 000 images naturelles hors du jeu** : ImageNet ou COCO marchent, le **bruit gaussien marche mal**.
  - Perte totale : L = L_M + 5·L_TD, lr 5·10⁻⁴.
  - Résultat en oubli homogène sur CIFAR-100 : test 73,2 et forget 75,6, contre 77,3 et 76,9 pour le modèle réentraîné.
- **Loss-Free SSD** (2402.19308) **[lu]** :
  - Importance d'un paramètre sans labels : |∂‖f(x)‖²/∂θ|. Si I_f > α·I_D, on atténue le paramètre (λ = 1, α = 3,5).
  - Sur ViT, la méthode **sur-oublie** : forget 24 % contre 35 % pour le modèle réentraîné.
  - I_D est calculé sur tout le jeu d'entraînement, il faudrait le remplacer par un proxy.
- **Zero-shot unlearning** (Chundawat 2201.05629) et **Bad Teacher** (2205.08096) **[lu]** :
  - Zero-shot (bruit min-max, GKT) ne traite que l'oubli de classe, et coûte des milliers d'époques.
  - Bad Teacher a besoin de 30 % du retain.
  - Inapplicables ici.
- **PSAQ-ViT** (2203.02250) **[lu, abstract]** : on ne peut pas faire d'inversion par statistiques BatchNorm sur un ViT, il faut une variante par similarité de patchs. Option pour générer des proxys, coûteuse.
- SalUn (2310.12508) et NPO : non approfondis (abstracts seulement).

## 4. Pistes priorisées

| # | Piste | Coût (estimé) | d de la perte | KL(gold‖sub) | retain |
|---|---|---|---|---|---|
| T1 | **Cible « jamais vue » calibrée** : pour chaque image forget, la cible est la moyenne des softmax de l'origine (T = 2) sur des augmentations fortes (RandAug, recadrage 30–50 %, mixup avec une image hors jeu). Température globale choisie pour que la moyenne et l'écart-type de la perte (pseudo-labels = argmax de l'origine) collent à la perte d'images non vues. On minimise KL(cible‖élève) en ne touchant que la tête et les 2 à 4 derniers blocs | 1–2 h de dev, quelques minutes de GPU | ↓↓ si la calibration est bonne (c'est le terme visé) | ↓ : erreurs vers des classes confusables plausibles ; cible issue d'altérations, donc bonne aussi sur les copies altérées | neutre seul ; protégé par T2 |
| T2 | **Garde-fou retain façon SCAR** : KL ou Jensen-Shannon (orig‖sub) à T = 2 sur environ 10 000 images naturelles hors jeu, poids d'environ 5 fois la perte forget, plus L2 vers θ₀, premiers blocs gelés. À combiner avec T1 ou T3 | < 1 h, plus copie des images sur JURECA (licence à vérifier) | neutre | légèrement freinée | ≈ 1, porte des 70 % tenue |
| T3 | **Effacer puis réparer sans retain** (Amnesiacs/MSG transposés) : bruit ou réinitialisation de la tête et des derniers blocs, éventuellement restreint par un masque de saillance (gradient forget / gradient hors jeu), puis distillation KL à T = 2 de l'origine sur les images hors jeu et sur des forget très altérées (pas les forget propres). Multiplicateur ×0,1 sur les poids hors masque | 2–3 h, ~30 min GPU | remonte vers le niveau test, sans viser l'uniforme | ↓ : le modèle retombe sur la généralisation du professeur | préservé par la distillation |
| T4 | **Arrêt ou interpolation sur la cible de perte** (SCRUB+R, appliqué à nos soupes) : choisir le α de la soupe origine ↔ modèle effacé quand la moyenne et l'écart-type de la perte forget atteignent la cible de T1. Le clip punit le dépassement | < 30 min sur les checkpoints existants | réglage direct | ↓ si la cible n'est pas l'uniforme | réglable par α |
| T5 | **LF-SSD en sélecteur** : importance |∂‖f‖²/∂θ| sur forget contre hors jeu, utilisée seulement pour choisir les paramètres que T1 ou T3 ont le droit de toucher. Jamais seule (sur-oubli documenté) | 30 min | à contrôler | à contrôler | bien préservé |

Ordre conseillé :
1. T4 sur les checkpoints existants : c'est gratuit et ça teste l'hypothèse du §0.
2. T1 + T2.
3. T3 si T1 plafonne.

Validation locale :
- calibrer la cible de perte sur de vraies images non vues (par exemple les « 1 000 photos hors forget » d'`eval.py`) ;
- comparer la moyenne et l'écart-type de la perte forget du candidat à cette cible avant toute soumission.

**Point à confirmer :** l'usage d'images publiques hors jeu est-il permis ? Il faut le demander aux mentors (même question que pour les modèles externes de c1).

## Sources

- arxiv.org/abs/2406.09073 · 2410.01276 · 2404.12922 · 2402.19308 · 2403.01218 · 2302.09880 · 2201.05629 · 2205.08096 · 2203.02250
- Fils Kaggle (référencés, non lisibles par fetch) : kaggle.com/competitions/neurips-2023-machine-unlearning/discussion/458721, /459200, /459334, /459148, /458740
- github.com/sprintml/HackathonParis · HackathonVienna · hackathonStockholm · florianzainzinger1/cispa-hackathon-grand-finale-2026 · cispa.de
