# Task 1 — diagnostic et mandat technique pour Opus

> **Reprise du 22/11/2026 :** ce rapport ci-dessous décrit l'audit T1 du 21 au soir. L'état consolidé ultérieur (envois, champions, jobs et relais actif) se trouve dans [LIRE-DABORD](docs/reprise-claude-20261122/LIRE-DABORD.md) et dans le [rapport de reprise détaillé](docs/reprise-claude-20261122/RAPPORT-CODEX.md). Instantané vérifié vers 03:05–03:07 Paris : T1 0,6739 ; T2 0,20 ; T3 0,0941. Ne pas reprendre les anciennes priorités ci-dessous comme état courant sans lire cette mise à jour.

Audit du 21 novembre 2026, soirée, heure de Paris. Demande directe de Selyan ; lecture SSH de JURECA autorisée pendant l'audit. Aucun apprentissage, aucune inférence de modèle, aucun job GPU, aucune soumission et aucune modification du champion effectués par cet audit.

## Décision

**Priorité unique : apprendre une localisation dense à partir des features intermédiaires des décodeurs officiels, avant leur réduction spatiale, avec supervision de masque et accord au message connu. Commencer par les F PixelSeal et TrustMark.**

Le changement recherché est le passage d'un décodeur global à un extracteur supervisé de régions. Un nouveau U-Net entraîné de zéro sur RGB/passe-haut a déjà été essayé sur JURECA : le relancer à l'identique serait une erreur. Une nouvelle fusion des cartes finales, un Grad-CAM sur la dernière couche ou un changement de seuil ne fourniraient pas le signal qui manque.

Cette priorité est une **hypothèse à tester**, pas une méthode dont le gain serait acquis. Son potentiel est supérieur aux pistes restantes parce qu'elle conserve les poids qui savent lire le watermark et récupère les features fines que les cartes actuelles ont éliminées. Le premier objectif de validation est un gain global robuste d'au moins **+0,03** ; 0,70 et 0,80 restent des objectifs ultérieurs, sans preuve actuelle de faisabilité dans le temps restant.

Avant ce pilote, Opus doit corriger la provenance de la référence et l'export NCC décrit ci-dessous. C'est une préparation de quelques minutes, pas une seconde recherche algorithmique.

## 1. Ce qui a effectivement été audité

Sources : dépôt principal, worktrees `c1-probe-1b51d2`, `c1-reembed`, `eaic-wt-c1-rect`, `eaic-wt-c1-tmatch`, `eaic-wt-c1-pixseg`, scripts et rapports `task1_*` dans Downloads, contexte fourni par Selyan, puis scripts, résultats et logs JURECA. Aucun dépôt de concurrent du concours consulté.

Au départ et avant rédaction :

```text
checkout : /Users/selyan/dev/eaic-paris-2026
branche  : main
HEAD     : f209e20397736c53824a7a94ec73f2bd66d012eb
état     : modifications préexistantes c1/seg/gen_f.py, train_f.py, docs/experiences.md ;
           combine.py, export.py, kfold_f.py, docs/PLAN-NUIT.md non suivis.
```

Je n'ai changé ni branche ni historique et n'ai modifié aucun de ces fichiers. Les seuls ajouts de cet audit sont ce rapport et `docs/task1-audit-20261121/`. `relay-check.py` n'a pas pu observer les processus ; son absence de pairs ne prouve donc aucune exclusivité de session.

Le dossier JURECA de la tâche **n'est pas un dépôt Git**. En revanche, le SHA-256 de `work_segf/head_eval.py` est identique à celui du fichier du worktree local `c1-probe-1b51d2` :

```text
66f815c304cc061a136b16a7de57a7be00b030d6088bd9a1eebf996c99169f31
```

Worktrees de sources : `c1-probe-1b51d2` à `409e4ec85a5061302478621fd0407ff280faafe7` ; `eaic-wt-c1-pixseg` à `2bb6b4bb106ebeb21db4ba0c1c6753e1ca5f0b97`, dont `c1/pixseg/` est non suivi. Un SHA Git seul ne décrit donc pas tous les scripts expérimentaux.

Les scripts d'analyse CPU, résultats structurés, instantané des scores et boîtes, hashes et trace d'échec sont conservés dans `docs/task1-audit-20261121/`. Les données train et les cartes déjà présentes dans `c1/seg/data/` restent leurs entrées locales ; elles ne sont pas incorporées de nouveau au rapport.

## 2. Audit du 0,5614 : score confirmé, lien au fichier public incorrect

### Score et protocole

Fichier vérifié :

```text
/p/scratch/training2666/mouhali1/SLZ/Hackathon_Watermark_Localisation_Task/output/segf_ens3_3.npz
SHA-256 : c182e5dac66fec152ae4a5784433fd6866dfe5af935cf02f29e0b887f4bb0896
```

Il contient `names[510]`, `champ[3,510]`, `head[3,510]`, `choices`. **Ce sont des scores par image, pas des masques ni des probabilités OOF.**

```text
CHAMP : 0,5097164288
HEAD  : 0,5614248628
DELTA : +0,0517084340
HEAD par graine externe : 0,56055410 / 0,56029068 / 0,56342981
```

Les trois lignes correspondent à trois partitions externes ; à l'intérieur de chaque entraînement, `--ens 3` moyenne trois réseaux. Ce n'est pas simplement la moyenne de trois scores de modèles entraînés sur tout le train.

Le code lu met correctement à l'écart le fold externe :

- cinq folds externes stratifiés méthode × F/R ;
- réseaux entraînés sur les F de l'apprentissage externe ; normalisation apprise sur ces mêmes images ;
- quatre folds internes pour produire les prédictions qui servent à choisir le seuil du réseau ;
- choix du tri et des paramètres de boîtes sur l'apprentissage externe seulement ;
- réseau final de chaque fold entraîné sans ses images externes, y compris pour chacune des trois graines de l'ensemble.

**Vérification indépendante supplémentaire :** j'ai reconstruit sur CPU les boîtes et les seuils R à partir des noms, folds et fichiers de boîtes. Les scores des images routées R sont identiques aux scores sauvegardés pour les trois partitions : **écart maximal 0,0**, sur respectivement 235, 232 et 233 images. L'alignement des 510 noms est exact.

Cela étaye le score et le protocole ; l'absence de probabilités F sauvegardées empêche de recalculer indépendamment ses IoU pixel ou d'auditer visuellement les erreurs F exactes. Il faut conserver ces probabilités lors du prochain entraînement, avec les folds, paramètres, graines et hashes des entrées.

Le gain apparié sur les scores est +0,0517, IC95 descriptif par bootstrap d'images `[+0,0436 ; +0,0609]`. Cet intervalle ne corrige ni la sélection des configurations déjà explorées ni la dépendance entre modèles de CV. Le progrès contre l'ancienne règle est substantiel ; il ne garantit pas un gain privé de cette amplitude.

### La configuration exacte et l'échec d'export

L'accounting Slurm identifie la commande du job 15902663 :

```text
sbatch --parsable -p dc-gpu --time=00:25:00 work_segf/head.sbatch segf_ens3
  ff+fm@20 ff+fm@10 ff@20 ff+fm+hf:ncc_m,ncc_d@20 --seeds 0 1 2 --ens 3
```

La tâche 3 est donc **`ff+fm+hf:ncc_m,ncc_d`, 20 époques, ensemble de trois réseaux**, évaluée sur trois partitions externes. Le job global est **FAILED**, après 15 min 48 s. Le log prouve que l'évaluation complète et la sauvegarde NPZ ont précédé l'échec :

```text
ff+fm+hf:ncc_m,ncc_d (moy.) all=0.5614
...
T32, T128 = build_inputs(... load(a.test_hf) if a.test_hf else {}, ...)
AssertionError: ncc_m
```

`work_segf/head.sbatch` passe `--hf` pour le train, mais pas `--test-hf` lors de l'export automatique. Les cartes `output/segf_hf_test_0..3.npz` existent. La correction consiste à fournir le motif test correspondant et à valider les canaux requis avant tout entraînement. Elle ne nécessite aucune modification du modèle ni du champion existant.

### Le candidat public cité est un autre artefact

```text
c1_segf_ffm_e20_20261121.pt == segf_sub_1.pt
SHA-256 commun : 514470852e15f8f4fc4316932ee8ece46ea86218bd067c48f1f2b9181c5ab43f
```

Le job 15902420 a exécuté `segf_sub ff+fm@60 ff+fm@20 ff@60 ff@20 --seeds 0`. Son log tâche 1 annonce **0,5577 OOF**, configuration `ff+fm`, puis écrit `segf_sub_1.pt`. Le fichier public nommé n'est donc pas la réalisation test de la configuration 0,5614. L'audit confirme l'identité des fichiers, pas leur historique d'envoi au serveur. **0,5572 public reste un score communiqué par Selyan, non revérifié sur le leaderboard.**

Commande de reproduction algorithmique du 0,5614, à exécuter seulement après autorisation et coordination GPU, avec un nouveau nom de sortie :

```bash
.venv/bin/python work_segf/head_eval.py \
  --config 'ff+fm+hf:ncc_m,ncc_d' --epochs 20 --ens 3 --seeds 0 1 2 \
  --heats output/heats_train.npz --boxes 'output/boxes_train_[0-9].npz' \
  --ff 'output/seg_ff_train_*.npz' --hf 'output/segf_hf_train_*.npz' \
  --out output/audit_repro_segf_ens3_ncc.npz
```

Cette commande n'a pas été exécutée par l'audit. Un nouveau run peut légèrement différer : les poids d'entraînement, versions d'environnement et états déterministes complets ne sont pas enregistrés dans le NPZ ancien.

## 3. Goulot mesuré, méthode par méthode

Scores de la configuration 0,5614, moyennes des trois partitions, 85 images par ligne :

| Groupe | Ancienne règle | Tête actuelle | Diagnostic principal |
|---|---:|---:|---|
| PixelSeal F | 0,4588 | **0,5715** | Signal global excellent ; support local mal résolu, trous fins et faux trous. |
| TrustMark F | 0,4583 | **0,5393** | Support local/contextuel insuffisant ; ce n'est pas un échec général du décodeur. |
| StegaStamp F | 0,4666 | **0,5565** | Pas de features spatiales ONNX exploitées ; dépendance aux occultations et aux canaux ajoutés. |
| PixelSeal R | 0,5647 | **0,5842** | Localisation/échelle et régions irrégulières ; 17 régions de moins de 1 %. |
| TrustMark R | 0,6417 | **0,6415** | Meilleur signal de boîte ; erreurs de localisation, de forme et rejet des objets. |
| StegaStamp R | 0,4682 | **0,4757** | Plus gros déficit ; signal de décodage utilisable sur des rectangles, recherche actuelle peu efficace. |

### Classification F/R

Le diagnostic initial à 0,75 est confirmé : 12 erreurs sur 510, réparties PS F/R = 0/6, TM = 1/3, SS = 1/1. La règle majoritaire passe de 0,45179 à 0,45469 avec le vrai régime : seulement **+0,00290** dans ce diagnostic.

La configuration 0,5614 choisit des seuils plus bas, surtout 0,65 : elle route **20 à 23 vrais R vers la branche F**, et aucun vrai F vers R. Ce choix est appris pour la métrique de segmentation ; il n'optimise pas l'accuracy F/R. Ces cas doivent être rapportés séparément. Ils ne prouvent pas qu'un classifieur F/R parfait améliorerait automatiquement l'assemblage : certaines prédictions de tête F sont utiles sur des R mal décodés.

### Localisation et formes R

Le score **0,866862** de bbox GT est reproduit, contre 0,456477 en tout-0. La médiane de solidité proche de 1 ne signifie pas que tous les masques sont rectangulaires : seuls **80/255 sont exactement des rectangles sur la grille 128** — 28 PS, 28 TM, 24 SS. Une bbox reste toutefois une bonne approximation pour beaucoup de formes.

Les boîtes sauvegardées du pipeline principal atteignent seulement **0,5320 sur les vrais R si on les prédit systématiquement**. Même choisir avec la GT entre cette unique boîte sauvegardée et tout-0 ne donne que 0,5573 sur ce sous-ensemble. Ce dernier chiffre ne borne pas la tête actuelle, qui utilise aussi des prédictions F sur certains R.

Plus directement : sur les branches effectivement routées R, remplacer uniquement la décision accepter/rejeter par une décision oracle gagnerait **+0,0025 à +0,0034 au global**. **Le problème exige de nouvelles boîtes ou de nouveaux masques ; la simple calibration du seuil de la boîte existante ne peut pas apporter +0,10.** Un nouveau score n'est intéressant que s'il change la localisation ou le classement des propositions.

### Localisation et formes F

Les trois décodeurs lisent presque parfaitement le message global des F ; les diagnostics anciens indiquent 0,97–0,98 de bit accuracy. Le problème est de retrouver tous les pixels qui portent le signal, pas de décider s'il existe quelque part.

Tous les F ont des trous non rectangulaires ; la médiane est de deux composantes sur la grille. Le rectangle englobant tous les trous est mauvais : **0,5492** reproduit. L'oracle grille à pas 8 communiqué atteint **0,7482** parce qu'il choisit un rectangle favorable avec la GT, pas parce qu'il retrouve toute la forme. Le code Downloads confirme que cette valeur est un oracle restreint, **pas un plafond absolu**. À titre de contrôle indépendant, choisir le meilleur rectangle englobant une composante de trou ou tout-1 donne déjà 0,7461 sur les F.

Les petits trous ne sont pas les seules erreurs. Exemple actuel : `d6470fd6a771`, PS F, couverture GT 99,58 %, `acc_full=1`, tête 0,4208 contre ancienne règle 0,4979. Il existe donc des faux négatifs watermark importants même quand le trou véritable est minuscule. Les scores seuls ne permettent pas d'en dessiner les pixels exacts.

**Résolution et information sont distinctes.** Avec la GT moyennée par blocs puis remontée, une grille 8×8 atteint 0,7033, une grille 32×32 0,9043. Ce sont des diagnostics de représentation utilisant la GT. La sortie 128×128 de la tête ne recrée pas l'information locale perdue dans des features 8×8 ou des fenêtres de 64–256 px. Mais la seule taille de sortie n'explique pas non plus le score actuel : la qualité du signal local est centrale.

### Ce que demandent 0,70 et 0,80

La référence actuelle donne F = **0,555737**, R = **0,567113**, avec autant d'images dans chaque régime.

| Remplacement oracle, le reste fixé aux scores actuels | Global obtenu |
|---|---:|
| F parfaits | 0,7836 |
| R parfaits | 0,7779 |
| R remplacés par leurs bbox GT | 0,7113 |

Ainsi, avec les R actuels, il faudrait F ≈ **0,8329** pour 0,70 ; avec les F actuels, il faudrait R ≈ **0,8443**. **0,80 exige un progrès des deux régimes.** Ces scénarios utilisent les étiquettes et la GT ; ce ne sont ni des prédictions réalisables ni des estimations de leaderboard.

## 4. Expériences déjà faites : conclusions actualisées

### Les petites fusions sont dépassées par la vraie référence

J'ai lu `task1_F_fusion_cv.py`, `task1_F_morph_cv.py` et leurs JSON dans Downloads. Leurs métriques sont correctes ; les images externes sont mises à l'écart du fit. Le seuil ridge est appris sur l'apprentissage, pas sur le test. Leur choix F connu en fait cependant un diagnostic de sous-ensemble, pas une pipeline test complète. Leurs folds ne sont pas ceux de la référence principale.

La ridge **0,4710** ne reçoit pas toutes les occultations de `seg_ff` et n'est pas comparée à la tête actuelle, F **0,5557**. Son +0,0083 sur une ancienne base ne prouve aucun gain supplémentaire. Morphologie +0,0007 : écartée. Aucun de ces essais n'a été relancé.

### Ré-embarquement, forensique et templates

- Le pilote ré-embarquement de 120 F est déjà négatif : AUC PS/TM ≈ 0,50/0,51, SS 0,67 ; mIoU F à peine différente de tout-1. Pas de nouveau passage complet justifié.
- Nuance : les cartes NCC d'un autre générateur sont **déjà présentes dans le 0,5614**. Dans le même job, l'ensemble `ff+fm` atteint 0,5600 ; ajouter `ncc_m,ncc_d` atteint 0,5614. Cela ne justifie pas de dire que toute corrélation est sans information, mais ne démontre aucun saut de performance.
- TruFor a déjà échoué : cartes proches du hasard, pas de gain utile avec la tête. Ne pas remplacer cet échec par la promesse d'un détecteur générique d'édition.
- A1 montre que le watermark reste lisible hors zone ; il **ne démontre pas** que l'extérieur a été recollé pixel à pixel. Une restauration propre peut retirer le watermark sans produire de trace « image générée ».
- Le template gris a déjà été écarté sur TrustMark et peu utile sur StegaStamp. Le résidu dépend du contenu, du cadre et de l'échelle ; `embed(x',m)-x'` n'est pas nécessairement le résidu initial inconnu.

### Le pilote contrastif ne mesure pas encore la localisation

Les quatre JSON contiennent bien 41 cas évaluables et les top-1 annoncés. Une stratification supplémentaire, sans nouveau calcul modèle, change leur interprétation :

| Méthode, cas presque rectangulaires (`bbox_fill > 0,98`) | n | GT premier avec acc | avec normalized |
|---|---:|---:|---:|
| PixelSeal | 4 | 4 | 4 |
| TrustMark | 8 | 8 | 8 |
| StegaStamp | 11 | 10 | 10 |

Chez TrustMark, le progrès total de 8/13 à 10/13 vient de **deux cas irréguliers sur cinq**. C'est intéressant, mais ce n'est pas une preuve d'un meilleur classement des véritables rectangles. Les négatifs sont peu nombreux, de même taille que la bbox GT et contiennent au plus 1 % de pixels watermarkés ; la GT fournit le candidat positif. Le pilote prend les 15 premiers R par méthode, écarte les côtés <16 px et certains grands cadres sans négatifs admissibles : il ne représente donc pas tous les R difficiles. Cela ne teste ni la couverture de la recherche, ni les perturbations de bords, ni le maximum de milliers de faux candidats.

### La segmentation directe RGB est déjà expérimentée

Le README local de `c1/pixseg` dit encore « aucune mesure réelle ». Les sorties JURECA le contredisent. Les modèles PS/TM spécialisés consultés n'apprennent pas un localisateur fiable ; les pertes proches de 1,41 dans plusieurs runs sont compatibles avec un effondrement vers une solution presque constante. Le code ne reçoit pas le message connu. Les retournements après incrustation et les crops déplacent aussi des motifs sensibles au cadre. Un test factice de quelques bits faibles ne valide pas le transfert aux vrais watermarks.

Sur StegaStamp, un signal spatial existe. Mais les cinq folds de la **configuration fixe** `ss6k_a/b/c` donnent seulement :

```text
StegaStamp : 0,53323 ; F 0,52331 ; R 0,54315
Remplacement de SS dans HEAD graine externe 0 : gain global +0,00501
```

Les fichiers `cvsel` donnent 0,59265 sur SS et +0,02481 global, **avec un assemblage différent**, notamment une autre graine sur le fold 1. Ce résultat ne devient une validation honnête que si ce choix repose sur un critère fixé sans consulter les scores externes. Aucun tel critère indépendant n'est documenté dans les artefacts lus. Ne pas présenter l'assemblage des meilleurs folds/graines observés comme un nouveau score OOF validé.

## 5. Validité, fuites et limites

1. **La métrique n'est pas la cause du plafond.** Les huit tests CPU de `c1/test_metric.py` passent. Tout-0 = 0,2517897 et tout-1 = 0,2482103 sur les 510 masques. La reconstruction exacte de la branche R recoupe la métrique utilisée pour la référence. Les masques train ne sont ni vides ni pleins.
2. **La classe minoritaire pèse lourd.** Tout-1 sur un F donne environ 0,45 malgré 90 % de pixels corrects. Il faut rapporter IoU de chaque classe, moyenne par image, aire et performance sur les petits supports. Une AUC poolée de pixels ou la bit accuracy ne remplace pas le mIoU.
3. **`head_eval.py` utilise bien une calibration interne.** À l'inverse, `c1/seg/combine.py` ne forme pas une vraie CV imbriquée pour les prédictions réelles de `kfold_f.py` : seulement **58/255 F ont le même fold** entre le producteur OOF et la calibration externe. Les modèles produisant les données de calibration peuvent avoir appris sur les images de l'externe. Le score F à seuil fixé reste OOF ; l'étiquette « global strictement imbriqué » de cet assemblage est excessive. Le cas synthétique seul ne présente pas cette fuite de labels train par les poids.
4. **Une séparation par noms ne suffit pas à prouver l'absence de fuite de contenu.** Noms train/test disjoints ; 510 masques uniques et 170 messages uniques par méthode dans le train. J'ai aussi comparé les 1 530 images sur CPU : aucun couple candidat avec pHash à distance ≤ 6 et MSE RGB 32×32 < 0,001. C'est un contrôle borné, pas une preuve d'absence de toute image source commune ou de recadrages similaires.
5. **Le synthétique utilise des images test d'une autre méthode déjà watermarkées.** Il s'agit d'un apprentissage transductif sur les images test, sans leurs masques. Ce n'est pas une fuite de labels vers les F train, mais ce n'est pas un pool naturel propre : superposition d'un autre watermark, éditions préexistantes et biais de contenu. Sa conformité au règlement doit être établie si elle ne l'est pas déjà. Ne pas introduire de nouveaux usages du test sans cette vérification.
6. **Les 1 440 anciens composites F n'ont aucun `src2`.** La première source vérifiable n'intersecte pas le train ; la seconde ne peut pas être auditée directement. Le code actuel trace les deux sources, mais cette correction n'ajoute pas rétrospectivement la provenance aux anciens fichiers. Couverture synthétique F = 0,8828, réel F = 0,9058 ; la différence de formes et d'opérations importe davantage qu'une moyenne de couverture proche.
7. **Normalisations à reprendre dans le prochain pipeline.** `kfold_f.py` sans synthétique initialise sa normalisation sur tous les F ; fuite de distribution non supervisée. L'option cartes de `pixseg` calcule aussi ses statistiques avant le split. `head_eval.py`, lui, calcule ses statistiques sur son apprentissage. Les nouveaux folds doivent partager un manifeste immuable et toutes leurs statistiques de fit.
8. **Sélection répétée des expériences.** Une CV correcte pour un modèle ne rend pas indépendant le choix final parmi de nombreux modèles évalués sur les mêmes 510 images. Garder les comparaisons appariées, les résultats par groupe et par fold ; ne pas compter les trois partitions ou les pixels comme autant d'images indépendantes.
9. **Prétraitement officiel vérifié par lecture.** PS 256, TM Q 245 sans ECC, SS 400 ; redimensionnements anticrénelés, signe des logits > 0. Les wrappers PS/TM `logits()` et `features()` sont décorés `no_grad`. Pour une branche entraînable ou des gradients, appeler les modules internes via un adaptateur dédié ; entourer seulement l'appel public d'un `enable_grad()` ne suffit pas.

## 6. Trois pistes classées, expériences éliminatoires

Les gains ci-dessous sont des **plages de travail hypothétiques**, non des prédictions. Les durées de nouveaux développements sont estimées. Aucun budget GPU n'est autorisé par ce rapport.

| Rang | Piste | Gain global recherché | Développement / GPU estimés | Risque |
|---|---|---|---|---|
| **1** | **Localisateur dense initialisé avec les décodeurs officiels ; features fines et accord au message** | +0,03 à +0,10 ; davantage seulement si le signal se transfère très bien | 2–3 h code ; pilote 15–30 min A100, validation 1–3 h-GPU | Moyen à élevé : signal fin possiblement insuffisant et domaine synthétique. |
| 2 | Localisation R : propositions du détecteur TrustMark Q + classement des crops au message, contrastif calibré à l'échelle de la recherche | +0,01 à +0,03 pour TM ; plafond de scénario bbox TM ≈ +0,037 global | 1–2 h ; pilote 10–40 min A100, puis coût des boîtes | Moyen : détecteur entraîné dans un autre domaine, régions non rectangulaires. |
| 3 | Raffinement par contours/objets, guidé par la preuve watermark et avec possibilité de conserver le masque initial | +0,01 à +0,04 si les frontières visuelles correspondent aux trous | 1–2 h ; pilote GrabCut CPU, SAM seulement si déjà disponible/autorisé | Élevé : une frontière d'objet n'est pas une frontière de présence du watermark. |

**Piste 1, rejet minimal.** Sur PS/TM F, faire un pilote d'une seule recette, avec deux ensembles de sources indépendants et un test réel tenu à l'écart du poids et du seuil. Exiger un apprentissage réel du signal sur un petit contrôle synthétique officiel, puis au moins **+0,08 de mIoU F moyen** contre la référence sur le sous-ensemble réel de confirmation, accompagné d'un progrès de l'IoU des trous. Si le réseau n'apprend même pas le contrôle officiel, changer une seule fois de mode d'entraînement — tête seule puis dégel du dernier stage — et arrêter si le second essai échoue. Si l'apprentissage fonctionne mais pas le transfert réel, ne pas produire davantage du même synthétique.

**Piste 2, rejet minimal.** Séparer au préalable 30 R TrustMark de calibration et 30 R de confirmation, sans choisir les images pour leur bonne décodabilité. Construire les propositions sans GT : top-k Q, anciennes propositions, voisins de bords. Comparer boîte finale et mIoU, pas le classement de bbox GT injectées. Le score `normalized = sum(s*l)/(sum(abs(l))+eps)` doit réellement changer le classement des candidats. Calibrer aussi le maximum de la recherche avec des messages aléatoires, sur les mêmes propositions, pour contrôler les faux maxima. Abandon si moins de **+0,06 sur TM R** tenu à l'écart, soit environ +0,01 global, ou si le rappel géométrique des propositions est le vrai échec. N'augmenter alors pas simplement le nombre de boîtes.

**Piste 3, rejet minimal.** À partir de probabilités OOF existantes, fixer les graines et la règle de conservation sur 30 F de développement ; tester sur 30 autres F. Pas de bbox GT, de clic GT ou de choix de proposition par sa vraie IoU. Les pixels très confiants servent de graines ; le raffinement peut s'abstenir. Abandon si moins de **+0,04 F** ou si les petits trous/proches du bord se dégradent. Le pilote SLIC oracle déjà mentionné et l'échec de TruFor rendent cette piste moins prioritaire ; ne pas refaire une grille de morphologie.

### Pourquoi pas une nouvelle saliency en priorité ?

Pour PS/TM, à la dernière couche spatiale suivie d'un average pooling et d'une tête linéaire, le score signé du message s'écrit :

```text
S(x,m) = mean_k [(2m_k-1) * logit_k(x)]
       = constante + mean_position [w_message · feature(position)]
```

Le gradient par rapport à cette dernière feature est constant dans l'espace. Un Grad-CAM sur cette couche est donc proportionnel à la carte de marge signée déjà utilisée, à la rectification et aux constantes près. Il ne crée pas une nouvelle mesure. Les gradients d'entrée ou les perturbations optimisées sont différents, mais localisent une sensibilité ou un sous-ensemble suffisant au décodage, pas nécessairement **tout** le support watermarké. Leur L1/TV pourrait préférer une petite région diagnostique au vrai masque F.

## 7. Mandat opérationnel unique pour Opus

### Étape 0 — geler une référence cohérente, 10–20 min CPU

- Conserver les hashes du fichier public et du NPZ 0,5614 ; les nommer séparément dans les prochains rapports.
- Dans `c1/segf/head_eval.py` du worktree `c1-probe-1b51d2`, ajouter la sauvegarde des probabilités externes, `fold`, config complète, graines, versions et hashes. Ne pas écraser les anciens fichiers.
- Dans `c1/segf/head.sbatch`, transmettre `--test-hf 'output/segf_hf_test_*.npz'` pour les configs qui le demandent. Ajouter une validation de présence des clés et d'unicité des noms **avant** l'entraînement. Ne pas relancer le job du champion pour cette seule correction sans autorisation GPU.
- Définir une seule partition par nom : `real_eval.folds(method|full_watermark, 5, 0)`. Les comparaisons initiales doivent utiliser la ligne `head[0]` de l'instantané, pas la moyenne de trois partitions comme s'il s'agissait des mêmes poids/folds.

### Étape 1 — nouvelle entrée fine, pas un nouveau U-Net RGB de zéro

Réutiliser le travail de `eaic-wt-c1-pixseg` dans une branche d'implémentation attribuée par Opus, après vérification des travaux concurrents. Ne pas copier à l'aveugle le dossier JURECA mutable.

| Fichier cible | Modification demandée |
|---|---|
| `c1/pixseg/data.py` | Charger et conserver `message`, `src`, `src2`, split/source et paramètres d'incrustation. Manifeste de folds par nom ; filtrage des deux sources pour chaque externe. |
| `c1/pixseg/decoder_features.py` — nouveau | Adaptateurs des modules PS/TM officiels ; prétraitement identique au kit ; extraction des stages fins et grossiers, avec gradients possibles. Aucun changement global des wrappers officiels. |
| `c1/pixseg/model.py` | Ajouter un mode de backbone officiel et un petit décodeur de pyramide ; branche fine et preuve au message explicite. Garder l'ancien mode comme témoin, pas comme candidat à réoptimiser. |
| `c1/pixseg/train.py` | Mode F PS/TM pour le premier pilote ; poids officiels gelés au premier essai ; normalisation par apprentissage ; validation interne ; sortie OOF complète et manifeste. |
| `c1/pixseg/combine_eval.py` | Référence 0,5614 graine externe 0, mêmes noms/folds ; branche R conservée pour cette première ablation ; scores par image et bootstrap apparié. |
| `c1/pixseg/test_pixseg.py` | Tests de contrats ci-dessous, sans téléchargement ni GPU. |

**Architecture de départ prescrite :**

1. Entrée officielle 256 PS / 245 TM pour conserver l'échelle attendue ; pas de passage arbitraire du décodeur à 512 au premier essai.
2. PS : sorties des stages ConvNeXt de strides 4, 8, 16, 32. TM : `layer1..layer4`, résolutions observées à l'exécution. Garder le mode `eval` des normalisations du backbone gelé.
3. Projeter chaque niveau vers 32 canaux, fusion descendante et sorties au niveau le plus fin, puis masque 128. Ajouter RGB/passe-haut uniquement dans une petite branche de finition, et les cartes d'accord au message existantes comme contexte. Les skips fins, et non une interpolation de la dernière carte, sont l'expérience.
4. Perte initiale : BCE + mIoU doux des deux classes, déjà validée dans le dépôt. Échantillonner les images et présenter suffisamment de petits/grands trous ; ne pas remplacer la métrique par une accuracy de pixels.
5. Premier essai : tête seule. Si le contrôle officiel apprend insuffisamment, second et dernier essai : dégel du dernier stage, LR backbone dix fois plus faible que la tête et contrainte de préserver les logits globaux du décodeur initial. Pas de sweep supplémentaire.

**Synthétique :** utiliser les embedders du kit, jamais celui de WAM. Produire des paires propre/watermarké avec le même contenu et des supports variés ; inclure les objets et les trous au bord. Éviter les transformations RGB après incrustation dans le pilote : un flip de carte scalaire est valide, un flip du motif physique n'est pas automatiquement équivalent. Appliquer les éventuels flips aux sources **avant** incrustation. Sources de validation et leurs deux remplissages exclus de tout synthétique d'entraînement ; tracer les messages et paramètres. Réutiliser des composites anciens sans messages pour un apprentissage conditionné serait incorrect.

### Étape 2 — expérience éliminatoire et validation

**Avant GPU :** tests CPU des noms/folds, source/source2, canaux, forme 128, finitude, signe des logits et gradients possibles. Avec de petits tenseurs et poids factices, vérifier que le backbone gelé ne change pas, que la tête reçoit un gradient et que les masques sont alignés. Ces tests contrôlent le code, pas la qualité du watermark réel.

**Après autorisation GPU et coordination de l'unique créneau d'équipe :**

- Réserver les folds 0 et 1 pour confirmer cette nouvelle direction. Développer uniquement sur les folds 2, 3 et 4, y compris les poids et les seuils du pilote ; ne pas regarder les folds de confirmation pour choisir le nombre d'époques.
- Contrôle court : surapprendre 8 composites officiels et vérifier le transfert à d'autres sources synthétiques. Un exemple factice bit-faible n'est pas un substitut.
- Entraîner la recette fixée, 20 époques de tête au départ, sur PS/TM F de développement. Choisir le seuil sur une séparation interne au développement. Évaluer une seule fois les folds 0/1 et comparer par image à `head[0]`, à régime connu pour le diagnostic puis avec le routage courant pour le score global.
- Les folds de confirmation ont déjà participé aux recherches historiques ; ils ne constituent pas un nouveau test vierge. Cette séparation les tient néanmoins à l'écart de tout développement du **nouveau** localisateur. Une recherche imbriquée complète reste nécessaire pour un chiffre final défendable.

**Poursuivre** si le mIoU F moyen PS/TM progresse d'au moins +0,08 sur confirmation, avec gain dans les deux méthodes, amélioration de l'IoU des trous et sans effondrement sur les supports <1 %. Cela correspond à un ordre de grandeur de +0,027 global si les deux groupes gagnent pareil et que le reste reste identique ; calculer le vrai score routé.

**Abandonner** si les deux essais d'apprentissage échouent, si le gain n'existe que sur synthétique, si la confirmation est inférieure à +0,03 F, ou si le coût projeté dépasse le créneau disponible. Entre +0,03 et +0,08 F, conserver le résultat comme diagnostic, sans prolonger une série de réglages de seuil.

**Validation finale si le pilote passe :** reprendre les cinq folds externes de la référence avec calibration interne par fold, sans concaténer des OOF d'un autre split. Comparer à la référence correspondante sur les 510 images, donner les six groupes et les deux IoU, vérifier le gain sur les deux partitions de confirmation et les familles de tailles. Exiger **au moins +0,03 global** contre la configuration de référence à partitions correspondantes, intervalle apparié descriptif au-dessus de zéro, et absence de régression >0,01 sur un groupe non ciblé. Une amélioration robuste de +0,01 à +0,03 peut rester un candidat de secours ; elle n'étaye pas l'objectif de rupture.

Sorties attendues : `names`, `probs_oof`, `fold`, `train_names_hash`, `source_groups`, `config`, scores par image et par groupe, plus poids/configs. Après validation, un refit et un candidat test portent un nouveau nom, avec correspondance exacte à la recette évaluée. Aucun fichier n'est soumis automatiquement.

### Sources techniques qui motivent ce choix

Le [code et la carte PixelSeal officiels](https://github.com/facebookresearch/videoseal/blob/main/videoseal/cards/pixelseal.yaml) montrent un extracteur ConvNeXt et une tête globale. Le [papier Pixel Seal](https://arxiv.org/html/2512.16874v1) documente l'adaptation de résolution et l'atténuation perceptuelle ; ce n'est pas un détecteur de support au pixel déjà entraîné. L'adaptation supervisée proposée est notre hypothèse.

[WAM](https://arxiv.org/html/2411.07231v1) fournit une recette de présence et de message supervisés spatialement, mais ses poids détectent son propre watermark. Il inspire l'objectif d'apprentissage ; il ne constitue pas un localisateur prêt à l'emploi des trois méthodes du concours.

L'[implémentation Adobe](https://github.com/adobe/trustmark/blob/main/python/trustmark/trustmark.py) expose un détecteur de boîtes optionnel, les scores via son module interne et des poids Q/P. Le kit lu utilise bien Q. C'est la justification concrète de la piste 2 ; aucune mesure locale du détecteur Q n'a été trouvée, et ses poids bbox ne sont pas dans le répertoire `models` inspecté.

[What Breaks Local Watermarks?](https://arxiv.org/html/2609.16832v1) distingue récupération du payload et localisation, ainsi que les effets de transformations géométriques et génératives. Son adaptation PixelSeal n'est pas la preuve que les R du concours ont été fabriqués de la même manière : nos diagnostics de recadrage restent prioritaires pour le domaine présent.

## 8. Preuves reproductibles et travail restant

Commandes effectivement exécutées : état Git/log/worktrees ; `relay-check.py` ; lecture des sources ; tests de métrique ; analyses NumPy/SciPy locales ; SSH en lecture seule pour `ls`, `sha256sum`, `cat`, `sacct` et réduction CPU de résultats existants. Aucun `sbatch`, `srun`, chargement de modèle ou appel de soumission exécuté par moi.

```bash
.venv/bin/python c1/test_metric.py
# 8 tests passés

.venv/bin/python docs/task1-audit-20261121/audit_cpu.py
# Résultats complets : audit_results.json ; aucune inférence.

.venv/bin/python docs/task1-audit-20261121/analyze_champion.py
# HEAD 0.5614248628236677 ; BASE 0.5097164288446294
# Reconstruction branche R : max erreur 0.0 pour les 3 partitions.
```

Les analyses n'ont pas réentraîné les modèles et ne prétendent donc pas reproduire leur optimisation. Le rapport ne déclare aucun nouveau gain algorithmique. Les scores de scripts anciens non relancés sont identifiés comme tels.

Actions restantes pour Opus/Selyan : préserver séparément les deux références, corriger l'export NCC, fixer la partition et le manifeste, implémenter le pilote dense officiel, puis décider du budget GPU avant exécution. Pas de nouvelle recherche de seuils, pas de répétition TruFor/ré-embarquement/template gris, et pas de choix du meilleur modèle après lecture de chaque fold externe.

État final revérifié : HEAD et branche identiques au départ ; modifications suivies préexistantes inchangées dans le relevé Git. Un dossier non suivi `docs/task3-audit-20261121/` est apparu pendant le travail et n’a pas été touché par cet audit. Les deux analyses CPU ont été réexécutées depuis leur emplacement livré avec les mêmes résultats.

## Relais du 22 novembre, 03:08 Paris

Rapport du relais demandé par Selyan : [docs/relais-codex-20261122/RAPPORT-CODEX.md](docs/relais-codex-20261122/RAPPORT-CODEX.md).

- Champion T1 : **0,6739** fourni par Selyan ; SHA de `c1_ensE_ssall3.pt` vérifié (`f00fdc03…`). Les quatre jobs cités sont COMPLETED, code 0 ; file des trois comptes SLZ vide à 03:07.
- IoU2 mesuré sur les cinq folds : **+0,003381**, IC95 [+0,000347 ; +0,006574], donc DIAGNOSTIC selon la règle fixée. Ce comparatif porte sur l'ancienne famille 4k, pas sur le champion SSall3.
- Candidat SS 6k construit et vérifié **localement**, non soumis : `/private/tmp/eaic-relais-20261122/c1_ensF_ssall6k2.pt`, SHA `39a3fb2d…`. Gain dilué sur deux folds : +0,007623 ; validation partielle, différence de nombre de graines entre ensembles test. Les 680 autres masques sont identiques au champion.
- T2 : 989 succès / 990 tentatives refs ; **MCC 0 sur chaque moitié**, NO-GO. Zéro égalité stricte famille A sur les 10 000 prompts du dernier job. Aucun lot ep ni aucune API appelée par ce relais.
- Lanceur de validation 3k/6k folds 2–4 et troisième graine test préparé, syntaxe Bash vérifiée, **non lancé**. Lancement manuel par Selyan conformément à la consigne de ne pas écrire sur un système externe. Aucun Git d'écriture, aucune soumission, aucune modification distante.

**Mise à jour 03:12 après GO explicite de Selyan (« go lance ça alors ? »)** : lanceur transféré et SHA vérifié ; file des trois comptes d'équipe vide immédiatement avant `sbatch`. **Job 15904703 (`j12`) lancé à 03:11:09, RUNNING à 03:11:57, quatre GPU sur jrc0256, dc-gpu-devel**, limite 40 minutes. Les quatre journaux de configurations sont présents. Aucune soumission au serveur de notation ni collecte T2. Reçu : `docs/relais-codex-20261122/launch_15904703.json`. Entraînement et mesure finale encore en cours.

**Analyse live demandée à 03:17 :** SLZ **27e/38, général 0,319108**, T1 0,673907 / T2 0,20 / T3 0,094115. Depuis le snapshot 00:26, général +0,019416 : gains bruts +0,090804 au barème ancien, puis effet du changement de normalisation −0,071388. Si le pilote 6k reproduisait son gain au public, +0,008145 au général seulement. Une T1 première, T2/T3 inchangées, plafonnerait à 0,457332 ; top 6 actuel à 0,761185. Scénarios, preuve live et arbitrages à cadrer : [docs/relais-codex-20261122/CLASSEMENT-LIVE.md](docs/relais-codex-20261122/CLASSEMENT-LIVE.md). Aucun nouveau job ni API notée lancés pour cette analyse.
