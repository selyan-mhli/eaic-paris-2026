# Audit approfondi — ce que les fichiers permettent réellement de conclure

22 novembre 2026, relevés complémentaires à partir de 03:15 Paris. Complément au rapport historique : sources exécutées, filiation des masques, données synthétiques, robustesse du gain T1, fonctionnement du test T2 et mécanisme exact du champion T3. Aucun nouveau modèle entraîné, aucune requête notée ni soumission. Analyses CPU sur sorties existantes ; pour T3, rejeu des seules têtes sur des features déjà sauvegardées, sans passage dans le ViT.

## Les constats nouveaux qui changent la reprise

| Constat établi | Conséquence concrète |
|---|---|
| PixelSeal et TrustMark sont identiques au bit près dans les huit assemblages T1 examinés, de `c1_pixss_fit0` à E. | Tout le gain public de cette série, 0,5845 → 0,6739, vient des changements StegaStamp. Il ne faut pas créditer les trois méthodes de ce progrès. |
| Le 6k gagne sur 40/68 SS et perd sur 25 ; sans les cinq meilleures améliorations, le gain global extrapolé tombe à +0,00189. | Le résultat est encourageant mais concentré. La confirmation cinq folds est déterminante ; une moyenne positive sur deux folds ne suffit pas. |
| Les 12 000 synthétiques tirent leurs deux sources du test, sans aucun nom train. | La recette est transductive : elle utilise les images test sans leurs masques. Il faut documenter cette entrée réelle, et ne pas la présenter comme une recette entraînée uniquement sur les 510 images train. |
| Le filtre T2 dit « strict » a retiré le critère de quasi-égalité ; 4/7 sondes étiquetées tie ont un d0 de valeur absolue > 0,5. | Le test effectivement exécuté ne correspond qu'en partie à la motivation « égalité insensible à la température ». Le NO-GO mesuré reste valable ; il ne clôt pas toute la théorie possible. |
| Même un seuil G choisi après lecture ne donne pas un MCC strictement positif sur les deux moitiés avec cette règle de regroupement. | Baisser simplement le seuil après l'échec ne répare pas cette batterie. Ce diagnostic n'est pas une nouvelle méthode sélectionnée. |
| Push s175 modifie exactement deux tenseurs et 45 lignes de tête. Sur les autres classes, accord ≈99,88 % mais KL≈0,4506 dans le cache local. | Le mécanisme est ciblé dans les paramètres ; la normalisation softmax propage néanmoins des changements de distribution aux autres classes. Accord et retain serveur sont deux mesures différentes. |
| Les sources T1 du cluster sont identiques aux copies locales, mais ne sont pas suivies par Git ; `merge_ss.py` manque même dans ce worktree local. | Un SHA de branche ne suffit pas pour reproduire le champion. Les sources exécutées ont été archivées dans ce dossier. |

Les sorties chiffrées de ces contrôles sont dans `profondeur/`. Les copies des sources ne remplacent pas l'historique au moment exact de chaque job : ce sont les fichiers présents sur le cluster au relevé.

## 1. Filiation du champion T1 : ce qui a changé, réellement

J'ai chargé sur CPU huit fichiers de masques, réaligné les 1 020 noms et comparé chaque méthode :

`c1_pixss_fit0`, `c1_pre_ss_cnt2`, `c1_pre4k_ens5`, `c1_ensA9`, `c1_ensB11`, `c1_ensC10`, `c1_ensD_psyn4`, `c1_ensE_ssall3`.

**Tous ont exactement les mêmes 340 masques PixelSeal et les mêmes 340 TrustMark.** Tous ont aussi l'ordre de `release/test/metadata.json`. Les empreintes des masques par méthode sont conservées, distinctes des hashes du fichier torch, dont la sérialisation peut varier.

| Comparaison vers E | Cellules SS changées | Fraction des cellules SS | Images SS changées | Cellules PS/TM changées |
|---|---:|---:|---:|---:|
| `c1_pixss_fit0` → E | 639 405 | 11,4783 % | 336/340 | **0** |
| `c1_pre_ss_cnt2` → E | 326 996 | 5,8701 % | 334/340 | **0** |
| `c1_pre4k_ens5` → E | 198 572 | 3,5647 % | 334/340 | **0** |
| A9 → E | 193 018 | 3,4650 % | 332/340 | **0** |
| B11 → E | 184 619 | 3,3142 % | 329/340 | **0** |
| C10 → E | 170 999 | 3,0697 % | 329/340 | **0** |
| D → E | 196 961 | 3,5357 % | 333/340 | **0** |

Conséquence causale solide : pour une métrique déterministe évaluée sur le même sous-ensemble public, le progrès entre les deux soumissions ne peut venir que des prédictions SS modifiées. **Il serait toutefois faux de multiplier exactement le gain global public par trois** pour annoncer un gain SS public : le nombre exact d'images publiques par méthode n'est pas connu.

A9 et B11 sont distincts (27 483 cellules SS diffèrent, sur 291 images). Leur comparaison ne permet donc pas de transformer le score 0,6391 en attribution certaine à l'un ou l'autre. Le reçu qui relie ce score à un fichier manque toujours.

Preuve : `profondeur/t1-filiation-masques.json`. Cette filiation écarte l'idée d'une amélioration récente déjà livrée à PixelSeal/TrustMark ; elle ne démontre pas que leurs nouvelles pistes locales seraient inutiles.

## 2. Ce que signifie vraiment la validation T1

Les sources exactes `train.py`, `data.py`, `model.py`, `export.py`, `combine_eval.py` et les configurations pertinentes ont été copiées depuis `work_pixseg_v2/`.

### Séparation des exemples réels

Pour le fold externe k, avec `--inner-val 1` :

- 102 images externes sont évaluées ;
- 102 du fold `(k+1)%5` servent au seuil ;
- 306 servent à l'apprentissage ;
- les composites ayant l'une des deux sources dans les deux folds retirés sont écartés par `src` et `src2`.

Les poids et le seuil n'utilisent donc pas les masques du fold externe dans ce chemin de code. Le seuil fixe 0,5 utilisé par la comparaison 3k/6k évite aussi une sélection de seuil sur ces images. **Cela ne rend pas les expériences indépendantes des choix de configurations précédemment guidés par leurs scores.**

Le fit-all entraîne ensuite sur **510 images réelles**, au lieu des 306 des entraînements de CV. Les prédictions test E sont un ensemble de trois graines, tandis que la CV actuelle est une graine par fold. Même après cinq folds, la comparaison restera donc **3k contre 6k à une graine CV**, et non une mesure exacte « ensemble E trois graines contre ensemble F trois graines ».

Les noms de graines 50/51/52 et 60/61 sont des arguments du script. Les graines réellement passées à l'entraînement fit-all sont `100*seed+99`, soit **5099/5199/5299** et **6099/6199**. Les folds CV par défaut utilisent 0,1,2,3,4. Cette distinction évite une fausse commande de reproduction.

### Nature des entrées du modèle

Le réseau pré-entraîné n'est pas un segmentateur « RGB seul » au sens littéral : son entrée comporte **10 canaux**, RGB normalisé (3), méthode one-hot (3) et résidus passe-haut Y/R/G/B (4). Il ne consomme pas le message connu dans cette recette, ni les cartes `extra_maps` dans les configurations 3k/6k examinées. L'encodeur est un ConvNeXt Tiny timm pré-entraîné, avec décodeur spatial.

Dire « le pré-entraînement a débloqué SS » est compatible avec les comparaisons. Dire « une pure détection des retouches visuelles explique à elle seule le gain » dépasserait la preuve : le modèle voit aussi des résidus de signal et la méthode.

### Précision des probabilités sauvegardées

Les probabilités OOF sont calculées en float32 puis sauvegardées en float16 ; le champ `miou` est calculé avant cet arrondi. J'ai recalculé la métrique depuis les probabilités réellement stockées. Pour le fold 0 du 6k, écart maximal par image **0,0007133**, moyen **−0,0000120** au seuil appris ; fold 1, écart nul.

C'est trop petit pour expliquer le gain global de +0,0076, mais suffisant pour empêcher une égalité au dernier chiffre entre un log et un rejeu. Il faut comparer les modèles avec le même chemin numérique. Les huit tests existants de métrique passent à nouveau, y compris les cas de classes vides et l'égalité à la référence copiée du kit.

## 3. Origine du synthétique : vérification des 24 tranches

Le générateur utilise par défaut `--pool test`, en excluant comme source les images de la méthode qu'il va poser. Les masques synthétiques sont créés par la géométrie du composite ; aucun masque GT test n'est lu dans ce chemin.

Lecture effective des champs `src`, `src2`, `method`, `full`, `mask128` des **24 fichiers** :

| Méthode posée | Exemples | Sources principales distinctes | Sources secondaires distinctes | Sources venant du train |
|---|---:|---:|---:|---:|
| PixelSeal | 3 000 | 672 | 673 | 0 |
| TrustMark | 3 000 | 668 | 673 | 0 |
| StegaStamp | 6 000 | 680 | 680 | 0 |

Aucune source, principale ou secondaire, n'est de la méthode posée. Aucun nom n'est inconnu des métadonnées test. Les deux moitiés F/R sont construites par le générateur, et la couverture synthétique moyenne est proche de 0,481 dans les logs.

**Ce protocole apporte une information transductive : les images test sont vues sous d'autres watermarks dans le mélange partagé d'entraînement.** Les OOF de train ne mesurent pas exactement ce contexte d'exposition des images test. Cela doit figurer dans la description de la recette et dans les commandes de reproduction. Cet audit ne déduit ni un score privé ni une violation du règlement de ce seul constat.

Vérification supplémentaire : SHA-256 des **1 530 PNG officiels** (510 train, 1 020 test), **aucun groupe de fichiers identiques** ; aucun nom commun train/test. Cela exclut les doublons stricts examinés, pas des versions éditées d'une même photo source. L'exclusion par nom dans `synth_idx` ne détecterait pas ce dernier cas.

Preuves : `profondeur/jureca-provenance.json`, `profondeur/doublons-png.json`. Les images n'ont pas été recopiées dans le rapport.

## 4. Le gain 6k : reproductible sur les deux folds, mais concentré

Recalcul indépendant depuis les masques GT et les probabilités stockées : **68 StegaStamp**, 34 par fold, seuil fixé à 0,5. Les configurations 3k/6k ne diffèrent que par les pas, les folds et les noms de sorties ; aucune autre différence de paramètres enregistrés n'a été trouvée.

| Indicateur | Résultat |
|---|---:|
| SS 3k | 0,83373850 |
| SS 6k | 0,85660751 |
| Gain SS moyen | +0,02286901 |
| Contribution globale extrapolée SS seul | +0,00762300 |
| Images améliorées / dégradées / égales | **40 / 25 / 3** |
| Gain SS médian | +0,00816742 |
| Pire variation par image | −0,14940609 |
| Meilleure variation par image | +0,33275230 |
| Contribution globale fold 0 / fold 1 | +0,00526596 / +0,00998005 |

Diagnostic de concentration, **pas nouvelle règle de sélection** : en retirant les cinq meilleures améliorations et en recalculant la moyenne sur les 63 autres SS, la contribution globale devient **+0,00188609**. Sans les cinq pires, elle devient +0,01075343. Le gain n'est donc pas uniformément réparti entre images.

Sensibilité descriptive à un seuil commun aux deux recettes : de 0,4 à 0,6, contribution globale entre **+0,00753 et +0,00769**. La différence moyenne observée ne vient pas d'un heureux choix exact à 0,5. Ces cinq seuils ont été lus pour l'audit ; aucun nouveau seuil ni candidat n'a été choisi.

L'IC image-par-image déjà rapporté ne corrige ni les essais successifs, ni la dépendance induite par les entraînements, ni l'hétérogénéité de deux seuls folds. La confirmation cinq folds augmente la couverture ; elle ne fournit pas à elle seule une répétition de l'expérience avec d'autres graines ou une confirmation de l'ensemble test.

Preuve : `profondeur/diagnostic-t1.json`. Le job **15904703** existant complète les trois folds manquants et la graine test 62. Au relevé 03:22:53, le fold 4 6k et le refit 62 avaient écrit leurs sorties ; les autres tâches poursuivaient les folds restants. Aucun second job lancé par cet audit.

## 5. Pourquoi la batterie T2 n'a pas testé exactement sa prémisse

### Le critère d'égalité a disparu dans le filtre final

Dans `ties_search.py`, la sélection initiale exige une masse suffisante et **|d0| ≤ 0,5**, où d0 est le log-odds des deux chaînes les plus probables. Dans `strict_filter.py`, le filtre parcourt ensuite **toutes les lignes** et exige masse, stabilité de contexte et décalage fp8, **sans exiger |d0| ≤ 0,5**. Il remplace `ties_robust` par ce résultat. Le pick puise alors les sondes dans cette liste.

Les sept sondes finales étiquetées tie ont les d0 suivants :

`2,125 ; 5,3125 ; 1,625 ; 0,5 ; 2,625 ; 0,375 ; 0,125`.

**Trois seulement satisfont le critère initial d'égalité.** Pour la sonde blue/red à T=0,7, la probabilité locale de la catégorie majoritaire est proche de **99,8 %**. Un changement de température peut déplacer un log-odds de 5,31 ; le raisonnement « une égalité reste une égalité sous température » ne s'y applique pas.

Ce constat n'invalide pas les réponses collectées ni le MCC calculé. Il explique pourquoi appeler l'ensemble « sept égalités robustes » est trompeur. Une future expérience corrigée serait une **nouvelle** expérience à autoriser et à valider ; cet audit n'a changé ni les fichiers du banc ni le verdict de la collecte effectuée.

### La règle ne sépare pas les variantes, même en diagnostic de seuil

Recalcul du G exact, avant arrondi du JSON : tous les endpoints restent dans BASE au seuil figé ≈46,9. Le plus grand G global est celui du **watermark de décodage**, qui doit pourtant rester dans la base. Le G fp8 est plus faible que celui du plain dans les données réunies.

J'ai exploré, uniquement hors ligne, toutes les coupures possibles de ce scalaire sur les trois jeux (ensemble, pairs, impairs). **Aucun seuil commun de la règle « G bas=base, G haut=isolé » n'obtient un MCC signé strictement positif sur les deux moitiés.** Le meilleur minimum vaut zéro, atteint notamment par une partition constante. Baisser le seuil de 46,9 après avoir vu l'échec ne suffit donc pas.

La statistique utilise 3 catégories par sonde et environ cinq réponses par cellule expérimentale dans chaque moitié. Pour ref-01 dans une moitié, **33/33 effectifs de catégories attendus sont inférieurs à cinq**. Le seuil χ² asymptotique à 21 ddl n'a pas été calibré empiriquement dans ce régime clairsemé. Ce contrôle de taille n'est pas une preuve que le seuil est faux, mais il interdit de lui attribuer une garantie de taux d'erreur non mesurée.

Autres limites déjà présentes dans le code : fp8 simulé selon quatre schémas, test strict accepté si **au moins un** de ces schémas décale assez les logits, top-50 local pour reconstruire les probabilités des catégories, prompt système caché inconnu. La bonne identification de la famille ne certifie pas ces détails de déploiement.

Preuve : `profondeur/diagnostic-t2.json`. Verdict maintenu : **NO-GO de cette batterie**, pas impossibilité générale d'une méthode T2.

## 6. T3 : preuve dans les poids, puis contrôle sur le cache

### Les paramètres effectivement modifiés

Lecture CPU des state_dict origine, push de base, s125/s150/s175/s200 : **176 clés**, dont seules **`head.weight` et `head.bias`** diffèrent ; exactement **45 lignes de classe** sont modifiées. Tous les autres tenseurs sont identiques à l'origine.

L'interpolation `θ_s = θ_orig + s·(θ_push−θ_orig)` reproduit s125, s150 et s200 exactement dans les tenseurs float32 examinés ; pour s175, écart maximal **5,96e−8**, dû à l'arithmétique flottante. Le champion est bien une extrapolation de la même tête.

Ce n'est donc pas une soustraction uniforme « alpha=14 au logit gagnant de chaque image ». La variation est **Δz(x)=ΔW·f(x)+Δb** : elle dépend de l'image, et porte sur des lignes de classes entières. Les biais s175 baissent jusqu'à −0,5663 ; le reste de l'effet vient des poids.

### Ce qui est stable, et ce qui ne l'est pas

Rejeu CPU des têtes sur le cache existant : 3 500 forget, leurs altérations et 1 530 images hors forget. Pour les cinq têtes disposant d'un forward complet sauvegardé, **aucun désaccord top-1** entre rejeu et forward sur les forget ; écart absolu maximal de logits ≤0,015625. Le rejeu reste une approximation numérique fp16 de la tête, pas un nouveau forward exact de tout le ViT.

| Push | Accord sur forget | Accord hors forget, classes concernées (699) | Accord hors forget, autres classes (831) | KL des autres classes, T=2 |
|---|---:|---:|---:|---:|
| ×1,25 | 90,11 % | 17,45 % | 99,88 % | 0,2715 |
| ×1,50 | 77,63 % | 9,44 % | 99,88 % | 0,3588 |
| **×1,75** | **58,29 %** | **5,01 %** | **99,88 %** | **0,4506** |
| ×2,00 | 38,54 % | 2,29 % | 99,88 % | 0,5462 |

Le softmax normalise ensemble les 100 logits : garder une ligne de poids intacte ne garde pas sa probabilité intacte lorsque d'autres logits bougent. À ×1,75, le facteur `1/(1+KL)` du seul groupe « autres classes » vaut environ **0,6894**, malgré l'accord top-1 quasi parfait. Ce facteur est local et n'inclut pas le terme accuracy du vrai retain.

Sur les classes concernées hors forget, le mécanisme modifie beaucoup plus largement les décisions qu'il ne le fait sur les forget : accord **5 % contre 58 %** à ×1,75. La sélection des images et les marges initiales diffèrent ; cette comparaison ne permet pas d'inférer les modèles gold. Elle montre que le push n'est pas un effacement spécifique des 3 500 images.

Le maximum public observé à ×1,75 est donc un **compromis empirique**, pas une preuve que l'unlearning gold consiste à supprimer 45 classes ou que poursuivre l'intensité fonctionnerait. Le zéro sous 70 % d'accuracy cachée reste impossible à certifier avec de simples accords sur un cache non étiqueté.

Preuves : `profondeur/t3-poids-verifies.json`, `profondeur/t3-rejeu-cache.json`. Le cache entier avait déjà servi aux analyses antérieures : **il ne constitue pas une nouvelle confirmation indépendante**.

## 7. Reproductibilité : ce qui est sauvegardé, ce qui manque encore

Les cinq fichiers locaux présents parmi train/data/model/export/combine_eval sont **identiques octet pour octet** aux sources relevées sur le cluster. Ils sont pourtant tous **non suivis par Git** dans `eaic-wt-c1-pixseg`. `merge_ss.py`, présent sur le cluster, n'existe pas à cet emplacement local ; sa copie est maintenant conservée dans `profondeur/sources-executees/`.

La recette est donc récupérable, mais **HEAD `2bb6b4b` ne la décrit pas**. Les poids T1 sont sauvegardés comme state_dict sans config d'entraînement intégrée ; les paramètres existent séparément dans les NPZ et les logs. Les logs sont essentiels pour éviter de confondre les seeds de l'argumentaire et les graines réellement utilisées.

Environnement relevé à 03:21 : Python 3.11.18, torch 2.5.1, torchvision 0.20.1, NumPy 1.26.4, SciPy 1.17.1, timm 1.0.31, Pillow 13.0.0, onnxruntime-gpu 1.23.2. Ces versions décrivent l'environnement **actuel**, pas un verrou de dépendances capturé à chaque job historique. Le code active `cudnn.benchmark=True` et ne force pas les algorithmes déterministes : une relance n'est pas une promesse d'égalité bit à bit des poids.

Actions de pérennisation à préparer pour le responsable Git : suivre les sources réelles et leurs configurations, conserver le manifeste poids/NPZ/source/version, relier le reçu à ce manifeste. Aucun commit, aucune intégration ni opération Git d'écriture effectués ici.

## 8. L'effet réel sur le général, à tableau figé

Le snapshot officiel capturé par le relais vers **03:17** montre **27e, général 0,31910777**, avec nos trois scores inchangés. Le sixième est à **0,76118525**. La baisse depuis le relevé 03:05 vient de l'évolution des autres équipes et des extrêmes ; aucun de nos meilleurs scores n'a baissé.

| Un seul score change, autres équipes figées | Général calculé |
|---|---:|
| T1 → 0,70 | 0,34699 |
| T1 → 0,75 | 0,40041 |
| T1 → meilleur actuel 0,80327 | 0,45733 |
| T2 → 0,60 | 0,45946 |
| T2 → 1,00 | 0,59981 |
| T3 → 0,30 | 0,47523 |
| T3 → meilleur actuel 0,43958 | 0,58107 |

Même rejoindre le meilleur actuel sur une seule tâche ne suffirait pas au top 6 de cet instantané. C'est un constat comptable, pas un arbitrage de recherche ni une prévision : l'autre session travaille sur la priorité d'expérience demandée par Selyan. L'audit établit seulement que les scénarios du plan de 18:35 ne permettent plus de fixer l'ambition du matin.

Preuve et calcul reproductible : `profondeur/diagnostic-classement.json`, `analyser_classement.py`.

## 9. Vérifications relancées et limites restantes

Contrôles exécutés pendant cet approfondissement :

```text
c1/test_metric.py                         : code 0, 8 tests passés
test_npo_same_view.py                      : code 0, delta même vue 0 ; ancien calcul 0,0183
official_formula.py                       : code 0, cas à la main OK
mech/test_class_split.py                   : code 0, 18 lignes contrôlées
analyser_profondeur.py                     : code 0, gain 6k reproduit +0,00762300317
analyser_t3_cache.py                       : code 0, pas de désaccord top-1 sur les têtes de contrôle
analyser_classement.py                     : code 0, général du snapshot reproduit à moins de 1e-12
```

Sorties complètes dans `profondeur/tests-cibles.json` et les JSON de diagnostic. Aucun paquet installé ; les tests sont ceux déjà présents, sur CPU, sans poids téléchargés ni nouvelle suite ajoutée.

Restent non établis : l'artefact exact à 0,6391 ; les reçus bruts manquants ; la règle officielle de sélection en final ; le budget authentifié courant ; le quota projet courant ; le score privé ; le gain exact du nouvel ensemble test. Les sources et données supplémentaires réduisent les incertitudes de provenance et de calcul, **pas ces inconnues externes**.

À la reprise de Claude : lire d'abord le résultat final du job 15904703 et ses OOF, conserver le seuil et la recette figés, distinguer le challenger deux graines du potentiel trois graines, puis appliquer le critère annoncé au bon comparateur. Les corrections du banc T2, si poursuivies, sont un nouveau lot de travail ; le lot noté actuel reste NO-GO. Le champion T3 reste s175 tant qu'un autre envoi n'a pas établi mieux.
