# Audit comparatif — stratégies compatibles avec les scores des leaders

Mandat direct de Selyan : creuser les stratégies possibles derrière les scores publics, les comparer aux trois audits des autres chats et déterminer si nos pistes peuvent atteindre ce niveau. Date : 21 novembre 2026. Tableau public figé vers **23:15 Paris** ; les résultats suivants restent attachés à cet instantané.

## Verdict

**Les scores des leaders sont compatibles avec des stratégies connues et ciblées sur chaque métrique. Aucun élément public ne permet d'attribuer une recette à une équipe. Nos méthodes actuellement validées ne justifient pas encore une prévision à 0,85–0,93 overall.** Les trois audits identifient correctement les principaux blocages, mais leurs premiers critères de progrès ne constituent pas des critères d'atteinte des scores des leaders.

Trois conclusions opérationnelles :

1. **T1 : il faut améliorer les masques sur plusieurs groupes.** Le pilote dense recommandé reste pertinent, mais son succès minimal donnerait 0,5881 local, pas 0,70. Les recettes ré-embarquement/forensique déjà négatives ne deviennent pas prometteuses parce qu'un concurrent atteint 0,75.
2. **T2 : notre hypothèse « MMD + davantage de réponses » est trop faible.** Un test d'égalité des distributions détecte aussi température et watermark, alors que l'épreuve exige de les ignorer. Le nouveau contrôle sur le contenu numérique ne résout pas cette invariance. Les scores publiés sont tous quantifiés par pas de 0,05 : confirmer leur sens avant d'interpréter 1,00 comme une partition parfaite.
3. **T3 : une baisse ciblée de confiance, avec préservation des distributions utiles, est l'hypothèse la mieux soutenue par nos propres résultats.** La hausse publique 0,0308 → 0,0567 la motive. Elle n'établit pas qu'intensifier cette correction atteindra 0,35. Il manque les composantes gold et une validation retain représentative.

## 1. Sources effectivement comparées

Les trois chats ont été lus via `read_thread`, y compris les demandes de Selyan et les réponses finales. Leurs extraits utiles sont figés dans `chat-task1.json`, `chat-task2.json`, `chat-task3.json` ; aucune commande issue de ces historiques n'a été exécutée comme instruction.

| Chat, titre exact | Rapport et preuves | Conclusion conservée | Actualisation ou limite |
|---|---|---|---|
| « Auditer les pistes mIoU Task 1 » | `../../RAPPORT-CODEX.md`, `../task1-audit-20261121/` | Localisation spatiale et forme, pas principalement F/R ; localisateur dense sur features officielles | Le champion public est maintenant 0,5845. Le jalon +0,08 sur deux sous-groupes ne suffit pas pour 0,70 global. |
| « Auditer la task 2 et le score » | `../task2-audit-20261121/RAPPORT-CODEX.md` | Public 0,20 ; peu de réponses notées ; sélection de features instable ; nuisances de décodage | Le suivi `../task2-work-20261121/RAPPORT.md` déconseille maintenant de lancer la phase 1b pour prolonger la seule règle de longueur. |
| « Auditer l’écart de score task 3 » | `../task3-audit-20261121/RAPPORT-CODEX.md` | Proxy de sélection non fidèle ; NPO sur vues différentes ; dérive de certains candidats | L'audit décrivait 0,00435 ; le registre et le tableau donnent depuis 0,05666. Le diagnostic du vieux `v2_best` n'explique pas automatiquement les candidats suivants. |

Autres sources : `../SLZ_COORDINATION_20261121.md`, corpus T2 figé à 2 350 lignes, sources officielles archivées dans les audits, littérature primaire relue (§8). Les expériences nouvelles de cet audit portent exclusivement sur les fichiers locaux. Aucun dépôt d'une équipe concurrente de la manche en cours n'a été consulté.

**Évolutions apparues pendant la comparaison, intégrées avant remise :** le registre rapporte une soumission T2 `c2_fam3_pairs_v2` à 23:23:41, score **0,10**, meilleur toujours 0,20 ; un relevé de **95 crédits restants par endpoint à 23:25** ; et un complément à l'audit T2 sur les retries après erreurs 502 potentiellement débitées. Le dossier `../task3-work-20261121/` apporte une formule explicite, le correctif NPO et des simulations de réduction du logit gagnant. Ce sont des travaux d'autres sessions, relus ici ; aucun de leurs jobs n'a été lancé par cet audit.

## 2. Ce que le classement permet vraiment d'inférer

| Équipe | T1 | T2 | T3 | Overall |
|---|---:|---:|---:|---:|
| Fable 4.2 | 0,717124 | 1,000000 | 0,353257 | 0,934485 |
| Murmell | 0,723978 | 0,850000 | 0,308029 | 0,852021 |
| Eugenia | 0,751079 | 0,850000 | 0,121892 | 0,717610 |
| Skesiea | 0,654212 | 0,000000 | 0,402509 | 0,596125 |
| SLZ | 0,584524 | 0,200000 | 0,056662 | 0,325632 |

Source : [classement officiel](http://34.147.229.55/leaderboard_page), HTML intégral dans `leaderboard-2315.html`, extraction dans `analysis_results.json`. Le tableau public ne révèle ni méthode, ni coût, ni nombre de tentatives sans amélioration. L'horodatage marque le meilleur résultat enregistré, pas le début du développement.

La normalisation est `(score-min)/(max-min)` par tâche, suivie de la moyenne. Extrêmes observés : T1 [0,293352 ; 0,751079], T2 [0 ; 1], T3 [0 ; 0,402509]. `analyze.py` reproduit tous les overall des équipes complètes à moins de 1e-12.

Cela explique le score élevé de Fable : trois performances proches du haut du tableau, sans nécessité d'obtenir 0,93 brut à une tâche. La T3 a aussi progressé : le premier de l'ancien audit était à 0,342216 ; le maximum figé est maintenant 0,402509.

### Point nouveau : granularité de la T2

Les **31 valeurs détaillées** T2 sont exactement des multiples de 0,05 ; les valeurs distinctes sont 0 ; 0,05 ; 0,10 ; 0,20 ; 0,40 ; 0,50 ; 0,60 ; 0,65 ; 0,70 ; 0,85 ; 1. Ce constat porte sur les attributs HTML `title`, pas seulement sur l'affichage à deux décimales.

**Inférence :** une quantification, un arrondi ou une autre transformation de publication est plausible. L'HTML ne permet pas d'en connaître la règle ; une propriété particulière de l'évaluation n'est pas exclue. Un 1,00 publié ne prouve donc pas à lui seul toutes les décisions correctes, encore moins sur les paires privées. À vérifier dans une clarification officielle ou un reçu déjà existant ; aucune soumission de sondage n'est proposée.

Les déductions du suivi T2 fondées sur la formule exacte sont à conserver avec leurs hypothèses : CSV et reçu correctement associés, même formule sur le serveur, transformation de publication connue. Une simulation conditionnée par un a priori ne démontre pas la vérité des familles.

## 3. T1 : comment obtenir 0,72–0,75 ?

### Hypothèse principale : exploiter finement le détecteur fourni

Une recette compatible avec ces scores serait : features intermédiaires des extracteurs officiels à plusieurs résolutions, comparaison au message connu, petite tête de segmentation supervisée, données synthétiques compatibles avec l'embedder, et traitement spécialisé par méthode/régime. L'entraînement peut s'appuyer sur les 510 masques réels sans nécessiter de reconstruire tout un détecteur à partir de zéro.

Le mécanisme attendu est précis : conserver un signal local avant le pooling, puis apprendre les frontières des régions watermarkées. **C'est l'explication techniquement la plus cohérente avec notre goulot mesuré ; ce n'est pas une information sur la méthode des concurrents.** Le papier [WAM](https://arxiv.org/abs/2411.07231) démontre le principe d'un extracteur produisant conjointement segmentation et message. Son réseau est entraîné avec son propre embedder : ses poids ne constituent pas un détecteur universel immédiatement applicable aux trois méthodes de l'épreuve.

Une seconde recette pourrait mieux résoudre les R par propositions de régions puis décodage du message sur les crops. Le [code officiel Adobe](https://raw.githubusercontent.com/adobe/trustmark/main/python/trustmark/trustmark.py) contient bien un détecteur de boîtes Q/P et `localize`. Son efficacité sur nos images n'est pas mesurée. Il ne résout pas automatiquement les trous F ni les autres méthodes.

### Ce que nos essais éliminent déjà

- **F/R seul** : gain oracle seulement +0,00290 contre notre règle majoritaire triviale. Ce chiffre ne mesure pas le routage de la tête actuelle, mais réfute l'idée que ce seul tri nous apporterait +0,15.
- **Seuil des boîtes actuelles** : +0,0025 à +0,0034 global même avec décision oracle sur les branches R routées ; il faut de nouvelles localisations.
- **Canal de détection PixelSeal prêt à l'emploi** : checkpoint audité avec `lambda_det=0`, canal non entraîné et AUC proche de 0,5.
- **Ré-embarquement corrélé** : pilote déjà négatif sur PS/TM ; les NCC ajoutés à la tête n'ont apporté qu'environ +0,0014 dans la comparaison documentée.
- **Forensique générique et morphologie** : TruFor et les retouches morphologiques n'ont pas montré de gain utile.
- **U-Net RGB de zéro** : déjà essayé ; PS/TM n'apprennent pas un localisateur fiable dans les configurations auditées. Cela ne réfute pas un adaptateur fondé sur les extracteurs officiels et le message.

### Le niveau de progrès nécessaire

Référence locale vérifiée : **0,561425**, F **0,555737**, R **0,567113**. Les six groupes méthode × régime ont 85 images chacun. Il faut distinguer cette référence du champion public 0,584524, obtenu avec un autre assemblage.

| Scénario de diagnostic, sans nouvelle mesure | mIoU global |
|---|---:|
| +0,08 sur PS F et TM F seulement | **0,588092** |
| PS F et TM F parfaits, quatre autres groupes inchangés | **0,709633** |
| SS R parfait seul, cinq autres groupes inchangés | **0,648811** |
| Tous les R remplacés par leurs bbox GT, F inchangés | **0,711300** |
| F moyen 0,65 ; R moyen 0,75 | 0,700000 |
| F moyen 0,64 ; R moyen 0,80 | 0,720000 |
| F moyen 0,70 ; R moyen 0,80 | 0,750000 |

Les deux premières lignes montrent la limite du plan court : **même rendre parfaits les deux groupes prioritaires ne dépasserait pas 0,71**, toutes choses égales par ailleurs. Atteindre 0,75 par ce seul changement est impossible. Les dernières lignes sont des objectifs de sous-groupes, pas des prévisions.

**Verdict T1 : 0,70–0,75 est structurellement plausible, mais non démontré par nos candidats.** Le pilote dense reste un bon test de signal ; pour viser les leaders il faudra étendre un gain réel à davantage de groupes, notamment R et StegaStamp. Multiplier les graines du champion ne constitue pas une explication suffisante du saut recherché.

**Test décisif proposé à Opus :** une recette fixe sur PS/TM F, sources synthétiques disjointes, message fourni, mêmes folds que le champion, calibration entièrement interne ; contrôler mIoU par image et IoU des trous, pas seulement BCE ou bit accuracy. +0,08 sur F PS/TM tenus à l'écart reste un seuil de poursuite, pas une annonce à 0,70. Deux essais d'apprentissage maximum si le contrôle synthétique échoue. Après succès, mesurer une pipeline complète sur les six groupes avant d'annoncer une cible globale.

## 4. T2 : comment obtenir 0,85–1,00 ?

### Hypothèse principale : une empreinte discriminante et résistante aux déguisements

La recette plausible comporte deux niveaux : identifier les bases, puis distinguer quantification et LoRA au sein de chaque base avec des prompts répétés qui produisent des erreurs ou des choix caractéristiques. Il faut simultanément regrouper les endpoints dont seuls le système, la persona, le watermark ou le décodage changent.

Deux mécanismes méritent une expérience :

- Réponses à contenu contraint, correction/continuation de chaînes, choix de solutions à des tâches proches de la limite de compétence ; extraire les décisions ou erreurs plutôt que le ton et la longueur du texte.
- Comparer les profils obtenus sur plusieurs prompts et répétitions, avec une calibration explicite sur les déguisements connus, puis imposer une partition transitive. Une réponse identique n'est pas à elle seule une preuve du même modèle ; deux variantes peuvent réussir la même tâche simple.

Ce schéma peut théoriquement identifier toute la partition. Il ne bénéficie ici d'aucune mesure proche de 0,85 sur les endpoints notés.

### Correction de notre première hypothèse sur MMD

Le papier [Model Equality Testing](https://arxiv.org/html/2410.20247v2) vise l'égalité de **distributions de sorties**. Son annexe C.3 montre qu'un changement de température sur un même modèle est détecté avec une puissance proche de 1. C'est précisément un changement que notre tâche doit ignorer. Le résultat de 77,4 % de puissance annoncé dans le papier utilise environ 10 réponses sur chacun de 20–25 prompts : il ne signifie ni 77,4 % de score au hackathon, ni 10 requêtes totales, et dépasse notre budget de 100 réponses par endpoint.

**Conclusion : MMD brut n'est pas une solution directe.** Il peut servir après choix d'une représentation et d'une hypothèse nulle adaptées aux nuisances ; aucune invariance générale n'est établie pour notre correction de dispersion `corr`.

### Nouveau contrôle CPU, effectivement exécuté

`fingerprint_check.py` lit le corpus figé de 2 350 lignes. Il conserve la première tentative de chaque clé et utilise les 44 prompts pilot1/pilot2 complets : **1 584 observations**, neuf refs et quatre répétitions. Le seuil d'un clustering à lien moyen est ajusté sur les répétitions 0/1 puis évalué sur 2/3, et inversement.

Deux représentations fixées par construction : n-grammes bruts sur 44 prompts ; séquences numériques sur sept prompts dont le texte demande un résultat numérique déterminé. Les prompts en nombres français, chiffres romains ou nombres aléatoires sont exclus après vérification de leur libellé et avant lecture des résultats. Pas de sélection des meilleurs prompts d'après le score. Les sorties sans nombres restent représentées ; aucune paire difficile n'est supprimée.

| Représentation/statistique | MCC hors sélection 0/1 → 2/3 | Sens inverse |
|---|---:|---:|
| N-grammes, similarité moyenne | 0,000 | 0,000 |
| N-grammes, MMD non biaisée | 0,000 | 0,000 |
| Séquence numérique, similarité moyenne | **0,3315** | **0,0210** |
| Séquence numérique, MMD non biaisée | 0,000 | 0,1203 |

Les huit partitions ont été vérifiées avec `c2/kit/score_practice.py` : accord à l'arrondi de 0,001. Ce sont des MCC modèle sur les mêmes neuf refs ; les paires ne sont pas indépendantes. Deux répétitions par moitié sont peu nombreuses pour MMD. **Ce contrôle réfute la suffisance de ces recettes simples sur ce corpus, pas toute méthode d'empreinte ni une collecte mieux conçue.** Le meilleur cas numérique isole la LoRA forte, mais laisse fp8 et les autres LoRA fusionnés avec les cinq bases.

### Ce qu'exigerait réellement 0,85

Avec la formule annoncée et avant toute quantification, même si les familles sont parfaites, 0,85 total exige **MCC modèle ≥ 0,8333**. Sur la vérité des refs :

| Décisions correctes obtenues, autres endpoints fusionnés | MCC modèle | Total hypothétique avec familles parfaites |
|---|---:|---:|
| Isoler seulement LoRA 1,5 | 0,3315 | 0,3983 |
| Isoler LoRA 0,7 et 1,5 | 0,5241 | 0,5717 |
| Isoler aussi fp8 | 0,7338 | 0,7604 |
| Isoler aussi LoRA 0,3 | 1,0000 | 1,0000 |

Ce n'est pas une borne pour les 33 endpoints, dont les groupes sont inconnus. Cela montre néanmoins pourquoi la seule détection des variantes fortes ne suffit pas. Le suivi hors ligne estime 0,13–0,18 privé pour son meilleur candidat sous deux a priori ; ce sont des simulations, pas un plafond prouvé de toute méthode.

Le nouveau **0,10 public de `fam3_pairs_v2`** est cohérent avec le plancher d'une solution trouvant les familles mais pas les modèles. Il ne démontre pas que ce soit sa décomposition : une autre combinaison des deux MCC peut donner le même total, et la règle de publication reste inconnue. Il confirme surtout l'absence de progrès public au-delà de 0,20 pour ce candidat précis.

**Verdict T2 : la stratégie d'empreinte est compatible avec 0,85–1,00, mais nos données et règles actuelles ne la valident pas.** Simplement ajouter des prompts à la règle de longueur ne fournit pas cette trajectoire : le suivi mesure seulement 0,142 → 0,236 MCC hors sélection en passant de 5 à 32 prompts uniques.

**Test décisif proposé :** sur refs seulement, trois prompts contraints × dix répétitions, méthode fixée avant collecte, cinq répétitions pour régler et cinq pour vérifier, puis inversion. Mesurer séparément faux éclatements des cinq déguisements, séparation fp8 et LoRA faible. MCC ≥0,60 dans les deux sens serait un signal de poursuite ; pour soutenir une ambition proche de 0,85 total, demander ensuite ≥0,833 de MCC modèle dans les deux sens et un transfert sur d'autres prompts. Si ce test échoue, ne pas dépenser les endpoints pour la même recette. Une collecte notée éventuelle de 20–30 réponses par endpoint dépend d'abord du budget serveur réel et de ce résultat ; rien n'a été lancé.

**Contrainte de collecte supplémentaire :** le complément §8 de l'audit T2 documente un signalement d'erreurs 502 débitées transmis par Selyan et confirme les retries illimités du runner archivé, qui n'écrit que les réussites. Ce n'est pas une perte démontrée chez SLZ ; le relevé rapporté des 95 crédits est rassurant à cet instant. Avant le pilote puis toute collecte notée : plafond de tentatives, trace de chaque statut, arrêt sur erreur ambiguë et compteur rapproché ; valider exactement les mêmes limites de tokens que celles envisagées pour les endpoints. Le succès rapporté d'un prompt avec `max_tokens=8` ne démontre pas que 8 soit adapté à toutes les empreintes. Les longs textes de récitation ne doivent donc pas être prescrits sans test de transport et de troncature.

## 5. T3 : comment obtenir 0,30–0,40 ?

### Hypothèse principale : réduire la surconfiance avec une dérive limitée

Un modèle réentraîné sans les images forget peut continuer à prédire leur classe correcte. Il peut être surtout moins confiant, et distribuer le reste de la probabilité sur des classes proches. Dans ce cas, réduire sélectivement certaines marges, sans détruire les représentations, est une stratégie compatible avec un score élevé. C'est une hypothèse sur les gold, pas une propriété mesurée des gold de cette épreuve.

La pièce empirique la plus pertinente est notre propre progression documentée : `push_best` 0,0308 ; modification de tête amplifiée ×1,25, `push_s125`, **0,0567**. Les valeurs détaillées sont dans le tableau public et le registre de coordination. Cela motive de tester cette famille de corrections. **Le tableau ne prouve pas le mécanisme exact du gain**, puisque les composantes du score ne sont pas exposées et que modifier les poids d'une ligne de tête peut aussi déplacer les logits d'autres images.

Il existe aussi une justification analytique, cohérente avec le nouveau travail T3. Pour une image, une cible gold `g`, `q = softmax((z − δ e_y)/T)` et un indice `y` fixé, on a **`d KL(g || q) / dδ = (g_y − q_y)/T`**. Tant que le candidat surévalue la probabilité de `y`, abaisser uniquement ce logit réduit cette KL. Cette propriété n'exige pas de faire changer la classe gagnante. Elle ne garantit ni le terme de perte, ni le retain, ni qu'une modification de poids réalise une baisse isolée sur chaque image.

Le fichier `../task3-work-20261121/toy_alpha_output.txt` illustre une saturation puis une chute du produit quand l'intensité dépasse la cible supposée ; l'optimum varie avec le gold synthétique. Son intervalle alpha 10–12 pour certains gold supposés ne doit pas être présenté comme un optimum mesuré sur l'épreuve. Les sorties de `analyze_margin_historique.txt` ne contiennent pas encore de contrôle complet des `push_s*` dans leur référentiel ; les meilleurs candidats classés par ce proxy ne constituent pas des gagnants serveur.

Une recette plus complète pourrait combiner cibles de confiance par image, mises à jour limitées à la tête/derniers blocs, distillation de l'origine sur des images hors forget et augmentations adaptées aux altérations cachées. La cible ne doit pas être systématiquement « mauvais label » ou « uniforme ».

Le rapport de la [compétition NeurIPS](https://arxiv.org/html/2406.09073v1) documente le schéma effacer puis réparer chez de nombreuses méthodes, avec une dépendance importante à la phase de réparation. [SCAR](https://arxiv.org/html/2404.12922v1) montre une distillation sur images hors distribution sans retain. Ces résultats motivent le mécanisme de préservation, pas le chiffre 0,35 sur notre métrique. Le test ViT de SCAR cité dans l'ancienne veille porte sur l'oubli de classes ; sa cible vers une mauvaise classe ne doit pas être transposée telle quelle à l'oubli d'exemples.

### Pourquoi un score de 0,35 ne nécessite pas une imitation parfaite

La formule officielle est `A × B × R` : progrès normalisé sur moyenne/écart-type des pertes, progrès normalisé sur KL vers les gold, puis retain. Les deux composantes d'oubli sont chacune moyennées entre images propres et altérées avant multiplication.

- `0,25 × 0,25 × 0,90 = 0,05625`, proche numériquement de notre meilleur public ; **ce n'est pas sa décomposition mesurée**.
- `0,65 × 0,65 × 0,85 = 0,359125` : un progrès substantiel mais partiel sur les deux axes peut suffire au niveau de Fable.
- Pour 0,40 avec retain 0,90 et deux progrès égaux, chacun doit atteindre 0,6667. Améliorer seulement la perte en laissant le progrès KL presque nul ne suffit pas.

### Ce que les audits empêchent de promettre

- `eval.py` n'est pas un estimateur du score gold ; le modèle identité y a un score positif. Corriger son zéro ne suffirait pas à en faire une validation gold.
- Le défaut NPO « professeur propre / élève augmenté » est réel ; sa correction n'a pas de gain serveur mesuré.
- Les augmentations de forget ne sont ni un retain indépendant ni des cibles gold certifiées. Les photos watermark hors forget permettent un contrôle de dérive, pas une preuve d'accuracy ≥70 %.
- Le candidat agressif `v2_best` et la recette de tête `push` sont différents. Il serait faux d'attribuer la dérive 39,9 % d'accord du premier au second.
- La baisse du logit gagnant ne garantit pas à elle seule la conservation du top-1 après réentraînement, ni la bonne répartition sur les classes secondaires, ni un gain sur les altérations cachées.

**Verdict T3 : atteindre 0,30–0,40 par une correction ciblée et bien préservée est plausible ; nos preuves ne permettent pas de le prévoir.** Un simple balayage plus fort du coefficient `push` peut aussi saturer ou dégrader le modèle. La validation doit comparer des mécanismes, pas seulement multiplier les coefficients jusqu'à la prochaine hausse publique.

**Test décisif proposé :** analyser d'abord les candidats déjà calculés avec les mêmes images/vues : distribution des pertes, quantiles et dispersion, changements top-1, rangs des classes secondaires, KL vers l'origine sur plusieurs contrôles distincts, fp16. Sur une seule recette ensuite, comparer correction de marge et cible issue de vues modérément altérées, avec même contrainte de distillation et mêmes étapes. Ne pas sélectionner sur le produit scalaire trompeur du vieux proxy. Sans gold ou modèle d'ombre validé, la conclusion restera une présélection ; seul un reçu officiel pour un candidat choisi a priori pourra établir le gain public. Les anciens scores ne permettent pas de déduire les seuils gold cachés.

## 6. Peut-on atteindre 0,85 ou 0,93 overall avec ces stratégies ?

Oui **au sens d'un scénario chiffré compatible avec le classement**. Non **au sens d'une performance déjà justifiée par nos validations**.

| Scores T1 / T2 / T3 hypothétiques | Overall recalculé | Rang dans le tableau figé, SLZ remplacé |
|---|---:|---:|
| 0,5845 / 0,20 / 0,0567, actuels | 0,3256 | 20 |
| T2 seule à 1,00, autres actuels | 0,5923 | 11 |
| 0,70 / 0,85 / 0,30 | **0,8279** | 3 |
| 0,72 / 0,85 / 0,31 | **0,8508** | 3 |
| 0,72 / 1,00 / 0,35 | **0,9339** | 2 |

Le dernier scénario reproduit presque Fable. Il exige simultanément un grand progrès en T1, une quasi-résolution publique en T2 et un progrès majeur en T3. Aucune des trois stratégies simples testées ou des corrections de validation ne démontre à elle seule ce triplet.

En portant **une seule tâche à son maximum public actuel**, les deux autres inchangées, on obtiendrait : T1 **0,4469** overall ; T2 **0,5923** ; T3 **0,6120**. Aucun de ces changements isolés n'atteint 0,85. Ces calculs supposent les extrêmes inchangés ; de nouveaux résultats des autres équipes modifieront l'overall. Ils ne prédisent pas le classement privé.

## 7. Décisions proposées à Opus et actions restantes

Cet audit ne change pas l'architecture ni la campagne en cours. Il propose les vérifications suivantes, à arbitrer par Opus/Selyan dans la coordination existante.

| Ordre proposé | Action concrète | Preuve attendue avant extension | Pourquoi |
|---|---|---|---|
| 1 | Clarifier la quantification T2 et la règle de choix du fichier final avec documentation/reçus existants ou organisateurs | Formule effective, arrondi/publication et règle de sélection explicités | Évite d'optimiser une interprétation incertaine des 0,85/1,00. Aucun sondage par soumission. |
| 2 | Réconcilier les résultats des candidats T3 déjà produits dans la session propriétaire | Tableau identique de contrôles, hash de fichier, reçu rattaché | C'est la famille qui possède déjà un signal public positif ; éviter un nouveau run en doublon. |
| 3 | Poursuivre un seul pilote T1 dense tel que cadré par l'audit T1 | Gain hors entraînement PS/TM F, puis gain global mesuré | Valide une source de signal local ; annoncer séparément progrès minimal et ambition 0,70. |
| 4 | Tester le pilote T2 répété sur références avant budget noté | Invariance aux déguisements et détection des variantes faibles sur répétitions tenues à l'écart | Notre contrôle montre que la canonicalisation simple n'y suffit pas. |

Questions encore ouvertes : quelle transformation produit les paliers T2 ; quel fichier comptera en privé ; quelles données externes sont autorisées ; quelle cible de confiance reproduit le comportement gold ; quelles mesures nouvelles des sessions propriétaires existent après nos instantanés. Les affirmations « trois familles de onze » et « une famille déterministe » restent des hypothèses comportementales non validées par une vérité officielle. Une seule réponse par prompt ne démontre pas le déterminisme d'un endpoint. Le job margin et son évaluation push sont déjà pris en charge par la coordination : les tests T3 proposés ici doivent réutiliser leurs sorties avant toute relance.

## 8. Preuves, état et limites

État au départ : `/Users/selyan/dev/eaic-paris-2026`, branche `main`, HEAD `f209e20397736c53824a7a94ec73f2bd66d012eb`, worktrees existants, arbre déjà modifié par d'autres travaux. Aucune opération Git d'écriture, aucun commit, aucun changement de branche. Les seuls fichiers créés par ce mandat sont sous `docs/audit-strategies-20261121/`. Le rapport racine et les trois audits restent intacts de notre part.

Commandes effectives et sorties :

```text
python3 docs/audit-strategies-20261121/analyze.py
  Normalisation reproduite pour toutes les équipes complètes (écart < 1e-12).
  SLZ actuel: overall=0.325632, rang=20
  T1 0,72 ; T2 0,85 ; T3 0,31: overall=0.850757, rang=3
  T1 0,72 ; T2 1 ; T3 0,35: overall=0.933882, rang=2

python3 docs/audit-strategies-20261121/fingerprint_check.py
  numeric_sequence cross 0->1 heldout=0.3315
  numeric_sequence cross 1->0 heldout=0.0210
  numeric_sequence mmd   0->1 heldout=0.0000
  numeric_sequence mmd   1->0 heldout=0.1203

c2/kit/score_practice.py sur les huit partitions de contrôle
  8 concordances à l'arrondi de 0.001 ; practice-check-output.txt

git diff --check
  sortie vide, code 0
```

La première extraction du tableau a rencontré `No score yet` parmi les titres HTML ; le parseur a été corrigé pour distinguer les valeurs numériques. Les prompts numériques ont été vérifiés par leur texte avant lecture des résultats du contrôle ; la version finale exclut les trois formats non adaptés à cette représentation. Ce sont les seules corrections de protocole, sans recherche de variantes après consultation des scores.

Fichiers de preuve : `analysis_results.json`, `analysis-output.txt`, `fingerprint_results.json`, `fingerprint-output.txt`, huit CSV `ref-check-*.csv` et vérification par le kit. `manifest.json` donne les hashes des sources locales utilisées. Aucun nouveau modèle vision évalué, aucun modèle gold disponible, aucune réplication des entraînements initiaux.

Sources primaires consultées :

- [Classement et explication de l'overall](http://34.147.229.55/leaderboard_page).
- [Gao et al., Model Equality Testing, ICLR 2025](https://arxiv.org/html/2410.20247v2), notamment annexe C.3 et protocole d'échantillonnage.
- [Sander et al., Watermark Anything with Localized Messages](https://arxiv.org/abs/2411.07231), pour le principe de localisation apprise avec embedder associé.
- [Adobe, code TrustMark](https://raw.githubusercontent.com/adobe/trustmark/main/python/trustmark/trustmark.py), présence de `localize` et des poids bbox Q/P.
- [Triantafillou et al., bilan de la compétition NeurIPS unlearning](https://arxiv.org/html/2406.09073v1), phases d'effacement/réparation et limites de transfert.
- [Bonato et al., SCAR](https://arxiv.org/html/2404.12922v1), distillation hors retain ; le tableau ViT concerne l'oubli de classes.

Les scénarios arithmétiques sont reproductibles. Les méthodes possibles des leaders demeurent des hypothèses. Aucune collecte endpoint, aucune connexion JURECA, aucun GPU et aucune soumission n'ont été lancés dans cet audit comparatif.
