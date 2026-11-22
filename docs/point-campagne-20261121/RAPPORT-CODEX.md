# Point de campagne et proposition de changement de stratégie

Mandat direct de Selyan : état des marges, overall et progression des rangs, proposition de véritable changement de stratégie ; garder la branche T2 phase 1b comme option de test final. Relevés du 21 novembre 2026, 23:31–23:35 Paris. Aucune soumission, requête d'inférence, modification de job ou opération Git d'écriture dans ce mandat.

## Résultat actuel vérifié

Le job T3 `15903709` est **COMPLETED**, code `0:0`, de **23:24:59 à 23:33:40**, soit **8 min 41**. Les quatre variantes ont atteint 1 000 étapes et les **32 checkpoints** ont une ligne d'évaluation. Ces mesures remplacent les estimations de durée rapportées précédemment.

| Marge | CE propre cible | CE propre finale moyenne / écart-type | Classes changées, propre | Accord hors forget | Proxy retain hors forget |
|---|---:|---:|---:|---:|---:|
| 2,5 | 0,1269 | 0,0210 / 0,0330 | 0 % | 96,3 % | 0,9952 |
| 1,5 | 0,3096 | 0,0382 / 0,0612 | 0 % | 95,6 % | 0,9940 |
| 1,0 | 0,4655 | 0,0507 / 0,0812 | 0,11 % | 95,4 % | 0,9934 |
| 0,5 | 0,6770 | 0,0664 / 0,1059 | 0,29 % | 95,1 % | 0,9927 |

Les deux valeurs hors forget sont des mesures contre les prédictions de l'origine sur des photos externes, **pas** une accuracy sur le retain officiel. Elles ne certifient pas le passage de la porte cachée des 70 %.

Les CE réalisées n'atteignent que **9,8 à 16,5 %** des CE cibles construites. C'est un écart d'apprentissage mesuré, pas une estimation de la distance au gold inconnu. L'augmentation d'entropie n'établit pas à elle seule un progrès vers la distribution gold.

L'analyse CPU existante, relancée sur une copie figée des nouvelles tables, présélectionne :

1. `mg_mg05_s1000` : CE 0,0664 / 0,1059 ; changement de classe 0,29 % ; accord lowpass32 0,7506.
2. `mg_mg10_s1000` : CE 0,0507 / 0,0812 ; changement de classe 0,11 % ; accord lowpass32 0,763 environ.

Ce classement dépend de scénarios gold hypothétiques et des contrôles de dérive ; ce n'est pas un score serveur. Aucun push nouvellement mesuré n'entre encore dans cette comparaison. Le meilleur candidat entre familles reste à déterminer par le job `push_eval` déjà confié à l'orchestrateur. Les checkpoints et les scripts restent sous la propriété de leurs sessions ; aucune copie de modèle n'a été créée ici.

## Classement officiel lu à 23:31 Paris

Source : [classement officiel](http://34.147.229.55/leaderboard_page), copie exacte `leaderboard-current.html`, extraction `leaderboard-analysis.json`. Comparaison avec `../audit-strategies-20261121/leaderboard-2315.html`.

| Mesure | Vers 23:15 | À 23:31 | Bilan |
|---|---:|---:|---|
| T1 | 0,584524 ; 20e | 0,584524 ; 22e / 41 | Score stable, recul de 2 places |
| T2 | 0,200000 ; 12e | 0,200000 ; 14e / 32 | Score stable, recul de 2 places |
| T3 | 0,056662 ; 26e | 0,056662 ; 26e / 38 | Score et rang stables |
| Overall | 0,325632 ; 20e | **0,287107 ; 21e / 30 classées** | Recul de 1 place |

42 équipes sont présentes au panneau général, dont 30 ont un rang. Les minima et maxima observés deviennent : T1 **[0,403691 ; 0,751079]**, T2 [0 ; 1], T3 [0 ; 0,402509]. La normalisation `(score-min)/(max-min)` par tâche, puis moyenne sur les trois tâches, reproduit tous les overall complets avec un écart inférieur à 1e-12. Le minimum T1 de l'instantané précédent valait 0,293352 : son relèvement explique la baisse de notre overall malgré nos scores inchangés. La cause du changement de ce minimum n'est pas établie par les deux instantanés.

Il y a bien eu des gains plus tôt dans la campagne : T1 0,5307 → 0,584524 et T3 0,030824 → 0,056662. La dernière hausse T1 enregistrée date de 20:05 Paris ; T3 de 23:07. T2 reste sur la baseline à 0,20 de 16:48 ; le candidat récent à 0,10 n'a pas amélioré le meilleur. Ces gains historiques ne changent pas le constat de stagnation entre les deux derniers relevés. Aucun historique général complet ne permet d'affirmer une progression de rang sur toute la journée.

## Proposition à l'orchestrateur : changer les expériences suivantes

L'enchaînement marges → évaluation push → export NCC est déjà autorisé et appartient à l'orchestrateur. Cette lecture n'en a ni modifié ni doublé l'exécution. Les propositions ci-dessous portent sur les expériences suivantes ; ce rapport ne constitue pas une autorisation de soumission ou une levée du gel GPU.

### T1 : passer à une localisation apprise, puis la valider sur tous les groupes

La modification stratégique est de passer des scores de décodage agrégés et des boîtes actuelles à une **segmentation supervisée exploitant les features spatiales des extracteurs officiels et le message connu**. Le pilote existe déjà dans `/Users/selyan/dev/eaic-wt-c1-pixseg/c1/pixseg/dense/` : ne pas en créer un second. Il prévoit quatre variantes en un job de 50 minutes maximum, génération synthétique tracée et confirmation sur des images réelles mises à l'écart.

Donner la priorité au passage GPU de ce pilote une fois son rendu CPU validé, puis aux résultats par image. Si le contrôle synthétique échoue, au plus deux essais ciblés avant abandon. S'il réussit mais ne transfère pas aux vrais masques, ne pas étendre les entraînements sur la seule foi du synthétique.

La première recette porte sur PS/TM F. **+0,08 sur ces deux groupes n'ajoute que +0,0267 au global**, soit 0,5881 sur la référence locale de tête à 0,5614, avant tout nouvel assemblage. Pour viser 0,70, il faut ensuite un gain étendu aux rectangles et à StegaStamp, mesuré dans une pipeline complète sur les six groupes. L'objectif de travail F moyen 0,65 et R moyen 0,75 donnerait 0,70, mais ces nombres sont des objectifs, pas des prévisions.

Le principe de lecture locale est soutenu par [WAM](https://arxiv.org/abs/2411.07231), qui entraîne conjointement son propre embedder et son extracteur. Cela n'autorise pas à présenter ses poids comme un détecteur universel pour les trois watermarks du challenge.

Le NCC déjà en file est une livraison du candidat existant. Les audits ne mesurent qu'environ +0,0014 pour l'ajout NCC comparé. Il ne justifie donc pas une nouvelle campagne de balayage ou de graines comme voie principale vers +0,10.

### T3 : terminer le comparatif actuel, puis comparer un autre mécanisme

La famille push possède le seul signal serveur positif récent. Conserver son champion comme référence et finir la mesure déjà lancée est utile. En revanche, les marges atteignent mal leur propre cible : un nouveau balayage de marges ou une nouvelle amplification ne constituerait pas encore une stratégie de rupture.

**Expérience distincte proposée : NPO corrigé « même image professeur/élève », suivi d'une phase de réparation par distillation de l'origine sur photos hors forget.** Le correctif existe déjà dans `../task3-work-20261121/unlearn_v2_fixed.py` et son test CPU passe ; il n'a pas été validé sur le vrai ViT. Cette proposition réutilise ce travail au lieu de recoder un autre localiseur de confiance.

Avant GPU : séparer les sources hors forget entre apprentissage/réparation et contrôle ; une image et ses transformations restent dans le même lot. Vérifier que les sources sont autorisées par l'épreuve. Repartir du modèle original, garder une seule recette NPO, comparer la même recette avec/sans réparation et le champion push inchangé. Prévoir un budget pilote explicite, par exemple 30 minutes maximum dans un seul job à proposer au propriétaire T3, plutôt qu'une grille ouverte.

Exiger des distributions de pertes par image, des résultats sur altérations tenues à l'écart, le changement top-1, la KL et l'accord hors forget indépendant, ainsi que le contrôle fp16. Écarter les régressions dominées sur ces contrôles. Sans gold et sans retain étiqueté, il reste une présélection : on ne peut pas prédire 0,30 ni certifier la porte des 70 % avec ces proxys. Aucun checkpoint issu de cette expérience n'est autorisé à être soumis par ce rapport.

La réparation après effacement est analysée dans le [rapport NeurIPS unlearning](https://arxiv.org/html/2406.09073v1). La distillation sans retain à partir d'images hors distribution est étudiée par [SCAR](https://arxiv.org/html/2404.12922v1). Ces sources motivent la phase de réparation, pas une promesse de gain dans notre épreuve. La cible SCAR vers une mauvaise classe n'est pas transposée ici.

### T2 : garder phase 1b comme option finale, changer d'abord le signal

La branche distante **`feat/c2-phase1b` existe**, confirmée par le connecteur GitHub. Son README et son plan ont été relus à cette référence. Elle n'était pas dans les références locales initiales. La branche n'a été ni récupérée dans l'arbre de travail ni fusionnée. Le plan conserve des relevés anciens ; ils ne remplacent pas les compteurs actuels.

Conserver cette branche comme option demandée par Selyan. Avant de la déclencher : inspecter son runner effectif, corriger les relances illimitées, tracer chaque tentative, rapprocher les compteurs, vérifier la correction serveur. Le coût de 924 réponses vient du runner audité sur JURECA ; le plan Git distant décrit aussi une allocation différente, plus ancienne. **Recalculer le coût sur les IDs, répétitions et réponses déjà présents**, sans le déduire du seul nom « phase1b ».

Pour changer de stratégie : tester sur les références une empreinte de décisions/erreurs à contenu contraint, avec calibration explicite des déguisements et évaluation sur d'autres répétitions ET d'autres prompts. Une simple similarité de texte ou de longueur ne fournit pas actuellement le signal fp8/LoRA faible. Mesurer séparément le regroupement des cinq déguisements et l'isolement des variantes. Si le signal ne tient pas hors sélection, ne pas dépenser 924 réponses pour la même règle en fin de campagne. La branche reste utile comme infrastructure d'analyse et de collecte, sans devenir une expérience concluante par sa seule existence.

Le [papier Model Equality Testing](https://arxiv.org/html/2410.20247v2), annexe C.3, détecte aussi un changement de température : MMD brut ne répond donc pas directement à notre identité de modèle, qui doit ignorer ce changement.

## Effet potentiel au classement : scénarios, pas résultats

Recalcul avec les extrêmes et concurrents figés au dernier relevé :

| Hypothèse, autres scores inchangés | Overall | Rang simulé |
|---|---:|---:|
| T1 seule à 0,70 | 0,3979 | 16e |
| T2 seule à 0,60 | 0,4204 | 16e |
| T3 seule à 0,30 | 0,4886 | 13e |
| T1 0,70 ; T2 0,60 ; T3 0,30 | 0,7328 | 4e |

Cela situe l'effort nécessaire et l'intérêt d'un progrès important en T3. Les scores des autres équipes et les extrêmes peuvent changer ; aucun de ces rangs n'est une prévision privée ou une performance acquise.

## Preuves et limites de l'intervention

- Départ : checkout `/Users/selyan/dev/eaic-paris-2026`, branche `main`, HEAD `f209e20397736c53824a7a94ec73f2bd66d012eb`. Arbre déjà sale et nombreux worktrees d'autres sessions. Relevés `git status --short`, `git branch --show-current`, `git rev-parse HEAD`, `git worktree list`, `git log --oneline -20` effectués. Aucun changement des sources existantes ni de branche.
- Lectures JURECA : `squeue -u mouhali1,pelet1,zhang95`, `sacct -j 15903709 -X -o JobID,State,Elapsed,ExitCode,End`, journaux `logs/15903709_[0-3].out`, tables `res/score_15903709_*.txt`, script `margin_edit.py`. Réponses archivées dans `margins-snapshot.json` et copies `logs/`, `res/`.
- Sortie Slurm finale : `15903709 COMPLETED 00:08:41 0:0 2026-11-21T23:33:40`. `squeue` équipe vide au relevé de 23:34:48, avant le prochain lancement éventuel du script de l'orchestrateur.
- Actualisation du journal local de l'orchestrateur : `23:35:08 push_eval soumis : 15903813`. La chaîne a donc bien enchaîné après les marges, sans intervention de ce mandat.
- Commande d'analyse locale : `PYTHONDONTWRITEBYTECODE=1 python3 docs/task3-work-20261121/analyze_margin.py --job 15903709 --res-dir docs/point-campagne-20261121/res --logs-dir docs/point-campagne-20261121/logs`. Code retour 0 ; sortie intégrale dans `analyse-marges.txt`. Aucune erreur/alerte n'est affichée dans cette sortie.
- Classement : GET public officiel via `urllib.request`, 126 303 octets, lu à 22:31:02 UTC / 23:31:02 Paris. L'outil web ne résolvait pas l'URL par IP ; recours à la lecture HTTP directe. Pas d'authentification ni d'endpoint d'inférence consulté.
- GitHub : `search_branches` trouve `feat/c2-phase1b`, puis `fetch_file` lit `c2/PLAN.md`, `c2/README.md`, `c2/features.py` à cette référence. La tentative `c2/prompts_ep_1b.csv` renvoie 404 sur la branche distante ; ne pas supposer que le runner JURECA et la branche distante ont les mêmes fichiers.
- Pas de nouveau test de modèle sur CPU ou GPU, pas de requête T2, pas de soumission, pas de modification de la chaîne Slurm. Les données brutes et le rapport sont les seuls nouveaux fichiers créés par ce mandat.

Question à arbitrer par Opus/Selyan : après clôture des livraisons déjà autorisées, donner la priorité au pilote dense T1 et au comparatif NPO corrigé/réparation T3, plutôt qu'aux nouveaux balayages NCC/graines/coefficients. La mise en œuvre GPU reste du ressort de l'orchestrateur existant.
