# Audit Task 2 — travail réalisé, scores et blocages

21 novembre 2026, relevés à partir de 22:49, heure de Paris. Mandat direct de Selyan : « task 2 0.895 rien ne s'améliore […] gros audit de ce qui a été fait ».

Checkout : `/Users/selyan/dev/eaic-paris-2026`, branche `main`, HEAD de départ `f209e20397736c53824a7a94ec73f2bd66d012eb`. Arbre déjà modifié par les travaux c1 et les audits précédents. Aucun changement de branche, commit, fusion, soumission ou lancement de collecte. Les ajouts de cet audit restent dans `docs/task2-audit-20261121/`. Le rapport racine, consacré à c1, est conservé.

**Complément après transmission du signalement Murmell par Selyan : voir section 8.** Le runner archivé réessaie les 502/503/504 sans limite de tentatives, avec `MAX_TOKENS=256`. Une tentative échouée peut donc rendre le coût réel supérieur au nombre de réponses enregistrées, si le serveur la débite comme rapporté par Murmell. Aucun débit perdu de SLZ n'est établi par ce seul témoignage.

## 1. Fait

### Verdict

**Le score public Task 2 vérifié est 0,20, pas 0,895.** À 22:52 Paris, le classement officiel place SLZ 10e sur cette tâche, avec un meilleur résultat horodaté **16:48:05 Paris**. Le premier score est 1,00, partagé par plusieurs équipes. Source : [classement officiel](http://34.147.229.55/leaderboard_page), dont l'HTML est conservé dans `remote_evidence.json` et la ligne SLZ dans `leaderboard.json`.

Je n'ai retrouvé **aucun reçu, score practice enregistré ou résultat reproduit établissant un score de 0,895** dans les fichiers Task 2 accessibles. Il ne faut donc pas analyser la situation comme un plateau à 0,895. Les valeurs proches trouvées pendant l'audit sont des résultats de sélection de features ou des AUC, et ne constituent pas ce score. Le classement ne donne que les meilleurs résultats : son ancien horodatage **ne prouve pas qu'aucune tentative ultérieure n'a été soumise**.

**Le travail existe, mais les progrès du code et les nombreux essais sur références n'ont pas encore fourni une méthode fiable sur les endpoints notés.** Le blocage principal cumule :

- très peu de données notées visibles : 5 réponses par endpoint, aucune répétition ;
- des distances qui réagissent aux déguisements et au décodage, alors que ceux-ci ne changent pas l'identité ;
- une sélection de features qui sur-apprend les références ;
- des partitions de familles incohérentes dans les derniers CSV ;
- l'absence de lien traçable entre commande, CSV, reçu et score public.

Cela ne permet pas d'attribuer le blocage à une personne. Le dossier partagé contient les fichiers de Sophie (`pelet1`) et les collectes auxiliaires de `mouhali1`, avec plusieurs versions et plusieurs journaux.

### Ce que demande réellement la tâche

La docstring du vrai `task_template.py`, relue sur JURECA et conservée dans `runner_metadata.json`, confirme :

- 33 endpoints, donc 528 paires ;
- « même modèle » signifie même checkpoint, quantification et LoRA ; watermark, prompt système et décodage ne changent pas ce label ;
- famille = **même modèle de base**, ce qui ne se réduit pas nécessairement à la marque Qwen ou Llama ;
- score : 10 % MCC famille sur toutes les paires, 90 % MCC modèle sur les paires d'une même famille ; une partie des paires est publique et le reste sert au classement final ;
- 100 réponses par endpoint noté, 256 tokens en sortie, 1 024 en entrée, quatre requêtes simultanées au maximum ;
- les neuf références sont séparées des endpoints notés et ne sont pas scorées au serveur.

Sur les références, les cinq endpoints `01/03/05/06/08` sont le même modèle malgré les déguisements ; `02` est fp8 ; `04/07/09` sont des LoRA de forces 0,7/1,5/0,3. **Tous appartiennent à la même famille.** Le score practice mesure la décision modèle sur 36 paires. Il ne faut pas le comparer directement au total public sur les endpoints notés. Le MCC famille dégénère sur ces références toutes de même famille ; le total imprimé par notre helper n'est pas l'indicateur pertinent ici.

### Inventaire chronologique du travail retrouvé

| Étape | Travail établi | Ce que cela prouve |
|---|---|---|
| 16:47–16:48 | `baseline_random.csv` créé à 16:47 ; meilleur public à 16:48 | Proximité temporelle compatible avec la baseline. Aucun reçu ne prouve que ce fichier a obtenu 0,20. |
| 16:59–21:11 | Premier pilote de 32 prompts, puis seconde vague de 12 prompts ; journaux `c2run/` | Corpus final complet de **44 × 4 × 9 = 1 584 réponses uniques**, avec doublons supplémentaires. |
| Vers 17:05 | Sondes API sur ref : validation de schéma, routes, paramètres | Pas de logprobs exploitables démontrés. Les essais n'établissent pas une nouvelle méthode de fingerprinting. |
| 17:20, commit `1f2c907` | Plan, n-grammes/seq/Hamming, correction de dispersion, UPGMA, générateur synthétique | Pipeline hors ligne et CSV fonctionnels ; les scores synthétiques ne prédisent pas les performances réelles. |
| 19:01, commit `a0a94f7` | Features : mots, nombres, listes, ordre, longueur, divergence ; corrections d'affixes, données manquantes et validation | Correctifs présents dans les références Git récentes, absents du checkout `main` courant. |
| 19:25 | Phase 1a achevée sur 33 endpoints, `f01..f05` × une réponse | **165 réponses notées**, complètes, sans doublon ni réponse vide. |
| 19:50–19:53, commits `540e7ff` et `7e5027b` | Règle « gros groupe + isolés », analyse de longueur, plan 1b réduit, `ep1b.py` préparé | Une piste et un runner existent. Le journal noté accessible ne contient pas leurs nouvelles réponses. |
| 19:51 | `c2_lump_v1.csv` | Partition Qwen/Llama avec `ep-02`, `ep-17`, `ep-31` isolés comme modèles. Score public associé inconnu. |
| 21:56 | `reference_answers.jsonl`, propriétaire `pelet1` | 225 réponses : 25 prompts, une seule répétition par endpoint. |
| 22:35 | `ref_pilot_minimal.jsonl` et six CSV de références | 5 prompts × 4 répétitions × 9 ref = 180 réponses ; métriques reproduites ci-dessous. |
| 22:36 | `ep_first_hamming.csv`, `ep_first_ngram.csv` et matrices de distances | Nouveaux candidats accessibles, avec des partitions très différentes. Aucun reçu retrouvé. |

Ces heures sont les horodatages des fichiers, des logs ou des commits selon la ligne ; un mtime ne prouve ni la commande exacte ni une soumission. Les traces plus anciennes parlaient de 0 requête et d'aucune soumission : ces phrases du README sont périmées.

### Constat P1 — Les endpoints notés n'ont pas les répétitions requises

`c2run/ep/ep_answers.jsonl` contient exactement **5 réponses pour chacun des 33 endpoints**, uniquement `repeat=0`. Deux prompts concernent l'identité déclarée et la date de connaissances ; les trois autres testent la copie d'une chaîne, le comptage de lettres et une liste multilingue.

Dans `analyze.py`, la statistique de correction de dispersion n'est utilisée qu'avec au moins deux réponses par endpoint. Avec ces données, le programme retombe sur une distance simple entre textes. **Choisir `--stat corr` ne permet donc pas d'estimer la variabilité du décodage sur cette collecte.** Un texte différent ne suffit pas à conclure à un checkpoint différent.

Le runner 1b existe et vise 15 prompts × 2 répétitions. Comme `f03/f04` sont déjà partiellement obtenus, son coût théorique est **28 réponses supplémentaires par endpoint, soit 924 au total**. Il n'a pas laissé de résultat dans le journal inspecté. Ce constat n'exclut pas une collecte dans un emplacement privé inaccessible.

**Budget :** les données notées retrouvées documentent 165 réponses **enregistrées**, soit 5 % du budget nominal de 3 300. Ce nombre ne borne pas la consommation totale si des erreurs sont débitées sans réponse enregistrée. Il ne permet donc pas de déduire 95 réponses restantes par endpoint. Je n'ai pas utilisé d'authentification pour lire ces compteurs. Aucun nouveau budget n'a été consommé par l'audit. Un relevé ultérieur est signalé dans le registre de coordination ; voir section 8.

### Constat P1 — Les dernières partitions se contredisent massivement

| Fichier existant | Familles | Modèles | Paires prédites « même modèle » / 528 |
|---|---:|---:|---:|
| `c2_lump_v1.csv` | 2 | 5 | 235 |
| `ep_first_hamming.csv` | 4 | 5 | 379 |
| `ep_first_ngram.csv` | 9 | 32 | 1 |

Hamming et n-grammes se contredisent sur **378 décisions modèle** et 135 décisions famille. Les deux sorties changent donc beaucoup plus que quelques paires ambiguës.

La sortie Hamming fusionne notamment des endpoints déclarés Llama et Qwen dans le même modèle. La sortie n-grammes les mélange aussi dans sa grande famille, tout en déclarant presque chaque endpoint différent comme modèle. Sous réserve de la fiabilité des identités déclarées, ces regroupements contredisent le diagnostic famille préalable. **Le changement de représentation modifie presque toute la réponse au lieu de résoudre les variantes fines.**

Le candidat `c2_lump_v1.csv` respecte le découpage Qwen/Llama proposé précédemment. Cela en fait un point de comparaison traçable, pas un gagnant démontré : les deux éventuelles bases Qwen ne sont pas identifiées et la séparation fp8/LoRA faible n'est pas résolue.

### Constat P1 — Les scores de sélection ne tiennent pas sur les répétitions gardées à part

J'ai rejoué les données figées avec le code corrigé de la branche c2 : top 15 features, vues par défaut, seuil et features choisis sur les répétitions paires puis évalués sur les impaires, et l'inverse. Les règles de retrait des affixes et phrases sont apprises sur la moitié de sélection seulement.

| Corpus | MCC sélection, plis 1 / 2 | MCC évaluation, plis 1 / 2 |
|---|---|---|
| Premier pilote, 32 prompts | 0,721 / 0,877 | **0,116 / 0,000** |
| Seconde vague, 12 prompts | 0,721 / 0,734 | **0,175 / 0,524** |
| Ensemble, 44 prompts | 0,721 / 0,877 | **0,037 / 0,021** |
| Pilote minimal de Sophie, 5 prompts | 0,495 / 0,486 | **0,000 / 0,000** |

**Ajouter les douze prompts n'améliore pas automatiquement le sélecteur : sa sélection combinée sur-apprend encore.** La seconde vague contient un signal partiel, mais son transfert est asymétrique. Ces évaluations restent sur les mêmes neuf endpoints ; même une amélioration stable ici ne prouverait pas un transfert vers d'autres modèles de base.

Sur les 44 prompts, aucune des **342 features** évaluées n'atteint 0,90 d'AUC globale ; le maximum est 0,873. Le critère initial de dix prompts fortement discriminants n'est donc pas rempli par ce corpus et ce jeu de vues. Ces vues issues d'un même prompt ne sont pas des expériences indépendantes.

Une exception trompeuse existe dans le pilote minimal : `m11:numset` atteint **0,905 d'AUC**, mais seulement sur **10 paires exploitables sur 36** : trois paires identiques et sept différentes. Les 26 autres sont manquantes, notamment toutes les paires impliquant la LoRA faible `ref-09`. L'AUC ignore les valeurs manquantes : sans couverture minimale, elle peut donner un classement flatteur à une feature inutilisable sur une grande partie du problème. Preuve détaillée : `partial-feature-support.json`.

### Constat P1 — Les sorties practice enregistrées ne valident pas le clustering

Le scoreur officiel du kit, exécuté sur les six CSV existants, donne :

| CSV de références | MCC practice | Erreurs sur 36 paires |
|---|---:|---:|
| Hamming corr | 0,495 | 12 fusions erronées |
| Hamming MMD | 0,495 | 12 fusions erronées |
| n-grammes MMD | 0,495 | 12 fusions erronées |
| n-grammes corr | 0,273 | 9 séparations erronées |
| séquence corr | 0,037 | 11 erreurs |
| séquence MMD | 0,037 | 11 erreurs |

Les trois premières sorties mettent `ref-04` et `ref-07` **dans une autre famille commune**. C'est doublement incorrect : toutes les références partagent la même famille, et ces deux LoRA sont deux modèles différents. Elles regroupent aussi fp8 et LoRA faible avec le modèle de base.

La calibration du seuil et l'évaluation sur les mêmes références sont optimistes. Les noms `corr` ou `MMD` ne rendent pas une partition valide ; il faut examiner les paires erronées et les regroupements.

### Constat P2 — La longueur fournit un signal partiel, avec des limites déjà mesurables

Sur les **15 prompts déjà fixés** dans le plan 1b, le ratio de longueur sur les références isole `ref-04` et `ref-07` au seuil 1,25, sur les répétitions 0–1 comme sur 2–3. Dans les deux cas : **MCC practice 0,524**, dix vraies paires identiques conservées, onze fusions erronées restantes.

Cela confirme le signal des LoRA fortes sur ces références. Cela **ne revalide pas indépendamment la sélection des 15 prompts ni le seuil** : ceux-ci ont été ajustés à partir des références. Je n'ai pas reproduit dans cet audit une nouvelle sélection de prompts sur chaque moitié. La règle continue à confondre fp8, LoRA faible et base ; les onze erreurs sont listées dans `audit_results.json`.

Sur les cinq prompts notés actuels, les plus grands ratios sont `ep-02` ×2,407, `ep-17` ×2,002 et `ep-31` ×1,944 dans leurs familles proposées. Ce sont des observations à confirmer, pas une preuve de LoRA. D'autres endpoints dépassent 1,25 sur cette seule réponse : appliquer directement la règle prévue pour quinze prompts répétés ajouterait des isolés sans la stabilité demandée.

### Constat P2 — La correction `corr` n'est pas une invariance au décodage

La documentation décrit la distance `1 - Kab/sqrt(Kaa*Kbb)` comme insensible au bruit de température. Cette propriété n'est pas générale : changer la distribution de décodage ne multiplie pas nécessairement les trois similarités par le même facteur.

Contre-exemple analytique conservé dans `dedup_sensitivity.json` : pour deux sorties orthogonales, une distribution de probabilités `(0,99 ; 0,01)` comparée à `(0,5 ; 0,5)` donne une distance **0,286**, même en supposant les mêmes poids. Ce n'est pas une mesure de l'API ; cela réfute seulement l'invariance générale revendiquée. Les références watermark/persona/température doivent donc rester des contrôles négatifs explicites.

### Constat P2 — Journaux et versions empêchent une reproduction exacte des résultats anciens

Le journal ref principal a **1 744 lignes**, mais **1 584 clés uniques** `(endpoint,id,repeat)`. Il contient 160 lignes supplémentaires et **152 clés dont les réponses diffèrent**. `group()` conserve la dernière occurrence, donc l'ordre du fichier influe sur les résultats.

Reproduction sur le premier pilote :

- dernière occurrence : MCC d'évaluation **0,116 / 0,000** ;
- première occurrence : **0,000 / 0,000**, comme les deux valeurs publiées dans les notes antérieures.

Ce n'est pas une corruption prouvée ni une fraude ; c'est une reprise de collecte sans identité de tentative distincte. Les mêmes clés désignent plusieurs réponses stochastiques. La différence explique pourquoi une ancienne métrique peut cesser d'être reproductible à partir du fichier courant.

Les corpus de Sophie et la collecte auxiliaire ont également des clés qui se chevauchent avec des réponses différentes : **173 des 180 clés communes** entre le pilote minimal et le journal auxiliaire sont différentes. **Les concaténer directement ferait perdre des observations** par dédoublonnage. Conserver campagne, tentative et configuration ; ne réindexer les répétitions qu'après vérification de la compatibilité des prompts et conditions.

Trois copies d'`analyze.py` ont des hashes distincts :

| Copie | SHA-256 |
|---|---|
| Checkout `main` courant | `2e5ee6da11fa50e473df74743106b7244c0c68f612391e2aba68ae2d0743fa60` |
| Branche c2 à `7e5027b` | `e2bbb678481f9262eeb0d6a66a9b45642cdbc83d511ec06038c8add4c4d42657` |
| `JURECA:c2run/analyze.py` | `4c86d9d53971ea472a2fbaae532361c732c4f32bff45db83aacf05f663e58c41` |

La dernière copie ne propose même pas le noyau Hamming : elle ne peut pas être la source directe d'une exécution `--kernel hamming`. Le rejeu des paramètres par défaut des autres copies ne reproduit pas non plus les derniers CSV : Hamming diffère sur 61 décisions famille et 56 décisions modèle ; n-grammes sur 26 et 2. Cela **n'établit pas un bug du CSV** : les paramètres réels, le corpus exact ou le chemin du script manquent. Sauvegarder le CSV seul ne suffit pas.

### Constat P2 — Une règle du plan 1b est ambiguë au point de bloquer son application

Le plan propose un 2-means sur les **Qwen non isolés**, puis exige que chaque groupe contienne un isolé fort (`ep-02` ou `ep-17`). Littéralement, ces endpoints sont exclus de la population à regrouper : la condition ne peut pas être satisfaite. Si l'intention est de réaffecter les isolés après l'ajustement, cette étape et sa règle doivent être écrites. Aucune implémentation complète de cette décision n'a été retrouvée.

Par ailleurs, Qwen/Llama sont des indications de lignée, pas une validation du checkpoint de base au sens de la tâche. Le LOO annoncé sur les identités déclarées vérifie la cohérence d'un classifieur avec ces déclarations, pas une vérité indépendante sur les familles officielles. L'hypothèse « 33 = trois groupes de onze » reste une hypothèse ; elle ne doit pas devenir une contrainte de clustering sans preuve.

## 2. Pas fait

- Aucun nouveau prompt ref/ep, aucune sonde API, aucune soumission, aucune écriture distante.
- Pas de correction du pipeline expérimental ni de choix d'une nouvelle architecture : mandat d'audit.
- Pas de score public reconstitué ni de vérité cachée inférée. Aucun gain futur chiffré promis.
- Pas d'accès aux fichiers privés de Sophie : son répertoire personnel JURECA refuse la lecture. Les preuves proviennent du dossier partagé et de `c2run/` accessibles.
- Pas de vérification en direct du budget authentifié ; l'exécution de 1b ailleurs ne peut pas être exclue.
- Pas de rejeu coûteux de toutes les anciennes combinaisons séquence/MMD sur tous les corpus ; les CSV séquence existants ont été scorés avec le kit officiel. Les nouveaux rejeux de matrices couvrent Hamming et n-grammes.

## 3. Preuves

Tout est figé sous ce dossier :

- `remote_evidence.json` : contenu des journaux et CSV collectés, tailles, mtime et SHA-256 ; classement HTML public.
- `runner_metadata.json` : documentation officielle, paramètres non sensibles des runners et fonctions utiles, lignes d'authentification omises.
- `sources/` : copies distinctes du code local, de la branche c2 et de l'ancienne copie distante.
- `data/` : extractions des entrées existantes, pour les rejeux sans réseau.
- `audit.py`, `audit_results.json`, `audit-output.txt` : intégrité, partitions, scores, validations et règle de longueur.
- `compare_versions.py`, `version_replays.json`, `version-output.txt` : reproduction explicite par version et paramètres par défaut.
- `check_dedup.py`, `dedup_sensitivity.json`, `dedup-output.txt` : influence du dédoublonnage et contre-exemple analytique.
- `partial-feature-support.json` : couverture réelle de l'AUC 0,905.
- `tests-main.txt`, `tests-branche-c2.txt` : sorties de tests.

Commandes et sorties réelles :

```text
ssh -o BatchMode=yes -o ConnectTimeout=12 jureca 'hostname; ...'
  jrlogin03.jureca
  lectures des fichiers partagés et c2run réussies
  /p/home/jusers/pelet1/jureca : Permission denied

squeue -u mouhali1,pelet1 ...
  en-tête seul au relevé initial
  (ne renseigne pas les collectes CPU hors Slurm ni les autres nœuds de login)

.venv/bin/python -m pytest -q c2/test_analyze.py
  No module named pytest

python3 -m pytest -q docs/task2-audit-20261121/sources/main/test_analyze.py
  9 passed in 30.00s

python3 -m pytest -q docs/task2-audit-20261121/sources/branche-c2
  37 passed in 47.31s

python3 docs/task2-audit-20261121/audit.py
  pilote1_32_prompts validation [(0.7211102550927979, 0.11602387022306428), (0.8770580193070291, 0.0)]
  pilote2_12_prompts validation [(0.7211102550927979, 0.17541160386140586), (0.7337993857053429, 0.5241424183609592)]
  ensemble_44_prompts validation [(0.7211102550927979, 0.03739787960033829), (0.8770580193070291, 0.020965696734438367)]
  minimal_5_prompts validation [(0.4947274449181537, 0.0), (0.48617243480439776, 0.0)]

python3 c2/kit/score_practice.py <CSV existant>
  Hamming corr/MMD et ngram MMD : MCC 0.495 ; 24 paires correctes, 12 erronées
  ngram corr : MCC 0.273 ; 27 correctes, 9 erronées
  seq corr/MMD : MCC 0.037 ; 25 correctes, 11 erronées
```

Les tests verts vérifient le code sur des exemples synthétiques et construits ; ils n'établissent pas sa précision sur les réponses réelles. L'interpréteur `python3` déjà installé a servi aux tests, sans installation de dépendance.

## 4. Claims pris en défaut ou non établis

| Claim | Résultat de l'audit |
|---|---|
| « Task 2 à 0,895 » | Aucun élément accessible ne l'établit ; score public vérifié 0,20. |
| « Budget intact, aucune soumission » du README local | Périmé : 165 réponses notées et un résultat public existent. |
| « Plus de features / prompts améliore la solution » | Pas dans le sélecteur rejoué : 44 prompts donnent 0,037 / 0,021 hors sélection. |
| « La correction corr neutralise la température » | Pas une invariance générale ; contre-exemple explicite. |
| « AUC ≈ 0,905 = bon détecteur » | Seulement dix paires valides sur 36 pour cette feature du pilote minimal. |
| « Les derniers CSV sont des versions voisines » | 378 décisions modèle changent entre Hamming et n-grammes. |
| « Le score antérieur de 0,000 est faux car je retrouve 0,116 » | Non : première occurrence reproduit 0,000 ; le journal contient des reprises conflictuelles. |
| « La phase 1b a forcément déjà alimenté l'analyse » | Aucun résultat 1b dans le journal noté retrouvé ; seul le script existe. |
| « Aucun essai après 16:48 » | Impossible à établir : le classement masque les scores inférieurs au meilleur et les reçus manquent. |

## 5. Décisions laissées à Opus / Selyan

1. Retenir la règle de longueur conservatrice comme prochaine référence expérimentale, ou poursuivre la recherche d'un signal distinct pour fp8/LoRA faible ? Le signal de longueur est partiel ; il ne justifie pas une promesse de score élevé.
2. Autoriser le lot 1b de 924 réponses après rapprochement avec les compteurs actuels, ou adapter le lot aux données déjà collectées par Sophie ailleurs ?
3. Conserver provisoirement le découpage Qwen/Llama comme hypothèse de travail, en laissant ouverte l'existence de plusieurs checkpoints de base Qwen ?

## 6. Risques

- Dépenser de nouvelles requêtes avant d'avoir rapproché les journaux pourrait dupliquer un travail invisible et consommer un budget irréversible.
- Choisir le meilleur score calibré parmi des centaines de features masque les erreurs sur les déguisements ; le public partiel peut aussi favoriser une partition par hasard.
- Réutiliser un seuil de longueur d'un petit modèle ref sur d'autres bases peut isoler une persona ou une stratégie de décodage plutôt qu'un LoRA.
- Concaténer les journaux sans campagne écrase des réponses distinctes ; retoucher silencieusement un journal invalide les métriques anciennes.
- Les sources ont évolué dans plusieurs emplacements ; le SHA du dépôt seul ne décrit pas le fichier réellement exécuté.
- Ces fichiers d'audit sont locaux et non commités. Aucun envoi externe ou Git n'a été effectué.

## 7. Actions humaines et reprise concrète

**Ordre proposé pour reprendre, sans choisir une nouvelle architecture :**

1. **Rapprocher l'historique des soumissions.** Pour chaque essai déjà effectué, réunir CSV exact, SHA-256, commande, code exécuté, corpus, reçu HTTP, score et `improved`. En priorité : `c2_lump_v1.csv`, puis les deux `ep_first_*`. Le classement public ne suffit pas à raconter les échecs.
2. **Vérifier les compteurs courants et les autres journaux avec Sophie.** Les fichiers accessibles documentent 5 réponses/endpoint ; ne pas traiter 95 restantes comme un compteur confirmé.
3. **Figer un corpus et un seul chemin de code.** Conserver les originaux ; ajouter campagne/tentative ; dédoublonner explicitement ; sauvegarder les hashes. Les tests des corrections c2 sont déjà verts sur la copie auditée.
4. **Ajouter aux contrôles existants trois critères lisibles :** paires couvertes par feature, erreurs par type de déguisement, stabilité entre répétitions. Refuser le mélange connu de familles sans justification vérifiée ; regarder les partitions avant tout score scalaire.
5. **Résoudre l'ambiguïté du plan 1b sur les isolés et les deux bases Qwen.** Décrire l'éventuelle réaffectation après 2-means ; ne pas imposer trois groupes de onze par simple arithmétique.
6. **Si Selyan autorise le lot 1b après ces vérifications, le lancer une seule fois.** Le script et les prompts existent ; comparer ensuite la règle de longueur déjà fixée, ses deux répétitions et les déguisements contrôlés. L'exécution n'est pas déclenchée par cet audit.
7. **Préparer une soumission choisie sur ces contrôles, puis la faire déclencher par Selyan.** Conserver immédiatement son reçu et l'empreinte du CSV. Ne pas multiplier les coupures du clustering pour sonder le public.

Le premier objectif vérifiable est une expérience dont les données, la commande, le fichier et le résultat sont reliés. Le gain de score de cette reprise reste à mesurer.

## 8. Complément — signalement de 502 débités et retries du kit

### Nouvelle information et portée

Selyan a transmis un échange des organisateurs avec l'équipe Murmell. Je traite ce texte comme un **signalement fourni**, sans accès au fil original ni aux traces HTTP de cette équipe. Selon Murmell :

- sur `ref-01`, à 20:25 Paris, le prompt demandant le seul mot OK réussit avec `max_tokens=8` en 9,4 s, mais retourne un 502 vide vers 15,04–15,06 s avec 128 ou 256 ;
- entre 19:22 et 19:54 Paris, dix-huit erreurs 502 sur leurs endpoints notés auraient augmenté `budget.used` sans fournir de réponse ;
- à 20:34, les organisateurs disent vérifier ; à 23:11, ils mentionnent le timeout de 180 s du fichier de requêtes et la charge serveur.

**La réponse sur les 180 s ne démontre pas que le problème des 502 à 15 s a été corrigé.** Le timeout du client est une limite d'attente locale ; il n'empêche pas un intermédiaire ou le serveur de renvoyer un 502 plus tôt. La régularité autour de 15 s est compatible avec une limite intermédiaire, mais sa localisation exacte n'est pas démontrée ici. Aucune annonce de remboursement, de correction ni de valeur `max_tokens` garantie n'est contenue dans l'échange transmis.

### Défaut confirmé dans les sources déjà archivées

Dans `runner_metadata.json`, le vrai `task_template.py` et sa copie `c2run/ep/ep1b.py` ont la même logique pertinente :

- configuration `MAX_TOKENS=256` ;
- lecture du budget au début de `query()` ;
- POST avec `timeout=180` ;
- sur 502/503/504 : attente de 15 s, puis nouvelle tentative dans la même boucle, **sans plafond de tentatives** ;
- sur exception réseau : attente de 30 s et nouvelle tentative, également sans plafond ;
- seules les réponses HTTP 200 sont écrites dans le JSONL ; les erreurs ne reçoivent pas de trace structurée contenant endpoint, prompt, tentative, heure et budget avant/après ;
- le budget initial sert à limiter le nombre de travaux planifiés, **pas le nombre total de tentatives HTTP**. Il n'est pas relu après un 502.

Repères du script original `ep1b.py` : début de `query` ligne 92, budget initial ligne 99, boucle ligne 128, POST ligne 130, branche 502/503/504 ligne 158. Ce sont des numéros de la copie distante au moment de l'archivage, conservés par les métadonnées.

La simulation `reproduce_retry_risk.py` exécute uniquement cette fonction archivée avec une session factice : deux 502 vides puis un 200. Résultat : **3 tentatives HTTP, une réponse enregistrée, une seule lecture du budget**, avec `timeout=180` sur les trois appels. Si les erreurs sont débitées, le coût du scénario est trois unités pour une observation exploitable. Aucun appel réseau ni débit réel n'est effectué par cette reproduction. Sortie : `retry_risk_reproduction.json`.

### Ce que cela change pour SLZ

1. **Correction du vocabulaire budgétaire dans ce rapport :** 165 lignes sont 165 réponses enregistrées. Elles ne prouvent pas à elles seules une consommation de seulement 165 unités. Le témoignage ne prouve pas non plus une perte de budget chez SLZ.
2. Les logs historiques archivés `pilot.log` et `ep1a.log` ne contiennent pas de ligne d'erreur/retry correspondante. Leur absence ne remplace pas un relevé des compteurs : ils ne constituent pas un journal exhaustif des tentatives et ne couvrent pas les autres collectes.
3. Le fichier partagé `docs/SLZ_COORDINATION_20261121.md`, apparu après le relevé initial, **rapporte à 23:25 un GET des compteurs avec 95 restantes sur chacun des 33 endpoints**, soit 3 135 au total. Ce relevé rapporté est cohérent avec les cinq réponses stockées par endpoint et ne met pas en évidence une perte nette de budget noté à cet instant. Je n'ai pas refait ce GET ni récupéré sa réponse brute ; un éventuel remboursement antérieur n'est pas identifiable.
4. Le même registre rapporte désormais `c2_lump_v1.csv` soumis à 22:52 avec **0,05**, et les autres automates arrêtés. Cela complète l'audit initial où aucun reçu n'était accessible ; l'information provient du registre de l'autre session. Elle ne justifie pas, seule, une partition vraie précise en trois familles : un score scalaire bas peut avoir plusieurs causes.
5. Les erreurs de transport pourraient expliquer de la lenteur, des trous de collecte ou des crédits manquants. Elles **n'expliquent pas directement** les 378 désaccords de partition ni le sur-apprentissage mesuré sur les réponses déjà obtenues. Ces constats hors ligne restent valides pour le corpus figé.

### Reprise recommandée avant une nouvelle collecte

- Préparer une copie du runner qui s'arrête sur 502/503/504 et sur une réponse réseau ambiguë, conserve toutes les tentatives et impose un plafond global. Un POST expiré peut avoir été traité côté serveur : le rejouer automatiquement n'est pas une preuve de coût nul. Relire le compteur avant toute reprise ; aucune modification ou exécution distante réalisée ici.
- Demander aux organisateurs la confirmation de la correction du timeout intermédiaire et du traitement budgétaire des erreurs. Le message déjà transmis ne fournit pas cette confirmation. Aucun message envoyé par cet audit.
- **Ne pas généraliser `max_tokens=8` comme réglage sûr.** C'est un seul succès rapporté sur une référence et un prompt. Abaisser la limite modifie aussi les réponses, leur troncature et les features de longueur/récitation : toute calibration correspondante doit être refaite sur des références avec les mêmes paramètres, en conservant `max_tokens`, `finish_reason`, durée, statut et campagne.
- Rapprocher les compteurs de chaque endpoint avec les tentatives et réponses stockées, en gardant séparés crédits perdus, observations reçues et éventuels remboursements.

Ce complément documente un défaut concret du client de collecte, sans affirmer que le serveur est encore affecté ni que SLZ a subi les dix-huit erreurs rapportées par une autre équipe.
