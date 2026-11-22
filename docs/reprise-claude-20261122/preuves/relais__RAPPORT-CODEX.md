# Relais Codex — 22 novembre 2026, 03:08 Paris

**Mise à jour après GO Selyan, 03:12 :** le lanceur préparé ci-dessous a été transféré, son SHA vérifié, puis exécuté une seule fois après contrôle immédiat de la file d'équipe vide. **Job 15904703 (`j12`), démarré à 03:11:10, RUNNING à 03:11:57 sur jrc0256, quatre GPU de dc-gpu-devel.** Les quatre journaux de tâches existent et présentent les configurations prévues. Limite 40 minutes ; durée attendue de l'ordre de 20–25 minutes, à confirmer par les calculs. L'autorisation explicite de ce lancement remplace l'attente de lancement manuel décrite au point de 03:08. Preuve structurée : `launch_15904703.json`. Aucun appel de soumission au serveur de notation ni collecte ep T2. Les résultats d'entraînement et la mesure cinq folds restent à terminer.

## Résultat

Le champion Task 1 est **0,6739**, résultat fourni par Selyan pour `c1_ensE_ssall3.pt`. Son SHA256 est vérifié sur JURECA et sur la copie locale : `f00fdc039718b29fabe64cf2155e6967b89b12c0986f331bd1d3ee7304418c80`.

Les jobs de la nuit sont terminés. La mesure IoU2 sur cinq folds reste un diagnostic, la validation Task 2 échoue, et le candidat Task 1 à 6 000 pas est construit localement. **Au point de 03:08 : aucune soumission, requête API, écriture distante ou relance GPU par ce relais.** Le lancement ultérieur autorisé est consigné en tête du rapport.

## État du dépôt et de l'accès

- Checkout : `/Users/selyan/dev/eaic-paris-2026`.
- HEAD au départ et avant remise : `f209e20397736c53824a7a94ec73f2bd66d012eb`, branche `main`, derrière `origin/main` de 7 commits.
- Arbre déjà sale : `c1/seg/gen_f.py`, `c1/seg/train_f.py`, `docs/experiences.md` modifiés ; nombreux dossiers d'audit et fichiers non suivis. Aucun de ces fichiers source modifiés par ce relais. Aucun Git d'écriture, commit, changement de branche ou nettoyage.
- `relay-check.py` : pas de pair détecté, mais `ps` indisponible ; la concurrence n'est donc pas exclue.
- Au relevé final, un dossier non suivi `docs/reprise-claude-20261122/` est apparu, avec une collecte de preuves datée de 03:07–03:08. Il n'a pas été modifié par ce relais ; cette apparition confirme que le dépôt peut être utilisé simultanément malgré l'absence de pair détecté.
- La connexion initiale arrivait sur `jrlogin02`, où seuls `/p/home` et `/p/hpcmon` étaient montés. Les données n'étaient pas supprimées : `/p/project1` et `/p/scratch` n'étaient simplement pas montés sur ce nœud.
- Selyan a authentifié `jureca03.fz-juelich.de`, socket partagé `/private/tmp/eaic-codex-jrlogin03.sock`. `hostname` confirme `jrlogin03.jureca` ; les deux espaces de données y sont accessibles. Aucun secret consulté.

## Jobs terminés — preuve Slurm

Commande exécutée via SSH : `sacct -j 15904559,15904561,15904581,15904551 -n -o JobID,State,ExitCode,Elapsed`.

```text
15904551      COMPLETED      0:0   00:05:54
15904551.0    COMPLETED      0:0   00:05:33
15904559      COMPLETED      0:0   00:18:38
15904559.0    COMPLETED      0:0   00:18:18
15904559.1    COMPLETED      0:0   00:00:03
15904561      COMPLETED      0:0   00:05:35
15904561.0    COMPLETED      0:0   00:05:15
15904581      COMPLETED      0:0   00:10:03
15904581.0    COMPLETED      0:0   00:09:44
```

Fins des jobs : sondes A à 02:34:05 ; IoU2 à 02:53:07 ; SS 3k à 02:40:37 ; SS 6k à 02:50:55. Le relevé `squeue -u mouhali1,pelet1,zhang95` à 03:07 n'affiche que l'en-tête : **aucun job actif ou en attente pour ces trois comptes**.

## Task 1 — verdict IoU2

Sorties des folds 2–4 récupérées depuis `/p/project1/training2666/mouhali1/t1sr/`. Les folds 0–1 et les masques train étaient déjà présents sur le Mac.

Recette conservée : moyenne à parts égales des probabilités 4k et IoU2 ; seuil strict `> 0.5`, stockage float16 reproduit ; comparaison à 4k + autre graine 4k. Les différences hors StegaStamp sont nulles puisque la même règle s'y applique. La métrique utilisée est `c1/metric.py` ; aucun recalcul d'inférence ou apprentissage.

Contrôles : 510 noms uniques, ordre exact des noms et folds entre fichiers, aucun chevauchement de couverture, cinq folds complets, probabilités finies dans [0,1]. Le calcul sur les folds 0–1 reproduit **exactement** la différence des scores enregistrés par Claude : `0.004881512724812841`, écart `0.0`.

| Groupe | N | Gain moyen | IC95 bootstrap apparié, 2 000 tirages, graine 0 |
|---|---:|---:|---|
| Global, cinq folds | 510 | **+0,003381** | [+0,000347 ; +0,006574] |
| StegaStamp F | 85 | +0,007779 | [−0,002558 ; +0,017351] |
| StegaStamp R | 85 | +0,012507 | [−0,003070 ; +0,028338] |

Gains globaux par fold : `+0.008107`, `+0.001656`, `-0.001940`, `-0.001536`, `+0.010618`.

**Règle préenregistrée : DIAGNOSTIC**, donc décision Selyan si cette piste doit être poursuivie. Le gain est inférieur à +0,01. L'IC est descriptif et ne corrige pas les essais précédents ni la dépendance entre modèles des folds.

Cette comparaison porte sur l'ancienne famille 4k. **Elle ne prouve aucun gain face au champion SS synthétique toutes méthodes à 0,6739.** Aucun candidat IoU2 construit à partir de l'ancien champion.

Preuves et code : `iou2_f01.json`, `iou2_5fold.json`, `compare_cv.py`.

## Task 1 — candidat SS 6k préparé

Le job 15904581 a produit deux fichiers CV, folds 0 et 1, et deux prédictions test, graines 60 et 61. Configuration identique à `psyn_lr1_all`, hormis 6 000 pas au lieu de 3 000 : ConvNeXt tiny, crop 448, batch 16 dont 50 % réel, LR 1e-4, synthétique des trois méthodes, BCE 1 + IoU 1. Seuil conservé à 0,5.

Comparaison appariée d'une graine CV par configuration sur les 204 images couvertes, dont 68 StegaStamp :

| Groupe SS | 3k | 6k | Gain |
|---|---:|---:|---:|
| F + R | 0,833738 | **0,856608** | +0,022869 |
| F | 0,863760 | 0,873437 | +0,009677 |
| R | 0,803717 | 0,839778 | +0,036061 |

Si seules les images SS changent, gain dilué sur ces deux folds : **+0,007623**, IC95 descriptif [+0,001517 ; +0,014331]. Les deux folds sont positifs pour SS : +0,015798 / +0,029940 avant dilution.

Limites : deux folds seulement ; une graine CV de chaque côté ; le champion test est une moyenne de trois graines 3k et le candidat test une moyenne de deux graines 6k. Cette comparaison ne mesure donc pas directement le gain exact de ces deux ensembles et ne permet pas d'annoncer un futur score public. Les scores PS/TM du réseau 6k ne justifient pas leur remplacement : la comparaison ne porte pas sur la recette actuelle du champion pour ces méthodes.

Fichier préparé :

```text
/private/tmp/eaic-relais-20261122/c1_ensF_ssall6k2.pt
sha256 39a3fb2d5ff9f3b31c44b51886bcebc2ad21e446e7b0825d327ab0c186b4bc7c
taille 16735080 octets
```

Construction : partir du champion vérifié ; moyenne float32 des probabilités test s60/s61 ; remplacer seulement ses 340 images StegaStamp avec le seuil strict `> 0.5`. **680 autres images conservées bit à bit.** Les trois prédictions s50/s51/s52 ont d'abord reproduit exactement les masques SS du champion. Le candidat change 0,4543 % des pixels globaux, 1,3629 % des pixels SS, sur 330 images. Format rechargé avec `weights_only=True` : 1 020 noms uniques, tenseur bool de forme (1020,128,128), taille inférieure à 25 Mo. Fichier présent seulement sur le Mac, non transféré et non soumis.

Preuves et code : `compare6k.json`, `compare6k.py`, `c1_ensF_ssall6k2.json`, `build_candidate.py`. Le script livré `compare6k.py` a été réexécuté ; `cmp` avec le premier JSON : code retour 0.

## Task 2 — NO-GO

Le journal distant `ties_refs.jsonl` contient **990 tentatives, 989 réponses HTTP 200 et un HTTP 503** (ref-01, sonde 2, répétition 0). Le fichier contient 11 sondes, 7 égalités et 4 bascules : **990 était le bon nombre de tentatives**, les annonces 12 sondes / 1 080 requêtes étaient inexactes pour cet artefact.

La règle figée `ties_rule.py analyze` classe les neuf références dans BASE. MCC : **0,000 au total, 0,000 sur les répétitions paires, 0,000 sur les impaires**. Le seuil requis était 0,60 sur chaque moitié. Le fp8 et les LoRA ne sont pas séparés. Le test échoue.

Le premier lancement de l'analyse hors ligne a calculé les résultats mais échoué lors de l'export JSON, car `numpy.bool_` n'est pas sérialisable. Correctif **dans la copie temporaire uniquement** : `bool(G <= thr)` dans le champ exporté. La règle numérique et ses seuils sont inchangés. Second lancement : code retour 0, résultats dans `ties_result.json`.

Recherche famille A : filtre strict figé appliqué aux quatre pools du job 15904551, **0 sonde d'égalité stricte sur 10 000 prompts** (0/2 500 dans chacun des pools s11/s13/s17/s19). Chaque pool contient aussi 60 bascules fp8 ; elles ne valident pas la règle d'égalité stricte. Preuve : `A3_strict_counts.json`.

**Aucun lot ep lancé**, aucune réponse notée consommée par ce relais. Pas de recalibration des seuils après lecture des résultats. Les budgets serveur n'ont pas été relus par API pendant ce relais : seul le fait de ne pas avoir appelé d'API est établi ici.

## Suite concrète préparée et actions humaines

`validate_ss6k.sbatch` est prêt, **non lancé**. Il conserve les configurations existantes, complète les folds 2–4 des références 3k et 6k et utilise le quatrième GPU pour une troisième graine test 6k, s62. Les sorties vont sur `/p/project1/.../t1sr/`, dont le dossier existe déjà. Vérification `bash -n` : code retour 0. Le script n'a pas été testé par un lancement Slurm ; sa durée réelle n'est donc pas mesurée. Les 6k déjà lancés donnent environ dix minutes par fold ou refit.

Répartition proposée dans ce seul job de quatre GPU : GPU 0, CV 6k folds 2–3 ; GPU 1, CV 6k fold 4 ; GPU 2, CV 3k folds 2–4 ; GPU 3, refit 6k graine 62 et prédictions test. Avant lancement, Selyan vérifie la file d'équipe et les règles de partition applicables à cet instant. Pas de nouvel entraînement PS/TM spécifique ni de nouvelle méthode proposés.

La consigne du poste indique : « Pas de déploiement […] pas d'appel qui écrit sur un système externe. Tu prépares, Selyan déclenche. » Les copies vers JURECA et `sbatch` restent donc à déclencher par Selyan. La collecte Task 2 reste NO-GO même si les GPU sont disponibles. Le candidat SS 6k reste une proposition à validation partielle ; aucune recommandation de soumission immédiate n'est déduite de ces deux folds.

Commande Mac de transfert du lanceur, **non exécutée** :

```bash
scp -o HostName=jureca03.fz-juelich.de -o HostKeyAlias=jureca.fz-juelich.de \
  -o ControlPath=/private/tmp/eaic-codex-jrlogin03.sock \
  /Users/selyan/dev/eaic-paris-2026/docs/relais-codex-20261122/validate_ss6k.sbatch \
  jureca:/p/project1/training2666/mouhali1/t1sr/relay_validate_ss6k.sbatch
```

Depuis le terminal JURECA, après le relevé d'équipe et la décision de lancement, **non exécuté par Codex** :

```bash
cd /p/scratch/training2666/mouhali1/SLZ/Hackathon_Watermark_Localisation_Task
squeue -u mouhali1,pelet1,zhang95
sbatch /p/project1/training2666/mouhali1/t1sr/relay_validate_ss6k.sbatch
```

Après le job : récupérer les OOF et mesurer sans changer la recette ni le seuil. Si un candidat à trois graines est ensuite construit, conserver le fichier à deux graines et produire un nouveau manifeste ; ne pas réutiliser son SHA pour une autre recette.
