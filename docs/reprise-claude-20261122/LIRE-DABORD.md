# Reprise Claude — 22 novembre 2026, nuit

**Approfondissement demandé ensuite par Selyan : lire aussi [ANALYSE-PROFONDE](ANALYSE-PROFONDE.md).** Nouveaux contrôles : filiation des huit assemblages T1, provenance des 12 000 synthétiques, concentration du gain 6k, condition d'égalité omise par le filtre T2, poids exacts et dérive des distributions T3. Le snapshot 03:17 affiche 27e/37, général 0,319108, avec nos scores inchangés.

**Point vérifié à 03:05–03:07 Paris : T1 0,67390686 ; T2 0,20 ; T3 0,09411482 ; général 0,31968303, 26e/37.** Le général dépend des autres équipes et ne vaut pas la moyenne brute des trois scores.

**Actualisation 03:14 : job 15904703 (`j12`) RUNNING sur quatre GPU, lancé par le relais après GO Selyan à 03:11:10.** Il complète les folds manquants 3k/6k et la graine test 62. Ne pas le relancer. Le point « file vide » ci-dessous décrit uniquement 03:07. Résultats encore attendus, aucun nouvel envoi au classement constaté dans le relais.

## Les fichiers à reprendre

| Tâche | Champion public | Identité vérifiée |
|---|---|---|
| T1 | `output/c1_ensE_ssall3.pt`, soumis à 02:43:05 Paris, 0,6739 | SHA `f00fdc039718b29fabe64cf2155e6967b89b12c0986f331bd1d3ee7304418c80` |
| T2 | `output/baseline_random.csv`, 0,20 | SHA `54f021d8504f5d27b77eecbf6403db090d55d7d08631bcfa972b90f99922b03c` ; le dernier CSV documenté `c2_fam3_pairs_v2.csv` a fait 0,10 |
| T3 | `work_c3_eval/push_scale/push_s175.pt`, 0,0941 | SHA `54c2aeccbf68965890be1818172d0c48f89f24a6a15251b0bade6fe34dcb464f` |

Chemins distants : `/p/scratch/training2666/mouhali1/SLZ/{Hackathon_Watermark_Localisation_Task,model_equality,unlearning}/`.

## Ce qui a été achevé pendant le relais Codex

- **T1 IoU2, cinq folds : +0,003381**, IC95 descriptif `[+0,000347 ; +0,006574]`. DIAGNOSTIC. Comparaison entre anciennes recettes `4k+IoU2` et `4k+4k` : **ce gain n'est pas mesuré contre le champion ssall3**.
- **T1 6 000 pas : candidat local prêt, non soumis.** `/private/tmp/eaic-relais-20261122/c1_ensF_ssall6k2.pt`, SHA `39a3fb2d5ff9f3b31c44b51886bcebc2ad21e446e7b0825d327ab0c186b4bc7c`. StegaStamp seulement, deux graines test 60/61 ; 680 autres masques conservés. Sur deux folds, une graine CV par recette : SS `0,8337385 → 0,8566075`, contribution globale estimée `+0,007623`. **Pas de score public ni de validation cinq folds de cette recette à ce stade.** Fichiers et script de validation dans `docs/relais-codex-20261122/` ; l'autre chat continue ce travail, le relire avant toute relance.
- **T2 sondes : ÉCHEC.** 11 sondes réelles, 990 tentatives refs = 989 réponses + une 503 ; MCC `0/0` sur les deux moitiés, tous les endpoints regroupés. Aucune sonde A3 ne satisfait le filtre strict sur les 10 000 prompts. **Ne pas lancer le lot noté à partir de ce résultat.** Journaux notés attendus absents ; budget frais non relu par cet audit. Dernier compteur documenté : 95/endpoint.
- Derniers jobs 15904551, 559, 561, 581 terminés ; file équipe vide à 03:07. Le job 15904543 a échoué sur une ligne de configuration mal formée. Aucun GPU utilisé par cet audit.

## Avant de reprendre les actions

1. Lire le dernier état du chat **« Suivre les jobs GPU et sondes »** et `docs/relais-codex-20261122/` : c'est le relais de travail, avec le job 15904703 en cours. Cet audit ne lui a envoyé aucun ordre.
2. Relire le classement et `squeue` ; les chiffres ci-dessus sont un instantané. Vérifier la validation cinq folds 6k avant de présenter son gain comme confirmé. Ne pas resoumettre E ni reconstruire ses trois graines.
3. Conserver les champions et les candidats distincts. Les envois restent une action de Selyan selon les consignes de ce poste. La règle de sélection finale meilleure publique/dernière reste à confirmer auprès des organisateurs.
4. Reprendre Git dans une branche dédiée après examen des changements. Checkout partagé `main`, HEAD `f209e20397736c53824a7a94ec73f2bd66d012eb`, sale, sept commits derrière la référence locale `origin/main`. Pas de pull/reset/stash aveugle. Un lot d'audit est déjà committé dans `.claude/worktrees/audit-global`, branche `docs/audit-global-nuit`, SHA `38e69bb`.

## Rapport et preuves

[Audit détaillé](RAPPORT-CODEX.md) · [27 lignes de soumissions reconstituées](soumissions-reconstituees.csv) · [37 jobs du compte](jobs-verifies.csv) · [classement exact](classement-verifie.json) · [empreintes des preuves](MANIFESTE.json).

Les registres historiques restent incomplets : des envois ont un nom inconnu ou un reçu manquant. Le lien fichier ↔ 0,6391 demeure non établi. Le fichier champion E à 0,6739 est, lui, relié au reçu transmis et au classement.
