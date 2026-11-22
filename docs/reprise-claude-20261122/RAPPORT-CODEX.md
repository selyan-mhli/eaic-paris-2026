# Audit de reprise SLZ — 21–22 novembre 2026

> **Complément technique après demande d'approfondissement :** [ANALYSE-PROFONDE](ANALYSE-PROFONDE.md). Les preuves supplémentaires sont sous `profondeur/` : sources réellement exécutées, filiation des masques, 24 tranches synthétiques, recalcul de robustesse du 6k, défaut du filtre T2 et comparaison des poids/caches T3. Le présent rapport conserve les instantanés historiques avec leurs heures.

Rapport destiné à Selyan et à Claude. Relevés directs le **22 novembre, vers 03:05–03:07, heure de Paris** ; rédaction après ces relevés. Mandat : reconstituer le travail, les envois et les résultats pendant les interruptions de quota, et permettre une reprise sans perte de contexte. Le dossier couvre l'après-midi du 21 et la nuit du 22, avec un focus sur le dernier passage de relais après le résultat T1 à 0,6739. L'heure précise de la première rupture de crédits Codex n'est pas établie ; le dernier échange transmis annonce une reprise Claude à 03:51.

**État officiel confirmé : T1 0,67390686 ; T2 0,20 ; T3 0,09411482 ; général 0,31968303, 26e sur 37 équipes classées.** La progression récente est principalement T1 ; aucune nouvelle méthode T2 n'a franchi le critère de validation prévu. Un challenger T1 à 6 000 pas a été préparé par le relais actif, sans envoi confirmé.

**Actualisation 03:14 : le relais a reçu un GO Selyan et lancé le job 15904703 (`j12`), démarré à 03:11:10, quatre GPU de devel.** Lecture SSH finale : RUNNING. Ce job complète la comparaison 3k/6k sur cinq folds et la troisième graine test 62 ; résultats encore à lire. Il remplace l'attente de lancement décrite dans le snapshot initial. **Ne pas le dupliquer.** La preuve de lancement est dans `docs/relais-codex-20261122/launch_15904703.json`, la relecture SSH dans `preuves/jureca-actualisation-0314.json`.

## 1. Sources, portée et degré de preuve

Sources lues : README, consignes fournies, historique Git et worktrees, rapport T1 racine, rapports T2/T3, journal de coordination, expériences, ancien registre temporaire et son export dashboard, données structurées du relais Codex, cinq chats Codex pertinents et reçus transmis dans ces chats. Accès SSH en lecture seule à `jrlogin03`, uniquement à des fichiers nommés de résultats et aux informations Slurm. Classement public téléchargé sans authentification.

Trois degrés de preuve sont distingués :

- **Confirmé directement** : classement exact, état Slurm, existence/hash des artefacts, compte des réponses du journal, reçu texte conservé ou transmis.
- **Rapporté** : ligne d'un registre ou compte rendu d'une session, sans reçu individuel retrouvé. Le score d'un essai non meilleur n'apparaît pas au classement public courant.
- **Estimation/diagnostic** : CV, bootstrap, simulation ou hypothèse de mécanisme. Aucun de ces chiffres ne devient un score public par conversion.

Le CSV reconstitué comporte **27 lignes historiques**, dont des doublons et des artefacts inconnus. Ce n'est pas un export exhaustif du serveur. Une absence dans un chemin attendu ne prouve pas qu'aucun autre script n'a jamais tourné ailleurs. Les lectures de chats antérieurs restent partielles pour les deux chats ayant un curseur de pagination ; les conclusions actuelles reposent surtout sur les preuves de fichiers et le journal consolidé.

Cet audit a écrit uniquement ce dossier et un renvoi documentaire dans le rapport racine. **Aucun envoi, aucune requête LLM notée, aucun entraînement, aucun sbatch/scancel, aucune modification distante, aucun commit/push.** Aucun fichier d'authentification lu. L'autre chat actif fait son propre travail ; ne pas attribuer ses actions à cet audit.

## 2. Situation exacte au classement

| Tâche | Score exact SLZ | Rang | Dernière amélioration, Paris | Artefact associé |
|---|---:|---:|---|---|
| T1 | 0,6739068609009466 | 20e/42 | 22/11 02:43:05 | `c1_ensE_ssall3.pt` |
| T2 | 0,2000000000000000 | 25e/39 | 21/11 16:48:05 | `baseline_random.csv`, attribution du registre |
| T3 | 0,09411482372784334 | 30e/42 | 22/11 00:40:19 | `push_s175.pt`, attribution du registre |
| Général | 0,31968302840227303 | 26e/37 | instantané de la page | calcul normalisé |

Source officielle : [classement EAIC](http://34.147.229.55/leaderboard_page), copie `preuves/leaderboard-0308.html` **capturée vers 03:05** (le nom du fichier n'est pas son horodatage de capture).

Le général a été **recalculé indépendamment**, avec les minima/maxima du même instantané : T1 `[0,49129971 ; 0,80327448]`, T2 `[0,05 ; 1]`, T3 `[0 ; 0,43606466]`. Nos scores normalisés sont `0,58532667 / 0,15789474 / 0,21582768`, dont la moyenne vaut exactement `0,31968303`, à moins de 1e-12 près.

Cela explique pourquoi un ancien « général 0,33 » peut être supérieur au général actuel malgré le progrès T1 : le tableau et sa normalisation ont changé. Les annonces « 0,93 overall », « 0,895 T2 » ou « 0,85 » ne disposent ici d'aucune preuve comme scores SLZ ; ne pas les reprendre pour l'équipe.

## 3. Envois reconstitués

Heures de Paris. Les valeurs arrondies ci-dessous viennent des registres, sauf précision. Les détails, auteurs rapportés, préfixes de hash et degré de preuve sont dans `soumissions-reconstituees.csv` et `.json`.

### T1 — progression publique réelle

| Date/heure | Fichier | Public | Statut de preuve |
|---|---|---:|---|
| 21/11 18:22 | `seg_fsynth_rect_v1.pt` | 0,5307 | registre |
| 21/11 19:12 | même fichier, SHA `302ee55e…` | 0,5307 | doublon rapporté |
| 21/11 19:53 | `c1_segf_ffm_e20_20261121.pt` | 0,5572 | registre ; identité d'artefact auditée antérieurement |
| 21/11 20:05 | `c1_pixss_fit0.pt` | 0,5845 | registre |
| 22/11 00:48 | `c1_pre_ss_cnt2.pt` | 0,6137 | registre ; heure du classement rapportée |
| vers 22/11 01:09 | `c1_pre4k_ens5.pt` | 0,6377 | registre + hash distant revérifié |
| 22/11 01:44 | **fichier non établi** | 0,6391 | ancien classement rapporté ; reçu manquant |
| **22/11 02:43:05** | **`c1_ensE_ssall3.pt`** | **0,67390686** | **reçu utilisateur + classement actuel + hash distant** |

Depuis 0,5845, T1 gagne environ **+0,0894** ; depuis la référence 0,6391 du dernier relais, **+0,0348**. Le fichier E est présent, 16 734 655 octets, SHA complet `f00fdc039718b29fabe64cf2155e6967b89b12c0986f331bd1d3ee7304418c80`. Son mtime est **02:41:10** : l'annonce « prêt vers 02:52 » était une estimation, pas l'heure réelle de production.

Recette E documentée : ConvNeXt Tiny pré-entraîné, mélange réel + synthétique des trois méthodes, remplacement des **340 StegaStamp** avec moyenne des graines 50/51/52 ; autres méthodes conservées. Le relais actif a reproduit exactement la partie SS à partir de ces trois prédictions. Le progrès public valide cet assemblage ; il ne certifie pas automatiquement les nouvelles variantes.

Artefacts intermédiaires vérifiés, **sans score public attribué par déduction** : `c1_ensA9.pt` (`967408de…`), `c1_ensB11.pt` (`f31c98eb…`), `c1_ensC10.pt` (`28d285db…`), `c1_ensD_psyn4.pt` (`5cf39214…`). Leur présence ne suffit pas à dire lequel a produit 0,6391, ni s'ils ont tous été envoyés. Les hashes complets sont dans le snapshot JURECA.

### T2 — trois envois documentés, meilleur inchangé

| Date/heure | Fichier | Public | Lecture |
|---|---|---:|---|
| 21/11 16:48 | `baseline_random.csv` | **0,20** | meilleur encore affiché ; aléatoire du kit selon le registre |
| 21/11 22:52 | `c2_lump_v1.csv` | 0,05 | échec de cette partition, envoi sans GO explicite signalé par coordination |
| 21/11 23:23:41 | `c2_fam3_pairs_v2.csv` | 0,10 | GO explicite rapporté, 3 familles/28 modèles/5 paires |

Hashes distants revérifiés : baseline `54f021d8504f5d27b77eecbf6403db090d55d7d08631bcfa972b90f99922b03c` ; lump `b4ee773a97cd85054e7dde0788a0a4de3f181a432b772561614b85fdad88b089` ; fam3 `6799a4388f736b90626b3680a1ee30546a1382e1f4c4ff7d8135161064ba0f5c`.

**0,10 seul ne prouve pas l'exactitude des familles ou l'inutilité des cinq paires.** Le résultat ultérieur de vraisemblance en boîte blanche apporte une preuve indépendante en faveur des trois familles. Le score privé estimé des CSV reste une simulation. La question « meilleure publique ou dernière soumission retenue en final ? » reste sans réponse officielle retrouvée ; garder le meilleur public ne tranche pas cette question.

### T3 — gains du push, autres essais rapportés

| Date/heure | Fichier | Public | Statut |
|---|---|---:|---|
| ~21/11 17:34 | inconnu | 0,0026 | attribution Yihuan incertaine |
| ~18:40 | `soup_mid.pt` | 0,0012 | registre |
| 18:52 | `soups/soupB_l85.pt` | 0,0035 | registre |
| 19:15 puis 19:26 | `distill_out/m75/distill_r45.pt` | 0,0044 | second envoi doublon rapporté |
| ~19:46 | inconnu | ≤0,0044 | fichier et reçu manquants |
| 19:57 | `anchor_ckpt/anc_f2a_s0550.pt` | 0,0039 | mandat étendu sans GO signalé |
| 22:11 | `push_best.pt` | 0,0308 | Yihuan selon registre |
| 22:57 | `t1t2_p03_e2.pt` | 0,0091 | mandat étendu sans GO signalé |
| 23:07 | `push_s125.pt` | 0,0567 | automate, mandat étendu sans GO signalé |
| 22/11 vers 00:17–00:20 | `push_s150.pt` | **0,0827** | **reçu texte conservé**, heure exacte divergente entre notes |
| **22/11 00:40:19** | **`push_s175.pt`** | **0,09411482** | meilleur confirmé au classement, fichier attribué par registre |
| entre 00:40 et 02:20 | `push_s200.pt` | 0,0792 | registre, heure précise non retrouvée |
| même plage | `clsA_kill_p150.pt` | 0,0205 | registre |
| même plage | `clsF_f125_p175.pt` | 0,0936 | registre |
| même plage | `sep_out/sep_p250_l3_b8.pt` | 0,0317 | registre |

Le s175 est présent (343 629 067 octets), SHA `54c2aeccbf68965890be1818172d0c48f89f24a6a15251b0bade6fe34dcb464f`. Son lien au score vient du registre et du meilleur correspondant au classement ; **le reçu brut individuel s175 n'a pas été retrouvé**. Ne pas présenter tous les essais du tableau comme des reçus revérifiés.

Le message ancien « s150/s200 non soumis » décrit l'annulation des automates à 23:12 ; il ne décrit pas les envois manuels ultérieurs. Le registre temporaire de 23:09 n'a pas été tenu à jour après ce point.

## 4. Dernier relais : ce qui est fini et ce qui reste ouvert

### T1 IoU2 : résultat cinq folds disponible

Job **15904559** : terminé le 22/11 à 02:53:07, durée 18 min 38, quatre GPU alloués. Les OOF folds 2–4 et les prédictions test graines 40/41 sont présents dans `/p/project1/training2666/mouhali1/t1sr/`.

Résultat repris de `docs/relais-codex-20261122/iou2_5fold.json` : ensemble **4k+IoU2 contre 4k+4k**, 510 images, seuil 0,5, poids 0,5. Gain global `+0,00338096`, IC95 descriptif `[+0,00034698 ; +0,00657361]`. Gains SS F `+0,007779`, SS R `+0,012507` ; les IC des deux sous-groupes recouvrent zéro. Deux folds régressent (2 et 3).

Verdict prévu : **DIAGNOSTIC**, sous le seuil +0,01. **Comparateur ancien : cette expérience ne démontre aucun gain sur E à 0,6739.** Les poids test prêts ne justifient pas à eux seuls une soumission ni une addition de ce gain à celui du 6k.

### T1 6 000 pas : challenger prêt localement

Job **15904581** : terminé à 02:50:55, durée 10 min 03. Sorties OOF `pixseg_cv_psall6k_f0/f1.npz` et prédictions test `pixseg_test_psall6k_fit_s60/s61.npz` présentes.

Le relais actif a construit `/private/tmp/eaic-relais-20261122/c1_ensF_ssall6k2.pt`, SHA `39a3fb2d5ff9f3b31c44b51886bcebc2ad21e446e7b0825d327ab0c186b4bc7c`, 16 735 080 octets. Moyenne float32 des graines 60/61, seuil strict >0,5, **SS uniquement**. Les 680 autres masques sont identiques à E ; 0,4543 % des pixels globaux changent, sur 330 images.

Sur les **68 SS de deux folds**, comparaison une graine CV contre une graine CV à seuil fixé : `0,83373850 → 0,85660751`, soit `+0,02286901` sur SS et **+0,00762300 en contribution globale extrapolée** ; IC95 descriptif global `[+0,00151732 ; +0,01433143]`. Le calcul divise le gain méthode par trois, il ne divise pas les seuls 68 cas par 510 pour extrapoler à toute la méthode.

Limites : deux folds, configs déjà explorées, ensembles test différents (trois graines E contre deux F), pas de confirmation cinq folds de cette comparaison, pas de reçu F. **La CV ne prédit pas un score public précis.** Le script `validate_ss6k.sbatch` est préparé dans le dossier du relais. Vérifier si le chat actif l'a lancé entre-temps avant toute duplication ; cet audit ne l'a pas lancé.

### T2 sondes : verdict réel, pas une attente

Fichier distant `ties_refs.jsonl` SHA `40c48255d36b9063afbf7fc40f439fd66aab0072eaa08ba9abdf71ba15b4987e` : **990 tentatives, 989 statuts 200, une 503**, 110 tentatives par référence. La batterie réelle porte sur **11 sondes = 7 égalités + 4 bascules**, et non 12 ; la projection 1 080 requêtes des anciennes notes est dépassée. Journal terminé à 02:50:22.

`ties_result.json` donne **MCC 0 sur toutes les données et sur chacune des deux moitiés**. Tous les refs, fp8 et LoRA inclus, restent dans le même groupe. Les déguisements sont regroupés, mais la règle n'a aucune capacité de séparation à son seuil pré-enregistré. Le critère ≥0,60 sur les deux moitiés est manqué.

Job **15904551 j11**, terminé à 02:34:05 : 4 pools ×2 500 prompts famille A. Les compteurs nommés `ties_robust` dans les sorties brutes sont **insuffisants** pour appliquer le critère strict : le contrôle CPU du relais trouve **0 sonde stricte** pour chacune des graines 11/13/17/19. Les B/L prêts ne réparent pas l'échec refs ni l'absence A.

Conclusion opérationnelle : **le lot noté ne satisfait pas le critère prévu**. `ties_ep_A/B/L.jsonl` et `wb2/collect_ep.jsonl` sont absents aux chemins attendus à 03:07. Ce constat n'est pas une mesure fraîche du budget serveur.

## 5. T2 : données, budget et pistes déjà évaluées

Le pilote noté original existe au vrai chemin `/p/scratch/training2666/mouhali1/c2run/ep/ep_answers.jsonl` : **165 lignes, exactement cinq par endpoint**, SHA `0b2f0a2f1456be5254a5eb279c7aaa45045a1b847da3136626b491df8721e397`, toujours identique au snapshot antérieur. Une première recherche sous `SLZ/model_equality/c2run/` était négative car ce chemin est faux ; cette absence ne doit pas être interprétée comme une perte de données.

Dernier compteur explicitement documenté : **95 restants sur chacun des 33 endpoints, 3 135 au total**, relevé 23:25 selon coordination. **Pas de lecture fraîche du compteur authentifié par cet audit**, conformément à l'interdiction de lire des secrets. Le maintien du journal à 165 lignes et l'absence des nouveaux journaux appuient l'absence de collecte connue ; ils ne garantissent pas l'absence de crédits débités ailleurs ou sur erreur.

| Piste | Résultat disponible | État de reprise |
|---|---|---|
| N-grammes, hamming, affixes et choix sur le pilote | scores de sélection élevés, effondrement hors sélection ; CSV contradictoires | ne pas reprendre les scores calibrés comme validation |
| Longueur et isolement | ~0,245/0,254 hors sélection dans le protocole à cinq réponses ; variantes faibles invisibles | diagnostic partiel, candidat W prêt mais pas remplacement ferme |
| Listes aléatoires p3 | MCC 0/0 ou 0/0,273 | échec |
| Détection des déguisements p4 | 576 réponses refs, critère annoncé manqué dans les 12 directions | échec de cette batterie |
| Identification des bases par vraisemblance | A=Qwen2.5-3B, B=Qwen2.5-1.5B, L=Llama-3.2-1B, 11 chacun ; accord 528/528 avec fam3 | résultat utile, ne sépare pas les identités intra-famille |
| wb2 simulation GPU | MCC 0→1 / 1→0 : Qwen1.5B 0,0308/0 ; Qwen3B 0,5241/0 ; Llama1B 0/0,2306 | pas de collecte justifiée ; LoRA simulées aléatoires = substituts |
| Quasi-égalités v1/v2/v3 | échec refs final et aucune A3 stricte | verdict rendu, pas une attente API |

Le défaut ancien de retries 502/503/504 est confirmé par l'audit T2 antérieur : tentatives sans limite, seules réussites journalisées. Les scripts récents des sondes prévoient une tentative pour 5xx/réseau et des retries bornés pour 429. Ne pas relancer `ep1b.py` ou un ancien collecteur sans vérifier sa version et les compteurs. Les paramètres température/seed/logprobs ignorés relèvent du contrat de l'API décrit dans l'énoncé ; ce n'est pas une découverte récente démontrant un dysfonctionnement.

## 6. Autres pistes : ce qu'il ne faut pas refaire par oubli

**T1 :** canal détection PixelSeal non entraîné, ré-embarquement pilote négatif, TruFor déjà essayé sans gain utile, petites fusions/morphologies sans gain démontré contre la vraie référence. L'export NCC a été réparé par **15903835**, résultat sauvegardé ; il ne faut pas recommencer ce travail parce qu'un ancien audit décrivait encore `AssertionError: ncc_m`. La tête de référence à 0,5614 OOF inclut NCC et ens3 ; le fichier public 0,5572 est un autre artefact (`segf_sub_1`, ff+fm). L'ancien chiffre 0,5583 du registre ne remplace pas la provenance technique reconstruite dans le rapport T1.

Le dense TrustMark conserve un diagnostic positif mais incomplet : effet routé estimé +0,0091 dans l'ancien rapport, quelques R routés F sans prédiction. Cela ne prouve ni un gain nul ni un gain au-dessus de +0,01. Le pré-entraînement RGB + synthétique sur SS est la piste ayant effectivement produit le progrès public récent.

Décision par espérance de mIoU : gain imbriqué −0,0001, abandonnée. Réel seul à 8k : gain −0,0042 sur deux folds, abandonné seul ; l'ensemble 4k+8k n'est qu'un diagnostic et ne se confond pas avec le nouveau 6k **réel+synthétique**. Crop512/ConvNeXt Small du test SR n'ont pas gagné dans leur comparaison.

**T3 :** proxy donnant un score positif à l'origine, NPO professeur/élève sur vues différentes, et confusion accord/accuracy ont été diagnostiqués. Correctif NPO même vue et scripts d'analyse existent dans `docs/task3-work-20261121/`. Les contrôles `39,9 %` ou `73,6 %` sont des accords avec un modèle, pas l'accuracy cachée qui ouvre la porte retain à 70 %.

Les marges ont été mesurées (15903709), puis les mécanismes (15903971) ; leur seul signal local ne dépasse pas le champion public. La recherche sur tête est utile pour analyser les 45 classes concernées, mais les classes non concernées stables n'apportent pas une garantie sur la composition du set retain caché. Les push modifient des classes entières via la tête : la formule ancienne « seul le logit gagnant de chaque image, sans changement de classe » était trop simplifiée et a été contredite par les mesures de flips. Le résultat s200 à 0,0792 réfute la prolongation monotone du gain en augmentant l'intensité.

L'échec d'un candidat de séparation img2img n'établit pas à lui seul le mécanisme exact du gold. De même, le test A1 T1 prouve une lisibilité du watermark hors zone dans le diagnostic disponible, **pas un recollage pixel à pixel**. Le rapport global ancien utilise pourtant ce recollage comme fait ; conserver cette affirmation au rang d'hypothèse non établie.

## 7. GPU, erreurs et infrastructure

Snapshot Slurm à 03:06:57 : **aucun job dans squeue** pour `mouhali1,pelet1,zhang95`. Comptabilité disponible du **seul mouhali1**, depuis 21/11 22:00 : **37 jobs, 30 COMPLETED, 5 CANCELLED, 2 FAILED, 16,04 GPU-heures allouées**. Ce chiffre ne mesure pas l'activité réelle des cartes et ne couvre pas les jobs invisibles des autres comptes.

| Job | Objet | État et résultat |
|---|---|---|
| 15903709 | marges T3 | terminé, 8:41 |
| 15903813 | mesure push | terminé, 2:20 |
| 15903835 | export NCC | terminé, 6:48 |
| 15903857 | dense PS/TM | terminé, 12:10 |
| 15903971 | mécanismes T3 | terminé, 7:37 |
| 15904227 | séparation T3 | **FAILED 143:0**, 5:01 ; cause non réauditée |
| 15904283 | autre séparation T3 | annulé après 5:36 |
| 15904307 | bases T2 | terminé, 3:41 |
| 15904367 | simulation wb2 | terminé, 5:25, expérience négative |
| 15904400 / 419 | sondes T2 v1/v2 | terminés, 2:47 / 6:09 |
| 15904421 | variantes SS R | terminé, 17:08 |
| 15904515 | T1 réel+synthétique | terminé, 18:08 |
| 15904543 | fit T1 | **FAILED 143:0**, 29 s ; rank3 argparse, ranks0–2 terminés par srun |
| 15904551 | A3 T2 | terminé, 5:54 ; 0 sonde stricte selon analyse CPU |
| 15904559 | cinq folds IoU2 et fit | terminé, 18:38 |
| 15904561 | trois graines SS tous synthétiques | terminé, 5:35 ; sorties E présentes |
| 15904581 | SS 6k CV0/1 et fit60/61 | terminé, 10:03 |

Cause directe de 15904543 retrouvée : le glob StegaStamp se retrouve après `--synth-methods` plutôt que dans la liste `--synth`. `argparse` refuse ces arguments ; le runner de ce job tue les autres ranks. Ce n'est pas un échec d'optimisation du modèle. Le job 15904561 a ensuite produit les sorties nécessaires, il ne faut pas relancer 15904543 par réflexe.

Les logs A3 signalent `Python.h` manquant dans une compilation Triton, **puis poursuivent les passes et écrivent les résultats** ; le job est COMPLETED. Garder cette nuance avant de déclarer tous les résultats inutilisables ou d'installer globalement un paquet.

**Connexion :** l'alias `jureca` a envoyé le relais sur `jrlogin02`, où les espaces de données n'étaient pas accessibles. Selyan a ouvert un multiplexeur vers **jrlogin03** ; accès revérifié. Le socket partagé est `/private/tmp/eaic-codex-jrlogin03.sock`. Ne pas fermer ce master SSH ; aucun TOTP ni secret ne doit être copié dans le rapport.

**Inodes :** les anciens incidents de quota projet sont documentés (jusqu'à 564 libres), avec déplacement de sorties vers `/p/project1`. Le `df -i` actuel donne seulement les totaux globaux des systèmes de fichiers ; **il ne prouve pas que le quota `training2666` est rétabli**. `lfs quota` n'était pas disponible. Ne pas déclarer « problème résolu » à partir de df. Aucun nettoyage effectué ici.

## 8. Coordination et contradictions à corriger à la reprise

| Ancienne indication | État corrigé |
|---|---|
| T1 meilleur 0,5845 / 0,6377 / 0,6391 | E à **0,67390686**, reçu et classement |
| T3 0,0941 attribué « probablement clsA » | registre ultérieur : **push_s175** ; clsA rapporté à 0,0205 |
| tous les envois ont SHA, heure, auteur | faux : noms inconnus, préfixes seuls, heures en plage, reçus manquants |
| sondes refs toujours attendues vers 02:50 | fini : **989 réponses, une 503, MCC 0/0** |
| 12 sondes, 1 080 réponses refs | réel : **11 sondes, 990 tentatives** |
| A3 `ties_robust` positives = critère atteint | **0 stricte** selon le critère complet |
| IoU2 +0,0034 peut s'ajouter au champion | comparateur ancien ; pas de mesure contre E |
| s150/s200 programmés ou jamais envoyés | automates annulés, puis envois ultérieurs rapportés |
| aucune écriture en dehors d'un lanceur | historique de recouvrements ; le statut actuel n'efface pas l'incident |
| 95 crédits encore présents avec certitude | dernier compteur connu 23:25 ; pas de relevé authentifié frais |

Le journal documente un mandat T1 étendu à T2/T3 sans GO, des jobs soumis après un gel insuffisamment propagé et des doublons de fichiers. Les annulations et l'arrêt des automates sont rapportés ; **l'audit n'a pas réinspecté les PID actuels**, donc ne certifie pas l'absence de tout automate vivant. Aucun de ces incidents n'appelle à relancer une action externe depuis ce poste d'audit.

Les règles GPU ont évolué : `cispahack` seul au départ, exception T3 puis devel autorisé, ensuite parallélisme demandé par Selyan. Les recouvrements ne se jugent donc pas tous avec la règle initiale. La limite pratique de deux jobs devel par utilisateur a été observée ; au relevé présent la file est vide. Toute reprise doit repartir du mandat courant et des files réelles, pas d'un ancien en-tête sbatch.

## 9. Git et pérennité des livrables

Au départ et lors du relevé : checkout `/Users/selyan/dev/eaic-paris-2026`, branche **main**, HEAD **f209e20397736c53824a7a94ec73f2bd66d012eb**. Référence locale `origin/main` sept commits devant, tête `2bb6b4b` ; aucun fetch n'a été fait, donc cet écart décrit les refs locales, pas une interrogation fraîche de GitHub.

Changements préexistants suivis : `c1/seg/gen_f.py`, `c1/seg/train_f.py`, `docs/experiences.md`. Nombreux fichiers non suivis : scripts seg, rapports, coordination et résultats d'expériences. Le dossier `docs/relais-codex-20261122/` est produit par l'autre chat pendant cet audit. L'absence de processus dans `relay-check.py` n'est pas une preuve d'exclusivité : `ps` indisponible et chat actif observé.

Le worktree `.claude/worktrees/audit-global` est sur `docs/audit-global-nuit`, SHA **38e69bb**, propre au relevé. Ce commit contient sept fichiers : rapport global, coordination à cette date et scripts T3. **Il ne contient pas toute l'activité ultérieure ni les résultats 6k.** Ne pas considérer ce commit comme une sauvegarde exhaustive de la nuit.

Ne pas faire un pull/reset/stash global sur cet arbre partagé. Faire examiner et conserver les changements, puis intégrer dans une branche/PR dédiée par le responsable Git. Cet audit n'a modifié ni branche ni historique. Les poids/données demeurent hors Git, conformément au README.

Le challenger F est **dans `/private/tmp`**, pas encore un artefact durable dans le repo ou sur le cluster au relevé. Sa recette, ses hashes d'entrées et ses scripts sont conservés dans `docs/relais-codex-20261122/`. Selyan/Claude devra préserver le binaire avant nettoyage de tmp, ou le reconstruire avec les mêmes entrées. Ce rapport ne l'a ni déplacé ni envoyé.

## 10. Reprise proposée, avec les arbitrages laissés à Claude/Selyan

1. **Synchroniser l'état du relais** « Suivre les jobs GPU et sondes », puis lire `docs/relais-codex-20261122/`. **Le job de confirmation est maintenant 15904703, RUNNING au dernier relevé : lire sa sortie, ne pas lancer un doublon.**
2. **T1 : finir la preuve du challenger 6k**, comparaison appariée pertinente contre la recette du champion, couverture cinq folds et provenance des ensembles. Les critères déjà prévus (+0,01, ou diagnostic entre +0,003 et +0,01) restent des règles de décision à appliquer au bon comparateur. Aucun score futur promis. Soumission éventuelle par Selyan après décision, conservation d'E.
3. **T2 : acter le NO-GO de la batterie actuelle.** Reprendre éventuellement une autre méthode sur mandat distinct, en tenant compte des essais négatifs ; ne pas consommer le lot noté prévu uniquement parce que les GPU ont fini. Relire les compteurs via l'accès autorisé avant toute collecte future. Les bases identifiées et les corpus restent réutilisables.
4. **T3 : conserver s175 comme champion documenté.** Aucun nouveau mécanisme validé dans ce relais ; pas de nouvelle série automatique de coefficients à partir des scores publics.
5. **Sécuriser la continuité des fichiers**, sans nettoyage destructif : registre durable, candidat local, preuves, branche de reprise. Confirmer aux organisateurs la sélection finale et le calendrier si cela n'a pas déjà été fait ; aucune réponse officielle nouvelle n'a été retrouvée ici.

Questions restant ouvertes : quel fichier a donné 0,6391 ; quels sont les reçus bruts s175/essais T3 secondaires ; quel est le budget serveur actuel ; quel est le quota projet actuel ; quelle soumission compte en final ; le champion T1 est-il soumis avec une recette CV équivalente à son ensemble test. Ces questions sont consignées sans inventer de réponse ni bloquer la consolidation.

## 11. Commandes, sorties et contrôles

Preuves brutes dans `preuves/`, identités SHA-256 dans `MANIFESTE.json`. Les deux scripts de ce dossier utilisent uniquement la bibliothèque standard et n'installent rien.

Commandes réellement exécutées, sorties significatives :

```text
git status --short --branch
  main...origin/main [behind 7], trois fichiers suivis modifiés, rapports non suivis
git rev-parse HEAD
  f209e20397736c53824a7a94ec73f2bd66d012eb
git worktree list ; git log --oneline -20 ; git log --oneline HEAD..origin/main
  worktrees et sept commits d'écart archivés
python3 ~/.claude/scripts/relay-check.py --cwd /Users/selyan/dev/eaic-paris-2026
  dirty=true ; concurrence non vérifiée, ps indisponible
SSH jrlogin03 : squeue -u mouhali1,pelet1,zhang95
  en-tête seul à 03:04 puis 03:07
SSH : sacct -u mouhali1 -S 2026-11-21T22:00 -X -P ...
  37 lignes ; 30 COMPLETED, 5 CANCELLED, 2 FAILED
curl --fail --silent --show-error --max-time 20 http://34.147.229.55/leaderboard_page
  code 0 ; HTML officiel capturé
python3 docs/reprise-claude-20261122/collecter_preuves.py
  huit captures code 0 ; snapshot distant 162 881 octets au premier relevé
python3 docs/reprise-claude-20261122/synthetiser_preuves.py
  général reproduit à moins de 1e-12 ; 27 lignes du registre ; 16,04 GPU-h allouées
```

Le premier accès SSH sous sandbox a échoué ; la lecture réseau autorisée a ensuite fonctionné. Le navigateur web de recherche n'a pas ouvert la page IP ; le téléchargement HTTP direct a fourni le HTML. `date -Is` n'est pas accepté par le macOS local ; les heures utilisateur ont été relevées avec `TZ=Europe/Paris date` et les dates distantes explicites. Ces échecs d'outillage n'ont pas été traités comme des erreurs du projet.

Les analyses ML du relais actif ne sont pas présentées comme relancées par cet audit. Leurs JSON ont été lus, contrôlés pour cohérence, et copiés pour figer le contexte. Aucune suite de tests du projet n'a été lancée pour cette consolidation documentaire.
