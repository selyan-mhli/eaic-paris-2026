# Journal de la boucle 04:15 → 06:15

- 04:16 — Démarrage. Mandat « Soumettre si règle OK ». Champions : T1 `c1_ensG_ssall6k3.pt` 0,6788 ; T2 0,20 ; T3 `push_s175` 0,0941.
  Exploration T3 P1→P5 tenue par une autre session : P1 refusé HTTP 429 à 04:05:59 (soumission récente d'un autre envoi, attente 236 s).
  Cette boucle ne soumet pas de T3 tant que cette exploration tourne ; elle relit `recus_t3.jsonl` à chaque tour.
  `caffeinate` actif jusqu'à ~06:20.
- 04:17 — /goal « général ≥ 0,5 » posé par Selyan. Classement live 04:14 : SLZ 28e/46, général **0,329** (T1 0,688 norm 0,62 ;
  T2 0,20 norm 0,16 ; T3 0,094 norm 0,21). Il manque **0,51 de score normalisé** (somme sur 3). Exemples comptables à tableau figé :
  T1 0,75 + T3 0,20 exigerait encore T2 0,28 ; T3 seule devrait passer ~0,32 (tête du classement 0,45, nous 0,094 après une nuit de recherche).
  Pistes T1 en réserve : TrustMark dense +0,009 estimé (fermée, sous +0,01), graines SS supplémentaires (variance ~0,004) : aucune ne passe la règle.
  T3 : exploration P1→P5 guidée par le serveur tenue par la session « Statut du projet ». T2 : NO-GO (sondes à MCC 0).
  **Constat : seuil 0,5 hors de portée mesurée d'ici 06:15** (règle 5 de la lesson audit-global). Boucle maintenue en surveillance + T1 si une piste passe la règle.
  JURECA : lecture OK via le maître SSH existant ; file équipe vide à 04:16.
- 04:35 — T3 P2 (k 1,65) : public 0,0923 < 0,0941 (soumission 2885, session « Statut du projet »). P1 resté en 429, non noté. File JURECA équipe vide.
  Classement : 29e/46, général 0,319 (baisse par la normalisation, nos scores inchangés). Rien à soumettre côté T1 (aucune piste ≥ +0,01).
- 04:54:17 — **GO Selyan « lance le job TrustMark dense quand même »** : job **15905105** `slz_t1_tmd` (dc-gpu-devel, 1 GPU, 20 min max),
  `STAGES=measure` seulement, après relevé « 0 job équipe ». Départ estimé ~05:24. Code copié dans `work_pixd/dense/tm_candidate/`
  (sha tm_job.py 1b104450…, tm_job.sbatch c01aa8e1…, identiques au worktree eaic-wt-c1-pixseg ; dense_train.py/data.py distants = locaux).
  Pas d'export par le script : il recopierait l'ancien champion c1_pixss_fit0 (ancien StegaStamp). Si la mesure passe la règle,
  greffe des seuls TM test routés F sur le champion actuel `c1_ensG_ssall6k3.pt` (TM identiques au bit près jusqu'à G, ANALYSE-PROFONDE).
- 04:56 — **GO Selyan « soumets le TM dense même sous +0,01 »** (garde-fou conservé : pas d'envoi si gain mesuré ≤ 0).
  Export **15905107** `slz_t1_tmx` en file (afterok:15905105, STAGES=export FORCE=1, base fit0 da6abc31…).
  Greffe sur G : `docs/boucle-20261122/graft_tm.py` ; essai à blanc OK (TM de G = fit0 au bit près, sha G 5417d899… = champion soumis).
- 04:58 — Message de la session « Statut » : champion T1 réel = `output/c1_ensF_ss6k_tmf.pt` (0,6880), sha relevé **19d971dc5f4a08a58913f276bfa1b2ebd510a236cb53595f30974e9f684757e4**.
- 05:00 — **TM dense abandonné avant soumission** : OOF folds 0–1, TM F n=34, dense 0,589 contre psall6k@0,5 0,606 (déjà dans le champion) :
  −0,017 [IC95 −0,060 ; +0,025]. Garde-fou « pas d'envoi si gain ≤ 0 » appliqué. Jobs 15905105 et 15905107 annulés (scancel) à 05:02.
- 05:03 — OOF psall6k par méthode contre segf_sub_1 (PS/TM de E) : PS F +0,079 [+0,043 ; +0,124], PS R −0,000, TM F +0,140, TM R −0,014 (n=34 chacun).
  Candidat `c1_ensH_ps6k.pt` (build_ps6k.py) : champion + 176 PS routés F (couverture E > 0,5) = mean(psall6k s60,s61) > 0,5. sha da22188f…
  Estimation +0,013 T1 (routage parfait, borne basse +0,007) → règle passée (mandat « Soumettre si règle OK »).
- 05:04:12 — **SOUMIS T1 `c1_ensH_ps6k.pt` : public 0,6773** (meilleur 0,688 inchangé, improved False). **Le serveur contredit l'OOF** (−0,0107 contre +0,013 attendu).
  Hypothèses non testées : fit-all 2 graines ≠ CV 1 graine sur PS ; couverture E > 0,5 route mal sur le test ; split public (30 %) bruité. Pas de nouvel envoi PS sans explication.
- 05:09 — **GO Selyan « soumets aussi le PS sur les 340 images »** : `c1_ensI_ps6k_all.pt` (build_ps6k_all.py, 307 masques PS changés sur 340), sha 139242d3…
  1er envoi 05:09:26 → HTTP 429 (une soumission T1 venait de passer, attente 285 s, non notée). Relance automatique à ~05:14:20.
- 05:10 — **GO Selyan « soumets aussi le TM dense sur les 340 TrustMark »**. Prédictions test absentes (export annulé) : job **15905139** `slz_t1_tmx`
  (STAGES=export FORCE=1, devel 1 GPU) soumis 05:10:14 après relevé « 0 job ». Greffe prévue : `build_tmdense_all.py` (champion 19d971dc…,
  340 TM ← dense > 0,70). Rappel : non validé (OOF TM F −0,017 contre psall6k ; TM R jamais mesuré).
- 05:11 — **GO Selyan « soumets aussi le PS + TM dense combinés »** : base `c1_ensI_ps6k_all.pt` (139242d3…) + 340 TM ← dense > 0,70,
  sortie `c1_ensK_ps6k_tmdense.pt` (build_tmdense_all.py <dossier> <tmprobs> c1_ensI_ps6k_all.pt c1_ensK_ps6k_tmdense.pt). Attend l'export 15905139.
  Ordre d'envoi (≥ 5 min d'écart, limite serveur) : PS-340 (≈05:14) → TM-340 → PS+TM.
- 05:14:31 — **SOUMIS T1 `c1_ensI_ps6k_all.pt` (340 PS) : public 0,6777** (meilleur 0,688 inchangé). Quasi identique au routé (0,6773) :
  le psall6k fit-all dégrade PixelSeal d'environ −0,01 au public, quel que soit le routage → contredit l'OOF (+0,079 PS F). Combiné PS+TM hérite de cette perte.
- 05:15 — Export **15905139 COMPLETED** (1 min 44 s), tmprobs 340 TM (185 routés F par acc_full ≥ 0,65).
  Candidats : `c1_ensJ_tmdense_all.pt` (338 TM changés, sha 30597439…), `c1_ensK_ps6k_tmdense.pt` (PS-340 + TM dense, sha cd36e5f6…).
- 05:16 — **GO Selyan « soumets aussi le SS seul avec plus de graines »** : `c1_ensL_ss9.pt` = champion + SS = mean(2·s60, 2·s61, 2·s62, s50, s51, s52) > 0,5
  (build_ss9.py ; SS du champion reproduit au pixel près avec les 7 ; 315 SS touchées, 17 548 pixels ; sha 326e0fd8…).
  Envois séquentiels (≥ 5 min, retry sur 429) : SS9 → TM-340 → PS+TM.
- 05:20 — **GO Selyan « SS avec les graines s53 et s40 »** (s40 = sr_iou2_fit_s40, choix Selyan) : `c1_ensM_ss11.pt` = SS9 + pslr05 s53 + sr_iou2 s40 (poids 1),
  build_ss11.py, 314 SS touchées, 18 567 pixels, sha b3211c7d…. Envoi en file derrière la série SS9 → TM-340 → PS+TM.
- 05:22 — **GO Selyan « SS avec toutes les graines disponibles »** : `c1_ensN_ss_all.pt` (build_ss_all.py) = moyenne poids 1 des 37 pixseg_test_*.npz
  (output/ + t1sr/, dont 12 nt_fit0 non pré-entraînés), seuil 0,5 ; 325 SS touchées, 156 700 pixels ; sha a6d0b53a…. En file après SS11.
- 05:25 — **GO Selyan « SS sans les graines nt_fit0 »** : `c1_ensO_ss_pre.pt` (build_ss_all.py … nt_fit0 c1_ensO_ss_pre.pt) = moyenne poids 1 des 25
  prédictions pré-entraînées, seuil 0,5 ; 325 SS touchées, 143 201 pixels ; sha 7f8d3227…. 6e en file (~06:15). Délai serveur T1 réel : 10 min entre envois.
- 05:27 — **GO Selyan « SS pré-entraîné seulement psall et psall6k »** : `c1_ensP_ss_psall6.pt` = moyenne poids 1 de psall s50/51/52 + psall6k s60/61/62,
  seuil 0,5 ; 315 SS touchées, 16 712 pixels ; sha d2343a3c…. 7e en file (~06:25, après la fin prévue de la boucle : les envois en file continuent).
- 05:29 — **GO Selyan « SS avec psall6k seulement »** : `c1_ensQ_ss_6k3.pt` = champion + SS mean(psall6k s60/61/62) > 0,5 ; 319 SS, 28 578 pixels ;
  sha ec01eb42…. **SS identique au pixel près à celui de G** (0,6788) : le fichier = champion 0,688 avec le SS de G. 8e en file (~06:35).
- 05:24:40 — **SOUMIS `c1_ensL_ss9.pt` : public 0,6886, NOUVEAU MEILLEUR T1** (précédent 0,688 ; +0,0006). sha 326e0fd8…. Suite de la file : TM-340 (~05:35).
- 05:28 — **GO Selyan « SS9 avec la graine sr_iou2 s41 »** : `c1_ensR_ss9_iou41.pt` (build_ss9_iou41.py) = SS9 + sr_iou2 s41 (poids 1, total 10) ;
  317 SS, 17 221 pixels contre le champion 0,688 ; sha 0fc9aabb…. 9e en file (~06:45). caffeinate prolongé jusqu'à ~07:00.
- 05:30 — **GO Selyan « SS9 avec sr_iou2 s40 et s41 »** : `c1_ensS_ss9_iou4041.pt` = SS9 + sr_iou2 s40 + s41 (poids 1, total 11) ;
  318 SS, 18 883 pixels ; sha 34aaa09d…. 10e en file (~06:55). caffeinate prolongé jusqu'à ~07:20.
- 05:32 — **GO Selyan « SS9 avec pslr05 s53 »** : `c1_ensT_ss9_lr53.pt` = SS9 + pslr05 s53 (poids 1, total 10) ; 316 SS, 17 178 pixels ; sha 58c8be83…. 11e en file (~07:05).
- 05:34 — **GO Selyan « SS9 avec sr_iou2 s40 »** : `c1_ensU_ss9_iou40.pt` = SS9 + sr_iou2 s40 (poids 1, total 10) ; 318 SS, 17 660 pixels ; sha 941d28c8…. 12e en file (~07:15). caffeinate jusqu'à ~07:45.
- 05:36 — **GO Selyan « SS9 avec sr_iou2 s41 et pslr05 s53 »** : `c1_ensV_ss9_iou41_lr53.pt` (poids 1 chacune, total 11) ; 317 SS, 18 267 pixels ; sha 9c725a09…. 13e en file (~07:25).
- 05:34:52 — **SOUMIS `c1_ensJ_tmdense_all.pt` (TM dense 340) : public 0,6414** (meilleur 0,6886 inchangé) : forte perte, conforme à l'OOF TM F négatif.
- 05:40 — **GO Selyan « SS9 d'abord »** : 11 files arrêtées (TaskStop, aucun envoi en cours), R et Q recopiés (copie manquante), **file unique** relancée :
  R (SS9+s41) → U (+s40) → T (+s53) → S (+s40+s41) → V (+s41+s53) → M (SS11) → Q (6k3) → P (psall6) → O (pré) → N (toutes) → K (PS+TM),
  10 min d'écart, retry 429. Fin estimée ~07:30.
- 05:42 — **GO Selyan « SS9 avec s40, s41 et s53 »** : `c1_ensW_ss9_iou4041_lr53.pt` (poids 1 chacune, total 12) ; 321 SS, 22 975 pixels ; sha d9ad84e2….
  File relancée (aucun envoi en cours, R était en attente 429) : R → U → T → S → V → **W** → M → Q → P → O → N → K. Fin estimée ~07:40.
- 05:43 — **GO Selyan « SS9 avec s62 en poids 3 »** : `c1_ensX_ss9_s62w3.pt` (poids 2/2/3 pour s60/s61/s62 + psall ×1, total 10) ; 320 SS, 24 300 pixels ; sha 7d534647….
  File relancée (R en attente 429, aucun envoi actif) : R → U → T → S → V → W → **X** → M → Q → P → O → N → K. Fin ~07:50 ; caffeinate jusqu'à ~08:05.
- 05:45:04 — `c1_ensR_ss9_iou41.pt` : public **0,6886** (= meilleur, sans gain). File : U (+s40) ensuite (~05:55).
- 05:46 — T3 (session « Statut ») : P4 (T 1,2) 0,0754 ; **P5 (T 0,9) 0,0986, nouveau meilleur T3** (précédent 0,0941).
- 05:55:17 — `c1_ensU_ss9_iou40.pt` : public 0,6884 (< 0,6886). 06:05 — T (SS9+s53) en cours d'envoi.
- 06:07 — Classement : 31e/46, général **0,320** (T1 0,6886 ; T2 0,20 ; T3 0,0986).
- 06:10 — **GO Selyan « SS9 avec s62 en poids 4 »** : `c1_ensY_ss9_s62w4.pt` (poids 2/2/4 + psall ×1, total 11) ; 321 SS, 30 261 pixels ; sha 2f11a52e….
  Envoi séparé : part dès le score de X (s62 poids 3, ~06:46), donc avant M (SS11, qui attendra son 429). Envoi T en cours de notation, file non relancée.
- 06:05:14 — Envoi T (SS9+s53) **non parti** : SSH « Permission denied (keyboard-interactive) » (maître SSH expiré). Toutes les tâches de fond arrêtées (code 144), caffeinate compris.
- 06:13 — Nouveau maître SSH actif (pid 64023, ouvert hors de cette session), accès vérifié. **File relancée** : T → S → V → W → X → Y → M → Q → P → O → N → K
  (arrêt propre si SSH perdu). caffeinate jusqu'à ~08:25. Fin estimée ~08:15.
- 06:13:41 — `c1_ensT_ss9_lr53.pt` : public 0,6884 (< 0,6886).

## Résumé final (06:17)
- Champion T1 : **`c1_ensL_ss9.pt` 0,6886** (sha 326e0fd8…), champion 0,688 + graine psall6k s62 au SS. T3 : P5 0,0986 (session « Statut »). T2 : 0,20, inchangé.
- Classement 06:07 : 31e/46, général 0,320. Objectif « ≥ 0,5 » non atteint : hors de portée mesurée (constat dès 04:17).
- Échecs instructifs : PS psall6k (OOF +0,079 PS F) → 0,6773/0,6777 au public ; TM dense → 0,6414 (OOF négatif confirmé). Les variantes SS9 + graines plafonnent à 0,6884–0,6886.
- File d'envois T1 **toujours active** en arrière-plan (S, V, W, X, Y, M, Q, P, O, N, K ; fin ~08:15) ; s'arrête seule si SSH perdu ; caffeinate ~08:25.

## 07:22 — Arrêt sur ordre de clôture (Selyan, relayé par la session « Clôture du projet »)

File byoygrp44 et caffeinate bmrjxpo56 arrêtés à 07:22. Rien n'a été relancé. Aucune soumission n'était en vol au moment de l'arrêt : la file attendait les 10 minutes après M.

Ce qui est parti (score public T1) :

| Heure | Fichier | Score |
|---|---|---|
| 06:13 | c1_ensT_ss9_lr53.pt | 0,6884 |
| 06:23 | c1_ensS_ss9_iou4041.pt | 0,688 |
| 06:34 | c1_ensV_ss9_iou41_lr53.pt | 0,6883 |
| 06:44 | c1_ensW_ss9_iou4041_lr53.pt | 0,6873 |
| 06:54 | c1_ensX_ss9_s62w3.pt | **0,689** (nouveau meilleur) |
| 07:04 | c1_ensY_ss9_s62w4.pt | **0,6893** (meilleur final) |
| 07:14 | c1_ensM_ss11.pt | 0,6879 |

Ce qui n'est pas parti : c1_ensQ_ss_6k3, c1_ensP_ss_psall6, c1_ensO_ss_pre, c1_ensN_ss_all, c1_ensK_ps6k_tmdense.

Meilleur T1 retenu par le serveur : **c1_ensY_ss9_s62w4.pt, 0,6893** (SS9 avec s62 en poids 4).
