# SLZ — coordination campagne du 21/11 au soir

Source de vérité de l'orchestrateur (session Opus « campagne 5 h »). Mise à jour à chaque décision.
Clôture : dim. 22/11 12:00 Paris (à confirmer dans le règlement).

## Relevé 23:05 Paris (JURECA, `jrlogin03`)

### GPU — BLOCAGE
`squeue -u mouhali1,pelet1,zhang95` : **5 jobs SLZ, tous sur `dc-gpu` SANS `--reservation=cispahack`** (champ réservation `(null)`),
nœuds `jrc0353` / `jrc0378` hors liste de la réservation.

| Job | État | Nom | Script | Soumis |
|---|---|---|---|---|
| 15903636 | RUNNING | j1 | `work_pixseg_v2/run_bf.sbatch` (pixseg RGB, synth SS seul, `--fit-all --predict-test`, seed 4) | 22:57:57 |
| 15903637 | RUNNING | j1 | idem, seed 8 | 22:57:57 |
| 15903649 | PENDING | j1 | `work_pixseg_v3/run_bf.sbatch` | 23:01:46 |
| 15903650 | PENDING | j1 | `work_pixseg_v3/run_bf.sbatch` | 23:01:46 |
| 15903640 | PENDING | j3 | `unlearning/work_c3_eval/margin_job.sh` | 22:59:42 |

Aucune session CCD active ne les revendique (toutes « not running ») ; des processus `claude` CLI tournent sur ce worktree.
Propriétaire non identifié → **décision Selyan requise**. Aucun `scancel` fait par l'orchestrateur.
L'orchestrateur ne soumet aucun job tant que ce point n'est pas réglé.

### Inodes — CRITIQUE
`/p/scratch/training2666` : 4 400 000 total, 4 348 102 utilisés, **51 898 libres (1,2 %)**. Partagé entre toutes les équipes.
Interdit : décompresser, fichiers par échantillon, nouveaux venv. `export PYTHONDONTWRITEBYTECODE=1`.

### Mise en conformité 23:07–23:08 (GO Selyan)
`scancel` par l'orchestrateur : 15903650 (PENDING), 15903640 (j3 `margin_job.sh`, RUNNING depuis 1 min), 15903649 (j1 pixseg_v3, RUNNING depuis 38 s).
15903636 / 15903637 étaient déjà COMPLETED (6 min) : sorties **conservées** `output/pixseg_test_nt_fit0_s4..s11.npz` (8 graines SS supplémentaires, non fusionnées, non soumises).
Relevé 23:08 : **0 job SLZ** (mouhali1, pelet1, zhang95). Rien n'a été effacé.
Origine : session « Audit des projets et progression » (et ses sous-agents) et session c3-submit ; `margin_job.sh` porte le commentaire « dc-gpu sans réservation (backfill) » ; `head.sbatch` utilise `dc-gpu-devel` sans réservation. Les deux sessions ont confirmé le gel.
Sessions prévenues (SendMessage) : Audit/coordination, c3-submit, c1-probe, Task 2. **Sophie et Yihuan : pas de canal agent ; à prévenir par Selyan.**

### Automates de soumission actifs (Mac, sans GPU, hors contrôle de l'orchestrateur)
Scratchpad de la session Audit : `auto_c3.sh` (PID 33527) soumet `push_scale/push_s150.pt` ~23:18 puis `push_s200.pt` ~23:28 ; `auto_c2.sh` (PID 29422) soumet `model_equality/output/c2_fam3_pairs_v2.csv` (6799a438) à 23:22 ; `auto_c1.sh` (PID 94300) se termine sans rien soumettre.
Registre des soumissions de l'équipe : `/private/tmp/claude-501/-Users-selyan--claude-spares-spare-LgnnGN/730425da-a91c-484f-8a0e-3638d4840e30/scratchpad/soumissions.md`. **Ce registre est volatil : à recopier dans le dépôt.**

### Scores publics réels (registre + message de la session Audit, 23:10)
- T1 **0,5845** : `c1_pixss_fit0.pt` (sha da6abc31…) = e20 ff+fm + pixseg StegaStamp (4 réseaux, seuil 0,2), 20:05.
- T2 **0,20** (baseline aléatoire, 16:48). `c2_lump_v1.csv` → 0,05 (22:52).
- T3 **0,0567** : `push_scale/push_s125.pt` (tête de Yihuan `push_a8` ×1,25), 23:07. Précédent 0,0308 (`push`, 22:11). `t1t2_p03_e2` → 0,0091. Signal serveur : baisser le seul logit gagnant sans changer de classe fait monter le score ; l'intensité ×1,25 bat ×1.

### Prêt pour la réservation (non soumis)
- `work_segf/export_ncc.sbatch` (sha 9cce254c…) + `work_segf/preflight_ncc.py`, sources dans `docs/slz-jobs/`. 1 GPU, 30 min max, `cispahack`. Refait la partition externe 0 (contrôle ≈0,5606), écrit `c1_segf_ncc_e20_<job>.pt`, puis la fusion SS → `c1_ncc_pixss_fit0_<job>.pt`. Préflight passé sur le login : noms uniques, 1020 test, ncc_m/ncc_d finis.

### Classement des prochains jobs GPU
1. **T3 `margin_job.sh`** (4 GPU, 25 min) relancé avec `sbatch --reservation=cispahack` : version locale « même classe, marge m », m ∈ {2,5 ; 1,5 ; 1 ; 0,5}, la seule direction que le serveur valide. C'est le meilleur levier (T3 étroite au général).
2. **T1 `export_ncc.sbatch`** (1 GPU, ≤30 min) : gain attendu faible (0,5614 contre 0,5577 en OOF hors SS), mais aucun risque.
3. T1 `work_pixseg_v3` (cfg_a1 PixelSeal, cfg_a2 TrustMark ; 4 configurations chacun : encodeur officiel entraînable, entrée 512, encodeur gelé, message connu en entrée). Il porte sur le fold 0, en synthétique seul, 3 000 pas, durée non mesurée. Avant de le mettre en file : test CPU du code (`extmodel.py`, `train.py --ext`), jamais exécuté. Ses deux jobs doivent tenir dans **un seul job de 4 GPU**. Le critère de la session Audit (mIoU > 0,5 sur la réserve synthétique) **ne suffit pas** : l'audit exige un transfert sur des F réels avant de conclure.

## Tableau d'état

| Task | Champion confirmé | Agent | État | Prochaine action |
|---|---|---|---|---|
| T1 | public 0,5845 (`c1_pixss_fit0.pt`) ; OOF de la tête 0,5614 (`segf_ens3_3.npz`) | B (orchestrateur) | `export_ncc.sbatch` prêt, non lancé | GO GPU (rang 2) |
| T2 | 0,20 public (baseline) ; lump_v1 0,05 ; fam3_pairs_v2 soumis à 23:22 | C (lancé) | Réconciliation hors ligne | Provenance CSV, budget, validation par répétitions |
| T3 | public 0,0567 (`push_s125.pt`, 23:07) ; 1er 0,3422 | c3-submit + auto_c3 | balayage push en cours | GO GPU `margin_job` (rang 1) |

## Registre des sous-agents

| Agent | Mission | Fichiers possédés | Statut | Dernier résultat |
|---|---|---|---|---|
| C — T2 | Réconciliation, validation par répétitions, CSV candidat hors ligne | `docs/task2-work-20261121/` (nouveau) ; lecture seule ailleurs ; aucune requête endpoint | lancé 23:10 | — |
| A — T1 dense | en attente : risque de doublon avec la session qui lance `work_pixseg_v2/v3` | — | non lancé | — |
| D — T3 | en attente : `margin_job.sh` d'une autre session en file | — | non lancé | — |

## Règles
- Seul l'orchestrateur appelle `sbatch`, après `squeue` d'équipe immédiatement avant, et GO de Selyan.
- Un fichier = un propriétaire. Aucune requête endpoint T2, aucune soumission sans GO.
- Les sous-agents ne font ni Git ni Slurm.

### 23:14 — rechute
15903699 et 15903700 (`work_pixseg_v3/run_bf.sbatch`, cfg_a1 et cfg_a2) ont été soumis à 23:08:48 hors réservation par un sous-agent ml-dev de la session Audit, qui n'avait pas encore reçu l'ordre de gel. La session Audit les a mis en hold à 23:11, puis l'orchestrateur les a annulés (`scancel`) à 23:14. Ils n'avaient pas démarré, aucun fichier n'est perdu. Relevé : **0 job SLZ**.

### 23:12–23:20 — décisions Selyan appliquées
- **Automates de soumission arrêtés** à 23:12 sur ordre de Selyan (kill des PID exacts 33527 `auto_c3.sh` et 29422 `auto_c2.sh` ; ils étaient en sleep). **NON soumis** : `push_s150.pt`, `push_s200.pt`, `c2_fam3_pairs_v2.csv` ; fichiers conservés. La session Audit a coupé `auto_c1.sh` et sa tâche planifiée de soumission ; il ne lui reste qu'un relevé GPU en lecture seule toutes les 15 min.
- **Mandat** (citation de la session Audit) : le seul accord large de Selyan, à 19:52, ne concernait que la Task 1 (« tant que la task 1 n'atteint pas 0.7 on continue et tu submit tout seul »). Ont été soumis **par extension de ce mandat, sans GO explicite** : Task 3 `anc_f2a_s0550` (19:57), `t1t2_p03_e2` (22:57), `push_s125` (23:07, 0,0567) ; Task 2 `c2_lump_v1` (22:52, 0,05).
- **GPU : job 15903709 `slz_t3_margin`** soumis à 23:16 après relevé « 0 job équipe, 0 dans la réservation ». Script `unlearning/work_c3_eval/margin_job_cispa.sbatch` (sha f25e0f13…), copie de `margin_job.sh` avec `--reservation=cispahack`, `--time 00:40:00` et `PYTHONDONTWRITEBYTECODE=1`. Sorties dans le nouveau dossier `margin_out/mg_<tag>_s<step>.pt` (32 checkpoints d'environ 344 Mo) et `res/score_15903709_<tag>.txt`. Inodes avant lancement : 48 803 libres.
- **Ensuite** (GO de principe Selyan) : `work_segf/export_ncc.sbatch`, seulement après la fin complète de 15903709.

### Registre des sous-agents (mise à jour 23:20)
| Agent | Mission | Fichiers possédés | Statut |
|---|---|---|---|
| C — T2 | réconciliation hors ligne, CSV candidat, avis sur la collecte | `docs/task2-work-20261121/` | en cours |
| A — T1 dense | localisateur sur features officielles PS/TM, tests CPU, sbatch proposé non lancé | `/Users/selyan/dev/eaic-wt-c1-pixseg/c1/pixseg/dense/` | lancé 23:20 |
| D — T3 | formule officielle et proxy, correctif NPO avec test, `analyze_margin.py`, classement des candidats | `docs/task3-work-20261121/` | lancé 23:20 |

### 23:25 — Task 2 : compteurs et rendu de l'agent C
- **Compteurs serveur lus** (`get_budget` = GET `/models`, aucune réponse consommée) : **95 restantes sur chacun des 33 endpoints, 3 135 au total**. La phase 1b n'a jamais tourné.
- Agent C terminé : `docs/task2-work-20261121/RAPPORT.md`.
  - Corpus figé : `corpus.jsonl`, sha 883a2aa1…, sans dédoublonnage.
  - Sur les refs, les similarités de texte font pire que le hasard (AUC 0,27–0,41) ; seule la longueur est un peu au-dessus (0,60–0,61). Règle longueur + isolés, MCC hors sélection 0,25 ± 0,16 ; fp8 et LoRA 0,3 restent indétectables.
  - Le 0,05 de `lump_v1` réfute un découpage exact Qwen/Llama en 2 familles et va dans le sens de 3 familles Qa/Qb/L (11/11/11), découpage identique aux familles de `c2_fam3_pairs_v2`.
  - Score privé attendu (simulation conditionnée sur les scores publics, **estimation**) : fam3_pairs_v2 0,12–0,13 ; `cand_W_fam3_len_tau2` 0,13–0,18 ; baseline aléatoire, notre meilleur public, environ 0,05.
- Recommandation C : ne pas lancer la 1b ; faire d'abord un test gratuit sur les refs (3 prompts déterministes longs × 10 répétitions × 9 refs = 270 réponses, analyse fixée à l'avance), puis un lot ep de 20–30 réponses par endpoint seulement si le MCC hors sélection atteint 0,6.
- **Question pour les organisateurs** : le classement final retient-il la meilleure soumission publique ou la dernière ?

### 23:27 — Exception GPU autorisée par Selyan (T3 uniquement)
GO explicite de Selyan : basculer **seulement** le job 15903709 (`slz_t3_margin`) vers `dc-gpu-devel`, hors réservation (1 nœud, 4 GPU, 4 variantes, 40 min). Avant la modification, le job était encore PENDING et c'était le seul job de l'équipe. Commande : `scontrol update JobId=15903709 Partition=dc-gpu-devel Reservation=`. Après : PENDING (Resources) sur `dc-gpu-devel`, départ estimé 00:02, contre 00:20 dans `cispahack`. Même numéro de job, donc mêmes sorties (`margin_out/`, `res/score_15903709_*`, `logs/15903709_*`) ; aucun doublon. `margin_job_cispa.sbatch` n'est pas modifié (son en-tête indique toujours cispahack ; la partition a été changée par scontrol).
**Le gel général n'est pas levé** : tout autre job reste en `cispahack`, un seul à la fois. L'export NCC de la Task 1 partira après la fin complète de 15903709.

### 23:35 — 15903709 démarré ; rendu agent D ; défaut de collecte T2
- **15903709** est passé en CONFIGURING sur `jrc0352` (`dc-gpu-devel`). Le second GO de Selyan pour la bascule est arrivé alors que la bascule de 23:27 était déjà faite : aucune autre action, aucun doublon.
- **Agent D terminé** (`docs/task3-work-20261121/`).
  - Formule réimplémentée (`official_formula.py`). Abaisser le seul logit gagnant rapproche le terme KL tant que q_y > g_y ; c'est cohérent avec le serveur.
  - Optimum d'intensité **estimé sur un jouet** : alpha 10–12 si le gold atteint 75–90 % sur les forget. Alpha 16 est probablement au-delà.
  - Correctif NPO même vue `unlearn_v2_fixed.py`. Test vérifié par l'orchestrateur : |delta| = 0 sur la même vue, 0,0183 avec l'ancien calcul.
  - `analyze_margin.py` (test vérifié) est copié dans `work_c3_eval/` (sha 326fd13f…).
  - Classement non soumis : 1) `push_s150` (b5c38b8f) ; 2) meilleur checkpoint margin en zone retain connue ; `push_s200` déconseillé avant le résultat de s150.
  - **Prochain job GPU proposé par D** : `score_like.py` sur push_s075/best/s125/s150/s200, mesure seule, environ 5 min, 1 GPU, `cispahack`. Il place la famille push dans le même référentiel que margin.
- **Défaut de collecte Task 2 (Selyan)** : les scripts réessaient les 502/503/504 sans limite et n'enregistrent que les réussites. Si les erreurs sont débitées, une réponse utile peut coûter plusieurs crédits. Aucune perte établie pour SLZ (95/endpoint à 23:25). **Avant toute collecte ep** : couper les retries automatiques, journaliser chaque tentative, relire les compteurs, obtenir confirmation du correctif serveur. Détail : `docs/task2-audit-20261121/RAPPORT-CODEX.md` §8.

### 23:40 — File GPU autorisée (GO Selyan) : push_eval puis export_ncc
- Après la fin complète de 15903709 : `unlearning/work_c3_eval/push_eval.sbatch` (sha 95e09b14…), 1 GPU, `cispahack`, 20 min max. **Mesure seule** avec `score_like.py --original ../model.pt --off offforget.npz` sur push_s075, push_best, push_s125, push_s150, push_s200. Sortie : `res/score_<job>_push.txt`. Pas d'entraînement, pas de soumission.
- Ensuite : `work_segf/export_ncc.sbatch` (sha 9cce254c…), `cispahack`.
- Enchaînement lancé par l'orchestrateur : `docs/slz-jobs/chain_push_ncc.sh`, journal dans `docs/slz-jobs/chain_push_ncc.log`. Avant chaque sbatch, il vérifie qu'aucun job d'équipe n'est actif et abandonne s'il en trouve un.
- 15903709 à 23:38 : RUNNING, pas 225/1000 à 3,2 pas/s.

## Soumissions (section tenue par la session Audit)

Registre de toutes les soumissions de l'équipe, heures de Paris. Règle : un fichier n'est soumis que si son sha256 n'est pas déjà dans ce tableau. Délais mesurés : Task 1 et Task 3 environ 10 min chacune, Task 2 30 min ; les délais sont par tâche.
Scripts (clé jamais affichée, fichiers en mode 600) : Task 1 `work_c1/submit_c1.py <fichier>` (dossier de la tâche, `.venv`) ; Task 2 `output/submit_c2.py <csv>` (dossier model_equality, `.venv`) ; Task 3 `submit_c3.py <fichier>` (dossier unlearning/work_c3_eval, `../.venv`).

| Heure | Tâche | Fichier | sha256 (début) | Score local | Score public | Par |
|---|---|---|---|---|---|---|
| 16:48 | 2 | baseline_random.csv (aléatoire du kit) | — | — | 0,20 (chance) | Sophie |
| ~17:34 | 3 | inconnu | — | — | 0,0026 | Yihuan ? |
| 18:22 | 1 | seg_fsynth_rect_v1.pt | 302ee55e | 0,547 | 0,5307 | Selyan |
| ~18:40 | 3 | soup_mid.pt | 10ee30f6 | proxy | 0,0012 | Selyan |
| 18:52 | 3 | soups/soupB_l85.pt | 69435b1b | proxy | 0,0035 | Selyan |
| 19:12 | 1 | seg_fsynth_rect_v1.pt (doublon) | 302ee55e | 0,547 | 0,5307 | Audit |
| 19:15 | 3 | distill_out/m75/distill_r45.pt | af207210 | CE 0,03 | 0,0044 | Selyan (probable) |
| 19:26 | 3 | distill_out/m75/distill_r45.pt (doublon) | af207210 | CE 0,03 | 0,0044 | Audit |
| ~19:46 | 3 | inconnu (n'a pas amélioré) | — | — | ≤ 0,0044 | inconnu |
| 19:53 | 1 | c1_segf_ffm_e20_20261121.pt | 51447085 | 0,5583 | 0,5572 | Audit |
| 19:57 | 3 | anchor_ckpt/anc_f2a_s0550.pt | 93b67cce | ancré, 8,4 % changées | 0,0039 | Audit |
| 20:05 | 1 | c1_pixss_fit0.pt (e20 + pixseg StegaStamp, 4 réseaux, seuil 0,2) | da6abc31 | — | **0,5845** | auto_c1.sh |
| 22:11 | 3 | push_best.pt (tête, logit gagnant −8, Yihuan) | 34cfc901 | CE 0,087, 2,5 % changées | 0,0308 | Yihuan |
| 22:52 | 2 | c2_lump_v1.csv (2 familles, 1 gros groupe + 3 isolés) | b4ee773a | — | 0,05 | Audit |
| 23:23:41 | 2 | model_equality/output/c2_fam3_pairs_v2.csv (3 familles A/B/L de 11, 28 modèles, 5 paires) | 6799a4388f736b90626b3680a1ee30546a1382e1f4c4ff7d8135161064ba0f5c | attendu privé ≈0,12–0,13 (estimation agent C) | **0,10** (meilleur inchangé : 0,20) | orchestrateur, GO Selyan pour cette seule soumission |
| 00:17:48 → 00:20:40 | 3 | unlearning/work_c3_eval/push_scale/push_s150.pt (tête push ×1,5) | b5c38b8f262cfae27ad3dd08299522697dc331e62f0678080f89096bb8ebd423 | flip forget 22,4 % ; accord classes concernées 0,088, autres 0,998 | **0,0827** (précédent 0,0567, improved: True) | orchestrateur, GO Selyan pour cette seule soumission ; reçu docs/slz-jobs/recu_t3_push_s150.txt |
| 22:57 | 3 | work_c3_t1t2/out/t1t2_p03_e2.pt | 8b257641 | CE 0,106, 3,1 % changées | 0,0091 | Audit |
| 23:07 | 3 | work_c3_eval/push_scale/push_s125.pt (tête Yihuan ×1,25, alpha 10) | b28ea783 | non mesuré | **0,0567** | auto_c3.sh |

~~Programmé (scripts locaux sur le Mac, sans GPU) : Task 3 push_s150 (alpha 12, b5c38b8f) vers 23:18 et push_s200 (alpha 16, e6b09356) vers 23:28 ; Task 2 c2_fam3_pairs_v2.csv (6799a438 : familles A/B/Llama, paires 21-32, 26-28, 13-19, 06-22, 09-10, tout le reste différent) à 23:22.~~ **Annulé à 23:12 par l'orchestrateur sur ordre de Selyan : push_s150 et push_s200 non soumis ; c2_fam3_pairs_v2 soumis à 23:23 sur GO explicite (ligne ci-dessous).**
Préparé, non soumis : push_scale/push_s075.pt (alpha 6, 5ca028ef) ; prédictions StegaStamp nt_fit0_s4..s11 (8 réseaux de plus, non fusionnés).

Leçons serveur :
- Task 3 : changer la classe prédite ou étaler l'entropie fait baisser le score (ancré 0,0039, distillation 0,0044 à 0,0091) ; baisser le seul logit gagnant de la tête sans changer de classe le fait monter, et davantage à alpha 10 qu'à alpha 8.
- Task 2 : l'hypothèse « un gros groupe + isolés », vraie sur les endpoints de référence, est fausse sur les ep (0,05). Il y a trois familles de 11, et une famille Qwen répond de façon déterministe.
- Task 1 : la validation imbriquée de la tête F (0,5583) a prédit le public à 0,001 près (0,5572).

### 23:50 — Changement de stratégie : leviers Task 2 et Task 3
Classement public relevé vers 23:45 : SLZ **21e sur 30**, général 0,29 (T1 0,58 ; T2 0,20 ; T3 0,06). En général recalculé, porter une seule tâche au maximum du tableau donnerait : T2 0,59 ; T3 0,61 ; T1 0,45 (audit `docs/audit-strategies-20261121/`).
- **T2** : GO Selyan pour les requêtes **refs uniquement**. Agent « banc T2 » lancé ; dossier `docs/task2-bench-20261121/`, journaux distants `c2run/bench_*.jsonl`. Contraintes : PARALLEL ≤ 2, aucune relance sur 5xx, 3 au plus sur 429. Critère : MCC modèle ≥ 0,83 dans les deux sens, avec séparation du fp8 et de la LoRA 0,3. Aucun ep, aucune soumission. `q3.py` est terminé (72/72) : aucune autre collecte active.
- **T3** : on arrête de monter le coefficient push à l'aveugle. Prochaine expérience : comparaison de mécanismes (cible de confiance par image contre marge), avec la même préservation.
- Questions à poser aux organisateurs (Selyan) : scores T2 arrondis par pas de 0,05 ? Le classement final retient-il la meilleure soumission ou la dernière ?

### 23:35 — Résultat de 15903709 (marges T3), puis push_eval 15903813 en file
15903709 COMPLETED en 8 min 41 s (23:24:59 → 23:33:40). Analyse : `analyze_margin.py`, TSV dans `work_c3_eval/res/margin_15903709.tsv`.
- Au pas 1000, CE forget de 0,021 (m 2,5) à 0,066 (m 0,5). Classes changées ≤ 0,3 %. Retain proxy correct : off 0,95–0,96, off_ret 0,99, lowpass32 0,75–0,80. fp16 sans aucun changement de classe.
- La CE progresse **linéairement avec les pas, sans saturer**. Le mécanisme n'est pas plafonné, simplement sous-entraîné à 1000 pas.
- **Signal défavorable** : l'entropie forget à T=2 passe de 0,097 (origine) à 1,37 (mg05), alors que la CE ne monte qu'à 0,066. La marge aplatit les classes secondaires (cible = second logit + m), autrement dit elle « étale l'entropie », ce que le serveur a pénalisé. push garde la structure des classes secondaires.
- Recommandation proxy : `mg_mg05_s1000`, puis `mg_mg10_s1000`. **Non soumis.** À départager contre push_s125 avec `push_eval`.
- 15903813 `slz_t3_pusheval` soumis par l'enchaînement à 23:35:08, PENDING dans `cispahack`. Seul job de l'équipe.

### 23:42 — push_eval 15903813 (COMPLETED 2 min 20 s) ; export NCC 15903835 soumis à 23:41:20
| push | CE forget moy/éc | flip forget | entropie T2 | public |
|---|---|---:|---:|---:|
| s075 (α 6) | 0,023 / 0,129 | 0,6 % | 0,58 | — |
| best (α 8) | 0,087 / 0,329 | 2,5 % | 0,95 | 0,0308 |
| s125 (α 10) | 0,270 / 0,684 | 9,9 % | 1,43 | **0,0567** |
| s150 (α 12) | 0,680 / 1,212 | 22,4 % | 1,95 | — |
| s200 (α 16) | 2,579 / 2,475 | 61,5 % | 2,76 | — |
Lecture : « changer de classe fait baisser » est **réfuté** (s125 change 9,9 % et bat best). Selon la session Audit, s125 dérive fortement hors forget (accord 0,61). Hypothèse Audit transmise à l'agent T3 : la tête push abaisse des classes entières ; un décalage de biais par classe proportionnel au nombre de forget retirées (β·n_c) est à tester par score_like sans entraînement, en variante (e) du job « mécanismes ».

### 23:47 — GO Selyan : prochains jobs sur `dc-gpu-devel`
La réservation est saturée : `--test-only` du pilote dense dans `cispahack` annonçait un départ à 01:21. Selyan autorise `dc-gpu-devel` pour les prochains jobs. **Un seul job d'équipe à la fois reste la règle.**
- Pilote dense T1 : copie `c1/pixseg` (avec dense/, sans v3/) dans `work_pixd/` (49 fichiers). `work_pixd/dense/pilot_devel.sbatch` = copie de `pilot.sbatch` en partition devel, sans réservation (4 GPU, 50 min). `--test-only` annonce un départ immédiat. Il sera soumis automatiquement à la fin de l'export NCC 15903835, après un relevé « 0 job d'équipe ».
- Ensuite : le job T3 « mécanismes » (4 GPU) dès qu'il sera prêt et relu.

### 23:49 — Export NCC T1 15903835 COMPLETED (6 min 48 s) ; pilote dense 15903857 lancé (devel, 4 GPU)
- La partition externe 0 refaite donne **0,5606**, identique au `head[0]` de l'audit (0,56055) : la reproductibilité est confirmée. Champion de l'ancienne règle 0,5088.
- `output/c1_segf_ncc_e20_15903835.pt` (sha 1652bd57…, 1020 masques, NON soumis), puis fusion SS pixseg (4 réseaux, seuil 0,2) → `output/c1_ncc_pixss_fit0_15903835.pt` (sha **52ea4f3f4c379f7bd464ddeee764185e322b6f4e9fec2f97739dd100a6aaa970**, NON soumis). Le champion public `c1_pixss_fit0.pt` (da6abc31…) reste intact.
- Décision : le gain attendu par rapport au champion public n'existe que sur la partie hors SS (0,5606 contre environ 0,5577 OOF pour ff+fm, sur des évaluations différentes, donc pas un apparié). C'est **sous la règle +0,01** : candidat conservé, non proposé à la soumission.
- 15903857 `pilot_devel.sbatch` soumis à 23:48:49 après relevé « 0 job », CONFIGURING sur jrc0255.

### 23:52 — Job T3 « mécanismes » prêt, en attente derrière le pilote 15903857
- Agent D terminé : `docs/task3-work-20261121/mech/` (8 fichiers) copié dans `work_c3_eval/mech/`. Variante devel `mech/mech_job_devel.sbatch` (4 GPU, 30 min, `slz_t3_mech`). `--test-only` annonce un départ immédiat. Tests CPU relancés par l'orchestrateur : `test_mech` OK, `test_head_search` OK (écart d'émulation 0,00036).
- Rangs : 0 = cache des features + `head_search` (vérifié sur push_best/s125/s150) + biais β·n_c + balayage push hors ligne + contrôle sur la moitié tenue à l'écart des 5 références ; 1 = NPO même vue → réparation ; 2 = cible de confiance par image → réparation ; 3 = idem avec lr ×3.
- Calibration du proxy (terme moyenne/écart-type) : l'ordre s125 > best > t1t2 est respecté, mais best et t1t2 sortent presque à égalité alors que le serveur les sépare d'un facteur 3,4 ; le terme de prédiction manque, la colonne `rest_kl` doit le combler. **Le retain local classe à l'envers du serveur** : s125 a les pires accords et le meilleur score.
- Soumission automatique à la fin de 15903857, après relevé « 0 job » (règle d'un seul job maintenue, Selyan n'a pas répondu sur 2 jobs en parallèle). Inodes : 33 897 libres, en baisse rapide (autres équipes) ; le job crée environ 25 fichiers.
- Candidat T3 suivant selon D : push_s150 (meilleur terme moyenne/écart-type, hors zone retain connue). **Non soumis.** À trancher après `hold_<job>_ref` et `head_search`.
- Consigne Selyan (23:53) : un seul job à la fois ; le job T3 part juste après le pilote T1 (enchaînement en place). **Critère de comparaison des mécanismes : écart entre la cible demandée et la cible atteinte (CE forget moyenne/écart-type), à préservation comparable** (ctl_agree / ctl_ret sur la moitié tenue à l'écart).
- Décision Selyan (23:56) : **push_s150 non soumis**. On attend le contrôle sur la moitié tenue à l'écart du job mécanismes (`res/hold_<job>_ref.txt`), puis on compare à push_s125 à préservation comparable. **push_s125 reste le champion T3 confirmé** (0,0567).

### 23:59 — Tableau de bord GPU mis à jour (session Audit)
Relevé nvidia-smi de 15903857 à 23:58:13 : **GPU 88 % en moyenne** (94/83/91/84 %), mémoire GPU 6 %. Bilan sacct depuis 14:00 : 84 jobs, 70,1 h GPU allouées. **Correction** : push_eval et l'export NCC ont reçu le nœud entier (4 GPU alloués) pour un seul GPU utilisé, soit 3 GPU perdus pendant 9 min. Règle pour la suite : ne jamais lancer un job à 1 GPU seul ; regrouper 4 tâches indépendantes par job (une par GPU).

- Consigne Selyan (00:00) : continuer jusqu'aux confirmations T1 ; pour les prochains jobs, **mesurer le débit avant d'augmenter la concurrence** ; regrouper les travaux utiles sans imposer 4 tâches systématiquement ; T3 « mécanismes » prioritaire après le pilote.

### 00:02 — Pilote dense T1 15903857 COMPLETED (12 min 10 s) → **DIAGNOSTIC** ; mech T3 15903971 lancé à 00:01:39 (devel, 4 GPU)
Confirmation sur les folds 0–1 (34 F par méthode, mode tête seule, appariés contre `head[0]`). `compare.py` sur `docs/task1-dense-pilot-15903857/`, sortie `compare.txt`.
| Méthode | dense | head[0] | gain | IC95 apparié descriptif | IoU trous |
|---|---:|---:|---:|---|---:|
| PixelSeal F | 0,582 | 0,569 | +0,013 | [−0,017 ; +0,042] | 0,26 |
| TrustMark F | 0,589 | 0,525 | **+0,063** | [+0,026 ; +0,107] | 0,28 |
Gain F moyen **+0,038**, entre les seuils (abandon < 0,03, poursuite ≥ 0,08) : **DIAGNOSTIC**, comme fixé avant le job. Le mode « dernier stage » n'apporte rien (dev 0,590/0,581 contre 0,597/0,589 en tête seule). Lecture : signal réel sur TrustMark F, non démontré sur PixelSeal. Effet global maximal si l'on remplace seulement les TM F : environ +0,063 × 85/510 ≈ +0,011 (non mesuré dans la pipeline routée).

### 00:08 — Job mech T3 15903971 : entraînements terminés, évaluations en cours
- Objectif fixé par Selyan (00:07) : **0,60 au général (top 10), puis 0,74 (top 6)**. T2 et T3 sont prioritaires ; le candidat TrustMark ne doit pas les retarder.
- `head_search` **validé** : écart max 0,00029 contre `score_like` sur push_best/s125/s150. Émulation de push ×1,25 identique à push_s125. Les éditions de tête se mesurent désormais hors ligne.
- **Constat T3** : sur la moitié tenue à l'écart, `ctl_agree` classe les 4 points serveur exactement à l'**inverse** de leur score (s125 0,62 < best 0,66 < t1t2 0,87 < anc 0,88). Le terme moyenne/écart-type, lui, ne les classe pas (anc 0,82 contre s125 0,28). Plus le modèle dérive sur les classes oubliées, plus le serveur récompense, ce qui va dans le sens de l'hypothèse « gold moins confiant sur des classes entières ».
- Le biais par classe seul (`bias_lin_0.1` : CE 0,23/0,71, flip 8,1 %, ctl 0,617) **reproduit presque exactement le profil de push_s125** (0,27/0,68 ; 9,9 % ; 0,62) : push ≈ décalage de confiance par classe.
- NPO même vue + réparation (`mech_ab_r_s0300`) : CE forget 0,92/2,63, flip 16,6 %, proche des statistiques gold supposées (1,14/2,66). Préservation encore à mesurer (`hold_15903971_ab`).

### 00:10 — Règle de sélection head_search FIGÉE (consigne Selyan), avant toute nouvelle lecture du contrôle
- La moitié « contrôle » de `offforget_all` (765 images, graine 20261121) sert **une seule fois** : la comparaison finale de ≤ 2 candidats (NPO+réparation contre push_s125), demandée par Selyan. Elle ne sert **pas** à choisir les variantes de `head_search`.
- Choix d'une variante head_search : uniquement sur les stats forget (CE moyenne/écart-type, flip, altérées) et sur la moitié « réparation » (`rep_agree`), avec la contrainte « rep_agree ≥ celui de push_s125 (0,625) ». Ensuite, la plus forte CE forget. Toute variante choisie ainsi n'est mesurée sur le contrôle qu'en comparaison finale.
- Lu jusqu'ici sur le contrôle : `hold_15903971_ref` (5 références), puis `ctl_agree` des variantes head_search (déjà affiché dans `score_15903971_headsearch`). **Ces variantes ne sont donc plus « indépendantes » du contrôle** ; on le signale si l'une d'elles est proposée.
- 15903971 COMPLETED en 7 min 37 s. GPU libre depuis environ 00:09.
- Mesures de préservation sur le **contrôle** (lecture autorisée : comparaison finale NPO contre push_s125) :
  - `mech_ab_r_s0300` (NPO + réparation) : CE forget 0,81/2,74, flip 12,8 %, **ctl_agree 0,736**, ctl_ret 0,733, rep_agree 0,941.
  - push_s125 : 0,27/0,68, flip 9,9 %, ctl_agree 0,620, ctl_ret 0,652.
  - (c)/(d), cible de confiance : très conservateur (CE ≤ 0,43, ctl ≥ 0,78) ; le mécanisme n'atteint pas sa cible.
- Comparaison par classes (45 classes concernées contre les autres) confiée à l'agent T3, sur la moitié réparation seulement ; le contrôle final ira dans un fichier séparé, lu une seule fois. Petit job GPU à suivre.

### 00:22 — Job class_split T3 15903998 soumis (devel, 1 GPU, 15 min max), après relevé « 0 job »
- `mech/class_split.py` + `class_split.sbatch`, test CPU OK. Il compare origine, push_s125, mech_ab_r_s0300 et mech_ab_r_s0100 sur les 45 classes concernées contre les autres (forget + moitié réparation ; contrôle dans `res/ctl_final_15903998.txt`, lu une seule fois).
- **Calibration** (agent D, sur des colonnes déjà mesurées) : ni le terme moyenne/écart-type ni rest_kl ne reproduisent l'ordre serveur. **mech_ab_r_s0300 ressemble à anc_f2a** (CE élevée, écart-type ≈ 2,7, rest_kl élevé, retain correct), dont le score serveur est 0,0039. C'est un risque, pas une mesure. NPO+réparation n'est donc pas soumis sans le découpage par classes.

### 00:24 — Candidats et règle de décision T3 FIGÉS avant la lecture de class_split 15903998 (consigne Selyan)
- **Candidats** : champion `push_scale/push_s125.pt` (0,0567 public) ; challenger `mech_out/15903971/mech_ab_r_s0300.pt`. Aucune nouvelle grille. `mech_ab_r_s0100` est mesuré pour information, mais il n'est pas candidat.
- **Transparence** : les 765 photos de la moitié contrôle ont **déjà été consultées** (`hold_15903971_*`). `ctl_final_15903998` apporte des éléments supplémentaires, mais **ce n'est plus une confirmation indépendante**.
- **Règle** : proposer le challenger à Selyan (sans soumission automatique) seulement si TOUTES ces conditions sont réunies :
  1. format, finitude et chargement strict conformes ; fp16flip ≤ 0,1 % ;
  2. classes concernées, vues propres ET altérées : KL origine→candidat ≥ celle de push_s125 (le serveur récompense jusqu'ici la dérive sur ces classes) ;
  3. classes non concernées (moitié réparation) : accord ≥ celui de push_s125 et KL ≤ celle de push_s125 (moins de dégâts collatéraux) ;
  4. aucune altération où les classes non concernées perdent plus de 5 points d'accord par rapport à push_s125.
  Sinon : push_s125 reste champion, rien n'est proposé. Même si la règle passe, le **risque « profil anc_f2a » (0,0039)** est rappelé dans la proposition.

### 00:30 — class_split 15903998 COMPLETED → **règle appliquée : challenger REJETÉ, push_s125 reste champion**
Moitié réparation (`res/class_split_15903998.txt`), vue propre, accord / KL T2 :
| | forget (concernées) | photos concernées (352) | photos autres (413) |
|---|---|---|---|
| push_s125 | 0,901 / 0,46 | **0,19 / 0,87** | **0,998 / 0,27** |
| mech_ab_r_s0300 | 0,872 / 0,46 | 0,92 / 0,05 | 0,959 / 0,05 |
- Condition 2 (KL sur les classes concernées ≥ push_s125) **échoue** : 0,05 contre 0,87 sur les photos concernées, et 0,4585 contre 0,4615 sur les forget propres.
- Condition 3 : sur le **contrôle** (déjà consulté, donc pas une confirmation indépendante), mech_ab_r_s0300 descend à **0,785 d'accord sur les classes autres (21,5 % de changements)**, contre 1,000 pour push_s125. La réparation sur-ajuste sa moitié (0,959 dessus contre 0,785 ailleurs). **Échec.**
- Fait mesuré clé : **push_s125 supprime presque entièrement les 45 classes concernées sur toute image (accord 0,19) et laisse intactes les autres (0,998)**. C'est le profil que le serveur récompense (0,0567), alors que NPO+réparation fait l'inverse.
- Émulation hors ligne depuis le cache (vérifiée : 0,188/0,868 contre 0,1875/0,8682 sur le GPU) : **push_s150** donne 0,088 / 1,22 sur les classes concernées, 0,998 / 0,35 sur les autres, 22,4 % de flips forget ; push_s200 donne 0,020 / 2,02 et 0,998 / 0,54. s150 = **plus de suppression de classe, même préservation des autres classes** que s125. Risque non mesurable : la porte retain des 70 % si le retain caché contient beaucoup d'images des 45 classes.

### 00:32 — Candidat TrustMark dense : vérification routée **NON passée** → pas d'export
Agent T1 (`c1/pixseg/dense/tm_candidate/`, mesure `docs/task1-dense-pilot-15903857/mesure_routee_pilote_cpu.json`). Comparaison au **champion exact** (`segf_sub_1`, base du public 0,5845), routage par `acc_full` tel qu'au test, seuils figés. Gain global **estimé, pondéré depuis la confirmation** : folds 0–1 **+0,0042** [IC +0,0015 ; +0,0073] ; folds 0–4 **+0,0091** [+0,0047 ; +0,0137]. TM R routés F : 2 (folds 0–1) et 5 (folds 0–4), sans prédiction dense, effet borné à [−0,0057 ; +0,0041]. **Sous la règle +0,01** : pas d'export, et le job de mesure (5–8 min) n'est pas lancé (il ne changerait pas la décision). Le travail est conservé (job prêt, export conditionné).

### 00:20 — push_s150 soumis (GO Selyan, une seule soumission) ; recadrage stratégique
- `push_scale/push_s150.pt` (sha b5c38b8f262cfae27ad3dd08299522697dc331e62f0678080f89096bb8ebd423) envoyé à **00:17:48**. Notation serveur en cours ; reçu dans `docs/slz-jobs/recu_t3_push_s150.txt`. push_s125 intact (sha vérifié avant l'envoi). Aucun envoi de push_s200, aucune série automatique.
- Recadrage (Selyan, 00:20) : **transformer les expériences terminées en décisions**, plus de nouvelles familles de proxys ni de grilles.
  - T3 : le critère « KL sur les classes oubliées ≥ push_s125 » est **retiré**. La décision NPO+réparation reste un **rejet** à cause de l'**effondrement sur les classes non concernées** (contrôle 0,785 d'accord, 21,5 % de changements, contre 1,000 pour s125), indépendamment de ce critère.
  - T2 : seuil d'action d'un **petit lot noté** = MCC ≥ **0,60 dans les deux sens**, déguisements regroupés, signal sur les variantes faibles ; **GO distinct** requis. Décision ferme à 01:15–01:30.
  - T1 : **piste TrustMark fermée pour la nuit** (+0,0091 estimé < +0,01).

### 00:21 — **T3 : push_s150 = 0,0827 public (nouveau meilleur, +0,026)**. push_s125 intact (sha b28ea783…).
Série serveur de la tête push : ×1 0,0308 ; ×1,25 0,0567 ; ×1,5 **0,0827**, soit environ +0,026 par pas de 0,25, encore linéaire. La porte retain des 70 % tient à ×1,5. Suite à décider par Selyan : aucune série automatique.

### 00:24 — Décisions Selyan : **push_s150 nouveau champion T3** ; série d'intensités CLOSE ; ×1,75 préparé, non soumis
- Ne pas présenter la progression ×1 → ×1,5 comme une loi linéaire : trois points, rien ne garantit la suite (porte retain, optimum gold inconnu).
- `push_scale/push_s175.pt` construit sur le login (CPU) : tête = origine + 1,75 × (push_best − origine). Formule vérifiée en reconstruisant push_s125 (erreur max 3e-8, autres tenseurs identiques). Poids finis, float32. **sha 54c2aeccbf68965890be1818172d0c48f89f24a6a15251b0bade6fe34dcb464f**. **NON soumis.**
- Mesures locales ×1,75 (cache des features, moitié réparation) : classes concernées, accord 0,051 / KL 1,61 ; autres classes, 0,998 / 0,44 ; flip forget 41,7 % ; CE forget 1,43/1,85 ; off_agree 0,55. Pour comparaison, ×1,5 : 0,088 / 1,22 ; 0,998 / 0,35 ; 22,4 %.
- **GPU** : 0 job depuis la fin de 15903998 (~00:29). Rien d'utile en file : T1 fermé, T3 sans entraînement en attente, T2 attend la décision du banc (01:15–01:30).

### 00:38 — T1 : job 15904061 `slz_t1_pre` (GO Selyan « lance la segmentation pré-entraînée T1 sur JURECA »), session Audit global
- Hypothèse (audit global 22/11) : A1 a montré que l'extérieur est recollé, donc la zone éditée est une région générée, détectable par une forensique apprise. Une dizaine d'équipes atteignent 0,74–0,77 en T1. L'option `--backbone` de pixseg n'avait jamais tourné.
- `work_pixseg_v2/pre_devel.sbatch` (copie de `run_bf.sbatch` : dc-gpu-devel, 55 min, HF_HOME du cache) + `configs_pre.txt` : réel seul, 5 folds de gated_eval, lr 3e-4, 2 000 pas, bs 16. GPU 0-2 : convnext_tiny (in12k) folds 0,1 / 2,3 / 4 ; GPU 3 : convnext_small folds 0,1. Poids téléchargés sur le login (timm/convnext_tiny.in12k_ft_in1k, convnext_small.in12k_ft_in1k, resnet50.a1_in1k).
- Relevé avant envoi : 0 job d'équipe ; `--test-only` départ immédiat.
- **Règle fixée avant lecture** : POURSUIVRE (fit-all + fusion, mesure routée contre le champion) si le mIoU OOF du réseau seul sur les 5 folds ≥ **0,58**, ou si un groupe F gagne ≥ +0,05 contre head[0]. ABANDON si < 0,52. Entre les deux : DIAGNOSTIC.
- Aucune soumission prévue par ce job.

### 00:56 — **Décision T2 : aucun lot ep** (banc refs rendu, `docs/task2-bench-20261121/RAPPORT.md`)
- API : seuls `max_tokens` et `messages` (user/assistant) sont honorés ; temperature, seed, top_p, logprobs, n et stop sont ignorés ; system est refusé ; environ 15 s par requête.
- Boîte noire (listes p3) : MCC hors sélection 0,000/0,000 (Hellinger) et 0,000/0,273 (Spearman). **Échec** : persona et température s'éloignent plus de la base que fp8 ou la LoRA 0,3.
- Boîte blanche (SmolLM2-360M identifié sur les refs) : 0,000/0,693 ; la seule configuration à environ 0,69/0,72 souffre d'une fuite de sélection. **Non validée.** Il faudrait environ 88 réponses par endpoint, les checkpoints exacts des 3 bases ep (non identifiés, aucun poids Qwen/Llama en cache sur JURECA) et un GPU.
- 470 réponses ref consommées (gratuites). 0 requête ep. 95/endpoint toujours intacts.
- Conformément au seuil fixé (≥ 0,60 dans les deux sens), **pas de lot ep**. T2 reste à 0,20 (CSV aléatoire).

### 01:00 — Levier T2 trouvé (hypothèse, à valider hors ligne) : détecter les DÉGUISEMENTS, pas les variantes
`practice_answers.csv` (colonne `kind`) : **seul le modèle de base porte des déguisements** (01 plain, 03 system prompt, 05 persona, 06 watermark, 08 temp 1,0) ; fp8 et LoRA sont « plain ». Si les ep suivent le même générateur : endpoints déguisés = un même modèle (la base), chaque autre plain = un modèle isolé. Sur les refs, cette partition donne un MCC de 0,72 (1,0 si l'on rattache le bon plain). Détecteurs visés : persona/system (identité, style), watermark (réponses répétées à l'identique), température (diversité). Agent banc T2 relancé : validation hors ligne sur les données refs existantes (seuil ≥ 0,60 dans les deux sens), lecture des 165 réponses ep, puis lot ep minimal préparé (≤ 16/endpoint), **sans lancement avant le GO de Selyan**.

### 01:00 — Hypothèse « déguisements » : premier verdict hors ligne NÉGATIF au budget visé
- Détecteurs sur les refs : identité par défaut absente (system/persona, mais la LoRA 1,5 la perd aussi), doublons exacts (watermark), similarité intra-prompt (temp). Partition P1 hors sélection : 0,222/0,231 avec seuils libres ; 0,734/0,734 avec un amendement post hoc, non fiable.
- Sous budget : 16 réponses/endpoint donnent 0,21/0,24 (P(≥ 0,60 dans les deux sens) = 0,01). **Pas de lot ep.**
- Dans les 165 réponses ep : identité perdue pour ep-08 et ep-20 (famille B) ; indices faibles pour ep-02, 14, 23 (A). Rien de détectable en famille L.
- Validation ciblée lancée sur les **refs** (GO refs) : `p4_disguise_battery.py`, 16 prompts conçus pour les détecteurs × 4 batteries × 9 refs = 576 requêtes ref, PARALLEL 3. Règle fixée en tête du fichier : MCC ≥ 0,60 dans ≥ 10 directions sur 12. Un lot ep (528) n'est envisagé que si ce test passe, puis GO de Selyan.

### 00:59 — Batterie p4 (refs uniquement) lancée par Selyan dans son terminal
`c2run/bench_p4_script.py` (benchlib + p4), PARALLEL 2, 576 requêtes ref, timeout 5400 s. Journal `c2run/bench_p4.jsonl`. Analyse automatique à la fin (`analyze_p4.py`, avec et sans ident) par l'orchestrateur. Réversible : refs gratuites, aucun budget ep consommé ; 2 des 4 places de requêtes simultanées occupées pendant environ 77 min.
Autre session active (Selyan), hors orchestrateur : jobs T1 `slz_t1_pre` (15904061, 73, 98, 127) et T3 `slz_t3_gen` (15904093) sur devel ; **2 jobs simultanés à 00:56** (98 et 127). Candidats T3 `push_scale/clsA_kill_p150`, `clsB_kill_p125`, `clsC_kill_p175`, `clsD_f250_p150` ; soumission de clsA programmée par Selyan depuis son terminal. Reçus à verser au registre.

### 01:16 — Classement public 01:14 (session Audit) : SLZ 26e/35, général **0,324**
- **T1 0,6137** (amélioré à 00:48:53, fichier inconnu, autre session) ; **T3 0,0941** (amélioré à 00:40:19, probablement `clsA_kill_p150`, soumis par Selyan depuis son terminal) ; T2 0,20. Max T1 du tableau : 0,7935.
- **Reçus manquants** : fichiers et sha256 de ces deux soumissions sont à verser au registre (Selyan / autre session).
- Recouvrements de jobs d'équipe (autre session, devel) : 00:38–00:42, 00:56–01:01, et plusieurs de moins d'une minute.

### 01:20 — GO Selyan : identification T2 en boîte blanche sur GPU
- Poids ouverts téléchargés sur le login (jrlogin03, nohup + timeout 5400) dans **/p/project1/training2666/mouhali1/hfmodels/** (et non sur scratch) : Qwen2.5-0.5/1.5/3/7B-Instruct, Qwen3-0.6/1.7/4B, unsloth Llama-3.2-1/3B-Instruct, unsloth Meta-Llama-3.1-8B-Instruct. Journal `dl.log`.
- **ALERTE inodes /p/scratch : 564 libres sur 4,2 M** (01:18). Tout nouveau fichier SLZ va sur /p/project1 (1,9 M libres). Les jobs qui écrivent des logs ou des sorties sur scratch risquent d'échouer, pour nous comme pour les autres équipes.
- Agent banc T2 : prépare `wb_ep_ident.py` + sbatch (devel, 4 GPU, ≤ 30 min), vraisemblance des 165 réponses ep sous chaque candidat, puis LLR fp8/LoRA sous la base identifiée. Sorties sur project1. Lancement par l'orchestrateur.

### 00:40–02:20 — session Audit global (soumissions lancées par Selyan depuis son terminal)
| Tâche | Fichier | Public | Note |
|---|---|---|---|
| 3 | push_scale/push_s175.pt (54c2aecc) | **0,0941** | nouveau meilleur T3 |
| 1 | output/c1_pre_ss_cnt2.pt (ed51874f) | 0,6137 | champion + convnext_tiny pré-entraîné (réel seul, 2 graines) sur les 340 SS ; prévu +0,03 |
| 3 | push_s200.pt | 0,0792 | optimum push ≈ ×1,75 |
| 3 | clsA_kill_p150.pt (580eae9f) | 0,0205 | supprimer les 9 classes à 130 forget : rejeté |
| 3 | clsF_f125_p175.pt | 0,0936 | facteur des 9 classes sans effet |
| 3 | sep_out/sep_p250_l3_b8.pt | 0,0317 | fine-tune « préserver les variations img2img » : hypothèse réfutée (le gold baisse toute la classe) |
| 1 | output/c1_pre4k_ens5.pt (48e467b7) | **0,6377** | 4 réseaux 4000 pas + 1 réseau avec cartes heats, SS seulement |
Jobs : 15904061/073/098/127/159 (T1 pré-entraîné), 15904093 (img2img sd-turbo, gen_out/), 15904196/205/227/266 (séparation T3), 15904283 annulé, 15904308 (T1 8000 pas). Sources : docs/slz-jobs/t3gen/, work_pixseg_v2/configs_pre*.txt. Inodes : caches uv/pycache nettoyés (≈9 800 fichiers) sur demande des organisateurs.

### 01:38 — T2 : batterie p4 = **ÉCHEC** (0/12 directions) ; boîte blanche phase 1 = **bases identifiées**
- p4 (576 réponses ref, 0 erreur) : MCC P1 de 0,27 à 0,72 selon la direction, aucune ≥ 0,60 avec déguisements ≤ 0,1 ; sans ident : 0/12. **Piste « déguisements en boîte noire » close.**
- Job 15904307 `j7` (devel, 4 GPU, 3 min 41 s, sorties `/p/project1/training2666/mouhali1/wb_ep/result_15904307.json`) : vraisemblance des 165 réponses ep sous 10 candidats. **Famille A = Qwen2.5-3B-Instruct (11), famille B = Qwen2.5-1.5B-Instruct (11), famille L = Llama-3.2-1B-Instruct (11)** ; accord 528/528 avec les familles de `c2_fam3_pairs_v2.csv`. Marges nettes (environ 0,4–0,5 nat/token).
- Phase 2 (température, LLR fp8, calibration, sous la base) : trop bruitée avec 5 réponses (se LLR8 de 6 à 70 mnat). Indices seulement : T estimée élevée pour ep-02 (1,2), ep-23 (1,4), ep-33 (1,1), ep-20 (1,0), ep-31 (1,0) ; LLR8 le plus haut pour ep-20 (+14 ± 6,5) et ep-03 (+6 ± 7).
- **Recouvrement de jobs** : 15904307 (orchestrateur) a démarré en même temps que 15904308/15904313 (autre session) : 3 jobs d'équipe en même temps à 01:33. Il faut un lanceur unique.
- 01:40 — GO Selyan pour **préparer** une collecte ep en boîte blanche : 30 réponses par endpoint (990), précédée d'une calibration par simulation GPU (base T 0,7/1,0, fp8 simulé, persona/système, LoRA aléatoire comme substitut) et d'une règle de partition figée. La collecte elle-même n'est lancée que sur GO de Selyan, budget irréversible : 95 → 65 par endpoint.
- 01:44 — Session « Audit de progression » (CPU seulement, aucun job) : décision par espérance de mIoU image par image sur les OOF `pixseg_cv_pre*`. Le champion T1 serait `output/c1_pre4k_ens5.pt` (01:09), **0,6377 public selon cette session, à vérifier au classement**. Jobs T1 `pre_devel60.sbatch` (15904308/13) = autre session de Selyan.
- 01:47 — **Correctif** : champion T1 public = **0,6377**, `output/c1_pre4k_ens5.pt`, sha **48e467b7c9aad71654ba490ace9a4201d22e98eef43e752a5576985ecbe4c99d** (vérifié). Classement relevé à 01:47 : SLZ **26e**, T1 0,64 / T2 0,20 / T3 0,09, général **0,33**. Étape CPU T1 de la session « Audit de progression » en cours ; règle fixée : POURSUIVRE si gain global apparié ≥ +0,01 et borne basse IC95 > 0.
- 01:46 — **Simulation T2 wb2 lancée : job 15904367 `j8`** (devel, 4 GPU, 30 min max, après relevé « 0 job »). Scripts `wb2/` copiés dans `/p/project1/training2666/mouhali1/wb2/`. Par base (Qwen2.5-3B, Qwen2.5-1.5B, Llama-3.2-1B), 9 endpoints simulés × 2 réplicats × 30 réponses ; même modèle = plain ×2, T 1,0, système, persona ; distincts = fp8, LoRA aléatoires rang 8 à 0,01/0,03/0,1 (**substituts**). Règle : fp8 si LLR8 > l8 ; sinon isolé si GAP hors [glo, ghi] ; sinon base. Sorties : `sim_15904367.out`, `rule.json`, `sim_report.json`. La collecte ep (990 réponses) attend le GO de Selyan.

### 01:52 — T2 : changement de stratégie, « sondes à quasi-égalité » (boîte blanche active)
- Canaux annexes vérifiés et **fermés** : latence rembourrée à 6–10 s, identique pour les 9 refs ; pas de champ usage ; id `chatcmpl-` aléatoire ; en-têtes nginx neutres.
- Nouvelle méthode (agent « ties ») : avec les poids de base, chercher hors ligne des prompts dont le 1er token est à quasi-égalité (p1 ≈ p2). Une égalité de logits survit à tout changement de température, alors que fp8 et LoRA la décalent. Plus des prompts « bascule fp8 » (bf16 et fp8 simulés donnent des tokens différents avec forte confiance). Interrogation `max_tokens=1` répétée.
- Étapes : (1) poids SmolLM2-360M ; (2) job GPU de recherche de sondes pour les 4 bases (lancé par l'orchestrateur) ; (3) **validation gratuite sur les refs** avec règle figée, seuil ≥ 0,60 dans les deux sens ; (4) seulement ensuite, lot ep ≤ 60/endpoint, sur GO de Selyan.
- En parallèle : simulation wb2 15904367 en cours.
- 01:52 — **T1, décision par espérance de mIoU (session « Audit de progression ») : ABANDON.** Gain global imbriqué −0,0001, IC95 [−0,0008 ; +0,0005], identique pour les variantes Monte-Carlo, ratio et Platt. OOF pre_cnt et prem sur 5 folds ; combine_eval reproduit (0,5688 / 0,5777). Scripts : scratchpad de cette session, `t1_emiou/`. Piste suivante suggérée au lanceur T1 : OOF 5 folds de la recette du champion (4000 pas, plusieurs graines) ; plus gros écart : StegaStamp R (0,58–0,62) contre F (0,72–0,73).
- 01:51 — **Simulation wb2 15904367 : ÉCHEC** (5 min 25 s). MCC hors sélection, sens 0→1 / 1→0 : Qwen1.5B 0,03/0,00, Qwen3B 0,52/0,00, Llama1B 0,00/0,23 ; bootstrap moyen 0,14–0,23. Les statistiques de vraisemblance (T, LLR8, GAP) ne séparent ni le fp8 (poids seuls, par canal) ni les LoRA aléatoires, **même en vérité connue**. → **Pas de collecte wb2** (on économise 990 réponses ep). Seule piste T2 restante : les sondes à quasi-égalité (`ties_search.py` en cours d'écriture).
- ~02:00 — **T1, 8000 pas contre 4000 pas** (convnext_tiny, réel seul, folds 0–1, combine_eval imbriqué ; session « Audit de progression », CPU). 8k seul : −0,0042 [−0,0122 ; +0,0040], SS F −0,041 → **ABANDON** (on garde les réseaux 4k). Ensemble 4k+8k : +0,0049 [+0,0002 ; +0,0102] → DIAGNOSTIC sous +0,01, confondu avec l'effet « 2 réseaux contre 1 » (le champion est déjà 4×4k). OOF : pre8k_f0 ec5ea675, pre8k_f1 d859ed12, pre_cnt4k_f01 45402676. Rien de soumis.
- 02:00 — **T2 : job 15904400 `j9`** (recherche des sondes à quasi-égalité, devel, 4 GPU, 25 min ; SmolLM2-360M + 3 bases ep), lancé après relevé « 0 job ». 0 requête API.
- 01:58 — **15904400 COMPLETED (2 min 47 s)**. Sondes dans `/p/project1/training2666/mouhali1/ties/ties_{smol,A,B,L}.json`. Sur 2588 prompts candidats par base : 184–211 quasi-égalités, 68–125 robustes, 56–80 bascules fp8. Point d'alerte : l'écart induit par le contexte (persona/système) sur les égalités, en médiane 1,8–3,8 logit, dépasse de loin le décalage fp8 (0,25–0,75). Le déguisement risque donc de masquer le signal ; seules les sondes « robustes » sont utilisables. Validation sur les refs : agent « ties ».
- 02:03 — Sondes v1 : **aucune égalité robuste au contexte** sur A/B/L (0/184, 0/211, 0/156). Le seul prompt système « You are a helpful assistant » décale une égalité 1,5 à 3 fois plus que le fp8. SmolLM2 : 0 égalité. **Job 15904419 `j10` (ties2, pool v2)** lancé après relevé « 0 job » : séquences à continuer, historique pré-rempli, long préambule. Critère GO/NO-GO figé : ≥ 8 sondes SmolLM2 avec masse top-2 ≥ 0,8 dans tous les contextes, décalage dû au contexte < 0,3 nat et |fp8| ≥ 0,25 nat. Sinon, **piste T2 close**.
- 02:01 — **T1 : job 15904421 `slz_t1_sr`** (session « Audit de progression », GO Selyan « vas-y lance ça »). Devel, 4 GPU, 45 min max ; pixseg CV folds 0–1, 4000 pas, 4 variantes visant les SS R (graine 1, crop 512, BCE 0,5 + IoU 2, convnext_small). Sorties dans /p/project1/.../t1sr/. Règle : POURSUIVRE si gain apparié dilué ≥ +0,01 et borne basse IC95 > 0. **2 jobs d'équipe en parallèle** (15904419 T2 + 15904421 T1), sur GO de Selyan pour la T1.
- 02:08 — **Sondes v2 (15904419) : critère GO rempli pour SmolLM2** (30 sondes avec masse top-2 ≥ 0,8 dans tous les contextes, décalage dû au contexte < 0,3 nat, |fp8| ≥ 0,25). Bases ep : A2 **0**, B2 10, L2 6. **La famille A (Qwen2.5-3B) n'a aucune sonde robuste.** 12 sondes choisies (`probes_smol2.json` : 7 égalités + 5 bascules ; `ties_rule.py` corrigé pour les prompts en liste de messages). **Validation refs lancée** sur jrlogin03 (nohup, timeout 5400) : 9 refs × 12 sondes × 10 répétitions = 1080 requêtes ref, PARALLEL 3, journal `/p/project1/.../ties/ties_refs.jsonl`. Analyse automatique `ties_rule.py analyze` à la fin (règle figée : MCC ≥ 0,60 sur chaque moitié, 03/05/06/08 dans le groupe base).
- 02:27 — Tableau de bord (session Audit), classement 02:24 : SLZ **28e/37, général 0,330**. **T1 0,6391** (amélioration à 01:44:59, fichier inconnu : reçu à verser) ; **T3 0,0941 = push_s175** (54c2aecc…, soumis hors orchestrateur), clsA 0,0205, push_s200 0,0792, clsF 0,0936, sep_p250 0,0317. Inodes /p/scratch remontés à 20 708 libres.
- 02:30 — T1 15904421 (session « Audit de progression ») : **DIAGNOSTIC**. BCE 0,5 + IoU 2 en ensemble : +0,0049 [+0,0007 ; +0,0093], SS R +0,030 ; crop 512 et convnext_small ABANDON ; variance entre graines ≈ 0,004. Sorties /p/project1/.../t1sr/.
- 02:31 — **T2 : job 15904551 `j11`** (sondes v3 pour la famille A seule : 4 graines de pool v2 = 10 000 prompts Qwen2.5-3B), objectif : trouver des sondes robustes pour A (0 en v2). En parallèle de 15904515 (T1, autre session) : Selyan demande les GPU « à fond ». Choix des sondes B2/L2 en CPU sur le login (`CUDA_VISIBLE_DEVICES=""` ; un premier essai avait touché le GPU du login par erreur, OOM, sans effet). Validation refs : 541/990 à 02:28.
- 02:32 — **T1 : job `slz_t1_sr2`** (session « Audit de progression », GO Selyan « Oui, confirme »). Devel, 4 GPU, environ 25 min. CV folds 2–4 IoU2 + fit-all IoU2 (graines 40/41, --predict-test) ; sorties /p/project1/.../t1sr/. Règle sur 5 folds, ensemble 4k+IoU2 contre 4k+4k : POURSUIVRE si ≥ +0,01 et borne basse > 0 ; DIAGNOSTIC entre +0,003 et +0,01 (décision Selyan) ; sinon ABANDON. **3 jobs d'équipe en parallèle** (j11 T2, slz_t1_psyn, slz_t1_sr2), sur demande de Selyan (« GPU à fond »).
- 02:36 — Limite Slurm : **2 jobs au plus en même temps par utilisateur sur devel** (QOSMaxJobsPerUserLimit). En file : slz_t1_sr2 15904559 et une 2e psyn 15904561. L'orchestrateur ne soumet plus rien sur devel avant le démarrage de 15904559. Sondes ep choisies : `probes_B2.json` (5), `probes_L2.json` (8). Script de collecte ep `ties_query_ep.py` copié sur /p/project1 : il exige `--go-selyan`, relit le budget avant et après, s'arrête après 5 erreurs consécutives ou 20 au total. **Non lancé.**
