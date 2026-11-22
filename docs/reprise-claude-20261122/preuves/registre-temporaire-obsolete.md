# Registre des soumissions SLZ (tenu par la session de coordination)

Règles : un fichier n'est soumis que si son sha256 n'est pas dans ce registre, que son format et son fp16 (Task 3)
sont vérifiés par la session qui le propose, et que le délai de la tâche est écoulé (délai par tâche : Task 1 et
Task 3 environ 10 min, Task 2 30 min ; une soumission Task 1 a été acceptée 20 s après un refus Task 3, le 21/11 à 19:53).
Scripts : Task 1 `work_c1/submit_c1.py <fichier>` (dossier Hackathon_Watermark_Localisation_Task, `.venv`),
Task 3 `submit_c3.py <fichier>` (dossier unlearning/work_c3_eval, `../.venv`). Heures de Paris.

## Soumis

| Heure | Tâche | Fichier | sha256 (début) | Score local | Score public | Par |
|---|---|---|---|---|---|---|
| 16:48 | 2 | baseline_random.csv (aléatoire du kit) | — | — | 0,20 (chance) | Sophie |
| ~17:34 | 3 | inconnu | — | — | 0,0026 | Yihuan ? |
| 18:22 | 1 | seg_fsynth_rect_v1.pt | 302ee55e | 0,547 | 0,5307 | Selyan |
| ~18:40 | 3 | soup_mid.pt | 10ee30f6 | proxy | 0,0012 | Selyan |
| 18:52 | 3 | soups/soupB_l85.pt | 69435b1b | proxy | 0,0035 | Selyan |
| 19:12 | 1 | seg_fsynth_rect_v1.pt (doublon) | 302ee55e | 0,547 | 0,5307 | coordination |
| 19:15 | 3 | distill_out/m75/distill_r45.pt | af207210 | CE 0,03 | 0,0044 | Selyan (probable) |
| 19:26 | 3 | distill_out/m75/distill_r45.pt (doublon) | af207210 | CE 0,03 | 0,0044 | coordination |
| ~19:46 | 3 | inconnu (n'a pas amélioré) | — | — | ≤ 0,0044 | inconnu |
| 19:53 | 1 | c1_segf_ffm_e20_20261121.pt | 51447085 | 0,5583 | **0,5572** | coordination |

| 19:57 | 3 | anchor_ckpt/anc_f2a_s0550.pt | 93b67cce | ancré, 8,4 % changées | 0,0039 | coordination |
| 20:05 | 1 | c1_pixss_fit0.pt (e20 + pixseg StegaStamp, 4 réseaux, seuil 0,2) | da6abc31 | — | **0,5845** | veilleur |
| 22:11 | 3 | push (tête, logit gagnant −alpha, Yihuan) | — | CE 0,087, 0 changée | **0,0308** | Yihuan |
| 22:52 | 2 | c2_lump_v1.csv | b4ee773a | — | 0,05 | coordination |
| 22:57 | 3 | work_c3_t1t2/out/t1t2_p03_e2.pt | 8b257641 | CE 0,106, 3,1 % changées | 0,0091 | coordination |
| 23:07 | 3 | work_c3_eval/push_scale/push_s125.pt (tête Yihuan ×1,25, alpha 10) | b28ea783 | non mesuré | **0,0567** | auto_c3.sh |

Programmé : push_s150 (alpha 12) ~23:18, push_s200 (alpha 16) ~23:28 (auto_c3.sh, PID 33527).
23:07 : gel GPU ordonné par Selyan via la session « EAIC Paris 2026 multi-agent orchestration » ; plus aucun sbatch hors de cette session.

Programmé : Task 2 c2_fam3_pairs_v2.csv (6799a438) à 23:22 (script auto_c2.sh).
Leçons serveur Task 3 : changer de classe ou étaler l'entropie fait baisser le score ; baisser le seul logit gagnant
sans changer de classe le fait monter. File Task 3 vidée : attendre les candidats « marge » de c3-submit ou de Yihuan.

Possiblement soumis par Selyan, à confirmer : c1_segf_ffm_e60_20261121.pt (3c257b16), soups/soupB_l95.pt (1575e4f0),
anchor_ckpt/anc_r05_lr2e5_s0400.pt (8494fc4d).

## File d'attente (ordre)

- Task 3 : 1) anchor_ckpt/anc_f2a_s0550.pt (93b67cce, programmé ~19:57) ; 2) work_c3_t1t2/out/t1t2_p06_e2.pt ;
  3) selon scores : anc_r1_lr1e5L_s0800.pt (eeebc09a) ou anc_f1a_s0600.pt.
- Task 2 : model_equality/output/c2_lump_v1.csv dès que la session Task 2 l'a écrit et vérifié.
- Task 1 : prochain candidat seulement s'il bat 0,5583 en validation imbriquée d'au moins +0,01.
