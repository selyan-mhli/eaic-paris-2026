# Task 2 — profilage du contexte caché (wbctx), refs seulement (22/11/2026)

Aucune requête API (ni ref ni ep), aucune soumission, aucun sbatch, aucune opération Git.

## Verdict : **NO-GO** (critère figé non atteint)

- Le critère MCC médian ≥ 0,45 dans les deux sens à 30 réponses échoue :
  - H0→H1 : **0,446** ;
  - H1→H0 : **0,331**.
- La précision d'isolement passerait : 0,79 et 0,92.
- Les scripts ep n'ont donc pas été déposés dans ce dossier. Des brouillons non testés existent hors dépôt, voir plus bas.

Note du coordinateur (05:20) : le MCC « même modèle » officiel se calcule sur les paires intra-famille, ce qui place le point mort vers P(distinct) ≈ 0,5. Le critère GO est resté inchangé, comme demandé.

## Règle figée
- Texte `RULE` en tête de `analyze_ctx.py`, sha256 **94a3e0eb2889f08372e7d17ef906be06ba5a9a4f07f0497f61275b29dc8dfd7b**.
- Fichier `analyze_ctx.py` final : 422fe699…, après l'amendement A1.
- Journal des sha, de l'amendement A1 (correction d'un TypeError numpy dans la clé de départage, faite avant tout résultat réel, test sur données aléatoires) et du pré-engagement sur les seuils ep : `RULE_SHA256.txt`.

## Calcul
- `compute_ctx.py` : SmolLM2-360M-Instruct sur MPS, batch 1. Contextes C0..C5 du brief :
  - C0 = gabarit par défaut, qui injecte « You are a helpful AI assistant named SmolLM, trained by Hugging Face » ;
  - C1 = system vide forcé ;
  - grille T de `wb_stats.py`.
- fp8 e4m3 simulé comme dans `wb_stats.py`, mais l'arrondi e4m3 est émulé sur le device : identique bit à bit au cast torch sur 10^6 valeurs, CPU et MPS, et 0 écart sur un lot test de 20 réponses.
- **Durée : 3 396 s (57 min)** pour 1 584 réponses × 6 contextes × (base + fp8). Le seuil des 40 min est dépassé : le lot test prévoyait 24 min, mais les réponses longues ont ralenti le calcul. Je n'ai pas basculé vers le fp8 au seul contexte retenu, parce que le contexte retenu dépend du tirage et que le calcul complet était déjà lancé.
- Sortie : `ctx_refs.npz` (sha256 37bcb18c…), un enregistrement par réponse, pas par token.

## Features (moitiés complètes, 88 réponses) — profilé
| ref | F1 H0/H1 | z1 | F2 mnat H0/H1 (se) | z2 | (c*,T*) |
|---|---|---|---|---|---|
| 01 plain | −0,640/−0,625 | +0,00/+0,85 | −25,8/−26,5 (4,4) | −0,41/−0,55 | C0,0,7 / C0,0,7 |
| 02 fp8 | −0,613/−0,699 | +0,32/+0,00 | **−8,0/−11,7** (3,4) | **+1,81/+1,70** | C0,0,7 / C0,0,7 |
| 03 sys | −0,602/−0,736 | +0,45/−0,43 | −23,7/−28,7 (4,5) | −0,13/−0,87 | C5,0,7 / C2,0,8 |
| 04 LoRA 0,7 | −0,583/−0,664 | +0,67/+0,40 | −16,9/−18,2 (2,1) | +0,71/+0,72 | C0,0,8 / C0,0,8 |
| 05 persona | −0,748/−0,726 | −1,28/−0,32 | −27,1/−18,5 (5,8) | −0,57/**+0,67** | C1,0,8 / C0,0,8 |
| 06 wm | −0,678/−0,640 | −0,46/+0,67 | −22,6/−27,0 (4,0) | +0,00/−0,62 | C0,0,7 / C0,0,7 |
| 07 LoRA 1,5 | −0,884/−0,821 | −2,89/−1,42 | **−11,9/−13,3** (2,2) | **+1,33/+1,46** | C0,0,9 / C5,0,9 |
| 08 temp 1,0 | **−1,873/−1,995** | **−14,6/−15,0** | −29,6/−22,9 (3,4) | −0,88/+0,00 | C0,1,0 / C0,1,0 |
| 09 LoRA 0,3 | −0,569/−0,605 | +0,83/+1,08 | −17,2/−25,9 (3,2) | +0,67/−0,45 | C0,0,7 / C0,0,7 |

Contrôle C0 seul : les mêmes valeurs, sauf pour 03, 05 et 07. Détail complet dans `analyze.log` et `cv_results.json`.

## Validation hors sélection (MCC paires des 9 refs ; précision = isolés vraiment distincts)
| Features | Sens | 88/88 MCC (isolés) | 30/30 MCC médian [q10, q90] | 30/30 précision moy. | tirages sans isolé / avec ≥1 déguisement isolé | 88→30 MCC médian, précision |
|---|---|---|---|---|---|---|
| **Profilé** | H0→H1 | 0,446 (02,04,05,07) | **0,446** [0,000 ; 0,734] | 0,79 | 27 / 106 sur 200 | 0,446, 0,78 |
| **Profilé** | H1→H0 | 0,734 (02,04,07) | **0,331** [0,000 ; 0,734] | 0,92 | 42 / 34 | 0,331, 0,93 |
| Contrôle C0 | H0→H1 | 0,446 (02,04,05,07) | 0,331 [0,000 ; 0,734] | 0,84 | 36 / 78 | 0,446, 0,80 |
| Contrôle C0 | H1→H0 | 0,331 (02) | 0,331 [0,000 ; 0,524] | 0,84 | 32 / 66 | 0,331, 0,88 |

Valeurs de référence : 07 seul = 0,331 ; 04+07 = 0,524 ; 02+04+07 = 0,734 ; les quatre distincts = 1,0 (recalculées avec `anlib.mcc`).

Fréquence d'isolement, profilé, 30/30 :
- H0→H1 : 07 0,83, 02 0,74, 04 0,41, **05 0,39**, 06 0,11 ;
- H1→H0 : 02 0,71, 07 0,46, 09 0,24, 04 0,20, 03 0,11.

## Lecture
1. **F1 n'isole rien.** s1 = −inf dans toutes les sélections 88/88. La logvraisemblance maximale reste dominée par 08 (temp 1,0), à z1 ≈ −15 : profiler T ne rend pas une réponse échantillonnée à T=1 aussi probable qu'une réponse à T=0,7, car l'entropie propre du texte baisse sa vraisemblance. Comme le déguisement le plus extrême est le plus bas, aucun seuil sur F1 n'isole sans toucher 08.
2. **Le profilage de contexte ne rattrape pas les déguisements.**
   - 03 (sys) choisit C5 ou C2, avec un gain de F1 faible (−0,618 → −0,602).
   - 05 (persona) reste au contexte par défaut. Sa F2 sur H1 (z2 +0,67) le fait isoler dans 39 % des tirages H0→H1.
   - Le prompt système réel des refs n'est probablement pas dans la grille. Je n'ai pas cherché à le deviner, pour ne pas ajuster la grille sur 03/05.
3. **Le seul signal robuste vient de F2 (fp8 simulé).** Il isole 02 dans 71–89 % des tirages et 07 dans 46–97 %, avec une précision de 0,79–0,93. Mais à 30 réponses, le MCC médian plafonne à 0,33–0,45. L'apport du profilage sur le contrôle C0 est de +0,115 de MCC médian dans un sens et nul dans l'autre (H1→H0, 0,331 des deux côtés), donc faible.
4. **Hors critère, à titre d'information.** Avec le point mort corrigé (P ≈ 0,5), une précision moyenne de 0,8–0,9 pourrait rendre l'isolement de 02/07 rentable en espérance sur les ep. Ce n'est pas le critère figé, et la fraction de tirages isolant au moins un déguisement (17–53 %) n'est pas négligeable. Je ne requalifie pas le verdict.

## Non vérifié
- Le transfert du fp8 simulé (e4m3 par tenseur) vers le vrai fp8 du serveur ep, en dehors de ref-02.
- La transposition des z intra-famille de 9 refs (4/9 distincts) à 11 ep, où la proportion de distincts est inconnue.
- Les intervalles des features : seulement l'erreur type par réponse. Les tirages sont faits au niveau prompt, pas de bootstrap supplémentaire.
- Les brouillons ep (`collect_ep.py`, `score_ep.py`, `freeze_thresholds.py`, `score_ep.sbatch`) se trouvent dans le scratchpad de session, hors dépôt. Ils n'ont **pas** été testés de bout en bout (pas de test CPU Qwen2.5-0.5B, puisque NO-GO). Seul le dry-run de `collect_ep.py` a tourné : 30 prompts, 990 requêtes, refus sans `--go-selyan`.

## Reproduire
```
cd docs/task2-wbctx-20261122
../../.venv/bin/python compute_ctx.py HuggingFaceTB/SmolLM2-360M-Instruct ctx_refs.npz   # 57 min MPS
../../.venv/bin/python analyze_ctx.py ctx_refs.npz cv_results.json                       # ~10 s
```
