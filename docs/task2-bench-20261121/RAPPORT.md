# Task 2 — banc d'empreintes sur ref-01..09 (21-22/11/2026)

Aucune requête ep, aucune soumission, aucun sbatch. Journaux : `c2run/bench_p1.jsonl`, `bench_p2.jsonl`, `bench_p3.jsonl` (+ `bench_p1_fail401.jsonl` : 1 GET refusé, clé mal parsée).

## Point d'étape 23:58 Paris (demandé par le coordinateur)

1. **API** : seuls `max_tokens` (>=1, y compris 1) et `messages` (rôles user/assistant, 1 à 32 messages, 1024 tokens d'entrée) sont honorés. `temperature`, `seed`, `top_p`, `logprobs`/`top_logprobs`, `n`, `stop` sont ignorés silencieusement (pas de champ logprobs, 1 seul choix, réponses toujours échantillonnées). Le rôle `system` est refusé (400 « not counted »), et `/completions` renvoie 404. `finish_reason` vaut toujours `stop`. Latence d'environ 15 s par requête, quel que soit `max_tokens` : environ 7,5 réponses/min à PARALLEL=2. Le pré-remplissage assistant n'est pas concluant.
2. **Méthode** : aucune n'est encore validée hors sélection. En cours : (a) p3, des listes « aléatoires » (4 prompts × 12 répétitions, analyse pré-enregistrée dans `p3_lists.py`/`analyze_lists.py`) ; (b) une piste en boîte blanche. Le modèle des refs est identifié comme **SmolLM2-360M-Instruct** (−0,50 nat/token contre −0,82 à −0,96 pour 4 autres candidats). La boîte blanche teste la vraisemblance des réponses existantes : T estimé, LLR contre un fp8 simulé, écart de calibration (`wb_stats.py`, `analyze_wb.py`), sans aucune requête.
3. **Réponses consommées** (200 OK, à 23:57) : ref-01 29, ref-02 13, ref-03 12, ref-04 12, ref-05 12, ref-06 12, ref-07 17, ref-08 23, ref-09 12. p3 vise 48 réponses par ref, fin vers 00:45 Paris.
4. **Recommandation ep (oui ou non)** : estimée vers 01:15–01:30 Paris, après les analyses de p3 et de la boîte blanche.

Observation : ref-06 (watermark) répète souvent la réponse exacte (22 cellules sur 48 avec un doublon, contre 0 à 6 ailleurs), ce qui évoque un échantillonnage à clé dépendant du contexte. Cela fournit un détecteur de déguisement, pas d'identité.

## Point 00:30 Paris — boîte blanche (hors ligne, 0 requête)

Corpus figé pilot1/pilot2 (44 prompts × 4 répétitions × 9 refs, 1re tentative), modèle SmolLM2-360M-Instruct, Mac M3 (MPS).

| Endpoint | T estimé A/B | LLR fp8 simulé (mnat/token) A/B | LLR LoRA distillée s=0,2 (rép. 2/3, adaptateur entraîné sur 07 rép. 0-1) |
|---|---|---|---|
| 01 plain | 0,7/0,7 | −19,3/−22,9 | +1,5/−0,4 |
| 02 fp8 | 0,7/0,7 | **−9,4/−9,3** | +7,2/+6,2 |
| 03 sys | 0,7/0,7 | −18,6/−18,3 | +4,7/+7,2 |
| 04 LoRA 0,7 | 0,8/0,8 | −14,4/−18,5 | **+16,5/+16,1** |
| 05 persona | 0,8/0,7 | −26,4/−21,2 | +2,9/+0,1 |
| 06 watermark | 0,7/0,7 | −18,4/−22,5 | −0,8/−0,4 |
| 07 LoRA 1,5 | 0,9/0,9 | −11,9/−11,6 | **+30,9/+36,5** |
| 08 temp 1,0 | 1,0/1,0 | −26,2/−24,9 | +9,0/+5,4 |
| 09 LoRA 0,3 | 0,7/0,7 | −15,0/−21,7 | +5,1/+9,6 |

Erreur type : environ 2 mnat (LLR fp8, 88 réponses) et environ 1,6 mnat (LLR LoRA, 44 réponses).

- Règle pré-enregistrée 1 (seuils sur GAP et LLR8, moitiés A/B) : MCC hors sélection **0,000 / 0,693**.
- Règle pré-enregistrée 2 (LLR8 + LLR LoRA, 4 directions, 1 répétition = 44 réponses par moitié) : **0,486 / 0,396 / 0,446 / 0,693**. Elle n'atteint pas 0,60 dans les deux sens.
- Lecture : fp8, LoRA 0,7 et LoRA 1,5 se séparent nettement en moyenne. LoRA 0,3 reste dans le bruit des déguisements (sys, temp). La v2, avec un profil en T sous chaque hypothèse, est en cours.
- GPU : non utile avant l'échéance. Aucun poids Qwen ni Llama n'est en cache sur JURECA (`~/.cache/huggingface` absent, `SLZ/.cache` vide de modèles). La boîte blanche ep exigerait de télécharger les checkpoints sur un nœud de login, puis d'identifier les bases sur les 165 réponses ep1a.

## Résultats finaux et recommandation (00:55 Paris)

### p3, listes en boîte noire (432 réponses, 0 non-200)
- Règle pré-enregistrée : Hellinger **0,000 / 0,000**, Spearman **0,000 / 0,273** hors sélection. La normalisation intra dégénère (affinité intra proche de 0 sur 3 répétitions).
- Contrôle exploratoire sans normalisation : **0,000 / 0,000**. Distances à ref-01 : persona 0,69 et temp 0,71, contre fp8 0,55 et LoRA 0,3 0,51. **Les déguisements dominent : échec.**

### Boîte blanche v2 (LLR profilés en T), même règle
| Découpage | s | Sens 1 | Sens 2 | Erreurs notables |
|---|---|---|---|---|
| 1 rép. (44 réponses), 4 directions | 0,2 | 0,486 / 0,396 | 0,222 / 0,693 | déguisements éclatés jusqu'à 0,7 |
| 1 rép., 4 directions | 0,5 | 0,693 / 0,396 | 0,446 / 0,693 | |
| 2 rép. (88 réponses) A→B / B→A | 0,2 | **0,373** | **1,000** | A→B : 03 et 08 isolés, 07 avec fp8 |
| 2 rép. A→B / B→A | 0,5 | **0,693** | **0,721** | A→B : LoRA 0,3 fusionnée, 07 avec fp8 ; B→A : 03 (sys) isolé |

**Le seuil révisé n'est pas atteint proprement :**
- s=0,5 n'a été retenu qu'après avoir vu s=0,2 sur les deux moitiés ;
- la Δ̂ LoRA est apprise sur ref-07, qui est aussi évaluée ;
- dans le sens B→A, un déguisement (sys) est isolé ;
- la règle exige 88 réponses par endpoint ; avec 44, elle descend à 0,22–0,69.

### Recommandation ep : **NE PAS lancer de lot ep maintenant**
- Aucune méthode en boîte noire testée n'atteint 0,60 dans les deux sens.
- La boîte blanche a du signal pour fp8 et les LoRA ≥ 0,7, mais elle exige trois choses non disponibles :
  - les checkpoints exacts des 3 bases ep (non identifiés, aucun en cache sur JURECA) ;
  - un GPU ;
  - une LoRA forte par famille pour distiller Δ̂.
- Elle demande aussi environ 88 réponses par endpoint (au-delà de 60).
- Prochaine étape gratuite, en lecture seule (job GPU à lancer par le coordinateur) : identifier les bases ep par vraisemblance des 165 réponses ep1a existantes sous Qwen2.5-{0.5,1.5,3,7}B-Instruct et Llama-3.{1-8B,2-1B,2-3B}-Instruct. Les poids doivent d'abord être téléchargés sur un nœud de login. Coût API : 0. Les scripts `wb_stats.py` et `wb_lora2.py` sont adaptables.

## Hypothèse « détecter les déguisements » (01:25 Paris, 0 requête)
Features (`disguise_feats.py`, `disguise_cv.py`, `disguise_budget.py`) :
- `hf` : taux de mention « Hugging Face/SmolLM », c'est-à-dire la fuite du prompt système par défaut ;
- `dup` : doublons exacts entre répétitions ;
- `sim` : similarité intra-prompt.

Chaque feature isole son déguisement sur le corpus complet :
- `hf` = 0 pour 03 et 05 dans les deux moitiés, mais aussi 07 (LoRA 1,5) en B ;
- `dup` pilote ≥ 0,18 pour 06 seul ;
- `sim` minimale pour 08 (0,24 / 0,22).

Résultats hors sélection :
- Seuils libres, objectif MCC : **0,222 / 0,231**. Le surajustement inclut ref-01 dans le groupe.
- Seuils choisis sur la justesse déguisé/non (amendement post hoc, départage par marge), partition P1 (déguisés groupés + autres isolés) : **0,734 / 0,734**. P2 (+ plain le plus proche) : **0,524 / 0,524**, car le plus proche est ref-09. Le 0,734 tient en partie au hasard : ref-01 est marqué déguisé par son taux de doublons, ce qui le rattache au bon groupe, et 07 ou 04 sont mal classés.
- **Sous budget** (pilote seul, K prompts × 2 répétitions par moitié, 200 tirages) :
  - 8 réponses : 0,18 / 0,24 ;
  - **16 réponses : 0,21 / 0,24, P(≥ 0,60 dans les deux sens) = 0,01** ;
  - 32 réponses : 0,34 / 0,26 ;
  - 88 réponses : 0,734 / 0,222.
- **Seuil non atteint sous budget : pas de lot ep.**

Côté ep (ep1a, une réponse par prompt) :
- famille B : ep-08 (« Anthropic/OpenAI ») et ep-20 (aucune société) perdent l'identité Qwen, donc persona ou prompt système probable ;
- famille A : ep-02, ep-14 et ep-23 ne citent pas Qwen (signal faible) ;
- famille L : non applicable (Llama n'a pas d'identité par défaut dans son prompt système) ;
- copies f03 anormales : doublement de caractères pour ep-12, 14 et 17 ; unicode mathématique pour ep-10, 19, 29 et 31.

Préparé, **non lancé** : `p4_disguise_battery.py`. Batterie ciblée de 16 réponses par endpoint (identité ×8, copie ×4, nombre aléatoire ×4), 4 batteries sur les 9 refs = 576 requêtes ref (environ 75 min). Règle et critère sont fixés en tête du fichier.

## Boîte blanche ep sur GPU : préparée, non lancée (01:35 Paris)
- `wb_ep_ident.py` + `wb_ep.sbatch` : dc-gpu-devel, 1 nœud × 4 GPU, 30 min, bf16, `HF_HUB_OFFLINE=1`.
- Environnement : venv `SLZ/unlearning` (torch 2.15) + `SLZ/wheels/pylib_sd.tgz` (transformers 4.57.6) extrait dans `/tmp` du nœud.
- Règle fixée en tête du script : base = argmax de la moyenne nat/token profilée en T ; famille = même base ; comparaison par paires avec `c2_fam3_pairs_v2`.
- Étape 2 : T_hat, LLR fp8 simulé et GAP sous la base.
- Test CPU local (Qwen2.5-0.5B + SmolLM2-135M, 165 réponses ep, 2 rangs simulés) : pipeline complet, fichiers attendus produits.
- Limite mesurée sur ce test : l'erreur type du LLR8 vaut 10 à 20 mnat avec 5 réponses par endpoint, contre une séparation d'environ 10 mnat sur les refs. L'étape 2 est donc non concluante sur ep1a seul.
- Gabarits vérifiés : Qwen2.5 inclut « You are Qwen, created by Alibaba Cloud… » ; Qwen3 a un bloc think vide ; Llama 3.1 inclut « Cutting Knowledge Date: December 2023 / Today Date: 26 Jul 2024 ».

## wb2 : collecte ep en boîte blanche, préparée et non lancée (02:05 Paris)
Dossier `wb2/` :
- `wb2_lib.py` : prompts figés B1–B3, `max_tokens` 256 ;
- `wb2_sim.py` + `wb2_sim.sbatch` : calibration sur 3 bases × 9 endpoints simulés × 2 réplicats × 30 réponses ; les LoRA sont des **substituts aléatoires** ; le watermark n'est pas simulé ;
- `wb2_collect.py` : 990 réponses ep, PARALLEL 4, journal `/p/project1/.../wb2/collect_ep.jsonl`, budget lu avant et après ;
- `wb2_analyze.py` + `wb2_analyze.sbatch` : applique la règle figée, écrit le CSV et son sha.

Tests CPU (SmolLM2-135M, tailles réduites) : la simulation, le calage de la règle et l'analyse tournent de bout en bout. **Aucun MCC de simulation n'est encore mesuré** à la taille réelle.
