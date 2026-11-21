# c2 — Task 2 : Model Equality Testing

- **Propriétaire :** Sophie (GitHub SophiePelet, Jülich pelet1)
- **Données et env (JURECA) :** `/p/scratch/training2666/mouhali1/SLZ/model_equality/`
  (`task_template.py`, `query_reference.py`, `score_practice.py`, `practice_answers.csv`, `submission.py`)
- **But :** pour 33 endpoints LLM, un label de famille et un label de modèle.
  Même modèle = même checkpoint + quantification + LoRA ; watermark, system prompt et
  décodage ne changent pas l'identité.
- **Métrique :** 0,1 × MCC famille + 0,9 × MCC modèle sur 528 paires. Public partiel.
- **Soumission :** CSV `endpoint,family,model`, une toutes les 30 min (une erreur compte aussi).

## Budget de requêtes : la ressource rare

- 100 réponses par endpoint pour **tout** le week-end, 256 tokens en sortie, 1 024 en entrée,
  4 requêtes simultanées, `temperature`/`seed` ignorés, messages système refusés.
- Aucune requête sans plan écrit dans ce dossier. Toutes les réponses journalisées en jsonl
  (`data/` est ignoré par Git : garder les journaux sur JURECA, dans `output/`).
- S'entraîner d'abord sur `ref-01..09` avec `score_practice.py` (illimité).

## État au 21/11 16:30 et premiers pas

- **Budget intact** : 0 requête dépensée sur les 42 endpoints (33 `ep-*` + 9 `ref-*`), vérifié via `/v1/models`.
- Aucune soumission c2 encore. Étape 1 : une soumission aléatoire valide
  (`RANDOM_SUBMISSION = True` dans `task_template.py`, puis `submission.py`) pour tester clé et format.
- Étape 2 : sur `ref-01..09` uniquement, tester des prompts discriminants et les scorer avec
  `score_practice.py`. Vérité terrain dans `practice_answers.csv` : ref-01/03/05/06/08 = même modèle S
  (plain, system prompt, persona, watermark, température) ; ref-02 = fp8 ; ref-04/07/09 = LoRA 0,7/1,5/0,3.
  Le but est de séparer quantif/LoRA tout en regroupant les déguisements.
- Étape 3 : écrire le plan de requêtes `ep-*` (ex. 20 prompts × 5 répétitions = 100) dans `c2/PLAN.md`
  en PR, relu avant de lancer quoi que ce soit.

Plan de requêtes : [PLAN.md](PLAN.md) (budget ref-* mesuré : 1 000 000, ep-* : 100). Analyse hors ligne :
`analyze.py` (tests : `pytest c2/test_analyze.py`, synthétique via `synth.py`). Copie du scoreur : `kit/`.

Fiche complète : [../docs/HACKATHON.md](../docs/HACKATHON.md).
