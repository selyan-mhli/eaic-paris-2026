# c2 — Task 2 : Model Equality Testing

- **Propriétaire :** à préciser
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

Fiche complète : [../docs/HACKATHON.md](../docs/HACKATHON.md).
