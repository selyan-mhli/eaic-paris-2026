# c2 — plan de requêtes

## Objectif

Identifier les familles (10 % du score) puis les checkpoints/quantifications/adapters exacts
(90 % du score) sur les 33 endpoints `ep-*`, sans dépasser 100 réponses par endpoint.

## Phase 1 — entraînement illimité

- Utiliser uniquement `ref-01..09`.
- Test initial : `c2/query_references.py --prompts c2/reference_prompts.txt`.
- Analyser : `c2/analyze_references.py --answers <journal>`.
- Mesurer les performances avec `score_practice.py`.
- Toutes les réponses sont journalisées dans
  `/p/scratch/training2666/mouhali1/SLZ/model_equality/reference_answers.jsonl`.
- Aucun endpoint `ep-*` ni soumission à cette phase.

## Phase 2 — échantillon des endpoints

À exécuter seulement après approbation explicite :

- 10 prompts maximement discriminants.
- 5 répétitions sur tous les endpoints = 50 réponses/endpoint.
- Utiliser des tâches déterministes, courtes et faciles à analyser.
- Journaliser chaque réponse dans `output/answers.jsonl`.
- Comparer les sorties normalisées, la longueur, les distributions de tokens et les erreurs.

## Phase 3 — confirmation ciblée

- Réserver jusqu'à 50 réponses supplémentaires par endpoint pour les paires incertaines.
- Conserver au moins 10 réponses/endpoint de marge si possible.
- Aucun sondage via les soumissions publiques.
- Ne jamais soumettre pendant cette phase.

## Soumission

- Construire un CSV valide avec une ligne pour chaque `ep-01..ep-33`.
- Toute soumission nécessite une approbation explicite précédente.
- Vérifier CSV, contrainte famille/modèle et 30-minute cooldown.
