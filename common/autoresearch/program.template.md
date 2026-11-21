# program.md — <cN> : <nom du challenge>

Copier dans `cN/program.md` et remplir. C'est le seul texte que l'humain règle ;
l'agent ne modifie que `cN/solve.py`.

## Objectif

- Métrique : <nom exact, tiré de l'énoncé>. Direction : <max|min>.
- Score de la baseline : <à remplir après le premier run>.
- Ce qui compte aussi pour le jury : <qualité visuelle, temps, taille… ou rien>.

## Fichiers

- `solve.py --out <dir>` : modifiable. Produit la sortie au format de soumission dans `<dir>`.
  Lit `AR_BUDGET_S` et s'arrête de lui-même avant (le run est tué à 2× le budget).
- `eval.py --pred <dir>` : figé. Réimplémente la métrique sur la validation locale et affiche
  `score: <float>`. Testé une fois à la main sur un cas connu avant le premier run.
- Données : <chemins>. Split de validation : <fixe, seedé, jamais utilisé pour régler à la main>.

## Budget et arrêt

- Budget par run : <300> s. Où : <local | JURECA, un job par run>.
- Arrêt : <N> runs, ou <K> runs sans `keep`, ou l'heure <HH:MM>, au premier atteint.
  Fin dure : réservation cispahack, 2026-11-22T14:00 (heure du cluster) ; hacking jusqu'au
  dimanche 12:00.

## Interdits propres à ce challenge

- <ex. pas d'accès aux labels de test, pas de modèle externe non autorisé, taille max…>
- Pas de nouveau paquet : seulement ceux de l'env `Hackathon`.

## Pistes à explorer

- <idées de l'équipe, du papier de référence, des repos des organisateurs>
