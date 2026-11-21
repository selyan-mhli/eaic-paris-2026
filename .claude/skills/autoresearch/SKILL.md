---
name: autoresearch
description: Boucle d'expériences autonome façon karpathy/autoresearch pour un challenge EAIC (cN/) - l'agent modifie solve.py, lance un run à budget fixe, mesure sur la validation locale figée, garde ou restaure, et recommence jusqu'à une condition d'arrêt. À utiliser quand un challenge a une baseline et une métrique locale fiables et qu'on veut l'améliorer sans humain (la nuit). Pas pour trouver la baseline ni pour soumettre au serveur.
---

# autoresearch (EAIC)

Adapté de l'idée de `karpathy/autoresearch` (une métrique, un fichier modifiable, un budget
fixe, garder ou jeter). Aucun code de ce dépôt n'est copié ; il n'a pas de licence.

## Préconditions (vérifier, sinon s'arrêter et le dire)

1. `cN/program.md` rempli à partir de `common/autoresearch/program.template.md` :
   métrique, direction, budget, conditions d'arrêt, interdits.
2. `cN/eval.py` existe, affiche `score: <float>`, et a été vérifié à la main sur un cas connu.
   **Il est figé : ne jamais le modifier pendant la boucle**, ni les données de validation.
3. `cN/solve.py --out <dir>` produit une sortie au format de soumission.
4. Un territoire par boucle : une seule boucle par dossier `cN/`.

## Boucle

```
python common/autoresearch/run_one.py cN --direction <max|min> --budget <s> --desc "<idée>"
```

Le script copie `solve.py` dans `cN/runs/<id>/`, l'exécute (tué à 2× le budget), lance
`eval.py`, ajoute une ligne à `cN/results.tsv` (`keep` / `discard` / `crash`), et restaure
`solve.py` depuis le meilleur run si le score n'est pas meilleur.

À chaque tour :
1. Lire `cN/program.md`, la fin de `cN/results.tsv` et le `solve.py` actuel.
2. Choisir **une** idée, la coder dans `solve.py`.
3. Tester d'abord en très court (`--budget 30`) si l'idée touche formes, device ou format.
4. Lancer le run. Lire la ligne de sortie, pas le log entier ; en cas de crash,
   `tail -n 50` du `run.log`. Erreur triviale → corriger et relancer ; idée cassée → passer.
5. Toutes les ~5 lignes `keep`, ajouter un résumé à `docs/experiences.md`
   (hypothèses gardées, gain, coût).

**Critère de simplicité** : à gain quasi nul, préférer le code le plus court. Un gain minuscule
qui ajoute beaucoup de complexité ne vaut pas le coup ; supprimer du code à score égal, si.

**Surapprentissage de la validation** : après beaucoup de runs, le meilleur score local devient
optimiste. Si `program.md` prévoit un second split « holdout », le passer seulement sur le
meilleur run, au plus une fois par heure, et noter l'écart.

## Arrêt (contrairement à l'original, la boucle s'arrête toute seule)

S'arrêter à la première condition atteinte : celles de `program.md`, ou 2026-11-22 11:30
(heure de Paris, avant la fin du hacking), ou 3 crashs de suite sur des idées différentes.
Ne pas demander « je continue ? » entre deux runs : la boucle tourne jusqu'à une condition
d'arrêt ou une interruption.

## Sur JURECA

Le dépôt doit être présent côté cluster (`/p/project1/training2666/mouhali1/…`) :
1. `rsync` du dossier `cN/` et de `common/` vers le cluster.
2. Un job par run, qui exécute `run_one.py` sur le nœud :
   `ssh jureca "cd <dépôt> && sbatch --wait -J j<N> --time=<budget×2+marge> common/jureca/main.sh"`,
   avec `main.sh` adapté pour appeler `run_one.py` au lieu de `sample_submission.py`.
3. `rsync` retour de `cN/results.tsv` et `cN/runs/<id>/`.
Règles du README (noms opaques, pas de GPU sur login, `HF_HUB_OFFLINE=1`). Si `ssh jureca`
échoue, s'arrêter et le signaler.

## Interdits

- Soumettre au serveur : c'est la session principale qui choisit quel run soumettre.
- Opérations Git (la restauration passe par les snapshots de `runs/`, pas par `git reset`).
- Installer des paquets, modifier `eval.py`, les données ou `third_party/`.

## Rapport de fin

Meilleur score et run (`cN/runs/<id>/solve.py`), nombre de runs keep/discard/crash, idées qui
ont marché, condition d'arrêt atteinte, et ce qui n'a pas été vérifié (holdout, score serveur).
