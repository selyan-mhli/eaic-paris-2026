# EAIC Paris 2026, dépôt d'équipe

Équipe SLZ. Fiche des 3 épreuves : [docs/HACKATHON.md](docs/HACKATHON.md).

```text
eaic-paris-2026/
├── c1/       Task 1 Watermark Localization  (un propriétaire)
├── c2/       Task 2 Model Equality          (un propriétaire)
├── c3/       Task 3 Unlearning              (un propriétaire)
├── common/   submit, chemins et outils JURECA, autoresearch
├── docs/     HACKATHON.md, experiences.md, deck de présentation
└── .claude/  agent ml-dev et skill autoresearch
```

Le code vit ici ; les données, poids et `.venv` restent sur JURECA dans
`/p/scratch/training2666/mouhali1/SLZ/` (voir `common/paths.py`). Chacun clone le dépôt dans
son propre espace JURECA, pas dans le dossier partagé.

## Règles d'équipe

- Une soumission valide sur chaque challenge le plus tôt possible, puis on améliore.
- Toute expérience a une ligne dans `docs/experiences.md` : hypothèse, changement, coût, score local, score serveur.
- Validation locale avant soumission ; le serveur masque les scores inférieurs à ton meilleur.
- Jamais de clé API dans le dépôt : variable `EAIC_API_KEY` seulement.
- Un propriétaire par dossier `cN/` ; tout passe par une PR (voir ci-dessous).
- Aucune collaboration avec une autre équipe (règle des organisateurs : disqualification).

## Workflow Git (PR obligatoire)

```bash
# une fois, dans TON espace JURECA (pas dans le dossier partagé SLZ)
cd /p/scratch/training2666/$USER && git clone https://github.com/selyan-mhli/eaic-paris-2026.git
# à chaque changement
git switch main && git pull --rebase
git switch -c feat/c3-scrub          # feat/cN-…, fix/cN-…, docs/…
git add c3/ && git commit -m "feat(c3): SCRUB baseline"
git push -u origin HEAD
gh pr create --fill                  # ou le lien affiché par git push
```

- Jamais de push direct sur `main` ; une PR par idée, petite, fusionnée vite (squash).
- La CI (ruff + tests) doit être verte ; le propriétaire du dossier relit (`.github/CODEOWNERS`).
- Code dans `cN/` du dépôt ; données, poids, `.venv` et sorties restent dans le dossier SLZ.
- Dépendance en plus : `uv add <paquet>` dans le dossier de la tâche, pas `pip install` à la main.

## Cluster JURECA

- Nœuds de calcul sans Internet : `HF_HUB_OFFLINE=1`, poids et datasets copiés à l'avance dans le cache.
- Quota d'inodes partagé : pas d'environnements conda en double, pas de milliers de petits fichiers.
- Ne pas faire tourner l'image sur les GPU des nœuds de login.
- Noms de jobs Slurm opaques (`-J j1`, `-J j2`) : les autres équipes les voient.
- Jobs longs lancés avant de dormir.
