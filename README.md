# EAIC Paris 2026 — équipe SLZ

Dépôt de l'équipe SLZ pour la manche parisienne du European AI Championship (CISPA, Fraunhofer AISEC,
tâches conçues par SprintML), les 21 et 22 novembre 2026. Trois épreuves de sécurité de l'IA, 24 heures,
un classement général sur les trois.

| | Épreuve | Responsable | Dossier |
|---|---|---|---|
| Task 1 | Localisation de watermark dans une image | Selyan (`selyan-mhli`) | [c1/](c1/README.md) |
| Task 2 | Model equality testing (quel endpoint sert quel modèle) | Sophie (`SophiePelet`) | [c2/](c2/README.md) |
| Task 3 | Unlearning d'un ViT-B/16 | Yihuan (`Yihuan0301`) | [c3/](c3/README.md) |

## Où lire quoi

| Besoin | Fichier |
|---|---|
| Qui fait quoi maintenant, priorités, classement relevé | [docs/PLAN-NUIT.md](docs/PLAN-NUIT.md) |
| Règles, métriques et formats des trois épreuves | [docs/HACKATHON.md](docs/HACKATHON.md) |
| Toutes les expériences : hypothèse, coût, score local, score serveur | [docs/experiences.md](docs/experiences.md) |
| Usage des GPU sur JURECA (4 tâches par nœud) | [docs/jureca-gpu-audit.md](docs/jureca-gpu-audit.md) |
| Résultats et reproduction de la Task 1 | [c1/NOTES.md](c1/NOTES.md), veille dans [c1/VEILLE.md](c1/VEILLE.md) |
| Plan de requêtes de la Task 2 | [c2/PLAN.md](c2/PLAN.md) |

## Ce qu'on doit rendre

Source : slides d'ouverture (slides 14 à 16) et les trois énoncés PDF. Aucun des quatre documents ne
demande de code, de rapport écrit ni de dépôt public.

| Livrable | Format | Limite | Échéance |
|---|---|---|---|
| Task 1 | `.pt` : `{"image_names": names, "masks": Tensor[N, 128, 128]}` en 0/1, chaque image une seule fois | 20 Mo selon l'énoncé, 25 Mo selon le README du kit : viser moins de 20 Mo | dimanche 12:00 |
| Task 2 | `.csv` : `endpoint,family,model`, une ligne par endpoint `ep-01` à `ep-33` | une soumission toutes les 30 minutes, une erreur compte aussi | dimanche 12:00 |
| Task 3 | `.pt` ou `.pth` : state dict float32, chargé en `strict=True` dans l'architecture du template, sans NaN | 450 Mo ; délai d'environ 10 minutes entre deux soumissions | dimanche 12:00 |
| Présentation | 6 minutes sur les solutions des trois tâches, devant le jury (équipe technique du CISPA) | **seulement les 6 meilleures équipes** | dimanche 13:00 |

Calendrier du dimanche 22/11 : 08:00 petit-déjeuner, **12:00 fin des soumissions**, 13:00 présentations,
14:00 délibération, 14:30 remise des prix (1er prix : 10 000 €), 15:00 fin.

Ce que cela implique pour nous :

- **Le deck de 6 minutes doit être prêt à 11:00**, même si la place dans les 6 premiers n'est pas
  acquise : méthode par tâche, validation locale, ce qui a marché et pourquoi.
- Le classement public porte sur environ 30 % des données ; **le classement final sur les 70 % restants**,
  révélés après la clôture. Les trois énoncés demandent de ne pas optimiser sur le score public.
- Le classement affiche le meilleur score de l'équipe par tâche. Les énoncés ne disent pas si le
  classement final retient la meilleure soumission ou la dernière : **question posée aux mentors**.
- Classement général : chaque tâche est ramenée entre 0 (moins bonne équipe) et 1 (meilleure), puis on
  fait la moyenne. Il faut un score sur les trois tâches pour être classé.

## État au 21/11, 18:54 (scores publics)

SLZ est 6e sur 14 équipes classées, à 0,379.

| | Score public | Rang | Meilleure équipe | Validation locale |
|---|---|---|---|---|
| Task 1 | 0,531 | 9e / 25 | 0,697 | 0,509 en validation croisée imbriquée ; 0,547 mesuré, non soumis |
| Task 2 | 0,20 | 2e / 16 | 0,40 | pilote sur les endpoints de référence : MCC 0,33 au mieux à 2 répétitions |
| Task 3 | 0,0035 | 20e / 27 | 0,147 | proxys seulement, plus sévères que le serveur |

Ces chiffres vieillissent vite : la version à jour est dans [docs/PLAN-NUIT.md](docs/PLAN-NUIT.md).

## Structure du dépôt

```text
eaic-paris-2026/
├── c1/        Task 1 : métrique, cartes de chaleur, recherche de rectangle, tête de segmentation, sondes
├── c2/        Task 2 : plan de requêtes, prompts, analyse hors ligne, copie du scoreur du kit
├── c3/        Task 3 : évaluation locale par proxys, mesure de dérive, soupes de poids
├── common/    chemins JURECA, script de soumission, job Slurm type, boucle autoresearch
├── docs/      fiche des épreuves, plan de nuit, journal d'expériences, audit GPU
├── .github/   CI (ruff + pytest), modèle de PR, CODEOWNERS
└── .claude/   agent ml-dev et skill autoresearch
```

Le code vit ici. Les données, les poids, les `.venv` et les sorties restent sur JURECA, dans
`/p/scratch/training2666/mouhali1/SLZ/` (voir `common/paths.py`). Chacun clone le dépôt dans son propre
espace JURECA, pas dans le dossier partagé.

## Démarrer

```bash
# une fois, dans TON espace JURECA
cd /p/scratch/training2666/$USER && git clone https://github.com/selyan-mhli/eaic-paris-2026.git

# à chaque connexion, dans le dossier de la tâche
jutil env activate -p training2666
cd /p/scratch/training2666/mouhali1/SLZ/<dossier_de_la_tâche> && source .venv/bin/activate
```

Dossiers de tâche : `Hackathon_Watermark_Localisation_Task/`, `model_equality/`, `unlearning/`. Chacun a
son `.venv`, son `requirements.txt` et son `output/`. Un job se lance avec `sbatch` sur la partition
`dc-gpu`, réservation `cispahack`, compte `training2666`.

Tests et lint, comme la CI :

```bash
ruff check --select E9,F63,F7,F82 .
pytest -q c1/test_metric.py c2/test_analyze.py c3/test_eval.py
```

## Règles d'équipe

- **Aucune collaboration avec une autre équipe** (solutions, code, clés) : c'est une disqualification.
  Le dépôt reste privé et on ne lit pas les dépôts des autres équipes.
- Jamais de clé API dans le dépôt : variable `EAIC_API_KEY` seulement. Ni poids ni données dans Git.
- Une soumission se fait avec l'accord explicite de Selyan, après relecture du chiffre qui la justifie.
- On décide sur la validation locale, pas sur le score public.
- Toute expérience a sa ligne dans `docs/experiences.md`, y compris les échecs et les soumissions.
- Task 2 : aucune requête sans plan écrit ; le budget de 100 réponses par endpoint ne se récupère pas.
- Task 3 : prévenir avant de soumettre, le délai entre deux soumissions est commun à l'équipe.

## Workflow Git

```bash
git switch main && git pull --rebase
git switch -c feat/c3-scrub          # feat/cN-…, fix/cN-…, docs/…
git add c3/ && git commit -m "feat(c3): SCRUB baseline"
git push -u origin HEAD
gh pr create --fill
```

- Pas de push direct sur `main` ; une PR par idée, petite, fusionnée vite en squash.
- La CI doit être verte ; le responsable du dossier relit (`.github/CODEOWNERS`).
- Dépendance en plus : `uv add <paquet>` dans le dossier de la tâche, pas de `pip install` à la main.

## Cluster JURECA

- Projet `training2666`, 40 nœuds de calcul (4 × A100 40 Go, 128 cœurs, 512 Go), 12 nœuds de login.
- Nœuds de calcul sans Internet : `HF_HUB_OFFLINE=1`, poids et jeux de données copiés à l'avance.
- Aucun calcul lourd sur les nœuds de login. Quatre tâches par nœud quand le job le permet.
- Noms de jobs opaques (`-J j1`, `-J j3`) : les autres équipes voient la file.
- Quota d'inodes partagé : pas d'environnements en double, pas de milliers de petits fichiers.
