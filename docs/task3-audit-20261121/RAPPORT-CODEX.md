# Audit Task 3 — 21 novembre 2026

Mandat direct de Selyan : expliquer l'écart entre 0,34 et notre 0,00, vérifier les résultats de Yihuan par SSH. Rapport séparé du RAPPORT-CODEX.md racine, consacré à la Task 1 et modifié par une autre session.

## Verdict

**Notre meilleur score public vaut 0,0043524934, affiché 0,00 par arrondi. Le premier vaut 0,3422158992 : environ 79 fois plus.** Vérification du classement officiel à 22:12, heure de Paris ; meilleur SLZ horodaté 19:26. Source : http://34.147.229.55/leaderboard_page ; capture structurée dans `leaderboard.json`.

Ce score positif signifie que **la meilleure soumission enregistrée a passé la porte serveur des 70 %**. On ne peut donc pas attribuer son faible score à une disqualification par cette porte. En revanche, des candidats agressifs présentent une forte dérive locale. Le fichier ayant obtenu 0,004352 n'est pas identifié dans les journaux accessibles : relier tous les défauts ci-dessous à cette soumission précise serait abusif.

**Le défaut transversal est la sélection des modèles : les objectifs et contrôles locaux ne reproduisent pas les deux composantes d'oubli du serveur, et le contrôle retain d'entraînement donne une fausse impression de stabilité.** Un écart dans le calcul NPO et plusieurs défauts de validation sont établis. Aucun gain après correction n'a été mesuré.

## Formule officielle vérifiée dans le vrai template

`sources/task_template.py:16` : sur les forget propres ET leurs altérations cachées, le serveur calcule séparément :

- Un progrès normalisé sur la moyenne ET l'écart-type des pertes : `1 - clip(distance / distance_originale, 0, 1)`.
- Un progrès normalisé sur `KL(p_gold || p_candidat)` à T=2, image par image.
- Il moyenne chaque composante entre les deux groupes, puis multiplie les deux composantes.
- Il multiplie par un retain fondé sur accuracy relative et `1/(1+KL)` ; accuracy cachée sous 70 % → zéro.

L'identité a donc un oubli nul. Monter la perte, l'entropie ou le nombre de classes changées ne garantit pas un progrès vers les gold. Exemple illustratif seulement : deux progrès de 10 % avec retain 90 % donnent 0,009.

## Constats classés

### P1 — `eval.py` ne peut pas servir de score de sélection

`c3/eval.py:87` utilise `exp(-distance)` vers les pertes de l'origine sur des augmentations fortes. Il ignore la distribution gold par image et ne normalise pas le progrès par la distance de l'origine. Le vrai modèle inchangé obtient **0,022315** dans les logs, alors que sa composante d'oubli officielle est zéro. Les soupes conservatrices ont un score proxy inférieur à celui de l'identité. C'est annoncé comme proxy dans sa documentation ; son usage pour classer les soumissions reste dangereux.

Le proxy retain par défaut mesure l'accord sur les forget augmentés, précisément un domaine où l'oubli doit aussi agir. Le jeu `offforget.npz` améliore la séparation des images mais vient des photos de la tâche watermark, hors distribution des 100 classes (`sources/work_c3_eval/make_offforget.py:1`). **Aucun de ces accords n'est une accuracy ni une preuve de franchissement des 70 %.**

Preuves : `remote_evidence.json`, entrées `work_c3_eval/res/orig_aug.txt`, `orig_off.txt`, `soupB_l10_off.txt`, `soupB_l50_off.txt`.

### P1 — Le contrôle retain d'entraînement ne détecte pas la dérive de `v2_best`

Le job 15901563 sélectionne `v2_best` avec KL patch-shuffle **0,0792**, CE **1,4045 ± 3,8130**, **18,17 %** de top-1 changés. Le sélecteur maximise les changements de top-1 dans une bande prédéfinie et ne contraint pas l'écart-type (`sources/unlearn_v2.py:232`).

Une évaluation distincte sur les 3 500 forget et 1 000 photos hors forget donne :

| Mesure | `v2_best` |
|---|---:|
| Accord sur photos hors forget | 39,90 % |
| KL origine → candidat sur ces photos | 1,9868 |
| Accord sur forget fortement augmentés | 44,94 % |
| CE forget moyenne / écart-type | 1,4056 / 3,8147 |
| Format float32 / chargement strict | conformes |
| Changements top-1 fp32 → fp16 sur forget | 0 % |

Ce sont des mesures de logs antérieurs, relues pendant cet audit ; aucune nouvelle inférence. Le contrôle patch-shuffle est une distribution utilisée à l'entraînement, pas une validation indépendante. Les photos hors forget révèlent une dérive forte, mais ne permettent pas de calculer l'accuracy cachée. L'écart-type élevé est préoccupant au vu de la formule ; les vraies statistiques gold sont inconnues.

Preuves : `output/15901563.out`, `work_c3_eval/res/v2_best_off.txt`, `v2_best_aug.txt` dans `remote_evidence.json`.

### P1 — Le ratio NPO mélange deux vues différentes

`sources/unlearn_v2.py:272` calcule la référence sur les images propres. À la ligne 316, l'élève reçoit un crop/jitter/effacement aléatoire, puis sa probabilité est comparée à cette référence propre aux lignes 319–327. Le ratio ne compare donc pas professeur et élève sur la même entrée.

Même avec des poids identiques, une augmentation difficile crée déjà une différence de log-probabilité. Le signal mélange l'effet de la corruption et celui de la modification des poids. Il faut comparer la référence sur **le même tenseur augmenté**, ou appliquer les deux modèles à la vue propre. Ce correctif est local et précis ; son impact sur le score reste à mesurer.

La reproduction CPU utilise un exemple construit : référence propre -0,000322, probabilité augmentée -4 ; différence -3,999678 avant toute modification, au lieu de zéro sur une même vue. Ce n'est pas la mesure d'un batch réel.

### P2 — Les derniers candidats existent, sans preuve suffisante de meilleur score

`push_best.pt`, créé à 21:12 heure de Paris, a un contrôle via le vrai forward fp16 : CE **0,087 / 0,329**, **2,46 %** de top-1 changés, **4,17 %** sur les altérations locales. Le log annonce KL patch-shuffle **0,5099**. Aucun résultat de contrôle sur `offforget` pour ce checkpoint n'a été trouvé dans les journaux collectés.

Le fit de `score_push.py:49` ajuste chaque ligne de tête à une baisse constante sur les exemples de sa classe, avec une colonne de biais. Il n'impose aucune contrainte sur les exemples des autres classes ni de maintien dans le solveur. La KL proxy est calculée après le fit et **n'est pas bornée** par la condition d'acceptation (`sources/score_push.py:118` et `:135`). Une baisse du logit gagnant ne garantit pas que les autres logits restent stables, contrairement à l'intention annoncée dans le commentaire initial.

Il existe aussi des candidats déjà calculés moins dispersés :

| Candidat | CE forget moyenne / écart-type | Top-1 changés forget | Accord photos hors forget |
|---|---:|---:|---:|
| `t1t2_p03_e2` | 0,1056 / 0,2644 | 3,14 % | 86,30 % |
| `t1t2_p03_e3` | 0,1744 / 0,3202 | 3,91 % | 84,20 % |
| `t1t2_p06_e1` | 0,0876 / 0,2979 | 2,66 % | 85,30 % |
| `t1t2_p10_e1` | 0,2441 / 0,5194 | 7,00 % | 83,90 % |

Tous passent format et finitude dans ces logs. Leur stabilité sur passe-bas et masquages est plus faible : les altérations des forget ne constituent toujours pas du vrai retain. Ce tableau justifie de les comparer, **pas de déclarer un gagnant serveur**. Ils peuvent avoir été soumis sans améliorer le meilleur : aucun reçu n'a été trouvé.

### P2 — Une colonne `ret` et le contrôle de format peuvent être mal interprétés

`sources/work_c3_eval/score_like.py:129` affiche `1/(1+KL)` dans `off_ret`, sans terme d'accuracy. Ce nombre est seulement la composante KL, pas le retain officiel complet.

`c3/eval.py:150` vérifie dtype, taille et chargement, mais ne rejette pas explicitement les poids NaN/Inf. Un `Linear` float32 contenant NaN obtient `format_ok=True` dans notre reproduction. Les sauvegardes de Yihuan vérifient déjà la finitude ; aucune preuve que ce défaut explique le score public.

## Reprise préparée depuis ce poste

Accès confirmé : `ssh jureca`, compte `mouhali1`, hôte `jrlogin03.jureca`, dossier partagé `/p/scratch/training2666/mouhali1/SLZ/unlearning`. Il n'est pas nécessaire de se connecter au compte personnel de Yihuan pour lire ses fichiers partagés. Le script mentionné par Selyan est vraisemblablement **`teammateslz.sh`**, présent à la racine SLZ et décrit comme déjà configuré dans le README partagé. Ce n'est pas un accès SSH à son ordinateur ; il n'a pas été exécuté.

`audit_candidates.sbatch` prépare une comparaison de dix fichiers existants sur quatre A100, avec empreintes SHA-256, mêmes vues et même script de mesure ; aucune soumission. Les chemins, le `.npz` et les imports locaux utilisés ont été vérifiés à distance. **Script validé syntaxiquement seulement, jamais transféré ni lancé.** Les sorties iraient dans des noms `output/task3_audit_<job>_<rang>.tsv` distincts des anciens résultats.

Actions humaines / Opus :

1. Récupérer auprès de Yihuan le nom exact et, si disponible, le SHA-256 du fichier associé au reçu public de 19:26, ainsi que les autres reçus. Le classement public seul ne donne pas cette provenance.
2. Lancer la validation préparée des candidats existants sur un nœud de calcul. Examiner les métriques séparément, sans utiliser le score scalaire de `eval.py` comme objectif.
3. Corriger la comparaison de vues NPO et la sélection des checkpoints : contrôler écart-type, comportement sur les altérations, maintien sur plusieurs ensembles indépendants, pas seulement entropie/flip.
4. Obtenir des organisateurs la possibilité d'un contrôle retain représentatif et étiqueté, si autorisé. Sans cela et sans gold, le score officiel reste impossible à reconstruire exactement localement.
5. Toute soumission reste une action de Selyan selon les règles du dépôt. Aucun score cible chiffré garanti par cet audit.

## Preuves et vérifications

État initial et final : checkout `/Users/selyan/dev/eaic-paris-2026`, branche `main`, HEAD `f209e20397736c53824a7a94ec73f2bd66d012eb`. Arbre déjà sale : modifications de c1 et `docs/experiences.md`, fichiers de Task 1 non suivis. Une autre session travaille sur la Task 1 ; `relay-check.py` n'observe pas les processus dans ce sandbox et n'établit aucune exclusivité. Aucune mutation Git, aucun code de production modifié. Ajouts de cet audit uniquement dans `docs/task3-audit-20261121/`.

Commandes et sorties :

```text
git status --short ; git branch --show-current ; git rev-parse HEAD ; git worktree list
  main ; f209e20397736c53824a7a94ec73f2bd66d012eb ; modifications locales préexistantes
ssh -o BatchMode=yes -o ConnectTimeout=12 jureca 'hostname; id; ...'
  jrlogin03.jureca ; uid=37359(mouhali1) ; lecture des scripts/logs de zhang95 réussie
squeue -u mouhali1,zhang95 ...
  en-tête seul : aucun job visible pour les deux comptes au moment des deux relevés
GET http://34.147.229.55/leaderboard_page
  Skesiea 0.34221589924218265 ; SLZ 0.004352493436899641
.venv/bin/python -m pytest -q -rs c3/test_eval.py
  échec : No module named pytest (aucune installation effectuée)
.venv/bin/python docs/task3-audit-20261121/reproduce.py
  voir reproductions.json : NaN accepté par check_format ; identité à proxy positif ; CE identiques avec KL différente
bash -n docs/task3-audit-20261121/audit_candidates.sbatch
  code de sortie 0 ; aucune exécution Slurm
```

`remote_evidence.json` contient les textes collectés, avec lignes liées aux identifiants d'accès omises. `sources/` fige les sources utiles au diagnostic. Les logs sont des observations historiques, pas des expériences relancées par cet audit. Aucun entraînement, accès au poste personnel de Yihuan, changement distant ou envoi au serveur de soumission effectué.
