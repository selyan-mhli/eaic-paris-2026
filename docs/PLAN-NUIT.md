# Plan de la nuit — EAIC Paris 2026 (équipe SLZ)

Source unique de « qui fait quoi » jusqu'à la fin. Depuis le 21/11 à 18:55, la coordination est tenue par
la session « audit » (elle a repris le rôle de « Organisation du projet et tasks », qui ne donne plus de
consignes). Les sessions de travail lui envoient leurs comptes rendus et lisent ce fichier avant de
commencer une étape. Le détail des épreuves reste dans [HACKATHON.md](HACKATHON.md), les mesures dans
[experiences.md](experiences.md).

État au 21/11, 18:56. Fin des soumissions : **dimanche 22/11, 12:00**. Démo figée à **08:00**. Deck prêt à
**11:00**. Seules les **6 premières équipes** au général présentent.

## Où on en est (leaderboard public, 18:54)

SLZ **6e / 14 classées** au général (0,379) : Task 1 0,531 (9e / 25), Task 2 0,20 (2e / 16),
Task 3 0,0035 (20e / 27). Leader : Fable 4.2 (0,555). Le général ramène chaque tâche entre 0 (moins bonne
équipe) et 1 (meilleure), puis fait la moyenne ; seules les équipes notées sur les 3 tâches sont classées.

Meilleurs scores par tâche : Task 1 0,697 (MasterChickenHack), Task 2 0,40 (Eugenia), Task 3 0,147
(WhatThePhoques). En Task 3, 7 équipes sont groupées entre 0,026 et 0,028 (méthode commune inconnue) et
7 sont à 0,000 (baseline du template = modèle d'origine).

Effet d'un gain de +0,01 sur le général : Task 3 +0,023, Task 2 +0,008, Task 1 +0,0075.

Simulation avec la règle officielle, les autres équipes figées (indicatif, scores publics de 18:42) :

| Si on obtient… | Général | Rang |
|---|---|---|
| Task 3 à 0,01 | 0,397 | 6e |
| Task 3 à 0,03 | 0,443 | 4e |
| Task 3 à 0,05 | 0,488 | 3e |
| Task 3 à 0,09 | 0,579 | 1er |
| Task 2 à 0,30 | 0,464 | 4e |
| Task 2 à 0,40 | 0,547 | 2e |
| Task 1 à 0,60 | 0,433 | 4e |
| Task 1 à 0,70 | 0,506 | 2e |
| Task 2 à 0,40 et Task 3 à 0,05 | 0,655 | 1er |

**Conclusion : la Task 3 a la plus forte pente, puis la Task 2, puis la Task 1.**

## Task 3 : ce qu'on sait depuis 18:45

- Formule du serveur (docstring de `task_template.py`) : retain = clip(acc / acc_origine, 0, 1) ×
  1 / (1 + KL(p_origine ‖ p_candidat)) à T=2 ; chaque sous-score d'oubli = 1 − clip(écart au gold ÷ écart
  de l'origine au gold, 0, 1), sur les forget **et** sur des copies altérées cachées ; porte : 70 %
  d'accuracy sur une validation cachée.
- `c3/eval.py` est plus sévère que le serveur : exp(−KL) au lieu de 1 / (1 + KL), et une porte à 0,72
  d'accord sur augmentations fortes. **Il ne sert plus à refuser une soumission**, seulement à vérifier
  format et fp16 et à ordonner les candidats.
- Les 3 500 images à oublier sont des images générées, concentrées sur 45 classes sur 100. Les 55 autres
  classes ne sont couvertes par aucune mesure locale ; `offforget.npz` est hors des 100 classes.
- Scores serveur connus (soupes origine / `v2_best`, famille B) :

| Fichier | Part de `v2_best` | Accord local (aug.) | Score public |
|---|---|---|---|
| inconnu (16:46, Yihuan ?) | — | — | 0,0026 |
| `soup_mid.pt` | 50 % | 0,82 | 0,0012 |
| `soupB_l85.pt` | 85 % | 0,58 | **0,0035** |

  Un accord local de 0,58 passe donc la porte du serveur, et le score monte avec la part de `v2_best`.
- Délai entre deux soumissions Task 3 : environ 10 minutes, **commun à toute l'équipe**.

## Ce qui reste, par ordre d'impact

| # | Action | Tâche | Certitude de gain | Responsable | Échéance | Décision Selyan |
|---|---|---|---|---|---|---|
| 1 | Soumettre `soupB_l95.pt` (perte forget 0,98 ± 3,04, accord 0,49), puis `soupB_l90.pt` s'il score 0 | 3 | Sans risque sur le public (seul le meilleur compte) ; situe la porte ; gain attendu faible | Selyan lance | dès 19:02 | go par soumission |
| 2 | Distillation de Yihuan (`distill.py`, job `j3d`) et entraînement ancré NPO + KL hors forget (c3-submit, jobs 15902389 et 15902392) | 3 | Incertain ; seule voie crédible vers le palier à 0,027 | Yihuan, c3-submit | candidats évalués avant 22:00 | go de soumission |
| 3 | Phase 1a ep-* (165 requêtes), puis évaluation des ref à 4 répétitions. Phase 1b (60 par endpoint) **seulement si** des prompts atteignent une AUC ≥ 0,9 sur les ref ; sinon changer les prompts d'abord | 2 | Seul moyen de bouger la Task 2 ; budget irréversible | Task 2 | 1a ce soir | 1b : go séparé |
| 4 | Tête de segmentation F (canaux fins, TruFor en mesure seulement) | 1 | Incertain ; plus grosse réserve de la Task 1 (oracle F parfait : 0,778) | c1-probe | fichier test à 05:00 | go de soumission |
| 5 | Carte de corrélation avec le watermark ré-embarqué : essai 2 images, pilote 120 F, complet si le mIoU F dépasse tout 1 (~0,45) | 1 | Incertain ; sans modèle externe | mesure Task 1 | pilote avant 22:00 | — |
| 6 | Recherche exacte du meilleur rectangle (Kadane 2D) sur les 255 images R, en local | 1 | À mesurer ; intégrer si ≥ +0,01 en validation imbriquée | Scrapping info | ce soir | — |
| 7 | Soumettre le fichier Task 1 à 0,547 en validation imbriquée (tête F synthétique + rectangles), après relecture | 1 | +0,04 mesuré en local | Selyan lance | après relecture | go de soumission |

Écarté : template matching sur les R (aucun signal), accélération `box_search` (aucun gain de score),
`soupE_l70` (oublie neuf fois moins que `soupB_l85`).

## File des PR

| PR | Contenu | État | Prochaine étape |
|---|---|---|---|
| #15 | c2 features + correctifs | ouverte | dernier correctif (seuil ≥ 5 prompts dans `fit_affixes`), fusion après CI verte |
| #16, #17, #18 | c3 évaluation, c1 probes, c1 box-speed | fusionnées | relectures Sol en retard : #18 non relue, corrections de #16 et #17 à passer en PR de suivi |
| à venir | `fix/c1-a1-wording`, c1 tête F, c1 ré-embarquement, c1 rectangle exact, scripts c3 de Yihuan | — | relecture avant chaque fusion |

Codex (Sol) : quota épuisé jusqu'à 21:29 ; d'ici là, repli sur l'agent `qa`.

## Règles

- Une soumission seulement avec le go explicite de Selyan, qui la lance lui-même depuis le login JURECA.
  Aucune session ne soumet.
- Task 3 : prévenir avant toute soumission (délai de 10 minutes commun à l'équipe).
- Task 2 : au plus 2 requêtes simultanées par personne (limite de 4 par équipe) ; personne ne requête
  pendant la dépense du budget ep-*.
- GPU : jobs à 4 tâches par nœud, durées courtes ; aucun calcul sur les nœuds de login. Aucune session
  n'annule un job à la place de Selyan.
- Mac 16 Go : 3 sessions actives au maximum, un seul Codex à la fois.
- Aucun dépôt ni code d'une autre équipe n'est lu (règle des organisateurs).

## Questions ouvertes (Selyan)

- Mentors : modèles externes (TruFor, DinoLizer) autorisés en Task 1 ? Le classement final reprend-il la
  meilleure soumission publique ou la dernière ? (Si c'est la dernière, resoumettre le meilleur fichier de
  chaque tâche avant 12:00.)
- Yihuan : quel fichier a été soumis en Task 3 à 16:46 ? Le prévenir que des soupes sont soumises.
- Sophie : 2 requêtes simultanées au plus, aucune pendant la phase 1a.
- Annuler ou non le job tête F 15902276 pour libérer la file au plan B de la Task 3 (avis de la
  coordination : le garder).
