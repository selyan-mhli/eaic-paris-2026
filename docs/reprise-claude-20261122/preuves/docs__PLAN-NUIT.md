# Plan de la nuit — EAIC Paris 2026 (équipe SLZ)

Source unique de « qui fait quoi » jusqu'à la fin. La session de coordination (« Organisation du projet et
tasks ») le tient à jour à chaque fusion ou décision. Les sessions de travail le lisent avant de commencer
une étape. Le détail des épreuves reste dans [HACKATHON.md](HACKATHON.md), les mesures dans
[experiences.md](experiences.md).

État au 21/11, 18:35. Fin des soumissions : **dimanche 22/11, 12:00**. Démo figée à **08:00**. Deck prêt à
**11:00**. Seules les **6 premières équipes** au général présentent.

## Où on en est (leaderboard public, 18:35)

SLZ **6e / 12** au général (0,384) : Task 1 0,531, Task 2 0,20, Task 3 0,003. Leader : Fable 4.2 (0,545).
Le général normalise chaque tâche entre 0 (moins bonne équipe) et 1 (meilleure), puis fait la moyenne.

Simulation avec la règle officielle, les autres équipes figées (indicatif, scores publics) :

| Si on obtient… | Général | Rang |
|---|---|---|
| rien de plus | 0,384 | 6e |
| Task 3 à 0,02 | 0,423 | 3e |
| Task 3 à 0,05 | 0,491 | 3e |
| Task 2 à 0,30 | 0,467 | 3e |
| Task 2 à 0,40 | 0,550 | 1er |
| Task 1 à 0,56 | 0,406 | 4e |
| Task 1 à 0,60 | 0,437 | 3e |
| Task 2 à 0,30 et Task 3 à 0,03 | 0,529 | 2e |

**Conclusion : les Tasks 2 et 3 font bouger le classement plus vite que la Task 1.** On ne lâche pas la
Task 1, mais elle n'est plus la seule priorité.

## Ce qui reste, par ordre d'impact

| # | Action | Tâche | Certitude de gain | Responsable | Échéance | Décision Selyan |
|---|---|---|---|---|---|---|
| 1 | Soumettre le meilleur mélange origine/v2_best qui passe les 4 contrôles, puis les suivants à chaque délai | 3 | **Aucun risque sur le public** (seul le meilleur compte) ; gain réel inconnu tant que le serveur n'a pas noté | c3-submit | dès la sortie du job j3 | go par soumission |
| 2 | Phase 1a ep-* (5 prompts famille × 33 = 165 requêtes, 5 % du budget) : **go donné à 18:40**. Phase 1b (60 par endpoint) après l'analyse du pilote | 2 | **Seul moyen de bouger la Task 2** ; risque : budget irréversible | Task 2 | 1a ce soir, 1b vers 20:30 | 1b : go séparé |
| 3 | Tête de segmentation F (+ canaux fins, TruFor en mesure) | 1 | Incertain ; plus grosse réserve de la Task 1 (oracle F parfait : 0,778) | c1-probe | fichier test à 05:00 | go de soumission |
| 3 bis | Carte de corrélation avec le watermark ré-embarqué (F) : code écrit par ml-dev (Codex à court de quota) dans `.claude/worktrees/c1-reembed`, tests synthétiques verts ; pilote JURECA | 1 | Incertain ; sans modèle externe | mesure c1 | pilote avant 22:00 | — |
| 4 | ~~Template matching sur les R TrustMark et StegaStamp~~ **écarté à 18:50** : oracle sans signal pour TrustMark, signal faible et redondant pour StegaStamp | 1 | — | — | — | — |
| 5 | Accélération box_search (seulement si l'équivalence est démontrée) | 1 | Aucun gain de score, gain de GPU | mesure c1 | PR ce soir | — |

## File des PR

| PR | Contenu | État | Prochaine étape |
|---|---|---|---|
| #15 | c2 features + correctifs | relecture Sol : 1 régression (affixes) | correctif, puis vérification Sol, puis fusion |
| #16 | c3 évaluation des candidats + soupes | ouverte | relecture Sol |
| #17 | c1 A1/A2 + veille v2 | relecture Sol : conclusion A1 à nuancer | correctif, puis fusion |
| à venir | c1 box-speed, c1 tête F, c1 template matching, c3 correctif eval.py | — | relecture Sol avant chaque fusion |

## Règles

- Une soumission seulement avec le go explicite de Selyan, et après relecture Sol du chiffre qui la justifie.
- Task 2 : au plus 2 requêtes simultanées par personne (limite de 4 par équipe) ; personne ne requête
  pendant la dépense du budget ep-*.
- GPU : jobs à 4 tâches par nœud, durées courtes ; aucun calcul sur les nœuds de login.
- Mac 16 Go : 3 sessions actives au maximum, un seul Codex à la fois.

## Questions ouvertes (Selyan)

- Mentors : modèles externes (TruFor, DinoLizer) autorisés en Task 1 ? Le classement final reprend-il la
  meilleure soumission publique ou la dernière ?
- Yihuan : quel fichier a été soumis en Task 3 à 16:46 ?
