# Audit global SLZ — de la création à dim. 22/11 00:40

Demande de Selyan : audit complet depuis la création (hauts, bas, ce qui a marché, ce qui a moins marché), stratégies non utilisées, recherche de solutions en ligne (papiers, dépôts, annales, profs). Lecture seule, à une exception près : le job 15904061 a été lancé sur ordre explicite de Selyan (§5).

Sources : historique Git (18 commits, 21/11 14:00 → 21/11 soir), `docs/experiences.md`, `docs/SLZ_COORDINATION_20261121.md`, les audits `RAPPORT-CODEX.md` (T1), `docs/task2-*`, `docs/task3-*`, `docs/audit-strategies-20261121/`, `docs/point-campagne-20261121/`, `c1/VEILLE.md` (v1+v2), worktrees `eaic-wt-c1-pixseg`, `eaic-wt-c3-veille`, et le classement public relevé à **00:25**. Les quatre recherches externes (T1, T2, T3, plus une recherche commune sur les éditions passées) ont été menées sans lire aucun dépôt d'une équipe de la manche.

## 1. Où on en est (classement public, 00:25)

| | T1 | T2 | T3 | Général |
|---|---:|---:|---:|---:|
| SLZ | 0,58 (24e/42) | 0,20 (16e/35) | 0,08 (25e/41) | **0,30 (26e/33)** |
| 1er général (Fable 4.2) | 0,72 | 1,00 | 0,35 | 0,91 |
| 6e (Eugenia) | 0,75 | 0,85 | 0,16 | 0,73 |
| Médiane du top 10 | ≈ 0,74 | ≈ 0,85 | ≈ 0,30 | — |

Constat clé, **nouveau** : en T1, une quinzaine d'équipes sont entre 0,71 et 0,77, dont plusieurs qui n'ont presque rien en T2 (Incoming PhD : T1 0,75, T2 0,00). **0,74 en T1 est un niveau « standard » de cette manche**, atteignable par une recette générique, alors que nous plafonnons à 0,58 avec une recette fondée sur le signal du watermark.

## 2. Chronologie : hauts et bas

| Heure (21/11) | Événement | Lecture |
|---|---|---|
| 14:00–15:00 | Squelette, agent ml-dev, autoresearch, briefs, CI | Bonne base : chaque épreuve a son responsable et sa validation prévue |
| 15:00–17:35 | T1 : métrique = référence du kit, cartes de chaleur, recherche de boîtes, CV imbriquée 0,5088 | **Haut** : la validation locale a prédit le public à 0,001 près |
| 16:48 | T2 : baseline aléatoire 0,20 | Encore aujourd'hui notre meilleur T2 |
| 17:00–17:20 | Fantôme c2 sur un autre login, OOM de gen_f, canal 0 PixelSeal inutile | **Bas** : 1 h 30 perdue pour une coéquipière ; leçon « chercher sur tous les logins » |
| 17:30–19:45 | T1 tête trous F (réel puis synthétique) : 0,5307 → 0,5572 | **Haut** : règle +0,01, ensembles refusés quand le gain est trop faible |
| 18:40–19:57 | T3 : soupes, distillation, ancrage → 0,001–0,004 | **Bas** : 6 soumissions à l'aveugle, sans proxy fidèle ; doublons de fichiers |
| 20:05 | T1 0,5845 (tête + pixseg StegaStamp) | Dernier gain T1 de la nuit |
| 22:11–00:20 | T3 « push » : 0,031 → 0,057 → **0,083** | **Haut** : le seul signal serveur positif, compris ensuite (suppression de 45 classes entières) |
| 22:52–23:23 | T2 lump_v1 0,05, fam3_pairs_v2 0,10 | **Bas** : soumissions sans validation locale ; mais elles ont appris « 3 familles de 11 » |
| 23:05–23:14 | 5 jobs hors réservation, inodes à 1,2 %, automates de soumission hors mandat | **Bas** : processus ; corrigé par le gel et un orchestrateur unique |
| 23:24–00:30 | Marges T3, mécanismes, class_split, pilote dense T1 (+0,009 estimé, fermé) | Bonne rigueur (règles figées avant lecture) mais **rendement faible** : beaucoup de mesure, aucun saut |

### Ce qui a bien marché
1. **Validation locale imbriquée en T1** : 0,5583 local → 0,5572 public. C'est le seul endroit où nous avons avancé vite et sans gaspiller de soumission.
2. **Règles fixées avant la lecture des résultats** (+0,01, POURSUIVRE/ABANDON) : elles ont évité de soumettre du bruit (ensembles de graines, NCC, TrustMark dense).
3. **T3 push** : modifier seulement la tête est peu coûteux, mesurable hors ligne (`head_search` reproduit le GPU à 3e-4 près) et c'est le seul mécanisme que le serveur récompense.
4. **Journal et registre** : chaque soumission a son sha, son heure et son auteur.

### Ce qui a moins bien marché
1. **Mauvaise répartition de l'effort, puis mauvaise idée de la difficulté.** La simulation de 18:35 a conclu que T2/T3 font bouger le classement plus vite que T1. C'est vrai à la marge, mais le classement montre maintenant que **T1 est la tâche où l'écart avec le top 10 est le plus facile à combler** : nous sommes 0,16 sous un niveau atteint par 15 équipes.
2. **Recette T1 enfermée dans le signal du watermark.** Tous nos canaux (accord par position, boîtes décodées, NCC, dense sur décodeurs) lisent le watermark. Nous n'avons jamais attaqué la tâche par sa face forensique, alors que notre propre test A1 (17:45) a établi que **l'extérieur des F est recollé tel quel**, donc que la zone éditée est une région générée.
3. **T2 : 0 requête ep dépensée de façon informative en 10 h.** Il reste 95/100 réponses par endpoint. L'analyse a beaucoup porté sur des proxys de texte que les audits ont réfutés (AUC sous le hasard).
4. **T3 : soumissions sans proxy fidèle** avant 22:00, puis balayage d'un seul coefficient.
5. **Processus** : trop de sessions parallèles (4 claude + Codex), automates de soumission hors mandat, jobs hors réservation, jobs à 1 GPU sur un nœud de 4.

## 3. Ce qu'on a oublié ou jamais essayé

### T1 (priorité 1 : le plus gros écart à combler)
| Piste | Statut | Pourquoi maintenant |
|---|---|---|
| **Segmentation supervisée à encodeur pré-entraîné (ImageNet) sur RGB réel** | option `--backbone` codée dans pixseg, **jamais lancée** ; tous les pixseg étaient from scratch sur le passe-haut | A1 dit « recollé » : la frontière édition/original est un artefact visuel apprenable. Recette générique cohérente avec les 0,74 de 15 équipes. **Lancée : job 15904061 (§5)** |
| Choix du masque par espérance de mIoU, image par image (Nowozin 2014) | jamais fait | 1 h CPU, sur les probabilités OOF ; la mIoU à 2 classes est très sensible à la décision par image |
| Perte Lovász par image | jamais faite | aligne l'entraînement sur la métrique |
| Synthétique d'édition réaliste (markbench : SD-inpainting, outpainting, splicing) **avec l'extérieur recollé** | notre synthétique ne fait que des watermarks | données proches du test pour l'encodeur pré-entraîné |
| AEROBLADE / erreur de reconstruction VAE | jamais fait (pas de VAE en cache) | complément forensique |

### T2
| Piste | Statut | Coût |
|---|---|---|
| Choix discrets équivalents (AgentProv, **sprintml/agent_prov**, le labo des organisateurs) : question à K réponses équivalentes, ordre mélangé, `max_tokens` ≈ 5, distribution des choix, MMD à noyau delta. Robuste aux personas (7 % de faux positifs contre 67 % pour MET) | jamais essayé | à calibrer sur les refs, qui ont un budget de 1 000 000 (vérifié dans les réponses : `limit 1000000`) |
| Suites binaires pseudo-aléatoires (FLIPS, arXiv 2606.03330) + statistiques NIST (runs) : rappel 87–99 % sur fp8/int4 | notre p3 « listes » s'en approche mais sans features NIST | 8–40 requêtes par endpoint |
| Rapport de vraisemblance fp8 multi-schéma en boîte blanche (SmolLM2-360M identifié) | en cours (`wb_stats.py`) | 0 requête |
| Compter les tokens d'entrée par les erreurs 400 (« X in the messages ») pour repérer les personas cachées | jamais essayé ; à vérifier sur une ref que les 400 ne sont pas débitées | ≈ 0 |
| Écarts de copie discrets déjà présents (prompt f03 : « ∂∫東京 » sans espace chez ep-05/11/15/27, répétition chez ep-21/32) | jamais exploité | 0 |

Vérifié par l'audit : les réponses ne contiennent **pas** de champ `usage`, l'`id` est aléatoire, `created` vaut 0. Aucune fuite de métadonnées.

### T3
| Piste | Statut | Coût |
|---|---|---|
| **Push stratifié par confiance** (RUM 2406.01257 ; corrélation confiance/mémorisation −0,8 à −0,9 sur ViT) : facteur fort pour les images très confiantes, faible pour les typiques | jamais essayé | < 1 h CPU avec `head_search` |
| **Filtration linéaire normalisante de la tête** (Baumhauer 2002.02730) : la masse retirée va vers la confusion naturelle entre classes au lieu de rester indéterminée | jamais essayé | 1–2 h CPU, forme fermée |
| Cible « non-membre de même classe » : q_c = softmax moyen de l'origine sur des photos hors forget prédites c (ReGUn adapté) | jamais essayé | 1–2 h, quelques min GPU |
| JiT (Lipschitz, zero-shot) | jamais essayé | priorité basse (risque de dérive façon NPO) |

## 4. Recherche externe : y a-t-il une solution en ligne ?

**Non, aucune solution publique quasi directe** pour aucune des trois tâches (vérifié) :
- Éditions passées CISPA/SprintML : Paris 2025 (détection Tree-Ring/BitMark/Stable Signature), Vienne (suppression BitMark), Stockholm, Varsovie (inversion de modèle), finale de juillet 2026 (watermark **texte**, score par token, aucun code gagnant publié). Aucune n'avait de localisation d'image, d'égalité de LLM ou d'unlearning ViT.
- HF SprintML (25 datasets) : `SprintML/unlearning` a été créé le 17/11 (c'est la manche en cours) ; `watermark_localization` ne contient que du texte ; les jeux du cours TML 2026 portent sur d'autres sujets.
- Papiers des organisateurs : rien sur la localisation ni sur l'unlearning. **Une seule ressource directement pertinente : `sprintml/agent_prov` (EMNLP 2026)** pour la T2.
- Les dépôts publics de concurrents de la manche (sinonchum/euro-ai-championship-paris, gokunguru, adembnn) n'ont pas été lus, conformément à la règle anti-collaboration.

Onglets du navigateur : Claude in Chrome n'était pas connecté, je ne les ai donc pas lus.

## 5. Action lancée sur ordre de Selyan

Job **15904061** `slz_t1_pre` (dc-gpu-devel, 1 nœud, 4 GPU, 55 min max), voir le registre de coordination à 00:38. Réel seul, 5 folds de gated_eval : convnext_tiny (folds 0,1 / 2,3 / 4) et convnext_small (folds 0,1). La règle de décision a été fixée avant la lecture : POURSUIVRE si l'OOF du réseau seul ≥ 0,58 ou si un groupe F gagne ≥ +0,05 ; ABANDON sous 0,52.

## 6. Recommandations pour les 11 h restantes (fin à 12:00, démo figée à 08:00)

1. **T1** : lire 15904061. S'il passe la règle, entraîner sur tout le train, faire la mesure routée contre le champion, ajouter la décision par espérance de mIoU, puis soumettre avec le GO de Selyan. C'est le levier au rendement le plus probable.
2. **T3** : push stratifié par confiance et filtration de tête, hors ligne avec `head_search`. Au plus 2 soumissions, choisies a priori.
3. **T2** : sur les refs (gratuites), tester AgentProv et FLIPS avec une analyse pré-enregistrée. Ne dépenser les ep que si le MCC hors sélection est ≥ 0,6 dans les deux sens.
4. Processus : une seule session lance les jobs et les soumissions ; trois sessions au plus en même temps.
