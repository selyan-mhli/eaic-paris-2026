# Classement live — 22 novembre 2026, 03:17 Paris

**SLZ : 27e sur 38 équipes au général, 0,319108.** Le sixième est à 0,761185 ; écart 0,442077. Le premier, Murmell, est à 0,974910. Les scores SLZ sont confirmés sur le [classement officiel](http://34.147.229.55/leaderboard_page), téléchargé à nouveau à 03:17:19. Aucun appel authentifié ni soumission.

| Tâche | SLZ | Rang | Maximum actuel | Dernière amélioration SLZ, Paris |
|---|---:|---:|---:|---|
| T1 | 0,673907 | 20e/42 | 0,803274 | 02:43 |
| T2 | 0,20 | 25e/40 | 1,00 | 21/11 16:48 |
| T3 | 0,094115 | 31e/42 | 0,439580 | 00:40 |

## Pourquoi le général bouge si peu

Relevé conservé vers 00:26 : T1 0,584524 ; T2 0,20 ; T3 0,082681 ; général 0,299692, 26e/33. Aujourd'hui : T1 **+0,089383**, T2 **inchangée**, T3 **+0,011434**, général **+0,019416**, rang 27e/38.

Le général normalise chaque tâche par `(score − minimum)/(maximum − minimum)`, puis moyenne les trois valeurs. Le calcul reproduit exactement les 38 scores globaux complets du relevé, erreur maximum calculée 0,0.

Décomposition séquentielle, en conservant d'abord le barème de 00:26 : nos améliorations brutes auraient donné **+0,090804 de général**. L'évolution des extrema entre ce relevé et le live retire ensuite **−0,071388** à nos scores actuels, d'où **+0,019416 net**. Cette décomposition dépend de l'ordre choisi ; elle décrit exactement ces deux instantanés, sans mesure continue entre les deux.

En T1, le minimum passe de 0,403691 à 0,491300 et le maximum de 0,770002 à 0,803274. En T2, le minimum passe de 0 à 0,05. Le maximum T3 passe de 0,402509 à 0,439580. Le barème s'est donc durci pendant nos améliorations. Les nouvelles équipes et leurs progrès modifient aussi les rangs.

Le gain T1 est réel. Le résultat de compétition reste insuffisant, car T2 conserve la baseline et T3 reste loin du haut du tableau.

## Ce qui peut bouger le classement, concurrents et extrema figés

Ces valeurs sont des **scénarios arithmétiques**, pas des performances prévues. Les rangs sont recalculés avec tous les autres scores du relevé figés ; ils peuvent changer dès la soumission d'une autre équipe.

| Scénario | Général | Gain général | Rang simulé |
|---|---:|---:|---:|
| Actuel | 0,319108 | — | 27e |
| Gain T1 +0,007623 du pilote 6k reproduit au public | 0,327253 | +0,008145 | 26e |
| T1 à 0,70, reste inchangé | 0,346987 | +0,027880 | 25e |
| T1 à 0,75, reste inchangé | 0,400410 | +0,081303 | 21e |
| T2 à 0,40, reste inchangé | 0,389283 | +0,070175 | 22e |
| T2 à 0,60, reste inchangé | 0,459459 | +0,140351 | 18e |
| T3 à 0,20, reste inchangé | 0,399401 | +0,080293 | 21e |
| T3 à 0,30, reste inchangé | 0,475231 | +0,156123 | 18e |
| T1 0,75 / T2 0,40 / T3 0,20 | 0,550879 | +0,231771 | 14e |
| T1 0,75 / T2 0,60 / T3 0,20 | 0,621054 | +0,301946 | 10e |
| T1 0,75 / T2 0,85 / T3 0,30 | 0,784603 | +0,465496 | 5e |

**Même une T1 première du tableau, avec T2/T3 inchangées, plafonne à 0,457332 de général** dans ce contexte. Les plafonds d'une seule tâche portée au maximum actuel sont T2 0,599810 et T3 0,581074. Une tâche normalisée plafonne à 1, y compris si l'équipe dépasse le maximum brut existant : prolonger linéairement les calculs au-delà de ce maximum serait faux. À scores adverses fixes, le top 6 exige donc d'améliorer au moins deux tâches.

À la marge, +0,01 brut vaut aujourd'hui +0,010685 de général en T1, +0,003509 en T2, +0,007583 en T3. T1 a le meilleur rendement par point brut ; T2/T3 disposent d'une réserve de gain plus grande. Ces deux constats sont compatibles et ne suffisent pas, seuls, à choisir une méthode.

## Conséquence pour les prochains travaux

Le job **15904703** autorisé reste une validation utile du petit gain 6k. Le pilote ne démontre pas une rupture : son +0,007623 global T1, s'il se transférait intégralement au public, représente seulement +0,008145 de général. Le gain exact du candidat à trois graines n'est pas encore mesuré.

La piste T2 des égalités est close selon son critère annoncé : MCC 0 sur les deux moitiés ; aucun lot ep justifié par ce résultat. L'identification des trois familles est acquise, la séparation des identités intra-famille reste le verrou.

En T3, continuer simplement à augmenter le coefficient push n'a pas de justification : s175 donne 0,0941, s200 est rapporté à 0,0792. Cela ne condamne pas toute amélioration T3, mais le réglage d'intensité seul a déjà plafonné dans les essais connus.

**Arbitrage à cadrer par Selyan/Opus : une expérience T3 visant un saut vers 0,20–0,30, ou une nouvelle méthode T2 validée hors sélection visant 0,40–0,60.** Ces chiffres sont des objectifs de classement, pas des scores soutenus par les validations actuelles. Aucun nouvel algorithme ou job hors mandat n'a été choisi ou lancé dans cette analyse. La validation 6k autorisée continue ; ses résultats doivent être mesurés avant d'en déduire une soumission.

## Preuves

- HTML : `leaderboard-live-0317.html`, SHA256 `17aec74d823e753fd42ec95232ee7b54fd8e1bae15b059a857d8862c6cc656a5` ; en-têtes de téléchargement conservés à côté.
- Calcul reproductible : `analyze_live.py` ; sortie complète avec extrema, scénarios, 38 équipes et historique : `leaderboard-live-analysis.json`.
- Source historique 00:26 : `/private/tmp/claude-501/-Users-selyan--claude-spares-spare-oGfpae/d60fc14b-9806-4319-b999-3844f0f1cede/scratchpad/lb2.html`, déjà conservée avant cette analyse.
- La consultation par outil web du site à adresse IP a échoué ; téléchargement HTTP direct en lecture seule réussi, puis analyse locale de l'HTML officiel. Les constats live ne proviennent pas d'un moteur de recherche.
