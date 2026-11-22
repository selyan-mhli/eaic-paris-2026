# Task 2 — analyse hors ligne, 21/11/2026 vers 23:10–00:20, heure de Paris

Mandat : analyser hors ligne et, si c'est justifié, proposer un candidat. Aucune requête ref ni ep, aucune soumission, aucun sbatch, aucune opération Git, aucune écriture distante. Sur JURECA, je n'ai fait que des lectures (`ls`, `find`, `cat`, `sha256sum`). Rien n'a été écrit hors de ce dossier. Pour cette raison, aucune ligne n'a été ajoutée à `docs/experiences.md` (ligne proposée au §8).

**Aucun MCC de ce rapport n'est un score serveur.** Les MCC « refs » sont mesurés sur ref-01..09 en validation croisée. Les espérances « endpoints » sont des estimations Monte Carlo qui dépendent d'un a priori explicite.

## 1. Corpus figé

`build_corpus.py` écrit `corpus.jsonl` : **2 350 lignes, sha256 `883a2aa1…dcd39`**. Il n'y a aucun dédoublonnage. Chaque ligne porte `campaign`, `source`, `line`, `repeat` et `attempt` (rang d'occurrence de la clé dans la source), plus les hash du prompt et de la réponse.

| Campagne | Lignes | Source |
|---|---:|---|
| pilot1 (f/m/c) | 1 312 | snapshot JURECA `c2run/ref_answers.jsonl` 23:08, sha `756e9209…`. Ses 1 744 premières lignes sont identiques octet par octet à la copie d'audit. |
| pilot2 (w) | 432 | idem |
| pilot3 (e01–e04) | 36 | idem, **partiel** : `q3.py` (autre session) était en cours d'écriture |
| sophie_ref25 | 225 | `reference_answers.jsonl` |
| sophie_minimal | 180 | `ref_pilot_minimal.jsonl` |
| ep1a | 165 | `ep_answers.jsonl` : 33 × 5, une seule réponse par prompt |

- 160 clés dupliquées, dont 152 avec des textes différents. Elles sont conservées et distinguées par `attempt`.
- Un même `pid` correspond toujours au même prompt d'une campagne à l'autre (vérifié par hash).

### Relevés JURECA en lecture seule

- **Phase 1b : aucune réponse produite.** `ep_answers.jsonl` a le même sha qu'à l'audit (`0b2f0a2f…`), avec un mtime de 19:25. `ep1b.py` (sha `c07042b7…`) écrit dans ce même fichier. Aucun autre jsonl ep n'existe dans `/p/scratch/training2666/mouhali1`. Le répertoire scratch de pelet1 n'existe pas.
- **Aucun reçu de soumission** n'a été trouvé.
- Fichiers nouveaux, tous de `mouhali1` :
  - `output/submit_c2.py` (22:52, sha `4fc4f897…`) ;
  - `output/c2_fam3_pairs_v2.csv` (22:54, sha `6799a438…`) ;
  - `c2run/q3.py`, `q3.log`, `prompts_ref_pilot3.csv` (23:00–23:05) ;
  - `ref_answers.jsonl`, qui grossit encore (23:08).
- Ces fichiers sont copiés dans `raw/`.

## 2. Procédure de décision et validation croisée sur les refs (`validate.py`)

On reproduit les conditions des endpoints : **une réponse par prompt f01..f05 par endpoint**. Ces réponses sont tirées dans pilot1 avec une seed fixe, 60 tirages par moitié.

- Moitiés : H0 = répétitions 0–1, H1 = répétitions 2–3.
- Grille des paramètres, fixée avant l'exécution :
  - 4 features : longueur, n-grammes de caractères, préfixe de 3 mots, mélange ;
  - 2 règles : « gros groupe + isolés » (tau de 1 à 4) ou lien moyen coupé à un quantile.
- Tout est choisi sur une moitié et évalué sur l'autre.
- Couverture des paires : 100 % pour chaque feature (les 5 prompts sont présents des deux côtés pour les 36 paires, dans chaque moitié).

| Feature | AUC même/différent, H0 / H1 |
|---|---|
| longueur | 0,600 / 0,613 |
| n-grammes | **0,269 / 0,414** (en dessous du hasard) |
| préfixe | **0,357 / 0,362** (en dessous du hasard) |
| longueur + n-grammes | 0,463 / 0,575 |

Les similarités de texte sont *anti-informatives* sur les refs. Les déguisements (persona, prompt système, température) changent plus le texte que fp8 ou LoRA.

| Sélection → évaluation | Choix | MCC sélection | **MCC hors sélection** | Tirages à MCC 0 |
|---|---|---:|---:|---:|
| H0 → H1 | len, isolate, tau 2,5 | 0,269 | **0,245 ± 0,159** | 23 % |
| H1 → H0 | len, isolate, tau 1,5 | 0,316 | **0,254 ± 0,158** | 10 % |

Taux d'erreur par type de paire, hors sélection (H0→H1 / H1→H0) :

| Type de paire | Erreur |
|---|---|
| déguisement–déguisement | 0,04 / 0,14 |
| base–LoRA 1,5 | 0,23 / 0,08 |
| base–LoRA 0,7 | 0,90 / 0,70 |
| base–fp8 | 0,98 / 0,93 |
| base–LoRA 0,3 | 0,98 / 0,93 |

**Seule la LoRA forte est détectée. fp8 et LoRA faible restent invisibles avec 5 réponses uniques.** Pour la suite, tau = 2,0 a été retenu : c'est le meilleur minimum sur les deux moitiés (0,266 / 0,290).

**Nombre de prompts** (`validate_moreprompts.py`) : on tire K prompts au hasard (sans sélection sur leur score), une réponse chacun, avec tau choisi sur une moitié.

| K | 5 | 10 | 15 | 25 | 32 |
|---|---:|---:|---:|---:|---:|
| MCC hors sélection | 0,142 | 0,206 | 0,192 | 0,216 | 0,236 |

La règle de longueur plafonne vers 0,24. Davantage de prompts à une réponse ne la débloque pas.

## 3. Application aux 33 endpoints (`apply_ep.py`)

Règle famille, sans paramètre ajusté :
- **L** si f01 cite Meta ou Llama, ou si f02 donne « December 2023 ».
- Sinon **Qa** si la réponse f02 commence par un nom de mois, **Qb** sinon.

Résultat : 11 / 11 / 11, accord **528/528** avec les familles de `c2_fam3_pairs_v2`. Le « 3 × 11 » sort des données, il n'a pas été imposé. Mais cette règle a été écrite après lecture des réponses ep, et les refs (une seule famille) ne peuvent pas la valider.

Isolés par la longueur (tau 2,0, dans chaque famille) : ep-02, ep-17, ep-19, ep-31.

| CSV | Familles | Modèles | Paires « même modèle » |
|---|---:|---:|---:|
| c2_lump_v1 | 2 | 5 | 235 |
| c2_fam3_pairs_v2 | 3 | 28 | 5 |
| ep_first_hamming | 4 | 5 | 379 |
| ep_first_ngram | 9 | 32 | 1 |
| baseline_random | 3 | 9 | 58 |
| W_fam3_lump | 3 | 3 | 165 |
| **W_fam3_len_tau2** | 3 | 7 | 127 |
| W_fam3_singletons | 3 | 33 | 0 |

Comparaison paire par paire :
- **fam3_pairs_v2 vs lump_v1** : 230 décisions modèle et 121 décisions famille diffèrent. Les 121 sont exactement les paires Qa–Qb. Les 5 paires de fam3 sont toutes déjà « même modèle » dans lump. Les 230 autres paires « même » de lump deviennent « différents ».
- W_len vs lump_v1 : 108 décisions modèle, 121 décisions famille.
- W_len vs fam3_pairs_v2 : 124 décisions modèle, 0 décision famille.
- Le détail est dans `apply_results.json`.

## 4. Ce qu'apprennent les scores publics observés

Ces conclusions supposent que la formule du `task_template.py` est exacte et que la fraction publique est un sous-ensemble de paires.

- **lump_v1 = 0,05 réfute la partition Qwen/Llama en 2 familles exactes.** Avec des familles exactes, le MCC famille vaut 1 sur tout sous-ensemble de paires. Le MCC modèle est tronqué à 0 ou plus. Le score serait donc au moins 0,10.
- 0,05 correspond à un MCC famille d'environ 0,5 et à un MCC modèle public d'environ 0. Ce MCC famille est cohérent avec Qa/Qb/L : lump_v1 aurait 0,62 contre Qa/Qb/L sur toutes les paires.
- **Les isolés de longueur ep-02, ep-17 et ep-31 n'ont donc apporté aucun signal positif sur les paires publiques.** C'est une preuve faible (environ 9 paires publiques les concernent), mais elle va contre eux.
- **baseline_random = 0,20** : son MCC famille vaut 0 contre les deux hypothèses de familles. Ses 0,20 viennent donc d'un MCC modèle public d'environ 0,22 obtenu par chance. Sous les a priori, c'est rare (p90 du score aléatoire d'environ 0,10).

### Interprétation du score de fam3_pairs_v2 (soumission prévue à 23:22)

- **Moins de 0,10** : le découpage Qa/Qb/L n'est pas exact. Cela invaliderait aussi W_len.
- **Environ 0,10** : familles cohérentes. Les 5 paires ne donnent rien en public, et seulement 1 ou 2 d'entre elles sont attendues dans le public.
- **Plus de 0,10** : au moins une paire publique est juste. Dans la simulation, le score privé attendu de fam3 reste quasi plat quel que soit ce score public (0,12–0,15) : **ne pas surinterpréter**.
- La réponse serveur ne montre peut-être le score que si `improved` (meilleur actuel 0,20). Il faut relever `score` dans le reçu.

## 5. Fusionner ou séparer quand le signal manque (MCC, cas des refs)

Ce sont des calculs exacts sur la vérité des refs (`t2lib.pair_mcc`) :

| Décision | MCC |
|---|---:|
| Tout fusionner, ou tout séparer | 0 |
| Isoler 07 | 0,331 |
| Isoler 04 et 07 | 0,524 |
| Isoler 04, 07 et 02 | 0,734 |
| Isoler 04, 07 + **une base déguisée** (01) | 0,231 |
| Isoler seulement 01 | −0,265 |
| Tout séparer sauf une vraie paire | 0,273 |
| Tout séparer sauf une fausse paire (01–02) | −0,105 |

**Ce que le MCC modèle récompense** : seules les décisions non constantes et justes rapportent. Quand le signal manque, le choix le plus sûr est de **mettre l'endpoint douteux dans le gros groupe**, parce que :
- un vrai variant laissé dans le gros groupe ne coûte que des faux positifs ;
- isoler à tort une base déguisée retire plusieurs vrais positifs et fait vite chuter le MCC ;
- en mode « tout séparer », une seule paire fausse rend le MCC négatif.

## 6. Espérance du score (`simulate.py`, 20 000 vérités par a priori, 30 % de paires publiques tirées au hasard)

A priori :
- familles H3 (Qa/Qb/L) ou H2 (Q/L), probabilité 1/2 chacune ;
- par famille, une base de taille 1 à 75 % de la famille, le reste réparti sur des variants ;
- mode `unif` : appartenance au hasard ;
- mode `len` : les isolés de longueur sont hors base avec une probabilité d'environ 0,8.

Le conditionnement ABC sur les deux scores publics observés retient 303 vérités (`unif`) et 184 (`len`). **P(H3 | observations) = 1,0 dans les deux cas.**

| Score privé attendu (ABC) | unif | len |
|---|---:|---:|
| c2_lump_v1 | 0,087 | 0,122 |
| c2_fam3_pairs_v2 | 0,124 | 0,130 |
| W_fam3_lump / W_fam3_singletons | 0,100 | 0,100 |
| **W_fam3_len_tau2** | **0,131** | **0,177** |
| baseline_random | 0,060 | 0,055 |
| ep_first_hamming | 0,034 | 0,068 |
| ep_first_ngram | 0,021 | 0,025 |

Pour W_len, le p10 est 0,10 et le p90 va de 0,21 à 0,30. **Ce sont des estimations qui dépendent de l'a priori, pas des mesures.** L'écart entre W_len et fam3_pairs_v2 (+0,007 à +0,047) est petit devant la dispersion. De plus, W_len reprend les isolés que lump_v1 n'a pas validés en public.

## 7. Recommandation sur les CSV

- **Candidat préparé : `cand_W_fam3_len_tau2.csv`.** Il est le meilleur en espérance dans les deux a priori, avec le même plancher que fam3 (0,10 si Qa/Qb/L est juste). Ses paramètres viennent de la validation croisée.
- **Je ne le recommande pas comme remplacement ferme.** Le gain est dans le bruit, il dépend de l'a priori et il s'appuie sur des isolés que le public n'a pas confirmés. Son score public attendu est d'environ 0,10, donc inférieur à 0,20.
- Attendre le score de fam3 (§4). Si ce score est inférieur à 0,10, aucun candidat 3 familles n'est fiable.
- **Point stratégique à vérifier avec les organisateurs** : quelle soumission compte pour le classement final ? Si c'est la meilleure publique, le 0,20 aléatoire sera retenu (environ 0,05 attendu en privé). Aucun candidat honnête n'a alors d'espérance publique supérieure à 0,20.

## 8. Collecte supplémentaire

- **Ne pas lancer la 1b actuelle (924 réponses) pour la règle de longueur.** Sur les refs, passer de 5 à 15 prompts n'apporte qu'environ +0,05 hors sélection, plafond vers 0,24. fp8 et LoRA faible restent invisibles.
- **D'abord, une expérience gratuite sur les refs, pré-enregistrée :**
  - 3 prompts déterministes longs (récitation de π, liste de nombres premiers, base64 ou inversion, déjà dans pilot3), 10 répétitions × 9 refs = **270 réponses ref**, sans coût de budget ep ;
  - analyse figée à l'avance : test à deux échantillons (MMD sur n-grammes ou position de la première erreur) entre endpoints, seuil choisi sur 5 répétitions et évalué sur les 5 autres ;
  - critère d'arrêt : MCC hors sélection d'au moins 0,6 **et** séparation base–fp8 ou base–LoRA 0,3 dans au moins 70 % des tirages.
- **Seulement si ce critère passe : lot ep de 2 ou 3 prompts × 10 réponses = 20 à 30 par endpoint (660 à 990 au total)**, analysé avec la même règle figée dans chaque famille. Le budget restant (95 par endpoint) n'a pas été vérifié par compteur.
- Gain : impossible à chiffrer avant le test sur les refs. La borne haute est une identification correcte des variants dans chaque famille (MCC modèle jusqu'à 1). Sinon, on reste sur le plancher d'environ 0,10 à 0,18.
- Ligne proposée pour `docs/experiences.md` : `2026-11-21 23:10 | c2 | longueur+isolement validé par moitiés, familles Qa/Qb/L | hors ligne, ~1 min CPU local | refs hors sélection 0,245/0,254 | non soumis | docs/task2-work-20261121/cand_W_fam3_len_tau2.csv`

## 9. Commandes, tests, limites

```
cd docs/task2-work-20261121
python3 build_corpus.py; python3 validate.py; python3 validate_moreprompts.py
python3 apply_ep.py; python3 simulate.py; python3 -m pytest -q test_t2lib.py   # 4 passed
```

Le test du MCC à la main reproduit 0,524 (kit officiel, isolement de 04 et 07).

Limites :
- les refs sont un petit modèle très bruité, et le transfert vers Qwen ou Llama n'est pas démontré ;
- la règle famille n'est pas validée ;
- le sous-ensemble public est inconnu ;
- pilot3 était incomplet au moment du snapshot.

## 10. JURECA indisponible depuis 23:17, ce qui reste à lire

Tous mes relevés distants datent d'avant 23:17 (le dernier snapshot est de 23:08) ; je n'ai pas retenté de connexion. Lectures à faire une fois l'accès rétabli par Selyan :

- le reçu ou la sortie de la soumission de `c2_fam3_pairs_v2.csv` prévue à 23:22 ;
- `c2run/ref_answers.jsonl` et `q3.log` une fois pilot3 terminé (72 réponses), puis relancer `build_corpus.py` ;
- le registre de soumissions de l'autre session, pour avoir les sha des CSV et les scores.
