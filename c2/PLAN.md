# c2 — Plan de requêtes (Model Equality Testing)

Statut : **proposition, aucune requête lancée.** Toute requête (ep-* comme ref-*) attend le go explicite de
Selyan. Rédigé le 21/11 vers 17:30.

## 0. Faits mesurés (gratuits)

`GET /v1/models` (lecture seule, ne consomme rien), le 21/11 vers 17:00 :

| Endpoints | used | limit |
|---|---|---|
| ep-01 … ep-33 | 0 | **100** chacun |
| ref-01 … ref-09 | 0 | **1 000 000** chacun |

- Les ref-* ne sont donc **pas** limités à 100 : la docstring de `query_reference.py` (« no query limit »)
  est confirmée par le compteur. Le pilote sur les ref ne coûte rien sur le budget noté ; il coûte
  seulement du temps.
- `owned_by` vaut `hackathon` partout : aucune fuite d'information par les métadonnées.
- Contraintes du kit : 256 tokens en sortie, 1 024 en entrée, 4 requêtes simultanées, pas de message
  système, `temperature`/`seed` ignorés. Une réponse = une unité de budget.

## 1. Le problème en une phrase

Séparer ce qui change les **poids** (fp8, LoRA 0,3/0,7/1,5, autre checkpoint) de ce qui ne change que la
**présentation** (system prompt, persona, watermark, température). Sur les ref : ref-01/03/05/06/08 = un
seul modèle S malgré 4 déguisements ; ref-02 = fp8 ; ref-09/04/07 = LoRA 0,3/0,7/1,5. 90 % du score est
la question « même modèle ? », 10 % la famille.

Conséquences sur le choix des prompts :

- Un test d'égalité de distributions pur (MMD brut) dit « différent » pour un endpoint à température 1,0 ou
  avec persona : c'est le piège. On mesure donc un écart **systématique** et on neutralise le reste
  (section 4).
- Les prompts à **faible entropie et longue sortie** (récitation, listes, calcul) sont les plus utiles :
  un watermark ou une température touchent peu les tokens quasi certains, alors qu'un LoRA ou une
  quantification déplacent les points où le modèle hésite ou se trompe (décimale de pi, verset, nombre premier).
- Les prompts créatifs (histoire, poème) servent surtout à la famille et au style ; un persona les déforme
  le plus, donc ils pèsent peu pour la question « même modèle ».

## 2. Répartition du budget par endpoint noté (100 réponses)

| Phase | Fichier | Prompts × répétitions | Réponses / endpoint | Total 33 ep | Rôle |
|---|---|---|---|---|---|
| 1a famille | `prompts_ep_family.csv` | 5 × 1 | 5 | 165 | identité, coupure, tokenisation, style |
| 1b modèle | `prompts_ep_model.csv` | 15 × 4 | 60 | 1 980 | distance « même modèle » |
| 2 ciblée | sous-ensemble de 1b (mêmes ids) | 5 × 5 de plus | 0 à 25 | ≤ 825 | seulement les endpoints dans une zone ambiguë |
| Réserve | — | — | 10 | 330 | erreurs, contrôle final, imprévu |
| **Total** | | | **≤ 100** | **≤ 3 300** | |

Pourquoi 15 × 4 plutôt que 60 × 1 ou 30 × 2 :

- La statistique (`corr`, section 4) a besoin d'au moins 2 réponses par endpoint et par prompt pour estimer
  la dispersion interne ; avec 2 elle est très bruitée.
- Mesure sur données synthétiques (33 endpoints, 3 graines, seuil automatique) : 15 × 4 et 25 × 4 donnent
  1,00 sur les 3 graines ; 30 × 2 donne 0,64 / 0,64 / 1,00. Le synthétique est volontairement simple :
  ce résultat montre seulement qu'à budget égal, la répétition stabilise la statistique. **Le choix
  final se fait sur les vraies réponses ref** (section 3, étape P2), en rejouant 15 × 4, 20 × 3, 30 × 2
  et 60 × 1 à partir du même pilote.
- Phase 2 : relancer `task_template.py` avec les mêmes ids et `REPEATS = 9` ajoute exactement les
  répétitions 4 à 8 (le kit saute les `(endpoint, id, repeat)` déjà présents). Coût : 5 prompts × 5 = 25.
  On ne la lance que pour les endpoints dont une paire tombe dans la bande ambiguë (distance entre
  la moitié et le double du seuil) ; toute la famille concernée reçoit les mêmes prompts.

## 3. Pilote sur ref-01..09 (coût sur le budget noté : 0)

| Étape | Contenu | Requêtes ref | Durée |
|---|---|---|---|
| P1 | `prompts_ref_pilot.csv` (32 prompts : 5 famille + 15 modèle + 12 candidats) × 10 répétitions × 9 ref | 2 880 | estimation non mesurée : 30 à 60 min à 4 en parallèle (à mesurer sur les 50 premières) |
| P2 | hors ligne : AUC par prompt, choix du noyau, de la statistique, du seuil, de l'allocation | 0 | quelques minutes |
| P3 (option) | 2ᵉ vague de candidats si moins de 10 prompts ont une AUC ≥ 0,9 | ≤ 3 000 | idem P1 |

P2, concrètement :

```bash
# sur JURECA, depuis le clone du dépôt
python c2/analyze.py output/ref_answers.jsonl --prefix ref- --truth c2/kit/practice_answers.csv \
    --t-family 10 --calibrate --rank-prompts --out output/ref_guess.csv
python c2/kit/score_practice.py output/ref_guess.csv            # MCC officiel des ref
# allocation : rejouer avec --max-prompts 15 --max-reps 4, 30 2, 60 1 et plusieurs --seed
# noyau/statistique : --kernel ngram|seq|both, --stat corr|mmd
```

Règles de décision (fixées avant de voir les réponses pour ne pas les ajuster après coup) :

1. Garder dans `prompts_ep_model.csv` les 15 prompts de meilleure AUC « même modèle » sur les ref, avec au
   plus 3 prompts du même type (récitation, liste, calcul, ouvert), pour ne pas tout miser sur un seul
   mécanisme.
2. Retenir le noyau/statistique dont le MCC ref est le plus stable entre graines de sous-échantillonnage,
   pas le meilleur sur une seule.
3. Le seuil calibré sur les ref (petit modèle) sert d'ordre de grandeur, pas de vérité pour les ep :
   les distances d'un modèle plus grand n'ont pas forcément la même échelle. Sur les ep, comparer le seuil
   ref, le seuil automatique et la structure attendue (dans chaque famille, un gros groupe « base +
   déguisements » et des variantes isolées, si les ep reprennent la structure des ref).
4. Ce que le pilote doit aussi établir : les ref « plain » sont-ils déterministes (réponses répétées
   identiques) ? Si oui, les réponses exactement égales entre ep plain sont une preuve forte
   (`analyze.py` les liste).

Hors kit, à décider par Selyan (requêtes ref seulement, quelques dizaines) : tester si l'API renvoie des
`logprobs`, accepte `n > 1` (et comment le compteur ref le décompte) ou un message `assistant` final
(préremplissage). Ces tests demandent un petit script ad hoc, pas encore écrit.

## 4. Analyse hors ligne (`analyze.py`)

1. Par endpoint, retrait du préfixe et du suffixe (en mots) communs à **toutes** ses réponses, tous prompts
   confondus : un vrai début de réponse dépend du prompt, une formule fixe vient d'un system prompt ou d'une
   persona.
2. Réponse → n-grammes de caractères 3-5 hachés (ou alignement `difflib` avec `--kernel seq`), noyau cosinus.
3. Pour chaque prompt et chaque paire (a, b) : `corr = 1 - Kab / sqrt(Kaa·Kbb)`, avec Kaa et Kbb sans la
   diagonale (non biaisés). Un bruit d'échantillonnage plus fort (température 1,0) réduit Kaa, Kbb et Kab
   ensemble, donc la distance reste près de 0 ; un écart systématique de poids l'augmente. `--stat mmd`
   donne le MMD² de Gao et al. pour comparaison.
4. Moyenne sur les prompts → matrice D, puis UPGMA coupé à un seuil : la sortie est une partition, la
   transitivité est donc imposée. Familles d'abord (prompts 1a + 1b), modèles ensuite à l'intérieur de chaque
   famille, ce qui garantit model ⊂ family.
5. Sortie : CSV `endpoint,family,model` ; `--prefix ep-` pour une soumission (le serveur refuse les lignes
   ref-), `--prefix ref-` pour `score_practice.py`.

Tests : `pytest c2/test_analyze.py` (synthétique seulement, voir `synth.py`). Le MCC calculé par
`analyze.score` est vérifié contre `c2/kit/score_practice.py` (copie du kit).

## 5. Déroulé proposé et points de validation

| # | Action | Coût ep | Go requis |
|---|---|---|---|
| 1 | Relire et fusionner ce plan | 0 | revue PR |
| 2 | Pilote P1 sur ref (copie privée de `query_reference.py`, clé lue dans `EAIC_API_KEY`) | 0 | **oui** |
| 3 | P2 hors ligne, mise à jour de `prompts_ep_model.csv` et de ce plan (PR) | 0 | revue PR |
| 4 | Phase 1a + 1b sur ep (`task_template.py`, `QUERY = True`) | 65 × 33 | **oui** |
| 5 | Analyse, première soumission réfléchie | 0 | **oui** (1 soumission / 30 min) |
| 6 | Phase 2 ciblée si des paires restent ambiguës | ≤ 25 × k | **oui** |
| 7 | Soumission finale choisie sur l'analyse, pas sur le score public | 0 | **oui** |

Lancement (étapes 2 et 4), dans un dossier à soi sur JURECA, pas dans le dossier partagé :

```bash
cp /p/scratch/training2666/mouhali1/SLZ/model_equality/{query_reference,task_template}.py ~/c2run/
# dans les copies : API_KEY = os.environ["EAIC_API_KEY"]  (jamais la clé en clair dans un fichier)
# query_reference.py : PROMPTS_FILE = c2/prompts_ref_pilot.csv, REPEATS = 10, ANSWERS_FILE = output/ref_answers.jsonl
# task_template.py   : QUERY = True, PROMPTS_FILE = c2/prompts_ep_family.csv, REPEATS = 1, puis
#                      PROMPTS_FILE = c2/prompts_ep_model.csv, REPEATS = 4, même ANSWERS_FILE = output/ep_answers.jsonl
```

Pièges du kit relevés en lisant le code :

- Avec un `.txt`, l'id d'un prompt est son numéro de ligne : deux fichiers `.txt` différents écrits dans
  le même `answers.jsonl` partagent les ids `0, 1, …` et le second serait **sauté** comme « déjà fait ».
  D'où les fichiers `.csv` avec des ids explicites et uniques (`f01`, `m01`, `c01`).
- `task_template.py` écrit aussi `submission.csv` (baseline « tous différents ») à chaque lancement : ne pas
  la soumettre par erreur.
- Les erreurs HTTP autres que 429/5xx abandonnent la requête sans réessayer : vérifier le nombre de lignes
  du jsonl après chaque phase (`analyze.py` affiche min/max de réponses par endpoint).

## 6. Ce qui reste à valider avec de vraies réponses

- L'AUC réelle des prompts, en particulier pour **fp8** et **LoRA 0,3** (les écarts les plus fins) ; si
  aucun prompt ne les sépare de S sur les ref, il faudra des sorties plus longues ou plus « fragiles »
  (P3), ou les pistes hors kit (logprobs).
- La force réelle des déguisements : un persona qui réécrit tout le texte n'est pas neutralisé par le
  retrait des formules fixes.
- La transposition du seuil ref → ep (petit modèle → modèles plus grands).
- La famille : sur des prompts très contraints, deux familles différentes peuvent donner le même texte ;
  la coupure famille automatique doit être relue à la main avec les réponses d'identité (f01, f02).
- Le temps réel par requête (estimé, pas mesuré).
