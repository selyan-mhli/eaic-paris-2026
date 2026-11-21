# EAIC Paris 2026 — les 3 épreuves

Source : les 3 énoncés PDF officiels (CISPA, superviseurs Adam Dziedzic et Franziska Boenisch),
reçus le 2026-11-21. Leaderboard commun : http://34.147.229.55/leaderboard_page
Setup JURECA commun aux 3 tâches : `git clone https://huggingface.co/datasets/SprintML/hackathon` (lire son README).

## Calendrier et format (slides d'ouverture)

- Sam. 14:30 début · Dim. **12:00 fin du hacking** · **13:00 : les 6 meilleures équipes présentent
  leur solution en 6 min** · 14:00 délibération · 14:30 remise des prix (1er : 10 000 €, Meta).
- Classement général sur les 3 tâches (Task 1 Watermark Localization, Task 2 Model Equality,
  Task 3 Unlearning). Jury : équipe technique du CISPA. Discord : https://discord.gg/urUG53tnbJ
- Il faut donc aussi **un deck de 6 min prêt dimanche 11:00** : méthode par tâche, validation
  locale, ce qui a marché et pourquoi.

## Setup JURECA officiel (slides 63-66)

- Projet `training2666`, 40 nœuds (4×A100 40 Go, 128 cœurs, 512 Go), réservation `cispahack`.
- **Un seul chef d'équipe** lance `hackathon_setup.sh` (télécharge les données, crée un `.venv`
  par tâche avec ses dépendances, caches uv/HF partagés) ; chaque coéquipier lance ensuite
  `teammate.sh` depuis **son propre compte**. Jamais les deux scripts sur le même compte.
- Dossier partagé : `/p/scratch/training2666/<lead>/<team>/{unlearning,model_equality,Hackathon_Watermark_Localisation_Task}/`,
  chacun avec `.venv`, `requirements.txt`, `output/`.
- Job : `jutil env activate -p training2666`, `cd` dans le dossier de la tâche,
  `source .venv/bin/activate`, `sbatch main.sh` (`--gres=gpu:1 --partition=dc-gpu --reservation=cispahack`).
- Notre env `Hackathon` et `common/jureca/main.sh` datent d'avant ce setup : les remplacer par
  les `.venv` officiels par tâche.

## Annonces Discord (2026-11-21)

- Équipe enregistrée via le formulaire des organisateurs. Les clés API de soumission sont
  envoyées à l'e-mail d'organisation (école) du **premier coéquipier listé**.
- **Interdit de collaborer entre équipes** (solutions, jetons…) : disqualification. Dépôt GitHub
  privé, pas de partage de code ni de clé hors de l'équipe.
- Accès au dataset watermark : validation manuelle sur Hugging Face, demande faite.

## Règle d'or commune

Le classement public ne porte que sur ~30 % des données ; le classement final sur les ~70 % restants.
Les trois énoncés le répètent : **ne pas optimiser sur le score public**. On décide avec la
validation locale ; le leaderboard sert à vérifier le format, à se situer face aux autres équipes,
et à détecter un écart grossier entre local et public. Chaque soumission est notée dans
`docs/experiences.md` avec son score local **et** son score public.

| | c1 — Localisation de watermark | c3 — Unlearning | c2 — Model equality |
|---|---|---|---|
| Entrée | 1 020 images test 512×512 (+ 510 train avec masques) | ViT-B/16 timm, 100 classes + 3 500 images à oublier | 33 endpoints LLM (+ 9 endpoints d'entraînement) |
| Sortie | `.pt` `{image_names, masks[N,128,128]}` ≤ 20 Mo | state dict float32 `strict=True` ≤ 450 Mo | CSV `endpoint,family,model` |
| Métrique | mIoU 2 classes, moyenne par image | forget_score × retain_score, porte 70 % de val. | 0,1·MCC famille + 0,9·MCC modèle |
| Validation locale | **forte** : 510 images train avec masques | **faible** : pas de gold ni de données retain | **moyenne** : ref-01..09 + `score_practice.py` |
| Ressource rare | GPU modéré | GPU (fine-tuning ViT) | **budget de requêtes : 100 par endpoint, pour tout le week-end** |
| Données | HF `SprintML/Hackathon_Watermark_Localisation_Task_Paris` | HF `SprintML/unlearning` | HF `SprintML/model_equality` |

## c1 — Localiser un watermark d'image (Task 1 officielle)

- Méthodes : PixelSeal, TrustMark, StegaStamp (340 images test chacune). Watermark posé sur un
  rectangle (ratio ≤ 16:1 pour PixelSeal/StegaStamp, ≤ 2:1 pour TrustMark), puis édité :
  outpainting, retrait/altération d'objet, ajout d'objet, transplantation d'objet.
- **Le test donne la méthode et le message de chaque image** : on peut mesurer localement si une
  zone porte ce message.
- Kit : wrappers unifiés `decode(images) -> bits` + cartes de features intermédiaires,
  downsampling 128×128 exact, mIoU local.
- Pistes, de la plus simple à la plus forte :
  1. Baselines triviales (tout à 1, tout à 0, rectangle central) pour calibrer l'mIoU.
  2. Fenêtres glissantes / occlusion : bit accuracy du message connu par patch → carte de chaleur → seuil.
  3. Saliency : gradient de la perte de décodage du message connu par rapport aux pixels.
  4. Petit réseau de segmentation sur les cartes de features des extracteurs, entraîné sur les
     510 images (+ données synthétiques générées avec les embedders du kit).
  5. Post-traitement par méthode : a priori rectangle (outpainting), trous (retrait/ajout).
- Rappel du talk Fernandez : PixelSeal travaille en 256×256 ; WAM traite la lecture comme une segmentation.
- Validation : k-fold sur les 510 images, **stratifié par méthode et par type d'édition**.

## c3 — Unlearning d'un ViT-B/16 (Task 3 officielle)

- Score final = forget_score × retain_score ; disqualification sous 70 % d'accuracy sur une
  validation cachée.
- forget_score : distribution de la perte (moyenne, écart-type) **et** distribution prédite
  image par image (KL, softmax à T=2) proches des modèles « gold » réentraînés sans ces images,
  sur les images et sur des copies altérées cachées.
- retain_score : accuracy et KL par rapport au modèle d'origine sur un set de contrôle caché.
- Pièges : la montée de gradient naïve ne suffit pas ; il faut imiter un modèle qui n'a jamais
  vu ces images (perte et mauvaises classes « naturelles »), pas un modèle cassé.
- Pas de données retain, pas de labels : les pseudo-labels viennent du modèle d'origine.
- Pistes : SCRUB (enseignant = modèle d'origine), SalUn (saillance des poids), NPO, unlearning
  zero-shot (bruit maximisant l'erreur), distillation sur images altérées ou synthétiques pour
  garder la KL retain.
- Validation locale : **à construire, c'est le vrai problème**. Proxys : stabilité des prédictions
  du modèle d'origine sur des images altérées (retain), perte sur les images à oublier comparée à
  la perte sur des augmentations fortes (forget). Le score public pèse plus ici que pour c1/c2,
  mais seulement en tendance.
- Serveur : forward en float16 autocast → vérifier le score en fp16 avant de soumettre.
- JURECA : nécessite `timm` (absent de l'env actuel).

## c2 — Model equality testing (Task 2 officielle)

- 33 endpoints (528 paires). Même modèle = même checkpoint + même quantification + même LoRA ;
  watermark, system prompt et stratégie de décodage ne changent pas l'identité.
- **Budget : 100 réponses par endpoint pour tout le week-end**, 256 tokens max en sortie,
  1 024 en entrée, 4 requêtes simultanées, `temperature`/`seed` ignorés, messages système refusés.
  **Toute requête dépensée est perdue : écrire le plan de requêtes avant d'en lancer une seule,
  et tout journaliser (jsonl).**
- Soumission : une toutes les 30 minutes.
- Entraînement : ref-01..09 (un petit modèle, mêmes déguisements), réponses dans
  `practice_answers.csv`, scoreur `score_practice.py` illimité.
- Pistes : famille via questions d'identité, tokenisation, style (10 % du score) ; modèle via tests
  de distribution sur les mêmes prompts répétés (MMD de Gao et al., test de rang RUT), prompts
  qui révèlent la quantification ou un LoRA (calcul, récitation longue, continuation exacte).
  Imposer la transitivité (partition) : le format le force.
- Ne pas utiliser les soumissions pour « sonder » les paires : c'est du sur-apprentissage du public
  (et l'esprit de la limite de 30 min).

## Répartition (à confirmer en équipe)

| Épreuve | Profil | Personne |
|---|---|---|
| c1 Localisation (Task 1) | vision, à l'aise avec les modèles de watermark | à préciser |
| c3 Unlearning (Task 3) | PyTorch, entraînement GPU sur JURECA | à préciser |
| c2 Model equality (Task 2) | LLM, statistiques, rigueur sur le budget | à préciser |

## Ordre de marche

1. Setup JURECA via `SprintML/hackathon`, données des 3 tâches.
2. **Une soumission valide par épreuve** (baseline du template) : teste clé, format, serveur.
3. Validation locale par épreuve (`cN/eval.py`), vérifiée par `qa`.
4. Itérations ; la nuit, boucle autoresearch sur c1 et c3 si la validation locale tient.
5. Matin : choix des soumissions finales sur le score local, pas sur le public.

## Questions pour les mentors

- c1/c3 : les modèles pré-entraînés externes (WAM, DINO, VAE…) sont-ils autorisés ?
- c1/c3 : le délai entre deux soumissions (30 min comme c2 ?).
- c2 : les endpoints de référence ref-01..09 ont-ils aussi un budget de 100 réponses chacun ?
- Une soumission en erreur de format consomme-t-elle le délai (oui pour c2 d'après l'énoncé) ?
