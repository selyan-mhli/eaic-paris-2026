<!-- ppt-master-schema: design-spec/v1 -->
# SLZ EAIC Paris 2026 - Design Spec

## I. Project Information

| Item | Value |
| --- | --- |
| Project Name | slz_deck — EAIC Paris 2026, équipe SLZ |
| Canvas Format | ppt169, 1280 × 720 |
| Page Count | 7 |
| Primary Language | fr-FR |
| Target Audience | Jury technique du CISPA (chercheurs en sécurité de l'IA, organisateurs des trois épreuves) ; ils connaissent les tâches et les métriques mieux que nous. |
| Communication Intent | Rendre compte de notre méthode par épreuve et de notre discipline de validation, d'abord ; expliquer ensuite honnêtement ce qui a bloqué en T2 et T3. |
| Desired Audience Outcome | Le jury comprend en 6 minutes ce que chaque épreuve nous a appris, juge notre validation locale crédible et retient un enseignement transférable. |
| Core Message / Ask / Action | Là où nous avions une validation locale fidèle (T1), chaque décision a été mesurée et le score a progressé de 0,51 à 0,68 ; là où elle manquait (T2, T3), nous avons protégé le budget plutôt que soumettre à l'aveugle. |
| Delivery Context | Présentation orale de 6 minutes avec un présentateur, projetée devant le jury ; lecture éventuelle après coup. |
| Artifact Afterlife | Trace de fin de hackathon pour l'équipe ; rien d'autre. |
| Reading Mode | presentation |
| Content Strategy | balanced (default, field left blank) |
| Design Style | Carnet de labo — custom mode + custom visual style |
| AI Image Acquisition Path | not applicable (image source: none) |
| Generation Mode | continuous |
| Spec Refinement | disabled |
| Speaker Notes | enabled — final Stage-2 proactive policy |
| Custom Animations | disabled — final Stage-2 proactive policy |
| Narration Audio | disabled — final Stage-2 proactive policy |
| Created Date | 2026-11-22 |

## II. Canvas Specification

| Property | Value |
| --- | --- |
| Format | PPT 16:9 |
| Dimensions | 1280 × 720 |
| viewBox | `0 0 1280 720` |
| Margins | 64 px left/right, 56 px top/bottom |
| Content Area | x 64–1216, y 56–664 |

## III. Visual Theme

### Theme Style

- **Mode**: custom
- **Mode Behavior**: Conclusion d'abord sur chaque page (titre = affirmation mesurée), puis la preuve chiffrée, puis la limite ; une page par épreuve, ouverte par le score et fermée par ce qu'elle nous a appris ; la dernière page tire l'enseignement transférable.
- **Visual style**: custom
- **Visual Style Behavior**: Fond papier clair, filets fins et colonnes de type revue scientifique, titres en serif, chiffres-clés en grand, lignes de source en petit sous chaque preuve ; une couleur d'accent par épreuve, le reste en gris encre ; aucun ornement hors filets.
- **Theme**: Cahier d'expériences : chaque page est une entrée datée avec sa mesure et sa source.
- **Tone**: Rigoureux, calme, honnête sur les échecs.

### Color Scheme

| Role | HEX | Purpose |
| --- | --- | --- |
| Background | #FAF8F3 | Page papier |
| Secondary background | #EFEBE1 | Encadrés de preuve, bandeau de règle |
| Primary | #1F2A44 | Titres, chiffres-clés neutres |
| Accent | #C2410C | T1 et mise en avant principale |
| Secondary accent | #2F6F73 | T2 / T3 et seconde série |
| Body text | #22252B | Texte courant |
| Secondary text | #6B6A66 | Lignes de source, légendes, notes |
| Divider | #D9D3C4 | Filets, axes, bordures |

## IV. Typography System

### Font Plan

| Role | Character (Reference) | Primary | English if non-English | Fallback tail |
| --- | --- | --- | --- | --- |
| Title | serif de revue | Georgia | Georgia | Times New Roman, serif |
| Body | sans neutre | Arial | Arial | Helvetica, sans-serif |
| Display | serif pour chiffres-clés | Georgia | Georgia | Times New Roman, serif |

- **Title stack**: Georgia, "Times New Roman", serif
- **Body stack**: Arial, Helvetica, sans-serif
- **Display stack**: Georgia, "Times New Roman", serif
- **Role rationale**: Display porte les chiffres-clés en très grand corps (0,6788, 528/528…), récurrents sur la plupart des pages.

### Font Size Hierarchy

| Purpose | Anchor Size (px) |
| --- | ---: |
| Body | 28 |
| Title | 44 |
| Subtitle | 32 |
| Annotation | 18 |
| Display | 96 |

## V. Layout Principles

### Deck-wide Direction

- **Hierarchy direction**: titre-affirmation en haut à gauche → chiffre-clé ou graphique → ligne de source en bas.
- **Composition tendency**: colonnes asymétriques de revue (preuve large, commentaire étroit), filets horizontaux fins pour séparer les zones.
- **Cross-page continuity**: un en-tête d'entrée de carnet (numéro d'épreuve + couleur d'accent) et une ligne de source en pied de page sur chaque page de contenu.
- **Spacing posture**: aéré, adapté à la projection.
- **Spacing anchors**: marge 64 px · espacement de bloc 32 px · gouttière 40 px · rayon 4 px · interligne 1,3

## VI. Icon Usage Specification

- **Primary bundled library**: tabler-outline
- **Stroke Width**: 2

| Icon Path | Suitable Scenarios |
| --- | --- |
| icons/tabler-outline/flask.svg | |
| icons/tabler-outline/ruler-measure.svg | |
| icons/tabler-outline/lock.svg | |
| icons/tabler-outline/target.svg | |
| icons/tabler-outline/check.svg | |
| icons/tabler-outline/x.svg | |

## VII. Visualization Reference List

| Page | Family | Template | Usage |
| --- | --- | --- | --- |
| P03 | chart | line_chart | Progression du score T1, étape par étape |
| P04 | table | record_table | Pistes T1 tranchées par la règle, avec leur gain mesuré |
| P06 | chart | column_chart | Score public T3 selon l'intensité du push |

## VIII. Image Resource List

| Filename | Dimensions | Ratio | Purpose | Type | Image pattern | Crop Policy | Acquire Via | Status | Reference | text_policy | page_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## IX. Content Outline

### Part 1: Ouverture

#### Slide 01 - Couverture

- **Audience move**: le jury ne connaît pas SLZ → il sait que notre fil rouge est la mesure avant la soumission.
- **Relationships**: none
- **Composition**: le chiffre 0,51 → 0,68 en très grand comme accroche, titre en serif dessous, équipe et événement en petit.
- **Title**: Mesurer avant de soumettre
- **Core message**: Avec une validation locale fidèle, chaque décision T1 a été mesurée : 0,51 → 0,68.
- **Content**: Accroche 0,51 → 0,68 (Task 1) · sous-titre « EAIC Paris 2026 — équipe SLZ » · « Watermark · Model equality · Unlearning »
- **Cover impact**: le saut 0,51 → 0,68 obtenu sans soumission à l'aveugle.

#### Slide 02 - Notre règle

- **Audience move**: le jury voit un score → il comprend la règle qui a décidé chaque soumission.
- **Relationships**: order — hypothèse → mesure locale appariée → décision (POURSUIVRE / DIAGNOSTIC / ABANDON) ; contrast — qualité de la validation locale par épreuve (forte T1, moyenne T2, faible T3).
- **Composition**: à gauche la boucle de décision en trois temps ; à droite les trois épreuves avec la qualité de leur validation locale.
- **Title**: Une règle fixée avant de lire le résultat
- **Core message**: On soumet seulement si le gain apparié est ≥ +0,01 et si la borne basse de l'IC95 est > 0.
- **Content**: Boucle : hypothèse · mesure appariée (bootstrap 2000) · décision · Règle POURSUIVRE : gain ≥ +0,01 et IC95 bas > 0 · Validation locale : T1 forte (510 images avec masques, CV 5 folds) / T2 moyenne (9 refs, budget 100 requêtes/endpoint) / T3 faible (ni gold ni retain)
- **Fact IDs**: source-deck.md « Contexte », « T1 », HACKATHON.md tableau des épreuves

### Part 2: Les épreuves

#### Slide 03 - T1 progression

- **Audience move**: le jury connaît la règle → il voit qu'elle a produit une progression régulière et prévisible.
- **Relationships**: order — étapes successives de la recette T1.
- **Composition**: courbe large à gauche ; colonne étroite à droite avec le chiffre 0,001 (écart local/public).
- **Title**: T1 : +0,17 en six étapes mesurées
- **Core message**: Le local a prédit le public à 0,001 près (0,5583 → 0,5572), donc chaque étape a été choisie hors ligne.
- **Content**: Courbe : 0,5088 CV imbriquée · 0,5572 tête F · 0,5845 + pixseg StegaStamp · 0,6137 encodeur convnext pré-entraîné · 0,6377 ensemble 4×4000 pas · 0,6739 ensemble E · 0,6788 StegaStamp 6000 pas · relevé 04:06 : 0,6880 (dernier envoi, hors registre) · Encadré : écart local/public 0,001 · Source : docs/experiences.md, registre de coordination
- **Visualization**: t1-progression — line chart, axe = étape, valeur = score (premier point = CV locale, les autres = public)
- **Native-ready**: t1-progression=yes

#### Slide 04 - T1 ce que la règle a rejeté

- **Audience move**: le jury voit la progression → il voit aussi les pistes coupées, preuve que la règle a filtré le bruit.
- **Relationships**: contrast — pistes retenues contre pistes abandonnées.
- **Composition**: tableau sobre à filets, une ligne par piste, verdict coloré.
- **Title**: T1 : la règle a coupé ce qui ne mesurait pas
- **Core message**: Plusieurs idées séduisantes ont été abandonnées sur mesure, sans coûter de soumission.
- **Content**: Lignes : 8000 pas contre 4000 → −0,004 → ABANDON · décision par mIoU espérée → −0,0001 → ABANDON · BCE 0,5 + IoU 2, 5 folds → +0,0034 → DIAGNOSTIC · StegaStamp 6000 pas → positif sur 5 folds → soumis (+0,0049 public) · Template matching, carte de ré-embarquement → pas de signal → écarté · Leçon : 15 équipes à 0,71–0,77 avec une recette forensique générique ; la nôtre lisait surtout le signal du watermark
- **Visualization**: t1-pistes — record table, colonnes piste / gain mesuré / verdict
- **Native-ready**: t1-pistes=yes

#### Slide 05 - T2

- **Audience move**: le jury attend un score T2 → il comprend ce qui est acquis (familles) et où est le verrou (variantes).
- **Relationships**: parent — 33 endpoints → 3 familles → variantes intra-famille ; contrast — acquis contre verrou.
- **Composition**: à gauche l'acquis (3 familles identifiées, 528/528) ; à droite le verrou et les tests qui l'ont confirmé.
- **Title**: T2 : familles identifiées, variantes non séparées
- **Core message**: On a refusé de dépenser le budget sans méthode validée sur les refs : on reste à 0,20.
- **Content**: Acquis : vraisemblance en boîte blanche → Qwen2.5-3B, Qwen2.5-1.5B, Llama-3.2-1B, 11 endpoints chacun, accord 528/528 · Verrou : fp8, LoRA, température, prompt système · Simulation GPU : MCC 0,00–0,52 même en vérité connue · Sondes à quasi-égalité de logits : 990 requêtes sur les refs, MCC 0,000 · Canaux annexes fermés : latence rembourrée, id aléatoire · Décision : 0 lot dépensé sur les endpoints notés · score 0,20
- **Fact IDs**: source-deck.md « T2 »

#### Slide 06 - T3

- **Audience move**: le jury sait que la T3 est sans gold → il voit le seul levier que le serveur a récompensé et ce qui a été réfuté.
- **Relationships**: order — intensité du push croissante ; contrast — push contre méthodes réfutées.
- **Composition**: histogramme des scores du push à gauche ; à droite la liste courte des réfutations.
- **Title**: T3 : seule la tête a fait bouger le serveur
- **Core message**: Sans gold ni retain, notre meilleur levier est une modification de 45 lignes de la tête, optimum à ×1,75.
- **Content**: Histogramme push : ×1,25 0,0567 · ×1,5 0,0827 · ×1,65 0,0923 · ×1,75 0,0941 · ×2,0 0,0792 · Réfuté : soupes / distillation / ancrage 0,001–0,004 · supprimer 9 classes 0,0205 · préserver les variations img2img 0,0317 (le gold baisse toute la classe) · Contexte : images forget générées par diffusion, 3 500 images sur 45 classes · Maximum du tableau 0,44
- **Visualization**: t3-push — column chart, catégorie = intensité, valeur = score public
- **Native-ready**: t3-push=yes

### Part 3: Conclusion

#### Slide 07 - Ce qu'on retient

- **Audience move**: le jury a vu les trois épreuves → il repart avec un enseignement transférable.
- **Relationships**: membership — trois enseignements.
- **Composition**: trois entrées numérotées sur filets, la première mise en avant.
- **Title**: Ce qu'on retient
- **Core message**: La validation locale fidèle est l'actif qui compte ; sans elle, mieux vaut protéger le budget que soumettre à l'aveugle.
- **Content**: 1. Une validation locale fidèle vaut plus que des soumissions à l'aveugle (T1 contre T3 avant 22:00) · 2. Figer la règle avant de lire le résultat a évité de soumettre du bruit · 3. Notre erreur : la T1 était l'écart le plus facile à combler, avec une recette forensique générique · Scores : T1 0,6880 · T2 0,20 · T3 0,0941
- **Closing impact**: l'enseignement n°1, en grand.

## X. Speaker Notes Requirements

- **Generation**: enabled
- **Filename**: match each SVG filename under `notes/`
- **Content**: une explication orale par page, uniquement à partir des faits de sources/source-deck.md ; aucun chiffre ajouté.
- **Total duration**: 6 minutes (environ 50 s par page)
- **Notes style**: conversationnel et précis
- **Presentation purpose**: Rendre compte de notre méthode par épreuve et de notre discipline de validation, d'abord ; expliquer ensuite honnêtement ce qui a bloqué en T2 et T3.
