<!-- ppt-master-schema: spec-lock/v1 -->
# Execution Lock

## canvas
- viewBox: 0 0 1280 720
- format: PPT 16:9

## communication
- primary_language: fr-FR
- audience: Jury technique du CISPA, qui connaît les épreuves et les métriques mieux que nous
- objective: Rendre compte de la méthode par épreuve et de la discipline de validation, puis expliquer honnêtement les blocages T2/T3, pour que le jury juge la validation crédible et retienne un enseignement transférable en 6 minutes
- core_message: Là où la validation locale était fidèle (T1), chaque décision a été mesurée et le score est passé de 0,51 à 0,68 ; là où elle manquait (T2, T3), nous avons protégé le budget plutôt que soumettre à l'aveugle
- consumption_mode: presentation

## mode
- mode: custom
- mode_behavior: Conclusion d'abord sur chaque page (titre = affirmation mesurée), puis la preuve chiffrée, puis la limite ; une page par épreuve, ouverte par le score et fermée par ce qu'elle nous a appris ; la dernière page tire l'enseignement transférable.

## visual_style
- visual_style: custom
- visual_style_behavior: Fond papier clair, filets fins et colonnes de type revue scientifique, titres en serif, chiffres-clés en grand, lignes de source en petit sous chaque preuve ; une couleur d'accent par épreuve, le reste en gris encre ; aucun ornement hors filets.

## colors
- background: #FAF8F3
- secondary_bg: #EFEBE1
- primary: #1F2A44
- accent: #C2410C
- secondary_accent: #2F6F73
- body_text: #22252B
- secondary_text: #6B6A66
- divider: #D9D3C4

## typography
- font_family: Arial, Helvetica, sans-serif
- title_family: Georgia, "Times New Roman", serif
- body_family: Arial, Helvetica, sans-serif
- display_family: Georgia, "Times New Roman", serif
- body: 28
- title: 44
- subtitle: 32
- annotation: 18
- display: 96

## icons
- library: tabler-outline
- stroke_width: 2
- inventory: tabler-outline/flask, tabler-outline/ruler-measure, tabler-outline/lock, tabler-outline/target, tabler-outline/check, tabler-outline/x

## page_rhythm
- P01: anchor
- P02: dense
- P03: dense
- P04: dense
- P05: dense
- P06: dense
- P07: anchor

## page_visualizations
- P03: chart/line_chart
- P04: table/record_table
- P06: chart/column_chart

## pptx_structure
- mode: flat

## forbidden
- `mask`, `<style>`, `class`, external CSS, `<foreignObject>`, `textPath`, `@font-face`, `<animate*>`, `<set>`, `<script>` / event attributes, `<iframe>`
- HTML named entities in text; write typography as raw Unicode and escape XML reserved characters
