# Source du deck SLZ — EAIC Paris 2026 (faits mesurés, 22/11 04:20)

Toutes les valeurs viennent de docs/experiences.md, docs/SLZ_COORDINATION_20261121.md,
docs/relais-codex-20261122/CLASSEMENT-LIVE.md et docs/reprise-claude-20261122/. Format : 6 minutes,
jury technique CISPA. Les scores publics portent sur 30 % des données ; le classement final sur 70 %.

## Contexte
- 3 épreuves : T1 localisation de watermark, T2 égalité de modèles (LLM), T3 désapprentissage d'un ViT-B/16.
- Équipe SLZ, un responsable par épreuve, orchestration par agents, calcul sur JURECA.
- Règle d'équipe : validation locale avant toute soumission, règle de décision fixée avant de lire le résultat
  (POURSUIVRE si gain apparié ≥ +0,01 et borne basse IC95 > 0 ; sinon DIAGNOSTIC ou ABANDON).

## Scores publics (dernier relevé)
| Tâche | SLZ | Rang au relevé 03:17 | Maximum du tableau |
|---|---|---|---|
| T1 | 0,6788 (soumis 03:55) | 20e/42 (à 0,6739) | 0,803 |
| T2 | 0,20 (baseline) | 25e/40 | 1,00 |
| T3 | 0,0941 | 31e/42 | 0,440 |
Général au relevé 03:17 : 0,319, 27e/38.

## T1 — Localisation de watermark (notre meilleure tâche)
- Données : 510 images train avec masques, 1 020 test 512×512 ; 3 méthodes (PixelSeal, TrustMark, StegaStamp),
  4 éditions (outpainting, retrait, ajout, transplantation d'objet). Le test donne méthode et message.
- Validation : CV imbriquée 5 folds stratifiée méthode × édition, gain apparié + bootstrap 2000.
  Le local a prédit le public à 0,001 près (0,5583 local → 0,5572 public).
- Progression publique : 0,5088 (CV) → 0,5572 (tête « trous » F) → 0,5845 (+ pixseg StegaStamp) →
  0,6137 (encodeur convnext_tiny pré-entraîné) → 0,6377 (ensemble 4×4000 pas) → 0,6739 (ensemble E) →
  0,6788 (StegaStamp 6000 pas, gain sur les 5 folds).
- Recette : canaux qui lisent le message connu (accord par position, boîtes décodées) + segmentation
  pixseg (convnext pré-entraîné) sur StegaStamp, routage par méthode.
- Abandonné par la règle : 8000 pas (−0,004), décision par mIoU espérée (−0,0001), crop 512, convnext_small,
  template matching, carte de ré-embarquement (pas de signal).

## T2 — Égalité de modèles (verrou)
- 33 endpoints, budget 100 requêtes par endpoint pour le week-end ; score 0,1·MCC famille + 0,9·MCC modèle.
- Acquis : les 3 familles identifiées en boîte blanche par vraisemblance (Qwen2.5-3B, Qwen2.5-1.5B,
  Llama-3.2-1B, 11 endpoints chacune), accord 528/528.
- Verrou : séparer les variantes d'une même famille (fp8, LoRA, température, prompt système).
  - Simulation GPU : statistiques de vraisemblance MCC 0,00–0,52 même en vérité connue → pas de collecte.
  - Sondes « quasi-égalité de logits » : critère GO rempli sur SmolLM2, puis validation sur les 9 refs :
    990 requêtes, MCC 0,000 → aucun lot dépensé sur les endpoints notés.
  - Canaux annexes fermés : latence rembourrée, pas de champ usage, id aléatoire.
- Leçon : on a protégé le budget, mais sans méthode qui sépare les variantes on reste à la baseline 0,20.

## T3 — Désapprentissage (seul signal serveur : la tête)
- Pas de données retain, pas de labels, pas de gold : validation locale faible.
- Images forget générées par diffusion (pas un dataset public) ; 3 500 images réparties sur 45 classes.
- Seul mécanisme récompensé : « push » sur la tête (45 lignes de classe modifiées, rien d'autre).
  Balayage : ×1,25 0,0567 ; ×1,5 0,0827 ; ×1,75 0,0941 ; ×2,0 0,0792 → optimum ≈ ×1,75.
- Réfuté par le serveur : soupes / distillation / ancrage 0,001–0,004 ; supprimer 9 classes 0,0205 ;
  préserver les variations img2img 0,0317 (le gold baisse toute la classe).

## Ce qu'on retient
1. Une validation locale fidèle (T1) vaut plus que des soumissions à l'aveugle (T3 avant 22:00).
2. Règles de décision figées avant lecture : elles ont évité de soumettre du bruit.
3. Erreur de répartition : T1 était le plus gros écart facile à combler (15 équipes à 0,71–0,77).
4. Processus : trop de sessions parallèles au début, corrigé par un orchestrateur unique et un registre.
