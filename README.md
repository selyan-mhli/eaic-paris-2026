# EAIC Paris 2026, dépôt d'équipe

Selyan, Yihuan, Sophie, Mohammed. Un dossier par challenge : `c1/`, `c2/`, `c3/`. Code partagé dans `common/`, notes et journal dans `docs/`.

## Règles d'équipe

- Une soumission valide sur chaque challenge le plus tôt possible, puis on améliore.
- Toute expérience a une ligne dans `docs/experiences.md` : hypothèse, changement, coût, score local, score serveur.
- Validation locale avant soumission ; le serveur masque les scores inférieurs à ton meilleur.
- Jamais de clé API dans le dépôt : variable `EAIC_API_KEY` seulement.

## Cluster JURECA (retours des éditions précédentes, à confirmer sur place)

- Nœuds de calcul sans Internet : `HF_HUB_OFFLINE=1`, poids et datasets copiés à l'avance dans le cache.
- Quota d'inodes partagé : pas d'environnements conda en double, pas de milliers de petits fichiers.
- Ne pas faire tourner l'image sur les GPU des nœuds de login.
- Noms de jobs Slurm opaques (`-J j1`, `-J j2`) : les autres équipes les voient.
- Jobs longs lancés avant de dormir.
