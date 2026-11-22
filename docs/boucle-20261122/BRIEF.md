# Boucle autonome 04:15 → 06:15 (22/11)

Mandat Selyan (04:14) : « Soumettre si règle OK ». Selyan absent, ne touche plus au PC.

## Fin
Arrêter la boucle à **06:15** au plus tard (`ScheduleWakeup stop`), avec un résumé final dans `JOURNAL.md`.

## À chaque itération (~20 min)
1. Lire `docs/SLZ_COORDINATION_20261121.md` (fin) et `JOURNAL.md` ici : ne rien refaire, ne pas marcher sur les autres sessions
   (exploration T3 P1→P5 tournée par une autre session : ne pas soumettre de T3 en double, lire `docs/t3-explore-20261122/recus_t3.jsonl`).
2. Relever les jobs (file mouhali1/pelet1/zhang95) et le classement.
3. Faire avancer UNE piste à la fois, par ordre d'impact : T3 (si l'exploration P1→P5 est terminée, raffiner autour du meilleur reçu),
   puis T1 (StegaStamp R, ensembles de graines). T2 : aucun budget ep.
4. Écrire une ligne datée dans `JOURNAL.md` (fait, mesure, décision) et une ligne dans `docs/experiences.md` par expérience.

## Règles dures
- Soumission T1 seulement si gain apparié mesuré hors fuite ≥ +0,01 et borne basse IC95 > 0 contre le champion courant ; T3 : variantes figées avant envoi, une à la fois, reçu versé.
- ≤ 2 jobs sur dc-gpu-devel (limite Slurm), relevé « file » avant chaque sbatch ; rien sur les nœuds de login ; jamais `ssh` hors des chemins déjà utilisés.
- Aucun secret affiché ; aucune opération Git ; aucune requête ep T2.
- Deux échecs sur la même approche → changer de piste.
