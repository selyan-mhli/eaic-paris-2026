#!/bin/bash
# Enchaînement autorisé par Selyan (21/11 ~23:40) : fin de 15903709 -> push_eval -> export_ncc.
# Un seul job d'équipe à la fois ; contrôle « zéro job » juste avant chaque sbatch. Aucune soumission.
set -u
J=jureca; O="-o BatchMode=yes -o ConnectTimeout=20"
alive() { ssh $O $J "squeue -j $1 -h -o %T" 2>/dev/null | grep -qE "PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED"; }
team_jobs() { ssh $O $J 'squeue -u mouhali1,pelet1,zhang95 -h | wc -l'; }
waitjob() { while alive "$1"; do sleep 60; done; ssh $O $J "sacct -j $1 -X -n -o JobID,State,Elapsed"; }
submit() { # $1 dossier, $2 script
  n=$(team_jobs); [ "$n" = "0" ] || { echo "$(date +%T) ABANDON : $n job(s) d'équipe actifs avant $2"; exit 2; }
  ssh $O $J "cd $1 && sbatch --parsable $2"
}
echo "$(date +%T) attente 15903709"; waitjob 15903709
P=$(submit /p/scratch/training2666/mouhali1/SLZ/unlearning/work_c3_eval push_eval.sbatch) || exit 2
echo "$(date +%T) push_eval soumis : $P"; waitjob "$P"
N=$(submit /p/scratch/training2666/mouhali1/SLZ/Hackathon_Watermark_Localisation_Task work_segf/export_ncc.sbatch) || exit 2
echo "$(date +%T) export_ncc soumis : $N"; waitjob "$N"
echo "$(date +%T) fin ; PUSH=$P NCC=$N"
