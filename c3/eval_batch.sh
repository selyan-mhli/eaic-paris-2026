#!/bin/bash
# Évaluation par lots des candidats c3, une tâche par GPU (4 en parallèle sur un nœud).
# Lecture seule sur unlearning/ ; tout est écrit dans work_c3_eval/ (res/, logs/).
# LIST = fichier texte de 4 lignes au plus, une par GPU, candidats séparés par des espaces.
# Chaque GPU passe sa ligne dans eval.py (×2) puis drift.py (cf. eval_task.sh).
# Usage (depuis work_c3_eval/) :
#   LIST=cands.txt sbatch eval_batch.sh
#   LIST=cands.txt MAXIMG=200 sbatch --time=00:10:00 eval_batch.sh   # essai court
# Mesuré (job 15901897, 10 fichiers, 2-3 par GPU) : 5 min 23 s.
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --job-name=j3
#SBATCH --output=logs/%j.out

cd /p/scratch/training2666/mouhali1/SLZ/unlearning/work_c3_eval || exit 1
unset PYTHONPATH
source ../.venv/bin/activate
export HF_HUB_OFFLINE=1 SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=8
export LIST=${LIST:?LIST=fichier de candidats requis} MAXIMG=${MAXIMG:-}
srun --kill-on-bad-exit=0 --cpu-bind=cores --output=logs/%j_%t.out bash eval_task.sh
