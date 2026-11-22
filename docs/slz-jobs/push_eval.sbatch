#!/bin/bash
# Évaluation seule (aucun entraînement, aucun checkpoint écrit) des 5 checkpoints push,
# même protocole que margin_job : score_like.py --original ../model.pt --off offforget.npz.
# GO Selyan 21/11 ~23:40, après la fin de 15903709. Propriétaire : orchestrateur SLZ.
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --job-name=slz_t3_pusheval
#SBATCH --output=logs/%j.out
set -euo pipefail
cd /p/scratch/training2666/mouhali1/SLZ/unlearning/work_c3_eval
unset PYTHONPATH
source ../.venv/bin/activate
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1
sha256sum push_scale/push_s075.pt ../output/push_best.pt push_scale/push_s125.pt push_scale/push_s150.pt push_scale/push_s200.pt
python -u score_like.py --original ../model.pt --off offforget.npz \
  push_scale/push_s075.pt ../output/push_best.pt push_scale/push_s125.pt push_scale/push_s150.pt push_scale/push_s200.pt \
  > "res/score_${SLURM_JOB_ID}_push.txt" 2>&1
cat "res/score_${SLURM_JOB_ID}_push.txt"
