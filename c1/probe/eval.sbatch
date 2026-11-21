#!/bin/bash
# A2 : absl_eval.py, un jeu de cartes par tâche (CPU seulement), 4 tâches sur un nœud.
#   sbatch work_c1_probe/eval.sbatch "base" "base+absl" "base+comb" "base+absl+comb"
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH -J j1
#SBATCH --output=output/%j.out
set -euo pipefail
cd /p/scratch/training2666/mouhali1/SLZ/Hackathon_Watermark_Localisation_Task
source .venv/bin/activate
export OMP_NUM_THREADS=8 SETS="$*"
srun --ntasks=4 --kill-on-bad-exit=0 --output="output/%j_%t.out" bash -c '
  read -ra S <<< "$SETS"; set_name=$(echo "${S[$SLURM_LOCALID]}" | tr "_" " ")
  exec python -u work_c1_probe/absl_eval.py --heats output/heats_train.npz --absl output/probe_absl_train.npz \
       --boxes "output/boxes_train_*.npz" --write-merged "output/probe_merged_$SLURM_LOCALID.npz" --sets "$set_name"'
