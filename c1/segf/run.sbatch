#!/bin/bash
# 4 tâches (1 GPU chacune) : srun python <script> [args] ; puis, si POST est défini, une commande CPU sur le nœud.
#   (réservation : sbatch -p dc-gpu --reservation=cispahack ...)
#   sbatch [--export=ALL,POST="python work_segf/chan_auc.py ..."] work_segf/run.sbatch work_segf/<script>.py [args]
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=00:40:00
#SBATCH -J j1
#SBATCH --output=output/%j.out
set -euo pipefail
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
cd /p/scratch/training2666/mouhali1/SLZ/Hackathon_Watermark_Localisation_Task
source .venv/bin/activate
srun --ntasks=4 --kill-on-bad-exit=0 --cpu-bind=cores --output="output/%j_%t.out" bash -c 'export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; exec python -u "$@"' _ "$@"
if [ -n "${POST:-}" ]; then echo "== POST"; eval "$POST"; fi
