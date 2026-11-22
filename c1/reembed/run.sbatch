#!/bin/bash
# Usage (depuis le dossier de la tâche, code copié dans work_c1/) :
#   sbatch work_c1/reembed/run.sbatch work_c1/reembed/maps.py --split train --only-full --limit 40 --out output/re_pilot.npz
# 4 tâches = 4 A100 ; shard = SLURM_PROCID, GPU = SLURM_LOCALID (lus par maps.py).
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH -J j1r
#SBATCH --output=output/%j.out
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8
cd /p/scratch/training2666/mouhali1/SLZ/Hackathon_Watermark_Localisation_Task
source .venv/bin/activate
nvidia-smi --query-gpu=name --format=csv,noheader
srun python "$@"
