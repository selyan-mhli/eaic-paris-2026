#!/bin/bash
# Usage (depuis le dossier de la tâche) : sbatch work_c1/j1_heats.sbatch <split> <out.npz> [limit]
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00
#SBATCH -J j1
#SBATCH --output=output/%j.out
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1
cd /p/scratch/training2666/mouhali1/SLZ/Hackathon_Watermark_Localisation_Task
source .venv/bin/activate
nvidia-smi --query-gpu=name --format=csv,noheader
srun python work_c1/real_heats.py --split "$1" --out "$2" --limit "${3:-0}" ${4:-}
