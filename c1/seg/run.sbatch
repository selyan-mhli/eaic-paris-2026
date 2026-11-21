#!/bin/bash
# Usage (depuis le dossier de la tâche) : sbatch [--ntasks-per-node=4] work_seg/run.sbatch work_seg/<script>.py [args]
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH -J j1s
#SBATCH --output=output/%j.out
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8
cd /p/scratch/training2666/mouhali1/SLZ/Hackathon_Watermark_Localisation_Task
source .venv/bin/activate
srun python "$@"
