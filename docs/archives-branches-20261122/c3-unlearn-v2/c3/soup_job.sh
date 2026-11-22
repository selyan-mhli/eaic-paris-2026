#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --time=00:30:00
#SBATCH --job-name=j3s
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err

cd /p/scratch/training2666/mouhali1/SLZ/unlearning || exit 1
unset PYTHONPATH
source .venv/bin/activate
srun python soup_search.py
