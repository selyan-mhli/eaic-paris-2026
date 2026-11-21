#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1            # passer a gpu:4 si le code utilise les 4 A100 du noeud
#SBATCH --cpus-per-task=128     # 128 coeurs physiques par noeud
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --time=01:00:00         # limite : un job plante ne brule pas le budget
#SBATCH --output=output/%j.out

# srun ne transmet pas toujours --cpus-per-task a l'etape : on le force
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# chemin en dur : $PROJECT n'est defini que si jutil a ete active avant sbatch
cd /p/project1/training2666/mouhali1/Hackathon
source .venv/bin/activate
srun python sample_submission.py
