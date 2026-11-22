#!/bin/bash
# Depuis unlearning/work_c3_eval : sbatch gen_var.sbatch   (4 GPU, une tranche des forget par GPU)
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:40:00
#SBATCH -J slz_t3_sep
#SBATCH --output=res/sep_%j.out
set -euo pipefail
cd /p/scratch/training2666/mouhali1/SLZ/unlearning/work_c3_eval
source ../.venv/bin/activate
export HF_HOME=/p/scratch/training2666/mouhali1/SLZ/.cache HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
srun --ntasks=4 --kill-on-bad-exit=0 --cpu-bind=cores --output="res/sep_%j_%t.out" bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  ARGS=("--push 1.75 --lam 1 --tag sep_p175_l1" "--push 2.5 --lam 1 --tag sep_p250_l1" "--push 2.5 --lam 3 --tag sep_p250_l3" "--push 3.5 --lam 3 --tag sep_p350_l3")
  python -u sep_ft.py ${ARGS[$SLURM_LOCALID]}'
