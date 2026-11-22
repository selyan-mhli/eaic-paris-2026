#!/bin/bash
# Depuis unlearning/work_c3_eval : sbatch gen_var.sbatch   (4 GPU, une tranche des forget par GPU)
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH -J slz_t3_gen
#SBATCH --output=res/gen_%j.out
set -euo pipefail
cd /p/scratch/training2666/mouhali1/SLZ/unlearning/work_c3_eval
source ../.venv/bin/activate
export HF_HOME=/p/scratch/training2666/mouhali1/SLZ/.cache HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
srun --ntasks=4 --kill-on-bad-exit=0 --cpu-bind=cores --output="res/gen_%j_%t.out" bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  L=/tmp/pylib_sd_${SLURM_JOB_ID}_${SLURM_LOCALID}; mkdir -p $L; tar xzf ../../wheels/pylib_sd.tgz -C $L
  export PYTHONPATH=$L
  python -u gen_var.py; rc=$?; rm -rf $L; exit $rc'
