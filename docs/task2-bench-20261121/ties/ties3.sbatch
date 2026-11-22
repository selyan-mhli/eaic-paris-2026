#!/bin/bash
# T2 sondes v3 : famille A (Qwen2.5-3B) seule, 4 graines de pool v2 (11, 13, 17, 19) = 10 000 prompts, 1 graine par GPU.
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:25:00
#SBATCH -J j11
#SBATCH --output=/p/project1/training2666/mouhali1/ties/%j.out
set -euo pipefail
W=/p/project1/training2666/mouhali1/ties
H=/p/project1/training2666/mouhali1/hfmodels
source /p/scratch/training2666/mouhali1/SLZ/unlearning/.venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
L=/tmp/pylib_ties_${SLURM_JOB_ID}
trap 'rm -rf $L' EXIT
mkdir -p $L; tar xzf /p/scratch/training2666/mouhali1/SLZ/wheels/pylib_sd.tgz -C $L
export PYTHONPATH=$L
echo "host $(hostname) job $SLURM_JOB_ID début $(date -u +%H:%M:%S)"
srun --ntasks=4 --kill-on-bad-exit=0 --cpu-bind=cores -l bash -c '
S=(11 13 17 19)
CUDA_VISIBLE_DEVICES=$SLURM_LOCALID python -u '"$W"'/ties_search.py --model '"$H"'/Qwen__Qwen2.5-3B-Instruct --pool v2 --seed ${S[$SLURM_LOCALID]} --tag A3s${S[$SLURM_LOCALID]} --out '"$W"
echo "fin $(date -u +%H:%M:%S)"
