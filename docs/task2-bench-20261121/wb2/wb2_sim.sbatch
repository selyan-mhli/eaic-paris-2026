#!/bin/bash
# Calibration par simulation, AUCUNE requête API. Depuis /p/project1/training2666/mouhali1/wb2 : sbatch wb2_sim.sbatch
# 3 bases (A Qwen2.5-3B, B Qwen2.5-1.5B, L Llama-3.2-1B) sur 3 GPU ; sorties : sim_<jobid>.out, rule.json, sim_report.json
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH -J j8
#SBATCH --output=/p/project1/training2666/mouhali1/wb2/sim_%j.out
set -euo pipefail
W=/p/project1/training2666/mouhali1/wb2; H=/p/project1/training2666/mouhali1/hfmodels
source /p/scratch/training2666/mouhali1/SLZ/unlearning/.venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
T=/tmp/wb2sim_${SLURM_JOB_ID}; L=/tmp/pylib_wb2_${SLURM_JOB_ID}; trap 'rm -rf $T $L' EXIT
mkdir -p $T $L; tar xzf /p/scratch/training2666/mouhali1/SLZ/wheels/pylib_sd.tgz -C $L; export PYTHONPATH=$L
echo "host $(hostname) job $SLURM_JOB_ID début $(date -u +%T)"
srun --ntasks=3 --kill-on-bad-exit=1 --cpu-bind=cores -l bash -c '
  B=( Qwen__Qwen2.5-3B-Instruct Qwen__Qwen2.5-1.5B-Instruct unsloth__Llama-3.2-1B-Instruct )
  CUDA_VISIBLE_DEVICES=$SLURM_LOCALID python -u '$W'/wb2_sim.py sim --base '$H'/${B[$SLURM_LOCALID]} --tmp '$T' --n 30'
python -u $W/wb2_sim.py fit --tmp $T --rule $W/rule.json --report $W/sim_report.json
echo "fin $(date -u +%T)"
