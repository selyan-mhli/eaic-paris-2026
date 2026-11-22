#!/bin/bash
# T2 boîte blanche sur les 165 réponses ep existantes. AUCUNE requête API. Lancement : sbatch wb_ep.sbatch
# Fichiers persistants (<= 5) dans /p/project1/training2666/mouhali1/wb_ep/ : wb_ep_ident.py, wb_ep.sbatch,
# <jobid>.out, result_<jobid>.json. Les fichiers intermédiaires restent dans /tmp du nœud et sont effacés.
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH -J j7
#SBATCH --output=/p/project1/training2666/mouhali1/wb_ep/%j.out
set -euo pipefail
W=/p/project1/training2666/mouhali1/wb_ep
H=/p/project1/training2666/mouhali1/hfmodels
ANS=/p/scratch/training2666/mouhali1/c2run/ep/ep_answers.jsonl
FAM=/p/scratch/training2666/mouhali1/SLZ/model_equality/output/c2_fam3_pairs_v2.csv
MODELS=$H/Qwen__Qwen2.5-0.5B-Instruct,$H/Qwen__Qwen2.5-1.5B-Instruct,$H/Qwen__Qwen2.5-3B-Instruct,$H/Qwen__Qwen2.5-7B-Instruct,$H/Qwen__Qwen3-0.6B,$H/Qwen__Qwen3-1.7B,$H/Qwen__Qwen3-4B,$H/unsloth__Llama-3.2-1B-Instruct,$H/unsloth__Llama-3.2-3B-Instruct,$H/unsloth__Meta-Llama-3.1-8B-Instruct
source /p/scratch/training2666/mouhali1/SLZ/unlearning/.venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
T=/tmp/wbep_${SLURM_JOB_ID}; L=/tmp/pylib_wbep_${SLURM_JOB_ID}
trap 'rm -rf $T $L' EXIT
mkdir -p $T $L; tar xzf /p/scratch/training2666/mouhali1/SLZ/wheels/pylib_sd.tgz -C $L
export PYTHONPATH=$L
echo "host $(hostname) job $SLURM_JOB_ID début $(date -u +%H:%M:%S)"
srun --ntasks=4 --kill-on-bad-exit=1 --cpu-bind=cores -l bash -c \
  "CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID python -u $W/wb_ep_ident.py phase1 --rank \$SLURM_LOCALID --world 4 --answers $ANS --models $MODELS --tmp $T"
python -u $W/wb_ep_ident.py merge1 --tmp $T --fam $FAM
srun --ntasks=4 --kill-on-bad-exit=1 --cpu-bind=cores -l bash -c \
  "CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID python -u $W/wb_ep_ident.py phase2 --rank \$SLURM_LOCALID --world 4 --answers $ANS --tmp $T"
python -u $W/wb_ep_ident.py final --tmp $T --dest $W/result_${SLURM_JOB_ID}.json --fam $FAM
echo "fin $(date -u +%H:%M:%S)"
