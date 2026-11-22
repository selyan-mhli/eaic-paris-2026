#!/bin/bash
# T2 sondes v2 (prompts dominés par le contexte in-situ) : recherche HORS LIGNE (0 requête API), 1 modèle par GPU.
# Copie préalable : ties_search.py et ties.sbatch dans /p/project1/training2666/mouhali1/ties/
# Sorties (5 fichiers) : <jobid>.out, ties_{smol,A,B,L}.json dans /p/project1/training2666/mouhali1/ties/
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:20:00
#SBATCH -J j10
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
M=(HuggingFaceTB__SmolLM2-360M-Instruct Qwen__Qwen2.5-3B-Instruct Qwen__Qwen2.5-1.5B-Instruct unsloth__Llama-3.2-1B-Instruct)
G=(smol2 A2 B2 L2)
CUDA_VISIBLE_DEVICES=$SLURM_LOCALID python -u '"$W"'/ties_search.py --model '"$H"'/${M[$SLURM_LOCALID]} --pool v2 --tag ${G[$SLURM_LOCALID]} --out '"$W"
python -u $W/ties_search.py --model $H/HuggingFaceTB__SmolLM2-360M-Instruct --tag smolv1all --out $W
echo "fin $(date -u +%H:%M:%S)"
ls -la $W
