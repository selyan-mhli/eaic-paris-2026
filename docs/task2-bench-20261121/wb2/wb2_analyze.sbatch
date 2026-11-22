#!/bin/bash
# Analyse post-collecte (AUCUNE requête). Prérequis : rule.json (simulation) et collect_ep.jsonl (collecte terminée).
# Depuis /p/project1/training2666/mouhali1/wb2 : sbatch wb2_analyze.sbatch ; sorties : analyze_<jobid>.out,
# cand_wb2_<jobid>.csv (+ _features.json avec sha256)
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:20:00
#SBATCH -J j9
#SBATCH --output=/p/project1/training2666/mouhali1/wb2/analyze_%j.out
set -euo pipefail
W=/p/project1/training2666/mouhali1/wb2
ID=/p/project1/training2666/mouhali1/wb_ep/result_15904307.json
FAM=/p/scratch/training2666/mouhali1/SLZ/model_equality/output/c2_fam3_pairs_v2.csv
source /p/scratch/training2666/mouhali1/SLZ/unlearning/.venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
T=/tmp/wb2an_${SLURM_JOB_ID}; L=/tmp/pylib_wb2a_${SLURM_JOB_ID}; trap 'rm -rf $T $L' EXIT
mkdir -p $T $L; tar xzf /p/scratch/training2666/mouhali1/SLZ/wheels/pylib_sd.tgz -C $L; export PYTHONPATH=$L
srun --ntasks=3 --kill-on-bad-exit=1 --cpu-bind=cores -l bash -c \
  "CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID python -u $W/wb2_analyze.py score --rank \$SLURM_LOCALID --world 3 --ident $ID --collect $W/collect_ep.jsonl --tmp $T"
python -u $W/wb2_analyze.py decide --ident $ID --rule $W/rule.json --fam $FAM --tmp $T --csv $W/cand_wb2_${SLURM_JOB_ID}.csv
