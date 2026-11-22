#!/bin/bash
# 8 processus template_match.py par nœud, 2 par GPU (recouvrement CPU/GPU), 32 cœurs chacun.
# Usage : sbatch work_c1_tmatch/j1_pack8_tmatch.sbatch <split> <préfixe> <nshards total> <premier shard> [args en plus]
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH -J j1
#SBATCH --output=output/%j.out
set -euo pipefail
SPLIT=${1:?split}; PREFIX=${2:?prefixe}; NSH=${3:?nshards}; FIRST=${4:?premier}; EXTRA=${5:-}
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 TM_RES_BS=128 SPLIT PREFIX NSH FIRST EXTRA
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
cd /p/scratch/training2666/mouhali1/SLZ/Hackathon_Watermark_Localisation_Task
source .venv/bin/activate
srun --ntasks=8 --kill-on-bad-exit=0 --cpu-bind=cores --output="output/%j_%t.out" bash -c '
  export CUDA_VISIBLE_DEVICES=$((SLURM_LOCALID / 2))
  SH=$((FIRST + SLURM_LOCALID))
  [ $SH -lt $NSH ] || exit 0
  echo "tranche $SH/$NSH GPU $CUDA_VISIBLE_DEVICES $(hostname)"
  exec python -u work_c1_tmatch/template_match.py --split "$SPLIT" --shard $SH --nshards $NSH \
       --out "output/${PREFIX}_${SH}.npz" $EXTRA'
