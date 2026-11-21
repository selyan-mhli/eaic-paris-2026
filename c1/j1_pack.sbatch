#!/bin/bash
# Usage (depuis le dossier de la tâche) :
#   sbatch work_c1/j1_pack.sbatch <split> <nshards> <premier_shard>
#   ex. test/8 : sbatch ... test 8 0   puis   sbatch ... test 8 4
#   [prefixe de sortie (défaut boxes_<split>)] [args en plus, ex. "--limit 2"]
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH -J j1
#SBATCH --output=output/%j.out
set -euo pipefail
SPLIT=${1:?split}; NSHARDS=${2:?nshards}; FIRST=${3:-0}; PREFIX=${4:-boxes_$1}; EXTRA=${5:-}
NTASKS=$(( NSHARDS - FIRST < 4 ? NSHARDS - FIRST : 4 ))
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /p/scratch/training2666/mouhali1/SLZ/Hackathon_Watermark_Localisation_Task
source .venv/bin/activate
export SPLIT NSHARDS FIRST PREFIX EXTRA
srun --ntasks="$NTASKS" --kill-on-bad-exit=0 --cpu-bind=cores \
     --output="output/%j_%t.out" bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID      # 1 GPU par tâche (pas d isolation cgroups)
  SHARD=$(( FIRST + SLURM_LOCALID ))
  python -c "import torch,sys; n=torch.cuda.device_count(); sys.exit(0 if n==1 else f\"{n} GPU visibles\")" || exit 3
  echo "shard $SHARD/$NSHARDS sur GPU $CUDA_VISIBLE_DEVICES ($(hostname))"
  exec python -u work_c1/box_search.py --split "$SPLIT" --shard "$SHARD" --nshards "$NSHARDS" \
       --out "output/${PREFIX}_${SHARD}.npz" $EXTRA'
