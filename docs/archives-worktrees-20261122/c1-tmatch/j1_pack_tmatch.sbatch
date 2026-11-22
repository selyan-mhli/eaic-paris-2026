#!/bin/bash
# 4 tranches de template_match.py sur un nœud, une par GPU.
# Usage (depuis le dossier de la tâche) : sbatch work_c1_tmatch/j1_pack_tmatch.sbatch <split> <préfixe sortie> [args en plus]
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH -J j1
#SBATCH --output=output/%j.out
set -euo pipefail
SPLIT=${1:?split}; PREFIX=${2:?prefixe}; EXTRA=${3:-}
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 SPLIT PREFIX EXTRA
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
cd /p/scratch/training2666/mouhali1/SLZ/Hackathon_Watermark_Localisation_Task
source .venv/bin/activate
srun --ntasks=4 --kill-on-bad-exit=0 --cpu-bind=cores --output="output/%j_%t.out" bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  python -c "import torch,sys; sys.exit(0 if torch.cuda.device_count()==1 else 3)" || exit 3
  echo "tranche $SLURM_LOCALID/4 GPU $CUDA_VISIBLE_DEVICES $(hostname)"
  exec python -u work_c1_tmatch/template_match.py --split "$SPLIT" --shard $SLURM_LOCALID --nshards 4 \
       --out "output/${PREFIX}_${SLURM_LOCALID}.npz" $EXTRA'
