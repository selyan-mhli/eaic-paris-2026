#!/bin/bash
# Validation préparée localement ; jamais lancée par cet audit.
# Depuis le dossier unlearning sur JURECA : sbatch /chemin/audit_candidates.sbatch
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --job-name=j3
#SBATCH --output=output/task3_audit_%j.out
set -euo pipefail
cd /p/scratch/training2666/mouhali1/SLZ/unlearning
unset PYTHONPATH
source .venv/bin/activate
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
srun --kill-on-bad-exit=0 --cpu-bind=cores \
  --output=output/task3_audit_%j_%t.tsv bash -euc '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  python -c "import torch; assert torch.cuda.device_count() == 1"
  case "$SLURM_LOCALID" in
    0) candidates=(output/push_best.pt output/push_a8.pt) ;;
    1) candidates=(output/head_a2.pt output/head_a4.pt) ;;
    2) candidates=(work_c3_t1t2/out/t1t2_p03_e1.pt work_c3_t1t2/out/t1t2_p03_e2.pt work_c3_t1t2/out/t1t2_p03_e3.pt) ;;
    3) candidates=(work_c3_t1t2/out/t1t2_p06_e1.pt work_c3_t1t2/out/t1t2_p10_e1.pt work_c3_t1t2/out/t1t2_p10_e2.pt) ;;
    *) exit 2 ;;
  esac
  sha256sum model.pt "${candidates[@]}"
  exec python -u work_c3_eval/score_like.py \
    --original model.pt --off work_c3_eval/offforget.npz "${candidates[@]}"
  '
