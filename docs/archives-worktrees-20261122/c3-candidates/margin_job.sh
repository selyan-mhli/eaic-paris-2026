#!/bin/bash
# margin_edit.py, 4 configurations en parallèle (une par GPU), puis score_like.py sur les
# checkpoints de chaque GPU. dc-gpu sans réservation (backfill), --time court.
# Usage (depuis work_c3_eval/) : CFGS="tag:opts;..." sbatch margin_job.sh
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:25:00
#SBATCH --job-name=j3
#SBATCH --output=logs/%j.out

cd /p/scratch/training2666/mouhali1/SLZ/unlearning/work_c3_eval || exit 1
unset PYTHONPATH
source ../.venv/bin/activate
export HF_HUB_OFFLINE=1 SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=8
export CFGS=${CFGS:-"mg25:--margin 2.5;mg15:--margin 1.5;mg10:--margin 1.0;mg05:--margin 0.5"}
srun --kill-on-bad-exit=0 --cpu-bind=cores --output=logs/%j_%t.out bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID   # les GPU ne sont pas isolés par Slurm
  IFS=";" read -r -a ALL <<< "$CFGS"; c=${ALL[$SLURM_LOCALID]}; [ -z "$c" ] && exit 0
  t=${c%%:*}; opts=${c#*:}
  python -u margin_edit.py $opts --tag $t || exit 1
  python -u score_like.py --original ../model.pt --off offforget.npz margin_out/mg_${t}_s*.pt \
    > res/score_${SLURM_JOB_ID}_$t.txt 2>&1'
