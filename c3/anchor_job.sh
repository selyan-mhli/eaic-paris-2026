#!/bin/bash
# Plan B c3 : unlearn_anchor.py (NPO v2 + distillation KL hors forget), 4 configurations en
# parallèle (une par GPU : ret_weight ∈ {1, 4} × lr ∈ {5e-6, 1e-5}), checkpoint toutes les
# SAVE étapes, puis drift.py sur les checkpoints de chaque GPU. Écrit dans work_c3_eval/.
# Usage (depuis work_c3_eval/) : sbatch anchor_job.sh  (partition devel : voir soup_job_devel)
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:28:00
#SBATCH --job-name=j3
#SBATCH --output=logs/%j.out

cd /p/scratch/training2666/mouhali1/SLZ/unlearning/work_c3_eval || exit 1
unset PYTHONPATH
source ../.venv/bin/activate
export HF_HUB_OFFLINE=1 SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=8
export STEPS=${STEPS:-600} SAVE=${SAVE:-60} CFGS=${CFGS:-}
# Mesuré (job 15902389) : 3,15 étapes/s par GPU ; 600 étapes + drift de 10 checkpoints = 5 min 45 s.
srun --kill-on-bad-exit=0 --cpu-bind=cores --output=logs/%j_%t.out bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID   # les GPU ne sont pas isolés par Slurm
  # CFGS : 4 configurations séparées par « ; », au format "tag:options".
  IFS=";" read -r -a ALL <<< "${CFGS:-r1_lr5e6:--ret-weight 1 --lr 5e-6;r1_lr1e5:--ret-weight 1 --lr 1e-5;r4_lr5e6:--ret-weight 4 --lr 5e-6;r4_lr1e5:--ret-weight 4 --lr 1e-5}"
  t=${ALL[$SLURM_LOCALID]%%:*}; opts=${ALL[$SLURM_LOCALID]#*:}
  python -u unlearn_anchor.py $opts --tag $t --max-steps $STEPS --save-every $SAVE
  python -u drift.py --images offforget.npz --forget ../forget_images.npz --original ../model.pt \
    anchor_ckpt/anc_${t}_s*.pt > res/drift_${SLURM_JOB_ID}_${t}.txt 2>&1'
