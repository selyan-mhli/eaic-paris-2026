#!/bin/bash
# Distillation vers une cible « image jamais vue » mélangée (distill_mix.py), 4 valeurs de
# --mix en parallèle (une par GPU), puis score_like.py sur les checkpoints r85/r65/r45 de chaque
# GPU. Écrit dans work_c3_eval/ (distill_out/<tag>/, res/, logs/).
# Usage (depuis work_c3_eval/) : sbatch distill_job.sh  [CFGS="tag:opts;..."]
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --job-name=j3
#SBATCH --output=logs/%j.out

cd /p/scratch/training2666/mouhali1/SLZ/unlearning/work_c3_eval || exit 1
unset PYTHONPATH
source ../.venv/bin/activate
export HF_HUB_OFFLINE=1 SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=8
# CFGS : 4 configurations séparées par « ; », au format "tag:options de distill_mix.py".
export CFGS=${CFGS:-"m15:--mix 0.15;m30:--mix 0.3;m50:--mix 0.5;m75:--mix 0.75"}
# Mesuré (job 15902463) : 4 configs (arrêt à ratio 0,45) + score_like de 3 checkpoints = 3 min 35 s.
srun --kill-on-bad-exit=0 --cpu-bind=cores --output=logs/%j_%t.out bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID   # les GPU ne sont pas isolés par Slurm
  IFS=";" read -r -a ALL <<< "$CFGS"; c=${ALL[$SLURM_LOCALID]}; [ -z "$c" ] && exit 0
  t=${c%%:*}; opts=${c#*:}
  python -u distill_mix.py $opts --tag $t
  ls distill_out/$t/distill_r*.pt >/dev/null 2>&1 || { echo "aucun checkpoint pour $t"; exit 0; }
  python -u score_like.py --original ../model.pt --off offforget.npz distill_out/$t/distill_r*.pt \
    > res/score_${SLURM_JOB_ID}_$t.txt 2>&1'
