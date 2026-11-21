#!/bin/bash
# Soupes origine/v2_best et origine/v2_ent03 (6 λ chacune), construites sur CPU dans le job,
# puis évaluées une tâche par GPU (eval_task.sh). Rien sur le nœud de login.
# Usage (depuis work_c3_eval/) : sbatch soup_job.sh
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
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
L="0.1 0.2 0.3 0.4 0.5 0.6"
srun --ntasks=2 --cpu-bind=cores --output=logs/%j_build_%t.out bash -c '
  O=(../output/v2_best.pt ../output/v2_ent03.pt); P=(soupB soupE)
  CUDA_VISIBLE_DEVICES= python -u make_soups.py --original ../model.pt --other ${O[$SLURM_PROCID]} \
    --prefix ${P[$SLURM_PROCID]} --out-dir soups --lambdas '"$L"'' || exit 1
S=soups
cat > soup_cands.txt <<LST
$S/soupB_l10.pt $S/soupB_l50.pt $S/soupE_l30.pt
$S/soupB_l20.pt $S/soupB_l60.pt $S/soupE_l40.pt
$S/soupB_l30.pt $S/soupE_l10.pt $S/soupE_l50.pt
$S/soupB_l40.pt $S/soupE_l20.pt $S/soupE_l60.pt
LST
export LIST=soup_cands.txt MAXIMG=
srun --kill-on-bad-exit=0 --cpu-bind=cores --output=logs/%j_%t.out bash eval_task.sh
