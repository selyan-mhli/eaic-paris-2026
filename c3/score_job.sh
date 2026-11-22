#!/bin/bash
# score_like.py sur une liste de candidats, une ligne de $LIST par GPU (4 en parallèle).
# Usage (depuis work_c3_eval/) : LIST=fichier.txt sbatch score_job.sh
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --job-name=j3
#SBATCH --output=logs/%j.out

cd /p/scratch/training2666/mouhali1/SLZ/unlearning/work_c3_eval || exit 1
unset PYTHONPATH
source ../.venv/bin/activate
export HF_HUB_OFFLINE=1 SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=8
export LIST=${LIST:?LIST=fichier de candidats requis}
srun --kill-on-bad-exit=0 --cpu-bind=cores --output=logs/%j_%t.out bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID   # les GPU ne sont pas isolés par Slurm
  read -r -a C < <(sed -n "$((SLURM_LOCALID + 1))p" "$LIST"); [ ${#C[@]} -eq 0 ] && exit 0
  python -u score_like.py --original ../model.pt --off offforget.npz "${C[@]}" > res/score_${SLURM_JOB_ID}_${SLURM_LOCALID}.txt 2>&1'
