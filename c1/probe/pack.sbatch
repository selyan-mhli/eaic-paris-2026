#!/bin/bash
# A1 (recolle.py, une méthode par GPU) + A2 (absl_maps.py train puis test) sur un nœud, 4 tâches.
# Usage (depuis le dossier de la tâche, scripts copiés dans work_c1_probe/) :
#   sbatch work_c1_probe/pack.sbatch <préfixe> ["--limit 4"]
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=00:45:00
#SBATCH -J j1
#SBATCH --output=output/%j.out
set -euo pipefail
PREFIX=${1:?prefixe}; EXTRA=${2:-}
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /p/scratch/training2666/mouhali1/SLZ/Hackathon_Watermark_Localisation_Task
source .venv/bin/activate
export PREFIX EXTRA
srun --ntasks=4 --kill-on-bad-exit=0 --cpu-bind=cores --output="output/%j_%t.out" bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  M=(pixelseal trustmark stegastamp)
  if [ "$SLURM_LOCALID" -lt 3 ]; then
    exec python -u work_c1_probe/recolle.py --method "${M[$SLURM_LOCALID]}" --out "output/${PREFIX}_recolle_${M[$SLURM_LOCALID]}.npz" $EXTRA
  else
    python -u work_c1_probe/absl_maps.py --split train --out "output/${PREFIX}_absl_train.npz" $EXTRA
    exec python -u work_c1_probe/absl_maps.py --split test --out "output/${PREFIX}_absl_test.npz" $EXTRA
  fi'
