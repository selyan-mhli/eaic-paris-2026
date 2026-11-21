#!/bin/bash
# head_eval.py, une configuration par GPU (4 tâches). Usage :
#   sbatch work_segf/head.sbatch <préfixe> "<cfg0>[@époques]" ... "<cfg3>[@époques]" [args communs...]
#   SUB=1 : écrit aussi output/<préfixe>_<tâche>.pt (fichier test, NON soumis ; entrées test du kit)
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00
#SBATCH -J j1
#SBATCH --output=output/%j.out
set -euo pipefail
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
cd /p/scratch/training2666/mouhali1/SLZ/Hackathon_Watermark_Localisation_Task
source .venv/bin/activate
export PREFIX=$1 C0=$2 C1=$3 C2=$4 C3=$5; shift 5
srun --ntasks=4 --kill-on-bad-exit=0 --cpu-bind=cores --output="output/%j_%t.out" bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; C=(); C=("$C0" "$C1" "$C2" "$C3"); X=${C[$SLURM_LOCALID]}; CFG=${X%@*}; EP=()
  [ "$X" != "$CFG" ] && EP=(--epochs "${X##*@}")
  SUBA=(); [ -n "${SUB:-}" ] && SUBA=(--test-heats output/heats_test.npz --test-boxes "output/boxes_test_*.npz" --test-ff "output/seg_ff_test_*.npz" --write-sub "output/${PREFIX}_${SLURM_LOCALID}.pt")
  exec python -u work_segf/head_eval.py --config "$CFG" --heats output/heats_train.npz --boxes "output/boxes_train_*.npz" \
       --ff "output/seg_ff_train_*.npz" --hf "output/segf_hf_train_*.npz" --out "output/${PREFIX}_${SLURM_LOCALID}.npz" "$@" "${EP[@]}" "${SUBA[@]}"' _ "$@"
