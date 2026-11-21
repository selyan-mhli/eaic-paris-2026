#!/bin/bash
# Évaluation par lots des candidats c3 avec c3/eval.py, 4 GPU en parallèle sur un nœud.
# Lecture seule sur unlearning/ (model.pt, forget_images.npz, output/*.pt) ; tout est écrit dans
# work_c3_eval/. Chaque GPU évalue sa liste de candidats, deux fois chacun :
#   aug : proxy retain par défaut (augmentations fortes des forget) ;
#   off : proxy retain sur offforget.npz (photos hors forget, cf. make_offforget.py).
# Usage (depuis work_c3_eval/) :
#   sbatch eval_batch.sh            # complet (3500 forget)
#   MAXIMG=200 sbatch --time=00:15:00 eval_batch.sh   # essai court
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --job-name=j3
#SBATCH --output=logs/%j.out

W=/p/scratch/training2666/mouhali1/SLZ/unlearning/work_c3_eval
cd "$W" || exit 1
unset PYTHONPATH
source ../.venv/bin/activate
export HF_HUB_OFFLINE=1 SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=8
export MAXIMG=${MAXIMG:-}

srun --kill-on-bad-exit=0 --cpu-bind=cores --output=logs/%j_%t.out bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID   # les GPU ne sont pas isolés par Slurm
  O=../output
  LISTS=("../model.pt $O/ckpt_ent04.pt" "$O/v2_best.pt $O/v2_ent03.pt" "$O/soup_safe.pt $O/ckpt_ent08.pt" "$O/soup_mid.pt $O/ckpt_ent12.pt")
  EXTRA=(); [ -n "$MAXIMG" ] && EXTRA=(--max-images "$MAXIMG")
  for c in ${LISTS[$SLURM_LOCALID]}; do
    n=$(basename "$c" .pt); [ "$c" = ../model.pt ] && n=orig
    echo "== GPU $CUDA_VISIBLE_DEVICES $(hostname) $n"
    python -u eval.py --candidate "$c" --task-dir .. "${EXTRA[@]}" > "res/${n}_aug${MAXIMG}.txt" 2>&1
    python -u eval.py --candidate "$c" --task-dir .. --retain-npz offforget.npz "${EXTRA[@]}" > "res/${n}_off${MAXIMG}.txt" 2>&1
    grep -h "^score" "res/${n}_aug${MAXIMG}.txt" "res/${n}_off${MAXIMG}.txt"
  done'

# Dérive hors forget de tous les candidats, un seul processus (GPU 0), après les évaluations.
O=../output
srun --ntasks=1 --gres=gpu:1 --output=logs/%j_drift.out python -u drift.py --images offforget.npz --forget ../forget_images.npz \
  --original ../model.pt $O/v2_best.pt $O/v2_ent03.pt $O/soup_safe.pt $O/soup_mid.pt \
  $O/ckpt_ent04.pt $O/ckpt_ent08.pt $O/ckpt_ent12.pt $O/ckpt_ent16.pt
