#!/bin/bash
# Corps d'une tâche srun (une par GPU) : évalue les candidats de la ligne SLURM_LOCALID+1 de $LIST
# avec eval.py (retain = forget augmentées, puis photos hors forget) puis drift.py sur ces mêmes
# candidats. Appelé par eval_batch.sh et soup_job.sh, depuis work_c3_eval/.
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID   # les GPU ne sont pas isolés par Slurm
python -c "import torch,sys; sys.exit(torch.cuda.device_count()!=1)" || exit 3
read -r -a C < <(sed -n "$((SLURM_LOCALID + 1))p" "$LIST")
[ ${#C[@]} -eq 0 ] && exit 0
EXTRA=(); [ -n "$MAXIMG" ] && EXTRA=(--max-images "$MAXIMG")
for c in "${C[@]}"; do
  n=$(basename "$c" .pt); [ "$c" = ../model.pt ] && n=orig
  echo "== GPU $CUDA_VISIBLE_DEVICES $(hostname) $n"
  python -u eval.py --candidate "$c" --task-dir .. "${EXTRA[@]}" > "res/${n}_aug${MAXIMG}.txt" 2>&1
  python -u eval.py --candidate "$c" --task-dir .. --retain-npz offforget.npz "${EXTRA[@]}" > "res/${n}_off${MAXIMG}.txt" 2>&1
  grep -h "^score" "res/${n}_aug${MAXIMG}.txt" "res/${n}_off${MAXIMG}.txt"
done
python -u drift.py --images offforget.npz --forget ../forget_images.npz --original ../model.pt "${C[@]}" \
  > "res/drift_${SLURM_JOB_ID}_${SLURM_LOCALID}.txt" 2>&1
