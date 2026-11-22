#!/bin/bash
# Task 3 « mécanismes » + cache de features pré-tête. Préparé par l'agent D
# (docs/task3-work-20261121/mech/), NON lancé. Copier mech/ dans work_c3_eval/mech/, puis
# depuis work_c3_eval/ :  sbatch --test-only mech/mech_job.sbatch && sbatch mech/mech_job.sbatch
#   rang 0 : cache_feats (features pré-tête, 1 npz) -> head_search --verify sur push_best/s125/s150
#            -> (e) balayage biais hors ligne -> contrôle « tenu à l'écart » des références notées
#   rang 1 : (a)+(b) npo_repair   : points (a) = phase d'oubli, points (b) = réparation
#   rang 2 : (c)+(d) conf_repair  : idem, cible de confiance par image, lr 3e-5
#   rang 3 : (c)+(d) conf_repair, lr 1e-4 (garde-fou de convergence : margin n'a atteint que
#            10-16 % de sa cible en 1000 pas à 5e-5)
# Fichiers : mech_out/<job>/ : <= 7 + 8 + 8 checkpoints (~7,9 Go) + 1 npz (~75 Mo).
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --job-name=j3
#SBATCH --output=logs/%j.out

cd /p/scratch/training2666/mouhali1/SLZ/unlearning/work_c3_eval || exit 1
unset PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=8
source ../.venv/bin/activate
OUT=mech_out/$SLURM_JOB_ID
[ -e "$OUT" ] && { echo "refus : $OUT existe déjà"; exit 1; }
mkdir -p "$OUT"
df -i /p/scratch/training2666 | tail -1
sha256sum mech/*.py ../model.pt
export OUT
srun --kill-on-bad-exit=0 --cpu-bind=cores --output=logs/%j_%t.out bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID          # les GPU ne sont pas isolés par Slurm
  J=$SLURM_JOB_ID
  if [ "$SLURM_LOCALID" = 0 ]; then
    python -u mech/cache_feats.py --out $OUT/feats_$J.npz --heads \
      push_s075=push_scale/push_s075.pt push_best=../output/push_best.pt push_s125=push_scale/push_s125.pt \
      push_s150=push_scale/push_s150.pt push_s200=push_scale/push_s200.pt || exit 1
    python -u mech/head_search.py --cache $OUT/feats_$J.npz --verify res/score_15903813_push.txt \
      --bias-sweep --push-scale 1 1.1 1.25 1.375 1.5 1.75 2 --tsv res/score_${J}_headsearch.txt \
      > logs/${J}_headsearch.out 2>&1
    python -u mech/mech_holdout.py --off-all offforget_all.npz --off offforget.npz \
      push_scale/push_s125.pt ../output/push_best.pt ../work_c3_t1t2/out/t1t2_p03_e2.pt \
      anchor_ckpt/anc_f2a_s0550.pt push_scale/push_s150.pt > res/hold_${J}_ref.txt 2>&1
    exit 0
  fi
  case $SLURM_LOCALID in
    1) v=npo_repair;  t=ab;  extra="" ;;
    2) v=conf_repair; t=cd;  extra="" ;;
    3) v=conf_repair; t=cdx; extra="--lr 1e-4" ;;
  esac
  python -u mech/mech.py --variant $v --tag $t $extra --out-dir $OUT --forget-minutes 10 || echo "ECHEC entraînement $t"
  ls $OUT/mech_${t}_*.pt >/dev/null 2>&1 || exit 1
  python -u score_like.py --original ../model.pt --off offforget.npz $OUT/mech_${t}_*.pt \
    > res/score_${J}_${t}.txt 2>&1
  python -u mech/mech_holdout.py --off-all offforget_all.npz --off offforget.npz $OUT/mech_${t}_*.pt \
    > res/hold_${J}_${t}.txt 2>&1'
ls -la "$OUT" res/*_${SLURM_JOB_ID}_* 2>/dev/null
df -i /p/scratch/training2666 | tail -1
