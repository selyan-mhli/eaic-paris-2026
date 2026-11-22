#!/bin/bash
# T3 distillation conditionnelle --fit-all (3500 forget en entraînement), 4 variantes, une par GPU.
# Préparé 22/11 ~06:15, NON lancé par l'agent. Depuis work_c3_eval/ :
#   df -i /p/scratch/training2666 | tail -1        # quota d'inodes du projet saturé par moments
#   sbatch --test-only cond_distill/cond_distill_all_devel.sbatch && sbatch cond_distill/cond_distill_all_devel.sbatch
#   GPU 0 : K2_l1_t175   GPU 1 : K4_l1_t175   GPU 2 : K4_l3_t175   GPU 3 : K4_l1_t200 (orig+2,0*(push_best-orig))
# Budget : 6000 pas, plafond 20 min d'entraînement (débit mesuré job 15905198 : 0,11-0,13 s/pas -> 11-13 min),
#          + chargement et éval (~2-4 min, estimation) ; --time 30 min.
# Sorties : cond_distill/out_all/<job>/<tag>/{student.pt, eval.json} ; journaux cond_distill/logs/.
# Choix FIGÉ (RULE de cond_distill_all.py) : parmi les variantes soumettables à enseignant x1,75 (tags *_t175),
# la plus forte accord control_affected ; K4_l1_t200 est rapportée, jamais choisie.
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --job-name=j3ca
#SBATCH --output=cond_distill/logs/%j.out

cd /p/scratch/training2666/mouhali1/SLZ/unlearning/work_c3_eval || exit 1
unset PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=8
source ../.venv/bin/activate
OUT=cond_distill/out_all/$SLURM_JOB_ID
[ -e "$OUT" ] && { echo "refus : $OUT existe déjà"; exit 1; }
mkdir -p "$OUT"
df -i /p/scratch/training2666 | tail -1
sha256sum cond_distill/cond_distill_all.py ../model.pt push_scale/push_s175.pt ../output/push_best.pt
export OUT
srun --kill-on-bad-exit=0 --cpu-bind=cores --output=cond_distill/logs/%j_%t.out bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID          # les GPU ne sont pas isolés par Slurm
  T175="--teacher push_scale/push_s175.pt"
  case $SLURM_LOCALID in
    0) tag=K2_l1_t175; extra="--K 2 --lam 1 $T175" ;;
    1) tag=K4_l1_t175; extra="--K 4 --lam 1 $T175" ;;
    2) tag=K4_l3_t175; extra="--K 4 --lam 3 $T175" ;;
    3) tag=K4_l1_t200; extra="--K 4 --lam 1 --push-best ../output/push_best.pt --teacher-scale 2.0" ;;
  esac
  python -u cond_distill/cond_distill_all.py --fit-all --model ../model.pt --forget ../forget_images.npz \
    --off-all offforget_all.npz --gen-glob "gen_out/var_15904093_*.npz" \
    --out-dir $OUT --tag $tag --seed 0 --steps 6000 --minutes 20 $extra || echo "ECHEC $tag"'
ls -la "$OUT"/*
python - <<PY
import json, glob
rows = []
for f in sorted(glob.glob("$OUT/*/eval.json")):
    r = json.load(open(f)); sv, st = r["metrics"]["student_vs_orig"], r["metrics"]["student_vs_teacher"]
    rows.append(r)
    print(r["tag"], "soumettable=", r["decision"]["submittable"], "pas=", r["steps_done"], "sha=", r["student_sha256"][:12],
          {k: round(v, 4) for c in r["decision"]["checks"].values() for k, v in c.items() if isinstance(v, float)})
ok = [r for r in rows if r["decision"]["submittable"] and r["tag"].endswith("_t175")]
if ok:
    best = max(ok, key=lambda r: r["metrics"]["student_vs_orig"]["control_affected"]["agreement"])
    print("CHOIX FIGE :", best["tag"], best["student_pt"], best["student_sha256"])
else:
    print("CHOIX FIGE : aucun candidat soumettable a enseignant x1,75")
PY
df -i /p/scratch/training2666 | tail -1
