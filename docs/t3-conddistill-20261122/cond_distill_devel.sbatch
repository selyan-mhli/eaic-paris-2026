#!/bin/bash
# T3 distillation conditionnelle à l'image, 4 variantes (une par GPU). Préparé 22/11, NON lancé par l'agent.
# Depuis work_c3_eval/ :  sbatch --test-only cond_distill/cond_distill_devel.sbatch
#                         sbatch cond_distill/cond_distill_devel.sbatch
#   GPU 0 : K=2 λ=1 enseignant push_s175        GPU 1 : K=4 λ=1 enseignant push_s175
#   GPU 2 : K=4 λ=3 enseignant push_s175        GPU 3 : K=4 λ=1 enseignant orig+2,0*(push_best-orig)
# Sorties : cond_distill/out/<job>/<tag>/{student.pt (~344 Mo), eval.json} ; journaux cond_distill/logs/.
# Budget : 17 min d'entraînement max (ou 4000 pas) + éval (~1-3 min, estimation) par variante.
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --job-name=j3cd
#SBATCH --output=cond_distill/logs/%j.out

cd /p/scratch/training2666/mouhali1/SLZ/unlearning/work_c3_eval || exit 1
unset PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=8
source ../.venv/bin/activate
OUT=cond_distill/out/$SLURM_JOB_ID
[ -e "$OUT" ] && { echo "refus : $OUT existe déjà"; exit 1; }
mkdir -p "$OUT"
df -i /p/scratch/training2666 | tail -1
sha256sum cond_distill/cond_distill.py ../model.pt push_scale/push_s175.pt ../output/push_best.pt
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
  python -u cond_distill/cond_distill.py --model ../model.pt --forget ../forget_images.npz \
    --off-all offforget_all.npz --gen-glob "gen_out/var_15904093_*.npz" \
    --out-dir $OUT --tag $tag --seed 0 --steps 4000 --minutes 17 $extra || echo "ECHEC $tag"'
ls -la "$OUT"/*
python - <<EOF
import json, glob
for f in sorted(glob.glob("$OUT/*/eval.json")):
    r = json.load(open(f)); m = r["metrics"]
    print(r["tag"], "soumettable=", r["decision"]["submittable"], "pas=", r["steps_done"],
          {s: (round(m["student_vs_orig"][s]["agreement"], 3), round(m["student_vs_orig"][s]["KL"], 3))
           for s in ("forget_diag", "altered_diag", "gen_diag", "control_affected", "control_other")
           if s in m["student_vs_orig"]})
EOF
df -i /p/scratch/training2666 | tail -1
