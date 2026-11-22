#!/bin/bash
# PRÉPARÉ, JAMAIS LANCÉ. Candidat « TrustMark dense seul » (recette figée du pilote 15903857, TM tête seule).
# Copie : c1/pixseg/dense/tm_candidate/ -> work_pixd/dense/tm_candidate/ (dense/ existant inchangé, même sha).
#   sbatch --test-only work_pixd/dense/tm_candidate/tm_job.sbatch
#   sbatch work_pixd/dense/tm_candidate/tm_job.sbatch                      # mesure, puis export SI EXPORT_OK
#   STAGES=measure sbatch work_pixd/dense/tm_candidate/tm_job.sbatch       # mesure seule
#   STAGES=export DECISION=output/dense/tmc_measure_<job>.json sbatch ...  # export seul (après T3, par ex.)
# Durée estimée (débit mesuré du pilote B1 : 18,7-19,9 it/s, bs 32) : mesure 354+320+362+776 = 1812 pas
#   ~ 1 min 35 s + chargements (~2-3 min) ; export ~1000 pas ~ 1 min + 340 images test : total ~6-8 min, limite 20 min.
# Sorties (7 fichiers) : output/<job>.out ; output/dense/tmc_measure_<job>{.npz,_heads.pt,.json,_pilot.json} ;
#   si export : output/c1_tmdense_pixss_<job>{.pt,_head.pt,_tmprobs.npz}. RIEN n'est soumis.
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH -J j1
#SBATCH --output=output/%j.out
set -euo pipefail
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
cd /p/scratch/training2666/mouhali1/SLZ/Hackathon_Watermark_Localisation_Task
source .venv/bin/activate
J=$SLURM_JOB_ID
C=work_pixd/dense/tm_candidate
PILOT=output/dense/res_trustmark_head_15903857.npz
CHAMP=output/c1_pixss_fit0.pt
STAGES=${STAGES:-measure,export}
M=output/dense/tmc_measure_$J
DECISION=${DECISION:-$M.json}
echo "stages $STAGES ; pilote $(sha256sum $PILOT) ; champion $(sha256sum $CHAMP)"

if [[ $STAGES == *measure* ]]; then
  python -u $C/tm_job.py measure --pilot $PILOT --out $M --device cuda
  # décision : rerun seul (toutes TM des folds 0-4, seuils figés du pilote), contre le champion exact segf_sub_1
  python -u $C/measure_routed.py --dense $M.npz --ref output/segf_sub_1.npz --ref2 output/segf_ens3_3.npz \
    --acc output/heats_train.npz --masks release/train/masks_128.npz --scope all --out $M.json | tail -6
  # variante rapportée : F du pilote, R du rerun
  python -u $C/measure_routed.py --dense $PILOT $M.npz --ref output/segf_sub_1.npz --ref2 output/segf_ens3_3.npz \
    --acc output/heats_train.npz --masks release/train/masks_128.npz --scope all --out ${M}_pilot.json | tail -6
fi

if [[ $STAGES == *export* ]]; then
  # tm_job.py export refuse sans "export_ok": true dans $DECISION (sauf FORCE=1, sur ordre explicite)
  python -u $C/tm_job.py export --pilot $PILOT --champion $CHAMP --heats-test output/heats_test.npz \
    --decision "$DECISION" --gate 0.65 --thr 0.70 --out output/c1_tmdense_pixss_$J --device cuda \
    $([ "${FORCE:-0}" = 1 ] && echo --force)
  [ -f output/c1_tmdense_pixss_$J.pt ] && sha256sum output/c1_tmdense_pixss_$J.pt $CHAMP || echo "pas d'export"
fi
ls -la output/dense/tmc_measure_$J* output/c1_tmdense_pixss_$J* 2>/dev/null || true
