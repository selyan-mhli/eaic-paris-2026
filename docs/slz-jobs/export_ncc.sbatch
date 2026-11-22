#!/bin/bash
# Export test de la config champion OOF 0,5614 : ff+fm+hf:ncc_m,ncc_d, 20 époques, --ens 3.
# Correctif : passe --test-hf (absent de head.sbatch, cause de "AssertionError: ncc_m" au job 15902663).
# --seeds 0 : refait une seule partition externe (contrôle de reproductibilité, attendu ~0,5606 = head[0]).
# Puis fusion StegaStamp pixseg comme le public 0,5845 (merge_ss.py, seuil 0,2, pixseg_test_nt_fit0_s0..3).
# Propriétaire : session orchestration SLZ (21/11). Rien n'est soumis.
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH -J slz_t1_ncc
#SBATCH --output=output/%j.out
set -euo pipefail
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1
cd /p/scratch/training2666/mouhali1/SLZ/Hackathon_Watermark_Localisation_Task
source .venv/bin/activate
J=$SLURM_JOB_ID
OUT=output/c1_segf_ncc_e20_$J.pt
python work_segf/preflight_ncc.py
python -u work_segf/head_eval.py --config "ff+fm+hf:ncc_m,ncc_d" --epochs 20 --ens 3 --seeds 0 \
  --heats output/heats_train.npz --boxes "output/boxes_train_*.npz" \
  --ff "output/seg_ff_train_*.npz" --hf "output/segf_hf_train_*.npz" \
  --test-heats output/heats_test.npz --test-boxes "output/boxes_test_*.npz" \
  --test-ff "output/seg_ff_test_*.npz" --test-hf "output/segf_hf_test_*.npz" \
  --out "output/segf_ncc_e20_oof_$J.npz" --write-sub "$OUT"
sha256sum "$OUT"
python work_pixseg_v2/merge_ss.py "$OUT" "output/c1_ncc_pixss_fit0_$J.pt" 0.2 output/pixseg_test_nt_fit0_s[0-3].npz
sha256sum "output/c1_ncc_pixss_fit0_$J.pt" output/c1_pixss_fit0.pt
