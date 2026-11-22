#!/bin/bash
# Contrôles « 45 classes concernées / autres » : origine, push_s125, mech_ab_r_s0300, mech_ab_r_s0100.
# Préparé par l'agent D, NON lancé. Depuis work_c3_eval/ :
#   sbatch --test-only mech/class_split.sbatch && sbatch mech/class_split.sbatch
# Volume : 4 modèles × (3 500 forget + 765 off réparation) × 5 vues ≈ 85 000 images, plus la moitié
# contrôle (4 × 765 × 5 ≈ 15 000) dans un fichier séparé. Débit mesuré dans le log (img/s).
# Estimation : ~3-5 min sur 1 A100 (score_like : ~22 000 images par modèle en ~25 s dans 15903813).
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --job-name=j3
#SBATCH --output=logs/%j.out
set -euo pipefail
cd /p/scratch/training2666/mouhali1/SLZ/unlearning/work_c3_eval
unset PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=16
source ../.venv/bin/activate
J=$SLURM_JOB_ID
C=(push_scale/push_s125.pt mech_out/15903971/mech_ab_r_s0300.pt mech_out/15903971/mech_ab_r_s0100.pt)
sha256sum mech/class_split.py "${C[@]}"
python -u mech/class_split.py --out res/class_split_$J.txt --final-ctl res/ctl_final_$J.txt "${C[@]}"
