#!/bin/bash
# Préparé uniquement : lancement manuel par Selyan depuis le dossier Task 1.
# Complète les folds 2–4 des deux configurations, sans modifier l'entraînement.
# Le quatrième GPU produit une troisième graine test 6k, comme les 3 graines du champion.
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:40:00
#SBATCH --job-name=j12
#SBATCH --output=/p/project1/training2666/mouhali1/t1sr/relay_ss6k_%j.out
set -euo pipefail
cd /p/scratch/training2666/mouhali1/SLZ/Hackathon_Watermark_Localisation_Task
source .venv/bin/activate
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1 HF_HOME=/p/scratch/training2666/mouhali1/SLZ/.cache
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
srun --ntasks=4 --kill-on-bad-exit=1 --cpu-bind=cores \
  --output=/p/project1/training2666/mouhali1/t1sr/relay_ss6k_%j_%t.out bash -c '
set -euo pipefail
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
python -c "import torch; assert torch.cuda.device_count() == 1"
COMMON=(--synth output/pixseg_syn_pixelseal_*.npz output/pixseg_syn_trustmark_*.npz output/pixseg_syn_stegastamp_*.npz
  --ref-heats output/heats_train.npz --bs 16 --log-every 3000 --backbone convnext_tiny
  --real-frac 0.5 --lr 1e-4 --out /p/project1/training2666/mouhali1/t1sr/pixseg)
case "$SLURM_LOCALID" in
  0) exec python -u work_pixseg_v2/train.py "${COMMON[@]}" --cv --folds 2,3 --steps 6000 --tag relay_psall6k_f23 ;;
  1) exec python -u work_pixseg_v2/train.py "${COMMON[@]}" --cv --folds 4 --steps 6000 --tag relay_psall6k_f4 ;;
  2) exec python -u work_pixseg_v2/train.py "${COMMON[@]}" --cv --folds 2,3,4 --steps 3000 --tag relay_psall3k_f234 ;;
  3) exec python -u work_pixseg_v2/train.py "${COMMON[@]}" --fit-all --predict-test --steps 6000 --seed 62 --tag relay_psall6k_fit_s62 ;;
esac'
