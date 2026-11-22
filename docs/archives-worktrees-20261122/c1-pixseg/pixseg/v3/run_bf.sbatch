#!/bin/bash
# Un nœud, 4 GPU, une tâche par GPU. Depuis le dossier de la tâche, code copié dans work_pixseg_v3/ :
#   sbatch work_pixseg_v3/run.sbatch gen [n_par_methode]
#       GPU 0-2 : synthétique d'une méthode chacun (data.py gen) ; GPU 3 : CV sur le réel seul (référence)
#   sbatch work_pixseg_v3/run.sbatch train work_pixseg_v3/configs_cv.txt
#       la ligne k du fichier = arguments de train.py pour le GPU k (4 lignes au plus, ligne vide = GPU libre)
# Sorties : output/<job>.out, output/<job>_<tâche>.out, output/pixseg_*.
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:25:00
#SBATCH -J j1
#SBATCH --output=output/%j.out
set -euo pipefail
MODE=${1:?gen|train}; ARG=${2:-}
if [ "$MODE" = train ]; then [ -f "$ARG" ] || { echo "fichier de configs absent : $ARG"; exit 2; }; fi
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /p/scratch/training2666/mouhali1/SLZ/Hackathon_Watermark_Localisation_Task
source .venv/bin/activate
export MODE ARG
srun --ntasks=4 --kill-on-bad-exit=0 --cpu-bind=cores --output="output/%j_%t.out" bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID      # 1 GPU par tâche (pas d isolation cgroups)
  python -c "import torch,sys; n=torch.cuda.device_count(); sys.exit(0 if n==1 else f\"{n} GPU visibles\")" || exit 3
  echo "tâche $SLURM_LOCALID sur GPU $CUDA_VISIBLE_DEVICES ($(hostname)), mode $MODE"
  if [ "$MODE" = gen ]; then
    METHODS=(pixelseal trustmark stegastamp)
    if [ "$SLURM_LOCALID" -lt 3 ]; then
      exec python -u work_pixseg_v3/data.py gen --method "${METHODS[$SLURM_LOCALID]}" --n "${ARG:-3000}" --out output/pixseg_syn
    fi
    REF=""; [ -f output/heats_train.npz ] && REF="--ref-heats output/heats_train.npz"
    exec python -u work_pixseg_v3/train.py --cv --real-frac 1 $REF
  fi
  LINE=$(sed -n "$(( SLURM_LOCALID + 1 ))p" "$ARG")
  [ -n "$LINE" ] || { echo "pas de config pour cette tâche"; exit 0; }
  echo "config : $LINE"
  exec python -u work_pixseg_v3/train.py $LINE'
