#!/bin/bash
# PROPOSÉ, JAMAIS LANCÉ. Pilote unique du localisateur dense PS/TM F, un nœud, 4 GPU, UN job.
# Depuis le dossier de la tâche, après copie de c1/pixseg/ (avec dense/) dans work_pixd/ :
#   sbatch --test-only work_pixd/dense/pilot.sbatch
#   sbatch work_pixd/dense/pilot.sbatch
# Étape A (génération, ~5-12 min estimées) : GPU0 PS graine 0, GPU1 PS graine 1, GPU2 TM graine 0, GPU3 TM graine 1,
#   1000 composites F tracés chacun (embedders du kit), sources train hors folds 0-1.
# Étape B (apprentissage) : GPU0 PS tête seule, GPU1 TM tête seule, GPU2 PS dernier stage, GPU3 TM dernier stage ;
#   chacun : contrôle -> dev (folds 2-4) -> confirmation (folds 0-1), recette fixée à l'avance.
# Sorties (une vingtaine de fichiers au total) : output/dense/syn_*.npz (8), output/dense/res_*.npz (4),
#   output/<job>.out, output/<job>_A<t>.out, output/<job>_B<t>.out. Aucune écriture par échantillon.
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:50:00
#SBATCH -J j1
#SBATCH --output=output/%j.out
set -euo pipefail
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /p/scratch/training2666/mouhali1/SLZ/Hackathon_Watermark_Localisation_Task
source .venv/bin/activate
mkdir -p output/dense
N=${N_SYN:-1000}
export N

# A. synthétique tracé ; une tâche par GPU
srun --ntasks=4 --kill-on-bad-exit=1 --cpu-bind=cores --output="output/%j_A%t.out" bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  M=(pixelseal pixelseal trustmark trustmark); SEED=(0 1 0 1)
  exec python -u work_pixd/dense/synth_msg.py --method ${M[$SLURM_LOCALID]} --seed ${SEED[$SLURM_LOCALID]} \
    --n "$N" --shard 500 --minutes 12 --out output/dense/syn --device cuda:0'

# contrôle : 4 x 2 tranches attendues, sinon arrêt avant l apprentissage
NF=$(ls output/dense/syn_*_s*_*.npz | wc -l)
[ "$NF" -ge 4 ] || { echo "synthétique incomplet : $NF tranches"; exit 4; }

# B. apprentissage ; une configuration par GPU
srun --ntasks=4 --kill-on-bad-exit=0 --cpu-bind=cores --output="output/%j_B%t.out" bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  M=(pixelseal trustmark pixelseal trustmark); MODE=(head head last last)
  m=${M[$SLURM_LOCALID]}; mode=${MODE[$SLURM_LOCALID]}
  exec python -u work_pixd/dense/dense_train.py --method $m --mode $mode \
    --synth "output/dense/syn_${m}_s*_*.npz" --epochs 20 --bs 32 --real-frac 0.25 \
    --out output/dense/res_${m}_${mode}_${SLURM_JOB_ID} --device cuda'
ls -la output/dense/
