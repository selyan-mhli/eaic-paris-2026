#!/bin/bash
# Étape lancée dans une allocation j1s existante (srun --overlap), 4 tâches = 4 GPU.
# Usage : srun --jobid=<id> --overlap -N1 -n4 --gres=gpu:4 bash work_seg/step_f.sh <script> [args]
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
cd /p/scratch/training2666/mouhali1/SLZ/Hackathon_Watermark_Localisation_Task
source .venv/bin/activate
python "$@"
