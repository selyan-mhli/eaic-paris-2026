#!/bin/bash
# 4 variantes de unlearn_v2.py en parallèle sur un nœud, une par A100.
# dc-gpu alloue toujours le nœud entier : un job --gres=gpu:1 laisse 3 GPU inutilisés.
# Usage (depuis /p/scratch/training2666/mouhali1/SLZ/unlearning) : sbatch unlearn_pack.sh
# Prérequis : unlearn_v2.py accepte --tag et l'ajoute aux noms de checkpoint
#   (sinon les 4 runs écrasent output/v2_entXX.pt).
# Essai court : --epochs 1 sur les 4 configurations. Le script d'origine n'avait été
# vérifié qu'avec bash -n.
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --job-name=j3p
#SBATCH --output=output/%j.out

cd /p/scratch/training2666/mouhali1/SLZ/unlearning || exit 1
unset PYTHONPATH
source .venv/bin/activate
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=8

# Un log par GPU : output/<jobid>_0.out à _3.out. srun renvoie un code non nul si un run échoue.
srun --kill-on-bad-exit=0 --cpu-bind=cores --output=output/%j_%t.out bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID   # les GPU ne sont pas isolés par Slurm
  CFG=(
    "--lr 1e-5 --epochs 1"
    "--lr 2e-5 --epochs 1"
    "--lr 5e-6 --epochs 1"
    "--lr 1e-5 --proxy-weight 4 --epochs 1"
  )
  python -c "import torch,sys; sys.exit(torch.cuda.device_count()!=1)" || exit 3
  echo "GPU $CUDA_VISIBLE_DEVICES ($(hostname)) : ${CFG[$SLURM_LOCALID]}"
  exec python -u unlearn_v2.py ${CFG[$SLURM_LOCALID]} --tag _g$SLURM_LOCALID'
