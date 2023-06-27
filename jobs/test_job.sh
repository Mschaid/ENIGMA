#!/bin/bash
#SBATCH -A p31961
#SBATCH -p gengpu
#SBATCH --gres=gpu:a100:2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 1:00:00
#SBATCH --mem=20G
conda activate enigma
python /projects/p31961/ENIGMA/scripts/plt.py