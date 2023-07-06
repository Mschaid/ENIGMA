#!/bin/bash
#SBATCH -A p31961
#SBATCH -p gengpu
#SBATCH --gres=gpu:a100:2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH --mem=100G
source activate enigma
python /projects/p31961/ENIGMA/src/modeling/experiments/binned_trial_experiment_01.py