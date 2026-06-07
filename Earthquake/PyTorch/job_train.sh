#!/bin/bash
# Slurm job example for one configuration/seed.

#SBATCH --job-name=eq_no
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=train_%j.out

# 32 configurations, dispatched by --index 0..31 (see: python train.py --help).
python -u train.py --index 0 --seed 42 --checkpoint-suffix _seed42 --device cuda
