#!/bin/bash
#SBATCH -J KdvPikan
#SBATCH --time=12:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32g
#SBATCH -A gk-h100-gcondo
#SBATCH -p h100-gcondo
#SBATCH --gres=gpu:1
#SBATCH -o PIKAN_%j.out
#SBATCH -e PIKAN_%j.err

source /users/skhodaka/gpu_env.venv/bin/activate

python -u PIKAN.py train.adam_epochs=100000 train.learning_rate=1e-3 train.checkpoint_name="PIKAN" train.checkpoint_interval=1000 train.epoch_resample=25000 \
        model.degree=7 model.num_layer=4 model.width_layer=64  model.skip_con=False weights.w1_org=1.0 weights.w2_org=25.0 train.double_precision=True  train.L1_epoch=200000 \
        model.seed=1

