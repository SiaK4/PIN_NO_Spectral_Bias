# !/bin/bash
# SBATCH -J pikdv_jax
# SBATCH --time=2:00:00
# SBATCH -N 1
# SBATCH -n 4
# SBATCH --mem=32g
# SBATCH GPU name
# SBATCH --gres=gpu:1
# SBATCH -o PIKAN_jax_%j.out

# source "Pytorch environment (if needed)"

python -u PIKAN_JAX.py train.adam_epochs=100000 train.learning_rate=1e-3 train.checkpoint_name="PIKAN_JAX" train.checkpoint_interval=1000 train.epoch_resample=25000 \
        weights.w1_org=1.0 weights.w2_org=25.0 train.double_precision=True  train.L1_epoch=200000 \
        train.switch_epoch=1000 model.degree=5 model.num_layers=4 model.width_layer=23 model.in_c=3

