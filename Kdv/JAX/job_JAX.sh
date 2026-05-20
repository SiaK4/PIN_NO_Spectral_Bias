#!/bin/bash
# SBATCH -J pikdv_jax
# SBATCH ## Time
# SBATCH ## Node
# SBATCH ## CPU
# SBATCH ## Memory
# SBATCH ## GPU
# SBATCH ## GPU number
# SBATCH -o PINN_jax_%j.out

# source "Directory to Pytorch environment (if needed)"

python -u PINN_JAX.py train.adam_epochs=20000 train.learning_rate=1e-3 train.checkpoint_name="PINN_JAX_SIREN" train.checkpoint_interval=5000 train.epoch_resample=25000 model.features="[50,50,50,50,50,50]" \
        model.skip_con=False weights.w1_org=1.0 weights.w2_org=25.0 train.double_precision=True  train.L1_epoch=200000 model.use_siren=True model.siren_w0=30.0 model.learning_w0=False \
        train.h_siren=False model.activation="tanh" train.switch_epoch=1000 train.seed=10

