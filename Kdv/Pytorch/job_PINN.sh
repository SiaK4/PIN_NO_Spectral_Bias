#!/bin/bash
#SBATCH -J pikdv_soap_SIREN
#SBATCH --time=2:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=32g
## SBATCH -A gk-h100-gcondo
## SBATCH -p a6000-gcondo
#SBATCH -A gk-l40s-gcondo
#SBATCH -p l40s-gcondo
#SBATCH --gres=gpu:1
#SBATCH -o PINN_Soap_SIREN_%j.out
#SBATCH -e PINN3_Soap_SIREN_%j.err

source /users/skhodaka/gpu_env.venv/bin/activate

python -u PINN.py train.adam_epochs=100000 train.learning_rate=1e-3 train.checkpoint_name="PINN_soap_SIREN" train.checkpoint_interval=1000 train.epoch_resample=25000 model.features="[100,100,100,100,100,100]" \
        model.skip_con=False weights.w1_org=1.0 weights.w2_org=25.0 train.double_precision=True  train.L1_epoch=200000 model.use_siren=True model.siren_w0=30.0 model.learning_w0=False \
        train.h_siren=False model.activation="tanh" train.slope_R_w=0.01 rba.use_rba=False rba.rba_par.method="polynomial" rba.rba_par.order=1 rba.rba_par.gamma=0.99 rba.rba_par.eta=0.05 train.seed=100

