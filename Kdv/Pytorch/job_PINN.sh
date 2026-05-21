## Slurm job example

#!/bin/bash
#SBATCH -J pikdv
#SBATCH --time=1:00:00
#SBATCH -N "Specify Node number"
#SBATCH -n "Specify CPU number"
#SBATCH --mem= "Specify Memory"
#SBATCH -"Specify GPU"
#SBATCH --gres=gpu:1 "Specify GPU number"
#SBATCH -o PINN_%j.out

source "Specfiy Pytorch environment (if needed)"

python -u PINN.py train.adam_epochs=100000 train.learning_rate=1e-3 train.checkpoint_name="PINN" train.checkpoint_interval=1000 train.epoch_resample=25000 model.features="[100,100,100,100,100,100]" \
        model.skip_con=False weights.w1_org=1.0 weights.w2_org=25.0 train.double_precision=True  train.L1_epoch=200000 model.use_siren=True model.siren_w0=30.0 model.learning_w0=False \
        train.h_siren=False model.activation="tanh" train.slope_R_w=0.01 rba.use_rba=False rba.rba_par.method="polynomial" rba.rba_par.order=1 rba.rba_par.gamma=0.99 rba.rba_par.eta=0.05 train.seed=100

