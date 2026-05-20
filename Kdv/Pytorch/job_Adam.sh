## Slurm job example

#!/bin/bash
#SBATCH -J "Specify job name"
#SBATCH --time=3:00:00 "Specify time"
##SBATCH -N "Specify Node number"
##SBATCH -n "Specify CPU number"
##SBATCH --mem= "Specify Memory"
##SBATCH "Specify GPU"
##SBATCH --gres=gpu:1 "Specify GPU number"

#SBATCH -o PINN_%j.out

source "Specify pytorch environment (if needed)"

python -u PINN_Adam.py train.adam_epochs=100000 train.learning_rate=1e-3 train.checkpoint_name="PINN3_Adam_SIREN" train.checkpoint_interval=1000 train.epoch_resample=65000 model.features="[100,100,100,100,100,100]" \
        model.skip_con=False weights.w1_org=1.0 weights.w2_org=25.0 train.double_precision=True  train.L1_epoch=200000 model.use_siren=False model.siren_w0=30.0 train.batch_ratio=0.1 \
        model.activation="tanh" train.slope_R_w=0.0 model.learning_w0=False model.init_alpha=2.0 rba.use_rba=True rba.rba_par.method="polynomial" rba.rba_par.order=1 rba.rba_par.gamma=0.95 rba.rba_par.eta=0.05 rba.rba_par.cap=40 \
        train.seed=42
