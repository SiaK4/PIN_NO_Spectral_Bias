## Slurm job example

#!/bin/bash
#SBATCH -J "Specify job name"
#SBATCH --time=3:00:00 "Specify time"
#SBATCH -N -N "Specify Node number"
#SBATCH -n "Specify CPU number"
#SBATCH --mem= "Specify Memory"
#SBATCH "Specify GPU"
#SBATCH --gres=gpu:1 "Specify GPU number"
#SBATCH -o PIKAN_%j.out

source "Specfiy Pytorch environment (if needed)"

python -u PIKAN.py train.adam_epochs=100000 train.learning_rate=1e-3 train.checkpoint_name="PIKAN" train.checkpoint_interval=1000 train.epoch_resample=25000 \
        model.degree=7 model.num_layer=4 model.width_layer=64  model.skip_con=False weights.w1_org=1.0 weights.w2_org=25.0 train.double_precision=True  train.L1_epoch=200000 \
        model.seed=42

