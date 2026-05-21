python -u PIKAN.py train.adam_epochs=100000 train.learning_rate=5e-4 train.checkpoint_name="PIKAN" train.checkpoint_interval=1000 train.epoch_resample=5000 \
        model.skip_con=False train.double_precision=True data.wave_data_dir="data/data4/" train.grad_norm_interval=500 \
        weights.w1_org=1.0 weights.w2_org=25.0 weights.w3_org=25.0 weights.w4_org=25.0 train.L1_epoch=30000 model.degree=5 model.num_layer=6 model.width_layer=13 \
        train.switch_epoch=1000
