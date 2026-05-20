import torch
import argparse
import os
import json

def get_config():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model settings
    model_config = {
        "in_c": 3,
        "out_c": 1,
        "degree": 5,
        "num_layer": 5,
        "width_layer": 64, 
        "x_left": -1.0,
        "x_right": 1.0,
    }

    # Training settings
    train_config = {
        "learning_rate": 4e-4,
        "adam_epochs": 10000,
        "epoch_resample": 10000,
        "L1_epoch": 30000,
        "display_every": 20,
        "checkpoint_interval": 200,
        "checkpoint_name": "PIKAN",
        "lambda_pde_avg": 1.0,
        "lambda_init_avg": 1.0,
        "lambda_SR_avg": 1.0,
        "grad_norm_interval": 500,
        "beta": 0.9,
        "double_precision": False,
        "slope_R_w": 1.0,
        "seed": 1234,
        "batch_ratio": 0.1,
        "switch_epoch": 1000

    }

    # Data config
    data_config = {
        "wave_data_dir": "/users/skhodaka/data/skhodaka/PIKAN_PINN_other/Kdv/data/data1/",
        "domain_data_file": "domain_data.npy",
        "init_data_file": "initial_data.npy"  
    }

    # Weights schedule
    weights_schedule = {
        "w1_org": 1.0,
        "w2_org": 100.0
    }

    config = {
        "device": device,
        "model": model_config,
        "train": train_config,
        "data": data_config,
        "weights": weights_schedule
    }

    return config


def update_config_from_cli(config):
    parser = argparse.ArgumentParser()
    
    # Catch all unknown arguments like --train.learning_rate 2e-4
    parser.add_argument('overrides', nargs='*', help='Config overrides in key=value format')
    args = parser.parse_args()
    
    for override in args.overrides:
        if '=' not in override:
            raise ValueError(f"Invalid override: {override}. Use key=value format.")
        key_path, value = override.split('=', 1)
        keys = key_path.split('.')
        
        # Convert to int/float/list if possible
        try:
            value_eval = eval(value)
        except:
            value_eval = value
        
        # Update nested dict
        d = config
        for k in keys[:-1]:
            if k not in d:
                raise KeyError(f"Config key '{k}' not found")
            d = d[k]
        d[keys[-1]] = value_eval
    
    return config

def save_config(config, save_dir, filename="config.json"):
    os.makedirs(save_dir, exist_ok=True)

    # Convert any non-JSON-serializable objects
    def make_serializable(obj):
        if isinstance(obj, torch.device):
            return str(obj)
        if isinstance(obj, (set, tuple)):
            return list(obj)
        return obj

    serializable_config = {
        k: {kk: make_serializable(vv) for kk, vv in v.items()} if isinstance(v, dict)
        else make_serializable(v)
        for k, v in config.items()
    }

    with open(os.path.join(save_dir, filename), "w") as f:
        json.dump(serializable_config, f, indent=4)

    print(f"✅ Saved config to {os.path.join(save_dir, filename)}")