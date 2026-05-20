import numpy as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
import json
import time
import torch.nn.functional as F
import argparse
from Model.NN1 import NN_Periodic

from Model.NN1 import xavier_init_weights
from Model.save_model import save_checkpoint
from Optimizer.soap_double import SOAP
from config import get_config, update_config_from_cli, save_config
import argparse
import random

## Load config
config = get_config()
config = update_config_from_cli(config)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

seed = config["train"]["seed"]
set_seed(seed)

device = config["device"]
model_cfg = config["model"]
train_cfg = config["train"]
data_cfg = config["data"]
weights_cfg = config["weights"]

save_dir = "checkpoints_" + config['train']['checkpoint_name']
save_config(config, save_dir)
slope_R_w = config["train"]["slope_R_w"]

def w_schedule(w1_org,w2_org, epoch):
    if epoch <=2000:
        w1, w2 = w1_org, w2_org
    if epoch > 2000 and epoch <=5000:
         w1, w2 = w1_org, w2_org/2
    if epoch > 5000 and epoch <= 10000:
        w1, w2 = w1_org, w2_org/10
    if epoch > 10000:
        w1, w2 = w1_org, w2_org/25
    return w1,w2

def resample_points(points, residuals, num_samples, epoch, epoch_resample):
    residuals = torch.abs(residuals.squeeze())

    temperature = max(0.2, 1.0 * (0.9 ** (epoch-epoch_resample)))
    alpha = 0.8  # how much weight to give residual-based probs
    residual_probs = torch.softmax(residuals / temperature, dim=0)
    uniform_probs = torch.ones_like(residual_probs) / residual_probs.numel()
    probabilities = alpha * residual_probs + (1 - alpha) * uniform_probs
    probabilities = probabilities / probabilities.sum()

    sampled_indices = np.random.choice(points.shape[0], size=num_samples, replace=False, p=probabilities.detach().cpu().numpy())
    return sampled_indices

def KDV(model,X):
    x = X[:,0:1]
    t = X[:,1:2]

    u = model(torch.concatenate((x,t),dim=1))

    u_t = 2* torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    KDV = u_t + u*u_x + (0.022**2)*u_xxx
    
    return KDV

def train_step(model, domain_data, init_data, epoch_resample, epoch):
    ## Residual inside the domain
    domain_residual = KDV(model, domain_data)

    ## Initial condition
    init_data_input = init_data[:,[0,1]]
    init_data_output = init_data[:,-1].detach()
    init_data_pred = model(init_data_input)
    init_data_residual = init_data_output.squeeze() - init_data_pred.squeeze()

    return domain_residual.squeeze(), init_data_residual

def random_sampling(data, ratio):
    n_samples = int(ratio * data.shape[0])
    indices = torch.randperm(data.shape[0])[:n_samples]
    sampled_data = data[indices]
    return sampled_data

def grad_norm(loss, model):
    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
    total_norm = torch.sqrt(sum(torch.sum(g**2) for g in grads))
    return total_norm.detach()

def train(model, epoch_number, learning_rate, domain_data, init_data, epoch_resample,L1_epoch, double=False):
    lambda_pde_avg = train_cfg["lambda_pde_avg"]
    lambda_init_avg = train_cfg["lambda_init_avg"]
    grad_norm_interval = train_cfg["grad_norm_interval"]
    beta = train_cfg["beta"]

    best_loss = float('inf')
    domain_loss_list = []
    init_loss_list = []
    total_loss_list = []
    relative_error_list = []

    w1_org = weights_cfg["w1_org"]
    w2_org = weights_cfg["w2_org"]

    ########## Double precision change ###########
    dtype = torch.float64 if double else torch.float32

    model = model.to(device=device, dtype=dtype) ### model to double precision
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.9)

    domain_data = torch.tensor(domain_data, requires_grad=True, device=device, dtype=dtype)
    init_data = torch.tensor(init_data, device=device, dtype=dtype)
    
    for epoch in range(epoch_number):
        start_epoch = time.time()

        if epoch == 10:
            optimizer = SOAP(params=model.parameters(), lr=learning_rate,
                             betas=(0.95, 0.95), weight_decay=0.0, precondition_frequency=10)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
            print(">>> Switched optimizer: Adam → SOAP at epoch 10")

        if epoch <= epoch_resample:
            domain_data_sample = random_sampling(domain_data, ratio=0.1)
        else:
            domain_data_sample = domain_data
        # Get the residuals
        domain_residual, init_data_residual = train_step(model, domain_data_sample, init_data, epoch_resample=epoch_resample, epoch=epoch)
        # Calculate the losses
        domain_samples = np.arange(domain_data_sample.shape[0])
        init_samples = np.arange(init_data.shape[0])
        
        if epoch > epoch_resample:
            domain_samples = resample_points(domain_data_sample, domain_residual, domain_data_sample.shape[0]//10, epoch, epoch_resample)
            init_samples = resample_points(init_data, init_data_residual, init_data.shape[0]//1, epoch, epoch_resample)
        
        if epoch< L1_epoch:
            domain_loss = torch.mean((domain_residual[domain_samples])**2)
            init_loss = torch.mean((init_data_residual[init_samples])**2)

        else:
            domain_loss = torch.mean(torch.abs(domain_residual[domain_samples]))
            init_loss = torch.mean(torch.abs(init_data_residual[init_samples]))

        # Optimizer step
        optimizer.zero_grad()
        w1, w2 = w_schedule(w1_org,w2_org, epoch)

        if (epoch+1) % grad_norm_interval == 0:
            with torch.no_grad():
                grad_pde = grad_norm(domain_loss, model)
                grad_init = grad_norm(init_loss, model)
            lambda_pde = (grad_pde + grad_init)/(grad_pde + 1e-9)
            lambda_init = (grad_pde + grad_init)/(grad_init + 1e-9)
            lambda_pde_avg = beta * lambda_pde_avg + (1-beta) * lambda_pde
            lambda_init_avg = beta * lambda_init_avg + (1-beta) * lambda_init

        # loss = w1*lambda_pde_avg*domain_loss + w2*lambda_init_avg*init_loss
        print("lambda_pde_avg:", lambda_pde_avg)
        print("lambda_init_avg", lambda_init_avg)

        ### slope recovery term
        alpha_loss = model.get_alpha_loss()
        print("alpha_loss:",alpha_loss)
        
        loss = w1*lambda_pde_avg*domain_loss + w2*lambda_init_avg*init_loss + slope_R_w*alpha_loss
        loss_noWeight = domain_loss + init_loss

        loss.backward()
        optimizer.step()

        ### Get the unweighted losses to save 
        domain_loss_nw = torch.mean(domain_residual**2)
        init_loss_nw = torch.mean(init_data_residual**2)
        total_loss_nw = domain_loss_nw  + init_loss_nw 
        
        domain_loss_list.append(np.mean(domain_loss_nw.item()))
        init_loss_list.append(np.mean(init_loss_nw.item()))
        total_loss_list.append(np.mean(total_loss_nw.item()))
        
        if (epoch+1)<40000:
            scheduler.step()

        epoch_time = time.time() - start_epoch
        if ((epoch+1)%display_every == 0) or (epoch==0):
            print(f"Total loss: {total_loss_nw.detach()}, Domain loss: {domain_loss_nw.detach()}, \
            init loss: {init_loss_nw.detach()}, \
                at epoch {epoch+1}, w1:{w1}, w2:{w2}")
            
            print(" <><><><><><><> epoch time:", epoch_time)
        if (epoch+1)% checkpoint_interval == 0:
            best_loss = float('inf')
            best_loss = save_checkpoint(model, epoch, loss, best_loss, checkpoint_name=checkpoint_name)

        with torch.no_grad():
            pred = model(domain_data[:,[0,1]])
            rel_error = (torch.norm(pred.squeeze() - domain_data[:,-1].squeeze())/torch.norm(domain_data[:,-1].squeeze())).detach().item()
            relative_error_list.append(rel_error)

        if (epoch+1)% 100 ==0:
            with open(f'domain_loss_{checkpoint_name}.json','w') as file:
                json.dump(domain_loss_list, file)
            with open(f'Init_{checkpoint_name}.json', 'w') as file:
                json.dump(init_loss_list, file)
            with open(f'Total_{checkpoint_name}.json', 'w') as file:
                json.dump(total_loss_list, file) 
            with open(f'Rel_error_{checkpoint_name}.json', 'w') as file:
                json.dump(relative_error_list, file)

    return domain_loss_list, init_loss_list, total_loss_list, relative_error_list

def train_lbfgs(model, domain_data, init_data, \
        domain_loss_list, init_loss_list, total_loss_list, relative_error_list,\
        max_iter=500, epoch_adam=10000, save_every=10):

    w1_org = 1.0
    w2_org = 1.0

    ############## Double precision #################
    dtype = torch.float64 if double else torch.float32

    # Make sure all inputs are tensors on the right device
    domain_data = torch.tensor(domain_data, requires_grad=True, device=device, dtype=dtype)
    init_data = torch.tensor(init_data, device=device, dtype=dtype)
    model = model.to(device=device, dtype=dtype)

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=max_iter,
        max_eval=max_iter,
        history_size=50,
        tolerance_grad=1e-12,
        tolerance_change=1.0 * np.finfo(float).eps,
        line_search_fn="strong_wolfe"
    )

    iteration_counter = [0]
    def closure():
        optimizer.zero_grad()

        # Compute residuals
        domain_residual, init_data_residual = train_step(
            model, domain_data, init_data, epoch_resample=0, epoch=0
        )

        # Loss terms
        domain_loss = torch.mean(domain_residual**2)
        init_loss = torch.mean(init_data_residual**2)

        # Weighted total loss
        w1, w2 = w_schedule(w1_org, w2_org, epoch=epoch_adam+iteration_counter[0])
        loss = w1*domain_loss + w2*init_loss

        # Backward
        loss.backward()

        # Detach for logging
        domain_loss_list.append(domain_loss.detach().item())
        init_loss_list.append(init_loss.detach().item())
        total_loss_list.append(loss.detach().item())

        with torch.no_grad():
            pred = model(domain_data[:,[0,1]])
            rel_error = (torch.norm(pred - domain_data[:,-1])/torch.norm(domain_data[:,-1])).detach().item()
            relative_error_list.append(rel_error)

        iteration_counter[0] +=1
        if iteration_counter[0] % 1 == 0:
            # Print losses
            print(f"[L-BFGS iter {iteration_counter[0]}] Total: {loss.item():.6f}, "
                    f"Domain: {domain_loss.item():.6f},"
                    f"Init: {init_loss.item():.6f}")
        
        best_loss = float('inf')
        if iteration_counter[0] % save_every == 0:
            best_loss = save_checkpoint(model, epoch_adam+iteration_counter[0], loss.detach().item(), best_loss, checkpoint_name=checkpoint_name)

        with open(f'domain_loss_{checkpoint_name}.json','w') as file:
            json.dump(domain_loss_list, file)
        with open(f'Init_{checkpoint_name}.json', 'w') as file:
            json.dump(init_loss_list, file)
        with open(f'Total_{checkpoint_name}.json', 'w') as file:
            json.dump(total_loss_list, file)
        with open(f'Rel_error_{checkpoint_name}.json', 'w') as file:
            json.dump(relative_error_list, file)

        return loss

    # Run L-BFGS
    optimizer.step(closure)

    return domain_loss_list, init_loss_list, total_loss_list, relative_error_list

# Load data from config
wave_data_dir = data_cfg["wave_data_dir"]
domain_data = np.load(os.path.join(wave_data_dir, data_cfg["domain_data_file"]))
init_data = np.load(os.path.join(wave_data_dir, data_cfg["init_data_file"]))

## If sampling the domain collocation points
# idx = np.random.choice(domain_data.shape[0], size=domain_data.shape[0]//4, replace=False)
# domain_data = domain_data[idx]

print("domain_data.shape:",domain_data.shape)
print("init_data.shape:",init_data.shape)

display_every = train_cfg["display_every"]
checkpoint_interval = train_cfg["checkpoint_interval"]
checkpoint_name = train_cfg["checkpoint_name"]
lambda_pde_avg = 1.0
lambda_init_avg = 1.0
grad_norm_interval = 500
beta = 0.95


model = NN_Periodic(
    in_c = model_cfg["in_c"],
    out_c = model_cfg["out_c"],
    features = model_cfg["features"],
    activation_name = model_cfg["activation"],
    num_frequencies = model_cfg["num_frequencies"],
    fourier_type = model_cfg["fourier_type"],
    fourier_scale = model_cfg["fourier_scale"],
    use_siren = model_cfg["use_siren"],
    siren_w0 = model_cfg["siren_w0"],
    x_left = model_cfg["x_left"],
    x_right = model_cfg["x_right"],
    skip_con= model_cfg["skip_con"],
    h_siren= model_cfg["h_siren"],
    learning_w0=model_cfg["learning_w0"]
)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params}")

model.to(device)
model.apply(xavier_init_weights)

domain_loss_list, init_loss_list, total_loss_list, relative_error_list = train(
    model,
    epoch_number = train_cfg["adam_epochs"],
    learning_rate = train_cfg["learning_rate"],
    domain_data = domain_data,
    init_data = init_data,
    epoch_resample = train_cfg["epoch_resample"],
    L1_epoch = train_cfg["L1_epoch"],
    double = train_cfg["double_precision"]
)

domain_loss_list_lbfgs, init_loss_list_lbfgs, total_loss_list_lbfgs, relative_loss_list_lbfgs = train_lbfgs(
    model, 
    domain_data,  
    init_data, 
    domain_loss_list, init_loss_list, total_loss_list, relative_error_list,
    max_iter=20000,
    epoch_adam=train_cfg["adam_epochs"],
    save_every=100,
    double = train_cfg["double_precision"]
)