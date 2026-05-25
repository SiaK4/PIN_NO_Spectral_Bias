import os
import sys
from don import DeepONet

import math
import time
import datetime
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
# from YourDataset import YourDataset  # Import your custom dataset here
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torchinfo import summary
import torchprofile
import matplotlib.pyplot as plt

import json, time, tempfile

from soap import SOAP

import pickle

torch.manual_seed(23)

scaler = GradScaler()

DTYPE = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

CMAP='gray'



os.makedirs("logs", exist_ok=True)
run_name = f"run"
json_path = f"logs/{run_name}.json"

metrics = {
    "epoch": [], "train_loss": [], "val_loss": [],
    "spec_err": [], "lr": []
}

def save_json_atomic(obj, path):
    dir_ = os.path.dirname(path)
    with tempfile.NamedTemporaryFile("w", dir=dir_, delete=False) as tmp:
        json.dump(obj, tmp)        # or indent=2 for readability
        tmp.flush(); os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)     # atomic on POSIX


# Define your custom loss function here
class CustomLoss(nn.Module):
    def __init__(self, Par):
        super(CustomLoss, self).__init__()
        self.Par = Par

    def forward(self, y_pred, y_true):
        # Implement your custom loss calculation here
        # loss = torch.mean((y_pred - y_true) ** 2)  # Example: Mean Squared Error
        y_true = (y_true - self.Par["out_shift"])/self.Par["out_scale"]
        y_pred = (y_pred - self.Par["out_shift"])/self.Par["out_scale"]
        loss = torch.norm(y_true-y_pred, p=2)/torch.norm(y_true, p=2)
        return loss

class YourDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_sample = self.x[idx]
        y_sample = self.y[idx]

        if self.transform:
            x_sample, y_sample = self.transform(x_sample, y_sample)

        return x_sample, y_sample


def preprocess(traj_i, traj_o, Par):
    x = traj_i.transpose(1,0,2,3) #sliding_window_view(traj_i[:,:,:,:], window_shape=Par['lf'], axis=1 ).transpose(0,1,4,2,3).reshape(-1,Par['lf'],Par['nx'], Par['ny'])[:, [0,-1]] # BS, 2, nx, ny
    y = traj_o.transpose(1,0,2,3) #sliding_window_view(traj_o[:,:,:,:], window_shape=Par['lf'], axis=1 ).transpose(0,1,4,2,3).reshape(-1,Par['lf'],Par['nx'], Par['ny'])            # BS, lf, nx, ny
    
    print('x: ', x.shape)
    print('y: ', y.shape)
    print()
    return x,y

def combined_scheduler(optimizer, total_epochs, warmup_epochs, last_epoch=-1):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

# def make_images(true, pred, epoch):
#     # T,P - bs, nt, nx, ny
#     sample_id = 0
#     t_id = 0

#     CMAP = "gray"
#     VMIN = 0
#     VMAX = 255


#     T = true[sample_id, t_id].detach().cpu().numpy()*256
#     P = pred[sample_id, t_id].detach().cpu().numpy()*256

#     fig, axes = plt.subplots(1,2, figsize=(20,5))
#     axes[0].imshow(T, cmap=CMAP, vmin=VMIN, vmax=VMAX)
#     axes[0].set_title("True")
#     axes[1].imshow(P, cmap=CMAP, vmin=VMIN, vmax=VMAX)
#     axes[1].set_title("Pred")

#     plt.tight_layout()


#     fig.suptitle(f"Epoch: {epoch}", fontsize=22, y=1.2)
#     plt.savefig(f"images/{epoch}.png", dpi=150, bbox_inches='tight')
#     plt.close()


def compute_power(true, pred, inp):
    BS, nt, nx, ny = true.shape
    
    # Compute the Fourier transforms and amplitude squared for both true and pred
    fourier_true = torch.fft.fftn(true, dim=(-2, -1))
    fourier_pred = torch.fft.fftn(pred, dim=(-2, -1))
    fourier_inp  = torch.fft.fftn(inp , dim=(-2, -1))

    # Get the squared amplitudes
    amplitudes_true = torch.abs(fourier_true) #** 2
    amplitudes_pred = torch.abs(fourier_pred) #** 2
    amplitudes_inp = torch.abs(fourier_inp) #** 2

    # Create the k-frequency grids
    kfreq_y = torch.fft.fftfreq(ny) * ny
    kfreq_x = torch.fft.fftfreq(nx) * nx
    kfreq2D_x, kfreq2D_y = torch.meshgrid(kfreq_x, kfreq_y, indexing='ij')
    
    # Compute the wavenumber grid
    knrm = torch.sqrt(kfreq2D_x ** 2 + kfreq2D_y ** 2).to(true.device)
    
    # Define the bins for the wavenumber
    kbins = torch.arange(0.5, nx // 2 + 1, 1.0, device=true.device)
    
    # Digitize knrm to bin indices
    knrm_flat = knrm.flatten()
    bin_indices = torch.bucketize(knrm_flat, kbins)

    # Reshape and flatten the amplitudes
    amplitudes_true_flat = amplitudes_true.view(BS, nt, nx * ny)
    amplitudes_pred_flat = amplitudes_pred.view(BS, nt, nx * ny)
    amplitudes_inp_flat  = amplitudes_inp.view(BS, nt, nx * ny)

    # Initialize Abins
    Abins_true = torch.zeros((BS, nt, len(kbins) - 1), device=true.device)
    Abins_pred = torch.zeros((BS, nt, len(kbins) - 1), device=pred.device)
    Abins_inp  = torch.zeros((BS, nt, len(kbins) - 1), device= inp.device)

    # Vectorized binning: sum up the values in each bin
    for bin_idx in range(1, len(kbins)):
        mask = (bin_indices == bin_idx).unsqueeze(0).unsqueeze(0)  # Create a mask for each bin
        Abins_true[:, :, bin_idx - 1] = (amplitudes_true_flat * mask).sum(dim=-1) / mask.sum(dim=-1)
        Abins_pred[:, :, bin_idx - 1] = (amplitudes_pred_flat * mask).sum(dim=-1) / mask.sum(dim=-1)
        Abins_inp[:,  :, bin_idx - 1] = (amplitudes_inp_flat  * mask).sum(dim=-1) / mask.sum(dim=-1)

    # Scale the binned amplitudes
    scaling_factor = torch.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
    Abins_true *= scaling_factor
    Abins_pred *= scaling_factor
    Abins_inp  *= scaling_factor

    return Abins_true, Abins_pred, Abins_inp

def plot_power_spectrum(power_inp, power_true, power_pred, inp, true, pred, epoch, err):
    f = 2
    fig, axes = plt.subplots(1, 4, figsize=(4*f, 1*f))
    # t_ls = np.arange(power_true.shape[1])
    # skip_t = 12
    # time_ls = t_ls[::skip_t][1:]

    sample_id=-2
    t_id = 0
    for i in range(1):
        x = torch.arange(true.shape[-2]//2)
        axes[0].loglog(x, power_true[sample_id,t_id], label='true', c='black')
        axes[0].loglog(x, power_inp[sample_id,t_id], label='NO', c='blue')
        # axes[0].loglog(x, power_pred[sample_id,t_id], label='adv. NO', c='red')
        # axes[i].set_title(f"t: {0}")
        axes[0].set_xlabel(r'$k$')
        if i==0:
            axes[0].legend()
        if i==0:
            axes[0].set_ylabel(r'$P(k)$')
    

    inp_sample = inp[sample_id, t_id]
    true_sample = true[sample_id, t_id]
    pred_sample = pred[sample_id, t_id]
    vmin, vmax = true_sample.min(), true_sample.max()
    im1 = axes[1].imshow(true_sample, vmin=vmin, vmax=vmax, cmap=CMAP)
    axes[1].set_title("True")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    im = axes[2].imshow(inp_sample, vmin=vmin, vmax=vmax, cmap=CMAP)
    axes[2].set_title("NO")
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    im = axes[3].imshow(pred_sample, vmin=vmin, vmax=vmax, cmap=CMAP)
    axes[3].set_title("adv NO")
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    fig.colorbar(im1, ax=axes[3])
    plt.tight_layout()


    fig.suptitle(f"Epoch: {epoch}, MSE: {err:.2e}", fontsize=22, y=1.2)
    plt.savefig(f"images/{epoch}.png", dpi=150, bbox_inches='tight')
    plt.close()

def error_metric(inp, pred,true, epoch, Par, is_plot=True):
    #re-normalize
    # true = true*Par['out_scale'] + Par['out_shift']
    # true = true*Par['out_scale'] + Par['out_shift']

    # inp = inp*Par['inp_scale'] + Par['inp_shift']
    # inp = (inp - Par['out_shift'])/Par['out_scale']

    power_inp, power_true, power_pred = compute_power(inp, true, pred)
    err = torch.mean( (torch.log(power_true)-torch.log(power_pred) )**2 )
    f_err = torch.norm(true-pred, p=2)/torch.norm(true, p=2)
    ref_err = torch.norm(true-inp, p=2)/torch.norm(true, p=2)
    if is_plot:
        plot_power_spectrum(power_inp.detach().cpu().numpy(), power_true.detach().cpu().numpy(), power_pred.detach().cpu().numpy(), inp.detach().cpu().numpy(), true.detach().cpu().numpy(), pred.detach().cpu().numpy(), epoch, err)
    return err, f_err, ref_err



# Load your data into NumPy arrays (x_train, t_train, y_train, x_val, t_val, y_val, x_test, t_test, y_test)
#########################
begin_time = time.time()
traj_i = np.load(f"../data/lr8_data.npy").astype(np.float32)/256 #[nt, nx, ny]
traj_i = np.expand_dims(traj_i, axis=0) #[1, nt, nx, ny]
traj_o = np.load(f"../data/hr_data.npy").astype(np.float32)/256 #[nt, nx, ny]
traj_o = np.expand_dims(traj_o, axis=0) #[1, nt, nx, ny]

print(f"traj_i: {traj_i.shape}")
print(f"traj_o: {traj_o.shape}")

print(f"Data Loading Time: {time.time() - begin_time:.1f}s")


traj_i_train = traj_i[:, :800]
traj_i_val   = traj_i[:, 800:900]
traj_i_test  = traj_i[:, 900:]

traj_o_train = traj_o[:, :800]
traj_o_val   = traj_o[:, 800:900]
traj_o_test  = traj_o[:, 900:]

Par = {}
# Par['nt'] = 100 
Par['nx'] = traj_i_train.shape[2]
Par['ny'] = traj_i_train.shape[3]
Par['nf'] = 1
Par['d_emb'] = 128

Par['lb'] = 1
Par['lf'] = 1

Par["ld"] = 512
Par["n_channels"] = 16
Par["k"] = 3
Par["DEVICE"] = device
Par["DTYPE"] = DTYPE

# Par['temp'] = Par['nt'] - Par['lb'] - Par['lf'] + 2

Par['num_epochs'] = 20000 #500 #50

begin_time = time.time()
print('\nTrain Dataset')
x_train, y_train = preprocess(traj_i_train, traj_o_train, Par)
print('\nValidation Dataset')
x_val, y_val  = preprocess(traj_i_val, traj_o_val, Par)
print('\nTest Dataset')
x_test, y_test  = preprocess(traj_i_test, traj_o_test, Par)
print(f"Data Preprocess Time: {time.time() - begin_time:.1f}s")

# sys.exit()


Par['inp_scale'] = np.max(x_train) - np.min(x_train)
Par['inp_shift'] = np.min(x_train)
Par['out_scale'] = np.max(y_train) - np.min(y_train)
Par['out_shift'] = np.min(y_train)

print(f"Par:\n{Par}")

with open('Par.pkl', 'wb') as f:
    pickle.dump(Par, f)

# sys.exit()
#########################

# Create custom datasets
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

x_val_tensor   = torch.tensor(x_val,   dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val,   dtype=torch.float32)

x_test_tensor  = torch.tensor(x_test,  dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test,  dtype=torch.float32)

train_dataset = YourDataset(x_train_tensor, y_train_tensor)
val_dataset = YourDataset(x_val_tensor, y_val_tensor)
test_dataset = YourDataset(x_test_tensor, y_test_tensor)

# Define data loaders
train_batch_size = 20
val_batch_size   = 20
test_batch_size  = 20
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size)



# Initialize your Unet2D model
model = DeepONet(Par).to(device).to(torch.float32) 

dummy_x = x_train_tensor[0:1].to(device)
print(f"dummy_x: {dummy_x.shape}")



dummy_y = model(dummy_x)
print(f"dummy_y: {dummy_y.shape}")



summary(model, input_size=(1,)+x_train.shape[1:] )



# Adjust the dimensions as per your model's input size
dummy_x = x_train_tensor[0:1].to(device)
dummy_input = dummy_x

# Profile the model
# flops = torchprofile.profile_macs(model, dummy_input)
# print(f"FLOPs: {flops:.2e}")


# Define loss function and optimizer
criterion = CustomLoss(Par)
# optimizer = optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
optimizer = SOAP(model.parameters(),  lr = 3e-3, betas=(.95, .95), weight_decay=.01, precondition_frequency=10)

# Learning rate scheduler (Cosine Annealing)
scheduler = CosineAnnealingLR(optimizer, T_max= Par['num_epochs'] * len(train_loader) )  # Adjust T_max as needed
# scheduler = combined_scheduler(optimizer, Par['num_epochs'] * len(train_loader), int(0.1 * Par['num_epochs']) * len(train_loader))


# Training loop
num_epochs = Par['num_epochs']
best_val_loss = float('inf')
best_model_id = 0

os.makedirs('models', exist_ok=True)
os.makedirs('images', exist_ok=True)



t0 = time.time()
for epoch in range(num_epochs):
    begin_time = time.time()
    model.train()
    train_loss = 0.0

    for x, y_true in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        optimizer.zero_grad()
        if True:
            y_pred = model(x.to(device))
            loss   = criterion(y_pred, y_true.to(device))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

        # Update learning rate
        scheduler.step()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y_true in val_loader:
            if True:
                y_pred = model(x.to(device))
                loss   = criterion(y_pred, y_true.to(device))
            val_loss += loss.item()

    val_loss /= len(val_loader)

        # Save the model if validation loss is the lowest so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_id = epoch+1
        torch.save(model.state_dict(), f'models/best_model.pt')

    if epoch+1 % 100 == 0:
        torch.save(model.state_dict(), f'models/model_{epoch+1}.pt')

    # make_images(y_true, y_pred, epoch)
    s_err, f_err, r_err = error_metric(y_pred, y_pred, y_true.to(device), epoch+1, Par, is_plot=True)
    
    time_stamp = str('[')+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+str(']')
    elapsed_time = time.time() - begin_time
    print(time_stamp + f' - Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}, Spec Err: {s_err:.4e}, best model: {best_model_id}, LR: {scheduler.get_last_lr()[0]:.4e}, epoch time: {elapsed_time:.2f}')

    # inside your epoch loop, after computing metrics:
    metrics["epoch"].append(epoch + 1)
    metrics["train_loss"].append(float(train_loss))
    metrics["val_loss"].append(float(val_loss))
    metrics["spec_err"].append(float(s_err))
    metrics["lr"].append(float(scheduler.get_last_lr()[0]))

    save_json_atomic(metrics, json_path)

print('Training finished.')
print(f"Training Time: {time.time() - t0:.1f}s")

# Testing loop
model.eval()
test_loss = 0.0
with torch.no_grad():
    for x, y_true in test_loader:
        if True:
            y_pred = model(x.to(device))
            loss = criterion(y_pred, y_true.to(device))
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4e}')

