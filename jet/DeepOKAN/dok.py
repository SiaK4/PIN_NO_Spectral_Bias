import math
import torch
import torch.nn as nn
import numpy as np
import sys
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops import rearrange


import argparse
from typing import List

parser = argparse.ArgumentParser(description='Tunning_parameters')
parser.add_argument('--Run_type'   , type=str  , default='KAN_Benchmark')
parser.add_argument('--Mode'       , type=str  , default='PINN')
parser.add_argument('--Eqn'   , type=str  , default='HM')

parser.add_argument('--use_RBA'    , type=int , default=1)
parser.add_argument('--Mod_MLP'    , type=int , default=0)
parser.add_argument('--Adaptive_AF'    , type=int , default=0)
parser.add_argument('--Weight_Norm'    , type=int , default=0)
parser.add_argument('--resample' , type=int  , default=0)
parser.add_argument('--L2_reg'    , type=int , default=0)
parser.add_argument('--beta_L2' , type=float  , default=0.001)

#Resnet_params
parser.add_argument('--Use_ResNet'    , type=int , default=0)
parser.add_argument('--Adaptive'    , type=int , default=0)
parser.add_argument('--Light'    , type=int , default=0)

#KAN Params
parser.add_argument('--Network'    , type=str , default='KAN')
parser.add_argument('--degree'    , type=int , default=7)

parser.add_argument('--lr_lambdas_0' , type=float  , default=0.001)
parser.add_argument('--gamma' , type=float  , default=0.999)
parser.add_argument('--k_samp' , type=float  , default=0.75)
parser.add_argument('--c_samp' , type=float  , default=0.75)

parser.add_argument('--batch_size' , type=int  , default=500)
parser.add_argument('--g_steps' , type=int  , default=100000)
parser.add_argument('--num_layer' , type=int  , default=7)
parser.add_argument('--width_layer' , type=int  , default=64)

args, unknown = parser.parse_known_args()
for arg, value in vars(args).items():
    print(f'{arg}: {value}')


# Define Chebyshev polynomials
def T0(x): return torch.ones_like(x)
def T1(x): return x
def T2(x): return 2 * x**2 - 1
def T3(x): return 4 * x**3 - 3 * x
def T4(x): return 8 * x**4 - 8 * x**2 + 1
def T5(x): return 16 * x**5 - 20 * x**3 + 5 * x
def T6(x): return 32 * x**6 - 48 * x**4 + 18 * x**2 - 1
def T7(x): return 64 * x**7 - 112 * x**5 + 56 * x**3 - 7 * x

def identity(X,X_min,X_max):
    return X

activation = torch.tanh
num_layer=args.num_layer 
width_layer = args.width_layer  # neurons/layer
initialization = 'xavier'
degree = args.degree
Network_type = args.Network
Use_ResNet = args.Use_ResNet

def glorot_normal(in_dim, out_dim):
    std = np.sqrt(2.0 / (in_dim + out_dim))
    return torch.randn(in_dim, out_dim) * std

def init_params(
    layers: List[int],
    initialization_type: str = 'xavier',
    Network_type: str = 'KAN',
    degree: int = 7,
    Use_ResNet: bool = False
) -> dict:

    def init_adaptive_params():
        F = 0.1 * torch.ones(3 * (len(layers) - 1))
        A = 0.1 * torch.ones(3 * (len(layers) - 1))
        return [
            {
                "a0": A[3*i], "a1": A[3*i + 1], "a2": A[3*i + 2],
                "f0": F[3*i], "f1": F[3*i + 1], "f2": F[3*i + 2]
            }
            for i in range(len(layers) - 1)
        ]

    def init_layer_mlp(in_dim, out_dim):
        if initialization_type.lower() == 'xavier':
            W = glorot_normal(in_dim, out_dim)
        elif initialization_type.lower() == 'normal':
            W = torch.randn(in_dim, out_dim)
        b = torch.zeros(out_dim)
        g = torch.ones(out_dim)
        return {"W": W, "b": b, "g": g}

    def init_layer_kan(in_dim, out_dim, degree=degree):
        std = 1 / (in_dim * (degree + 1))
        W = torch.normal(0.0, std, size=(in_dim, out_dim, degree + 1))
        b = torch.zeros(out_dim)
        g = torch.ones(out_dim)
        return {"W": W, "b": b, "g": g}

    # Select model type
    if Network_type.lower() == 'mlp':
        init_layer_params = init_layer_mlp
    elif Network_type.lower().startswith('kan'):
        init_layer_params = init_layer_kan
    else:
        raise ValueError(f"{Network_type} is not a valid option. Use 'mlp' or 'kan'.")

    print(f"Initializing: {Network_type} parameters.")

    params = [init_layer_params(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]

    # Extra mMLP params
    U1 = glorot_normal(layers[0], layers[1])
    b1 = torch.zeros(layers[1])
    g1 = torch.ones(layers[1])
    U2 = glorot_normal(layers[0], layers[1])
    b2 = torch.zeros(layers[1])
    g2 = torch.ones(layers[1])

    mMLP_params = [{"U1": U1, "b1": b1, "g1": g1, "U2": U2, "b2": b2, "g2": g2}]

    return {
        'params': params,
        'AdaptiveAF': init_adaptive_params(),
        'mMLP': mMLP_params
    }

# layers = [2] + num_layer*[width_layer] + [1]
# params = init_params(layers=layers,initialization_type=initialization.lower(),degree=degree,Network_type=Network_type,Use_ResNet=Use_ResNet) 


class KAN_Net7(nn.Module):
    def __init__(self, params, M1=0, M2=0, activation=torch.tanh, norm_fn=lambda x, M1, M2: x):
        super(KAN_Net7, self).__init__()
        self.activation = activation
        self.norm_fn = norm_fn
        self.M1 = M1
        self.M2 = M2

        # Register weights
        self.layers = nn.ModuleList()
        for layer_params in params["params"]:
            W = nn.Parameter(layer_params["W"])
            self.layers.append(nn.ParameterDict({"W": W}))

    def Cheby_KAN_layer7(self, x, W):
        input_dim = W.shape[0]
        output_dim = W.shape[1]

        x = self.activation(x)
        x = x.view(-1, input_dim, 1)
        # x_stack = torch.stack([
        #     T0(x), T1(x), T2(x), T3(x), T4(x), T5(x), T6(x), T7(x)
        # ], dim=2).squeeze(-1)
        x_stack = torch.stack([
            T0(x), T1(x), T2(x), T3(x), T4(x), T5(x), T6(x), T7(x)
        ], dim=2).squeeze(-1)
        W = W.to(x_stack.device)
        x_out = torch.einsum('bid,iod->bo', x_stack, W)

        return x_out

    def forward(self, x):
        x = self.norm_fn(x, self.M1, self.M2)

        for layer in self.layers:
            x = self.Cheby_KAN_layer7(x, layer["W"])

        return x
    

##########################################################################################################


# def exists(val):
#     return val is not None

# def cast_tuple(val, repeat = 1):
#     return val if isinstance(val, tuple) else ((val,) * repeat)

# # sin activation

# class Sine(nn.Module):
#     def __init__(self, w0 = 1.):
#         super().__init__()
#         self.w0 = w0
#     def forward(self, x):
#         return torch.sin(self.w0 * x)

# # siren layer

# class Siren(nn.Module):
#     def __init__(
#         self,
#         dim_in,
#         dim_out,
#         w0 = 1.,
#         c = 6.,
#         is_first = False,
#         use_bias = True,
#         activation = None,
#         dropout = 0.
#     ):
#         super().__init__()
#         self.dim_in = dim_in
#         self.is_first = is_first

#         weight = torch.zeros(dim_out, dim_in)
#         bias = torch.zeros(dim_out) if use_bias else None
#         self.init_(weight, bias, c = c, w0 = w0)

#         self.weight = nn.Parameter(weight)
#         self.bias = nn.Parameter(bias) if use_bias else None
#         self.activation = Sine(w0) if activation is None else activation
#         self.dropout = nn.Dropout(dropout)

#     def init_(self, weight, bias, c, w0):
#         dim = self.dim_in

#         w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
#         weight.uniform_(-w_std, w_std)

#         if exists(bias):
#             bias.uniform_(-w_std, w_std)

#     def forward(self, x):
#         out =  F.linear(x, self.weight, self.bias)
#         out = self.activation(out)
#         out = self.dropout(out)
#         return out


# class SirenNet(nn.Module):
#     def __init__(
#         self,
#         dim_in,
#         dim_hidden,
#         dim_out,
#         num_layers,
#         w0 = 1.,
#         w0_initial = 30.,
#         use_bias = True,
#         final_activation = None,
#         dropout = 0.
#     ):
#         super().__init__()
#         self.num_layers = num_layers
#         self.dim_hidden = dim_hidden

#         self.layers = nn.ModuleList([])
#         for ind in range(num_layers):
#             is_first = ind == 0
#             layer_w0 = w0_initial if is_first else w0
#             layer_dim_in = dim_in if is_first else dim_hidden

#             layer = Siren(
#                 dim_in = layer_dim_in,
#                 dim_out = dim_hidden,
#                 w0 = layer_w0,
#                 use_bias = use_bias,
#                 is_first = is_first,
#                 dropout = dropout
#             )

#             self.layers.append(layer)

#         final_activation = nn.Identity() if not exists(final_activation) else final_activation
#         self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

#     def forward(self, x, mods = None):
#         mods = cast_tuple(mods, self.num_layers)

#         for layer, mod in zip(self.layers, mods):
#             x = layer(x)

#             if exists(mod):
#                 x *= rearrange(mod, 'd -> () d')

#         return self.last_layer(x)

class Conv_Block(nn.Module):
    def __init__(self,in_c,out_c,k,num_groups=1):
        super(Conv_Block, self).__init__()
        
        self.block = nn.Sequential(
                     nn.Conv2d(in_c, out_c, k, padding='same'),
                     nn.GroupNorm(num_groups,out_c),
                     nn.GELU())

    def forward(self, x):
        return self.block(x)

class DeepOKan(nn.Module):
    def __init__(self,Par):
        super(DeepOKan,self).__init__()

        self.Par = Par
        
        n_channels = self.Par['n_channels']
        k  = self.Par['k']
        Nx = self.Par['nx']
        Ny = self.Par['ny']
        x = np.linspace(0,1,Nx)
        y = np.linspace(0,1,Ny)
        xx, yy = np.meshgrid(x, y, indexing='ij')

        coord = np.concatenate([xx.reshape(-1,1),yy.reshape(-1,1)], axis=1)
        self.coord = torch.tensor(coord, dtype=self.Par['DTYPE'], device=self.Par['DEVICE'])

        self.branch_net = nn.Sequential(Conv_Block(self.Par['lb'], n_channels, k),
                                 nn.MaxPool2d(2, stride=2), #64
                                 Conv_Block(n_channels, 2*n_channels, k),
                                 nn.MaxPool2d(2, stride=2), #32
                                 Conv_Block(2*n_channels, 4*n_channels, k),
                                 nn.MaxPool2d(2, stride=2), #16
                                 Conv_Block(4*n_channels, 8*n_channels, k),
                                 nn.MaxPool2d(2, stride=2), #8
                                 Conv_Block(8*n_channels, 8*n_channels, k),
                                 nn.MaxPool2d(2, stride=2), #4
                                 Conv_Block(8*n_channels, 8*n_channels, k),
                                 nn.Flatten(),
                                 nn.Linear(8*n_channels*(4*8), self.Par['lf']*self.Par["ld"])
                                 )
                                 
        # self.trunk_net  = nn.Sequential(nn.Linear(2, 200),
        #                                 nn.GELU(),
        #                                 nn.Linear(200,200),
        #                                 nn.GELU(),
        #                                 nn.Linear(200,200),
        #                                 nn.GELU(),
        #                                 nn.Linear(200, self.Par['lf']*self.Par["ld"])
        #                                 )

        # self.trunk_net  = SirenNet(dim_in=2,
        #                            dim_hidden=512,
        #                            dim_out=self.Par['lf']*self.Par["ld"],
        #                            num_layers=5,
        #                            dropout=0.05)

        # ----------------- TRUNK KAN NET -----------------
        # in_dim = 2 (x, y), out_dim = lf * ld
        dim_in  = 2
        dim_out = self.Par['lf'] * self.Par['ld']

        trunk_layers = [dim_in] + num_layer * [width_layer] + [dim_out]

        trunk_params = init_params(
            layers=trunk_layers,
            initialization_type=initialization.lower(),
            Network_type=Network_type,
            degree=degree,
            Use_ResNet=Use_ResNet
        )

        self.trunk_net = KAN_Net7(params=trunk_params)
    
    def forward(self, x, coord=None):
        if coord==None:
            coord = self.coord
        B, C, X, Y = x.shape
        N_COORD, N_DIM = coord.shape

        x = (x-self.Par['inp_shift'])/(self.Par['inp_scale'])

        # print(f"x: {torch.min(x)}, {torch.max(x)}")

        bn = self.branch_net(x)      #[BS, lf*ld]
        tn = self.trunk_net(coord)   #[NCOORD, lf*ld]

        # print(f"bn: {torch.isnan(bn).any().item()}")
        # print(f"tn: {torch.isnan(tn).any().item()}")

        # print(f"bn: {torch.min(bn)}, {torch.max(bn)}")
        # print(f"tn: {torch.min(tn)}, {torch.max(tn)}")

        bn = bn.reshape(-1, self.Par["lf"], self.Par["ld"], 1) #[BS, lf,ld, 1]
        tn = tn.reshape(-1, self.Par["lf"], self.Par["ld"])    #[NCOORD, lf,ld]
        tn = torch.permute(tn, (1,2,0)) #[lf,ld, NCOORD]
        tn = tn.unsqueeze(0).repeat(B, 1, 1, 1) #[B, lf,ld, NCOORD]

        out = bn*tn #[B, lf,ld, NCOORD]
        out = torch.sum(out, dim=2) #[B, lf, NCOORD]
        out = out.reshape(B, self.Par["lf"], self.Par["nx"], self.Par["ny"])

        out = out*self.Par["out_scale"] + self.Par["out_shift"]
        
        return out
