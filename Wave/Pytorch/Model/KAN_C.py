import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache
import argparse
from typing import List
import numpy as np
import math

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
parser.add_argument('--num_layer' , type=int  , default=6)
parser.add_argument('--width_layer' , type=int  , default=35)

args, unknown = parser.parse_known_args()
# for arg, value in vars(args).items():
#     print(f'{arg}: {value}')


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
    degree: int = 5,
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

layers = [2] + num_layer*[width_layer] + [1]
params = init_params(layers=layers,initialization_type=initialization.lower(),degree=degree,Network_type=Network_type,Use_ResNet=Use_ResNet) 


class KAN(nn.Module):
    def __init__(self, params=params, M1=0, M2=0, activation=torch.tanh, degree=5, norm_fn=lambda x, M1, M2: x):
        super(KAN, self).__init__()
        self.activation = activation
        self.norm_fn = norm_fn
        self.M1 = M1
        self.M2 = M2
        self.degree = degree

        # Register weights
        self.layers = nn.ModuleList()
        for layer_params in params["params"]:
            W = nn.Parameter(layer_params["W"])
            self.layers.append(nn.ParameterDict({"W": W}))

    def Cheby_KAN_layer(self, x, W):
        input_dim = W.shape[0]
        output_dim = W.shape[1]

        x = self.activation(x)
        x = x.view(-1, input_dim, 1)

        if self.degree == 7:
            x_stack = torch.stack([
                T0(x), T1(x), T2(x), T3(x), T4(x), T5(x), T6(x), T7(x)
            ], dim=2).squeeze(-1)
        
        elif self.degree == 5:
            x_stack = torch.stack([
                T0(x), T1(x), T2(x), T3(x), T4(x), T5(x)
            ], dim=2).squeeze(-1)
        
        else:
            x_stack = torch.stack([
                T0(x), T1(x), T2(x), T3(x)
            ], dim=2).squeeze(-1)           

        W = W.to(x_stack.device)
        x_out = torch.einsum('bid,iod->bo', x_stack, W)

        return x_out

    def forward(self, x):
        # x_input = x.clone()

        # x_spatial = x_input[:,0:1]
        # x_periodic = torch.cat([
        #         torch.sin(2*math.pi*(x_spatial-self.x_left)/self.period),
        #         torch.cos(2 * math.pi * (x_spatial - self.x_left)/self.period)
        #         ], dim=-1)
        
        # if x.shape[1] > 1:
        #     x_rest = x_input[:, 1:]
        #     x_input = torch.cat([x_periodic, x_rest], dim=-1)
        # else:
        #     x_input = x_periodic

        x_input = x

        x_input = self.norm_fn(x_input, self.M1, self.M2)

        for layer in self.layers:
            x_input = self.Cheby_KAN_layer(x_input, layer["W"])

        return x_input


    # def forward(self, x):
    #     x = self.norm_fn(x, self.M1, self.M2)

    #     for layer in self.layers:
    #         print("x.shape:",x.shape)
    #         x = self.Cheby_KAN_layer(x, layer["W"])

    #     return x

# model = KAN()
# print("number of model parameters:",sum(p.numel() for p in model.parameters() if p.requires_grad))

# x = torch.rand(10,2)
# output = model(x)
# print(output.shape)