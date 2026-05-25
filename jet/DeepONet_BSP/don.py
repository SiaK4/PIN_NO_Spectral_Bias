import math
import torch
import torch.nn as nn
import numpy as np
import sys
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops import rearrange



def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

# sin activation

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

# siren layer

class Siren(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0 = 1.,
        c = 6.,
        is_first = False,
        use_bias = True,
        activation = None,
        dropout = 0.
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation
        self.dropout = nn.Dropout(dropout)

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        out = self.dropout(out)
        return out


class SirenNet(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0 = 1.,
        w0_initial = 30.,
        use_bias = True,
        final_activation = None,
        dropout = 0.
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layer = Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first,
                dropout = dropout
            )

            self.layers.append(layer)

        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

    def forward(self, x, mods = None):
        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                x *= rearrange(mod, 'd -> () d')

        return self.last_layer(x)

class Conv_Block(nn.Module):
    def __init__(self,in_c,out_c,k,num_groups=1):
        super(Conv_Block, self).__init__()
        
        self.block = nn.Sequential(
                     nn.Conv2d(in_c, out_c, k, padding='same'),
                     nn.GroupNorm(num_groups,out_c),
                     nn.GELU())

    def forward(self, x):
        return self.block(x)

class DeepONet(nn.Module):
    def __init__(self,Par):
        super(DeepONet,self).__init__()

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

        self.trunk_net  = SirenNet(dim_in=2,
                                   dim_hidden=512,
                                   dim_out=self.Par['lf']*self.Par["ld"],
                                   num_layers=5,
                                   dropout=0.05)
    
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
