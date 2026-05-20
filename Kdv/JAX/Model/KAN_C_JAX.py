import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
import numpy as np
import argparse
from typing import List
import math

# Argument parser
parser = argparse.ArgumentParser(description='Tunning_parameters')
parser.add_argument('--Run_type', type=str, default='KAN_Benchmark')
parser.add_argument('--Mode', type=str, default='PINN')
parser.add_argument('--Eqn', type=str, default='HM')

parser.add_argument('--use_RBA', type=int, default=1)
parser.add_argument('--Mod_MLP', type=int, default=0)
parser.add_argument('--Adaptive_AF', type=int, default=0)
parser.add_argument('--Weight_Norm', type=int, default=0)
parser.add_argument('--resample', type=int, default=0)
parser.add_argument('--L2_reg', type=int, default=0)
parser.add_argument('--beta_L2', type=float, default=0.001)

# Resnet_params
parser.add_argument('--Use_ResNet', type=int, default=0)
parser.add_argument('--Adaptive', type=int, default=0)
parser.add_argument('--Light', type=int, default=0)

# KAN Params
parser.add_argument('--Network', type=str, default='KAN')
parser.add_argument('--degree', type=int, default=7)

parser.add_argument('--lr_lambdas_0', type=float, default=0.001)
parser.add_argument('--gamma', type=float, default=0.999)
parser.add_argument('--k_samp', type=float, default=0.75)
parser.add_argument('--c_samp', type=float, default=0.75)

parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--g_steps', type=int, default=100000)
parser.add_argument('--num_layer', type=int, default=6)
parser.add_argument('--width_layer', type=int, default=35)

args, unknown = parser.parse_known_args()


# Define Chebyshev polynomials
def T0(x): return jnp.ones_like(x)
def T1(x): return x
def T2(x): return 2 * x**2 - 1
def T3(x): return 4 * x**3 - 3 * x
def T4(x): return 8 * x**4 - 8 * x**2 + 1
def T5(x): return 16 * x**5 - 20 * x**3 + 5 * x
def T6(x): return 32 * x**6 - 48 * x**4 + 18 * x**2 - 1
def T7(x): return 64 * x**7 - 112 * x**5 + 56 * x**3 - 7 * x


def identity(X, X_min, X_max):
    return X


activation = jnp.tanh
num_layer = args.num_layer
width_layer = args.width_layer  # neurons/layer
initialization = 'xavier'
degree = args.degree
Network_type = args.Network
Use_ResNet = args.Use_ResNet


def glorot_normal(key, in_dim, out_dim):
    """JAX version of Glorot normal initialization"""
    std = np.sqrt(2.0 / (in_dim + out_dim))
    return random.normal(key, shape=(in_dim, out_dim)) * std


def init_params(
    key: random.PRNGKey,
    layers: List[int],
    initialization_type: str = 'xavier',
    Network_type: str = 'KAN',
    degree: int = 5,
    Use_ResNet: bool = False
) -> dict:
    """Initialize parameters for KAN or MLP network"""
    
    keys = random.split(key, len(layers) * 3)
    key_idx = 0

    def init_adaptive_params():
        F = 0.1 * jnp.ones(3 * (len(layers) - 1))
        A = 0.1 * jnp.ones(3 * (len(layers) - 1))
        return [
            {
                "a0": A[3*i], "a1": A[3*i + 1], "a2": A[3*i + 2],
                "f0": F[3*i], "f1": F[3*i + 1], "f2": F[3*i + 2]
            }
            for i in range(len(layers) - 1)
        ]

    def init_layer_mlp(in_dim, out_dim):
        nonlocal key_idx
        if initialization_type.lower() == 'xavier':
            W = glorot_normal(keys[key_idx], in_dim, out_dim)
            key_idx += 1
        elif initialization_type.lower() == 'normal':
            W = random.normal(keys[key_idx], shape=(in_dim, out_dim))
            key_idx += 1
        b = jnp.zeros(out_dim)
        g = jnp.ones(out_dim)
        return {"W": W, "b": b, "g": g}

    def init_layer_kan(in_dim, out_dim, degree=degree):
        nonlocal key_idx
        std = 1 / (in_dim * (degree + 1))
        W = random.normal(keys[key_idx], shape=(in_dim, out_dim, degree + 1)) * std
        key_idx += 1
        b = jnp.zeros(out_dim)
        g = jnp.ones(out_dim)
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
    U1 = glorot_normal(keys[key_idx], layers[0], layers[1])
    key_idx += 1
    b1 = jnp.zeros(layers[1])
    g1 = jnp.ones(layers[1])
    U2 = glorot_normal(keys[key_idx], layers[0], layers[1])
    key_idx += 1
    b2 = jnp.zeros(layers[1])
    g2 = jnp.ones(layers[1])

    mMLP_params = [{"U1": U1, "b1": b1, "g1": g1, "U2": U2, "b2": b2, "g2": g2}]

    return {
        'params': params,
        'AdaptiveAF': init_adaptive_params(),
        'mMLP': mMLP_params
    }


# Initialize layers and params
layers = [2] + num_layer * [width_layer] + [1]
# Note: You'll need to provide a key when calling this
# init_key = random.PRNGKey(0)
# params = init_params(key=init_key, layers=layers, initialization_type=initialization.lower(), 
#                      degree=degree, Network_type=Network_type, Use_ResNet=Use_ResNet)


class KAN(nn.Module):
    """JAX/Flax implementation of Chebyshev KAN"""
    layers: List[int]
    M1: float = 0.0
    M2: float = 0.0
    x_left: float = -1.0
    x_right: float = 1.0
    degree: int = 5
    activation: callable = jnp.tanh
    
    def norm_fn(self, x, M1, M2):
        """Normalization function - can be customized"""
        return x
    
    def Cheby_KAN_layer(self, x, W):
        """Apply Chebyshev KAN transformation for one layer"""
        # Get actual input dimension from x, not from W
        actual_input_dim = x.shape[-1]
        output_dim = W.shape[1]
        
        # Apply activation
        x = self.activation(x)
        x = x.reshape(-1, actual_input_dim, 1)
        
        # Stack Chebyshev polynomials based on degree
        if self.degree == 7:
            x_stack = jnp.stack([
                T0(x), T1(x), T2(x), T3(x), T4(x), T5(x), T6(x), T7(x)
            ], axis=2).squeeze(-1)
        elif self.degree == 5:
            x_stack = jnp.stack([
                T0(x), T1(x), T2(x), T3(x), T4(x), T5(x)
            ], axis=2).squeeze(-1)
        else:
            x_stack = jnp.stack([
                T0(x), T1(x), T2(x), T3(x)
            ], axis=2).squeeze(-1)
        
        # Einstein summation: batch, input_dim, degree -> batch, output_dim
        x_out = jnp.einsum('bid,iod->bo', x_stack, W)
        
        return x_out
    
    @nn.compact
    def __call__(self, x):
        """Forward pass through the KAN network"""
        # Ensure x is 2D: (batch_size, features)
        if x.ndim == 1:
            x = x[None, :]
        
        x_input = x
        
        # Compute period
        period = self.x_right - self.x_left
        
        # Extract spatial coordinate and apply periodic encoding
        x_spatial = x_input[:, 0:1]
        phase = 2 * jnp.pi * (x_spatial - self.x_left) / period
        x_periodic = jnp.concatenate([
            jnp.sin(phase),
            jnp.cos(phase)
        ], axis=-1)
        
        # Concatenate with remaining features if they exist
        if x.shape[1] > 1:
            x_rest = x_input[:, 1:]
            x_input = jnp.concatenate([x_periodic, x_rest], axis=-1)
        else:
            x_input = x_periodic
        
        # Apply normalization
        x_input = self.norm_fn(x_input, self.M1, self.M2)
        
        # Get actual input dimension after periodic encoding
        actual_first_layer_dim = x_input.shape[-1]
        
        # Pass through all layers - define weights inline
        for i in range(len(self.layers) - 1):
            # Determine input dimension for this layer
            if i == 0:
                # First layer receives the periodically encoded input
                in_dim = actual_first_layer_dim
            else:
                # Subsequent layers receive output from previous layer
                in_dim = self.layers[i]
            
            out_dim = self.layers[i + 1]
            std = 1 / (in_dim * (self.degree + 1))
            
            # Define parameter with proper initialization
            W = self.param(
                f'W_{i}',
                lambda key, shape, s=std: random.normal(key, shape) * s,
                (in_dim, out_dim, self.degree + 1)
            )
            
            x_input = self.Cheby_KAN_layer(x_input, W)
        
        return x_input


# Example usage:
"""
# Create model
model = KAN(
    layers=[2, 35, 35, 35, 35, 35, 35, 1],
    x_left=-1.0,
    x_right=1.0,
    degree=7,
    activation=jnp.tanh
)

# Initialize with dummy input
key = random.PRNGKey(0)
dummy_input = jnp.ones((1, 2))  # batch_size=1, input_dim=2
variables = model.init(key, dummy_input)

# Apply model
output = model.apply(variables, dummy_input)
"""