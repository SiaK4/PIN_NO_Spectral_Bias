from typing import Sequence, Optional, Union
import math
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import lecun_normal, zeros
from functools import partial

def get_activation(activation_name: str= 'swish'):
    if activation_name == 'swish':
        return lambda x: x * jax.nn.sigmoid(x)
    if activation_name == 'tanh':
        return jnp.tanh
    if activation_name =='relu':
        return jax.nn.relu
    if activation_name =='GELU':
        return jax.nn.gelu
    if activation_name =='sigmoid':
        return jax.nn.sigmoid
    if activation_name == 'adap_swish':
        return AdaptiveSwish()

class Swish(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x * torch.sigmoid(x)

class AdaptiveSwish(nn.Module):
    @nn.compact
    def __call__(self,x):
        beta = self.param("beta", lambda rng: jnp.array(1.0))
        return x * nn.sigmoid(beta*x)

### SIREN activation
def siren_kernel_init(w0: float = 1.0, is_first: bool = False):
    """
    Returns an initializer function for kernels consistent with SIREN paper.
    """
    def init(key, shape, dtype=jnp.float64):
        fan_in = shape[0]
        if is_first:
            bound = 1.0/fan_in
        else:
            bound = math.sqrt(6.0/fan_in)/w0
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)
    return init

class SIRENLayer(nn.Module):
    in_features: int
    out_features: int
    w0: float = 1.0
    is_first: bool = False
    h_siren: bool = False
    learning_w0: bool = False

    @nn.compact
    def __call__(self,x):
        if self.learning_w0:
            w0_param = self.param("w0", lambda rng: jnp.array(float(self.w0)))
            w0_val = w0_param
        else:
            w0_val = jnp.array(self.w0)

        # kernel_init = siren_kernel_init(jnp.array(w0_val), is_first=self.is_first)
        kernel_init = siren_kernel_init(self.w0, is_first=self.is_first)
        dense = nn.Dense(self.out_features, kernel_init=kernel_init, bias_init=zeros)
        y = dense(x)
        if self.h_siren:
            return jnp.sin(jnp.sinh(w0_val * y))
        else:
            return jnp.sin(w0_val * y)

class DenseLayer(nn.Module):
    in_channels: int
    out_channels: int
    activation: Union[str, nn.Module]

    @nn.compact
    def __call__(self,x):
        # dense = nn.Dense(self.out_channels, kernel_init=lecun_normal(), bias_init=zeros)
        dense = nn.Dense(self.out_channels, kernel_init=jax.nn.initializers.uniform(1e-3), bias_init=zeros)

        x = dense(x)
        if isinstance(self.activation, str):
            act = get_activation(self.activation)
            return act(x) if isinstance(act, nn.Module) else act(x)
        else:
            act = self.activation
            return act(x) if isinstance(act, nn.Module) else act(x)

class FourierFeatureLayer(nn.Module):
    in_channels: int
    num_frequencies: int = 4

    @nn.compact
    def __call__(self, x):
        freq = self.variable(
            "constants",
            "freq_bands",
            lambda: (jnp.arange(1, self.num_frequencies+1, dtype=jnp.float32) * math.pi)
        )
        x_proj = x[..., None] * freq
        sin_f = jnp.sin(x_proj)
        cos_f = jnp.cos(x_proj)
        sin_flat = sin_f.reshape((x.shape[0], -1))
        cos_flat = cos_f.reshape((x.shape[0], -1))
        return jnp.concatenate([x, sin_flat, cos_flat], axis=-1)

class RandomFourierFeatureLayer(nn.Module):
    in_channels: int
    num_features: int = 0
    scale: float = 1.0

    @nn.compact
    def __call__(self, x):
        # If no Fourier features, just return x unchanged
        if self.num_features == 0:
            return x

        # create fixed B buffer at first call
        def make_B():
            key = self.make_rng("params")
            B = jax.random.normal(
                key, (self.in_channels, self.num_features)
            ) * (self.scale * math.pi)
            return B

        B_var = self.variable("constants", "B", make_B)  # (C, F)
        B = B_var.value

        # x: (B, C) -> (B, C, 1)
        # B: (C, F) -> broadcast -> (B, C, F)
        x_proj = x[..., None] * B
        sin_f = jnp.sin(x_proj)
        cos_f = jnp.cos(x_proj)

        sin_flat = sin_f.reshape((x.shape[0], -1))
        cos_flat = cos_f.reshape((x.shape[0], -1))

        return jnp.concatenate([x, sin_flat, cos_flat], axis=-1)

#### ----- Main NN module ------ ####
class NN(nn.Module):
    in_c: int
    out_c: int
    features: Union[Sequence[int], int] = (100, 100, 100, 100)
    activation_name: str = "tanh"
    num_frequencies: int = 0
    fourier_type: str = "random"   # 'linear' or 'random'
    fourier_scale: float = 1.0
    use_siren: bool = False
    siren_w0: float = 30.0
    skip_con: bool = False
    h_siren: bool = False
    learning_w0: bool = False

    @nn.compact
    def __call__(self, x):
        #ensure feature is a sequence
        if isinstance(self.features, int):
            feats = [self.features]
        else:
            feats = list(self.features)
        
        #tentative number for here (to be modified)
        fourier_in_c = 2
        if self.fourier_type == "linear":
            fourier_layer = FourierFeatureLayer(fourier_in_c, self.num_frequencies)
            fourier_out_dim = fourier_in_c + 2 * self.num_frequencies * fourier_in_c
        elif self.fourier_type == "random":
            fourier_layer = RandomFourierFeatureLayer(fourier_in_c, self.num_frequencies, scale=self.fourier_scale)
            fourier_out_dim = fourier_in_c + 2 * self.num_frequencies * fourier_in_c
        else:
            raise ValueError(f"Unknown fourier_type: {self.fourier_type}")
        
        # x_in = fourier_layer(x)
        x_in = x
        
        ## Build the layers sequentially (similar to Modulelist in Pytorch)
        h = x_in
        activation = get_activation(self.activation_name)

        if self.use_siren:
            h = SIRENLayer(in_features=fourier_out_dim, out_features=feats[0], w0=self.siren_w0, is_first=True,
                           h_siren=self.h_siren, learning_w0=self.learning_w0)(h)
            for i in range(1, len(feats)):
                h_prev = h
                h = SIRENLayer(in_features=feats[i - 1], out_features=feats[i], w0=1.0, is_first=False,
                               h_siren=self.h_siren, learning_w0=self.learning_w0)(h)

                if i > 0 and self.skip_con:
                        h = h + h_prev
        else:
            # regular dense layers via DenseLayer wrapper
            h = DenseLayer(fourier_out_dim, feats[0], activation)(h)
            for i in range(1, len(feats)):
                h_prev = h
                # Note: original PyTorch loop used range(1, len(features)-1) when building layers
                # but appended final_layer separately; to match behavior we build all hidden layers here
                out_ch = feats[i]
                h = DenseLayer(feats[i - 1], out_ch, activation)(h)
                if i > 0 and self.skip_con:
                    h = h + h_prev

        # final linear layer
        out = nn.Dense(self.out_c, kernel_init=lecun_normal(), bias_init=zeros)(h)
        # out = nn.Dense(self.out_c, kernel_init=jax.nn.initializers.uniform(1e-3), bias_init=zeros)(h)

        return out

def get_alpha_loss(params: dict):
    """
    Reproduces the intent of your PyTorch `get_alpha_loss`:
      - collect trainable w0 from SIREN layers (params named 'w0')
      - collect betas from AdaptiveSwish (params named 'beta')
    params should be the Flax params pytree (a nested dict).
    Returns scalar jnp.array.
    """
    # Flatten params and collect relevant entries by name
    alpha_params = []

    def visit(path, v):
        # path is tuple of keys in param tree
        name = "/".join(str(p) for p in path)
        # check for 'w0' param name or 'beta'
        if path and path[-1] == "w0":
            alpha_params.append(v.reshape(-1))
        if path and path[-1] == "beta":
            alpha_params.append(v.reshape(-1))

    # traverse recursively
    def recurse(d, path=()):
        if isinstance(d, dict):
            for k, vv in d.items():
                recurse(vv, path + (k,))
        else:
            visit(path, d)

    recurse(params)

    # concatenate scalars into list
    if len(alpha_params) <= 1:
        return jnp.array(0.0)
    # convert to 1D array of scalars (take first element of each param leaf)
    vals = jnp.array([float(a.ravel()[0]) for a in alpha_params])
    D = vals.size
    # compute alpha_loss like your code:
    alpha_loss = 0.0
    for k, a in enumerate(vals):
        alpha_loss = alpha_loss + jnp.exp(a ** k)
    alpha_loss = (D - 1) / alpha_loss
    return alpha_loss
