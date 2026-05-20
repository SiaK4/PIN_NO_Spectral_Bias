# Python Language Realted Moduli
import functools as ft
from collections.abc import Callable
from typing import Any, TypeVar, NamedTuple
import sys, os
import time
# Numpy and Scipy Related package
import numpy as np
from pyDOE import lhs
import matplotlib.pyplot as plt 
import scipy 
import scipy.io as sio
from scipy.interpolate import griddata

# JAX and JAX backends related  and differentiable packages
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PyTree, Scalar
import lineax as lx
import equinox as eqx
import equinox.internal as eqxi
import optax
import optimistix as optx  
from soap_jax import soap
## Create Directory for saving Model
dir_path = "./checkpoints"
os.makedirs(dir_path, exist_ok=True)

# Global setup and variables:
np.random.seed(1234)
jax.config.update("jax_enable_x64", True)
Y = TypeVar("Y") 
Out = TypeVar("Out") 
Aux = TypeVar("Aux") 
#_____________________________________________
k_react = 0.1   # ← reaction coefficient (very important parameter!)
#_____________________________________________

print(f"2D Diffusion–Reaction PINN (k = {k_react})")
model_name = "diffusion_reaction_2D_periodic.eqx"
MODEL_FILE_NAME = f"./checkpoints/{model_name}"
#_____________________________________________
lr = 1e-4
width = 44
bb= 234

print(f"lr = {lr} , width= {width}, seed = {bb}")

class BFGSTrustRegion(optx.AbstractBFGS):                                                                                 
    rtol: float 
    atol: float 
    norm: Callable = optx.max_norm 
    use_inverse: bool = True
    search: optx.AbstractSearch = optx.LinearTrustRegion()
    descent: optx.AbstractDescent = optx.NewtonDescent()
    verbose: frozenset[str] = frozenset()


class SSBFGSTrustRegion(optx.AbstractSSBFGS):                                                                                 
    rtol: float 
    atol: float 
    norm: Callable = optx.max_norm 
    use_inverse: bool = True
    search: optx.AbstractSearch = optx.LinearTrustRegion()
    descent: optx.AbstractDescent = optx.NewtonDescent()
    verbose: frozenset[str] = frozenset()
    
class SSBFGSBacktracking(optx.AbstractSSBFGS):
    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = False
    search: optx.AbstractSearch = optx.BacktrackingArmijo() 
    descent: optx.AbstractDescent = optx.NewtonDescent()
    verbose: frozenset[str] = frozenset()



class BroydenTrustRegion(optx.AbstractSSBroyden):                                                                                 
    rtol: float 
    atol: float 
    norm: Callable = optx.max_norm 
    use_inverse: bool = True
    search: optx.AbstractSearch = optx.LinearTrustRegion()
    descent: optx.AbstractDescent = optx.NewtonDescent()
    verbose: frozenset[str] = frozenset()  


initializer = jax.nn.initializers.glorot_normal()                                                                                              
def trunc_init(weight: jax.Array, key: jax.random.PRNGKey) -> jax.Array: 
    out, in_ = weight.shape
    return initializer(key, shape=(out, in_))                                                                                         

def apply_periodic_embedding(x, y):
    freqs = [1, 3, 6]
    feats = []
    for k in freqs:
        feats += [
            jnp.sin(k*jnp.pi*x), jnp.cos(k*jnp.pi*x),
            jnp.sin(k*jnp.pi*y), jnp.cos(k*jnp.pi*y)
        ]
    return jnp.array(feats)

def init_linear_weight(model, init_fn, key):
    is_linear = lambda x_loc: isinstance(x_loc, eqx.nn.Linear)
    get_weights = lambda m: [x.weight
                             for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                             if is_linear(x)]                                                                                                                                                                                                                               
    get_bias = lambda m: [x.bias for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                          if is_linear(x)]
    weights = get_weights(model)
    biases = get_bias(model)
    sz_w = [x.size for x in weights]
    sz_b = [x.size for x in biases]
    print("w", sz_w)
    print("b", sz_b)
    print(f"Total params: {sum(sz_w) + sum(sz_b)}")                                                                                   
                                                                                                                                      
    new_biases = jax.tree_util.tree_map(lambda p_loc: 0.0 * jnp.abs(p_loc), biases)
    new_weights = [init_fn(weight, subkey) 
                   for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
    new_model = eqx.tree_at(get_weights, model, new_weights)
    new_model = eqx.tree_at(get_bias, new_model, new_biases)
    return new_model                                                                                                                  
                      
                                            


# Popular smooth manufactured solution (many papers use this)
def U_fn(x, y):
    return jnp.sin(3 * jnp.pi * x) * jnp.cos(3 * jnp.pi * y)

def lap_U_fn(x, y):
    return -18 * jnp.pi**2 * U_fn(x, y)

def F_fn(x, y):           # forcing term = Δu_exact - reaction term
    u = U_fn(x, y)
    return lap_U_fn(x, y) - k_react * u**2
    
    
class Helmholtz2D(eqx.Module):
    layers: list
    def __init__(self, key, units=20):
        key_list = jax.random.split(key, 40)
        self.layers = [
            eqx.nn.Linear(12, units, key=key_list[0]),  # ← Change input from 2 to 4
            eqx.nn.Linear(units, units, key=key_list[1]),
            eqx.nn.Linear(units, units, key=key_list[2]),
            eqx.nn.Linear(units, units, key=key_list[3]),
            eqx.nn.Linear(units, units, key=key_list[4]),
            eqx.nn.Linear(units, units, key=key_list[6]),
            eqx.nn.Linear(units, units, key=key_list[7]),
            eqx.nn.Linear(units, 1, key=key_list[11])
        ]

    def __call__(self, x, y):
        xy_embed = apply_periodic_embedding(x, y)  # ← Add embedding
        for layer in self.layers[:-1]:
            xy_embed = jax.nn.tanh(layer(xy_embed))
        return self.layers[-1](xy_embed)[0]


#_____________________________________________
    
@eqx.filter_jit
def diffusion_reaction_residual(network, xx, yy):
    u_fn = lambda _x, _y: network(_x, _y)

    u_x  = jax.grad(u_fn, argnums=0)
    u_y  = jax.grad(u_fn, argnums=1)
    u_xx = jax.grad(u_x,  argnums=0)(xx, yy)
    u_yy = jax.grad(u_y,  argnums=1)(xx, yy)

    u = u_fn(xx, yy)
    #                     Δu            - reaction       - forcing
    R = u_xx + u_yy - k_react * u**2 - F_fn(xx, yy)
    return R


@eqx.filter_jit
def loss_fn(network, weight_f, xy_r):
    R = jax.vmap(diffusion_reaction_residual, in_axes=(None, 0, 0))(
        network, xy_r[:,0], xy_r[:,1]
    )
    loss_f = jnp.mean(jnp.square(R))
    return loss_f      # ← you can also return (loss_f, {"loss_pde":loss_f})

#_____________________________________________



def load(filename, model):
    with open(filename, "rb") as f:
        return eqx.tree_deserialise_leaves(f, model) 
   
if __name__ == "__main__":
    print("PINN Simulation for 2D Helmholtz (Periodic via Fourier features)")

    # === Grid for testing & plotting ===
    Nx = Ny = 100
    x = np.linspace(-1, 1, Nx)
    y = np.linspace(-1, 1, Ny)
    XX, YY = np.meshgrid(x, y)
    xy_star = np.hstack((XX.flatten()[:, None], YY.flatten()[:, None]))
    xy_star = jnp.array(xy_star)

    u_star = jax.vmap(U_fn, in_axes=(0, 0))(xy_star[:, 0], xy_star[:, 1]).reshape(-1, 1)
    #q_star = jax.vmap(Q_fn, in_axes=(0, 0))(xy_star[:, 0], xy_star[:, 1])

    # === Collocation points (only interior, no boundary points) ===
    Nf = 30000
    lb = jnp.array([-1.0, -1.0])
    ub = jnp.array([ 1.0,  1.0])
    xy_f_np = lb + (ub - lb) * lhs(2, Nf)           # pyDOE returns numpy
    xy_f = jnp.array(xy_f_np)                       # convert to JAX

    # === Model initialization ===
    key = jr.PRNGKey(bb)
    key, init_key = jr.split(key)
    pinn = Helmholtz2D(init_key, units=width)          # input now 4 due to embedding
    pinn = init_linear_weight(pinn, trunc_init, init_key)

    # === Adam warm-up (optional, 1 step is fine) ===
    
    num_steps = 150000
    lr_schedule = optax.cosine_decay_schedule(init_value=5e-4, decay_steps=num_steps, alpha=0.001)
    optimizer = soap(
        learning_rate=lr_schedule,
        b1=0.8,
        b2=0.99,
        weight_decay=1e-6,
        precondition_frequency=5,
        precondition_1d=True,
        qr_dtype=jnp.float64,   # ← THIS IS THE KEY
    )
    
    
    opt_state = optimizer.init(eqx.filter(pinn, eqx.is_inexact_array))

    @eqx.filter_jit
    def train_step(network, state):
        #loss, grad = eqx.filter_value_and_grad(loss_fn)(network, 1.0, xy_f)  # only lambda_f, xy_f
        loss, grad = eqx.filter_value_and_grad(lambda net: loss_fn(net, 1.0, xy_f))(network)

        updates, new_state = optimizer.update(grad, state, network)
        network = eqx.apply_updates(network, updates)
        return network, new_state, loss

    print("Adam warm-up (1 step)...")
    pinn, opt_state, loss = train_step(pinn, opt_state)
    print(f"Initial loss: {loss:.6e}")

    # === Partition model for Quasi-Newton ===
    params, static = eqx.partition(pinn, eqx.is_inexact_array)

    # === Closure functions for Optimistix ===
    def loss_newton(dynamic, static_part):
        model = eqx.combine(dynamic, static_part)
        return loss_fn(model, 1.0, xy_f)                # only interior loss

    # =====================================================
    # Adam-only training (50,000 steps)
    # =====================================================
    num_adam_steps = 150000
    print(f"Starting Adam training for {num_adam_steps} steps...")
    
    t_start = time.time()
    for step in range(num_adam_steps):
        pinn, opt_state, loss = train_step(pinn, opt_state)
    
        if step % 100 == 0:
            print(f"[Adam {step:6d}] loss = {loss:.6e}")
    
    t_end = time.time()
    print(f"Adam training finished in {t_end - t_start:.1f} seconds")


    # === Final prediction ===
    u_pred = jax.vmap(pinn, in_axes=(0, 0))(xy_star[:, 0], xy_star[:, 1])
    u_pred = u_pred.reshape(-1, 1)

    # === Compute final errors ===
    rel_l2 = jnp.linalg.norm(u_pred - u_star) / jnp.linalg.norm(u_star)
    linf   = jnp.max(jnp.abs(u_pred - u_star))

    print("="*80)
    print(f"FINAL RESULT:")
    print(f"Relative L2 error : {rel_l2:.3e} → {rel_l2*100:.6f}%")
    print(f"L∞ error          : {linf:.3e}")
    print("="*80)

    # === Save model ===
    eqx.tree_serialise_leaves(MODEL_FILE_NAME, pinn)
    print(f"Model saved to {MODEL_FILE_NAME}")

    # === Plotting (optional - paste my previous plotting code here) ===
    # ... (the full plotting code from my last message)

    # Simple scatter plot
    plt.figure(figsize=(6,5))
    plt.scatter(xy_star[:, 0], xy_star[:, 1], c=u_pred.squeeze(), cmap="RdYlBu_r", s=10)
    plt.colorbar(label="u_pred")
    plt.title(f"PINN Prediction (Rel. L2 = {rel_l2:.2e})")
    plt.xlabel("x"); plt.ylabel("y")
    plt.axis("equal")
    plt.savefig("u_pred_periodic.png", dpi=300, bbox_inches='tight')
    plt.close()

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np_on_cpu  # just to be safe

# Make sure everything is on CPU for plotting
xy_star_np   = np_on_cpu.array(xy_star)          # shape (N, 2)
u_star_np    = np_on_cpu.array(u_star).flatten() # shape (N,)
u_pred_np    = np_on_cpu.array(u_pred).flatten() # from final vmap

# -------------------------------
# 1. Compute L2 and L∞ errors
# -------------------------------
abs_error    = np_on_cpu.abs(u_pred_np - u_star_np)
l2_error     = np_on_cpu.sqrt(np_on_cpu.mean((u_pred_np - u_star_np)**2))
rel_l2_error = l2_error / np_on_cpu.sqrt(np_on_cpu.mean(u_star_np**2))
linf_error   = np_on_cpu.max(abs_error)

print("="*70)
print(f"Final Relative L2 error : {rel_l2_error:.3e}  ({rel_l2_error*100:.6f} %)")
print(f"Final L∞ error          : {linf_error:.3e}")
print(f"Final Absolute L2 error : {l2_error:.3e}")
print("="*70)

# -------------------------------
# 2. Reshape for 2D plotting (Nx × Ny grid)
# -------------------------------
Ngrid = Nx#int(np_on_cpu.sqrt(len(u_star_np)))
X_plot = xy_star_np[:, 0].reshape(Ngrid, Ngrid)
Y_plot = xy_star_np[:, 1].reshape(Ngrid, Ngrid)
U_exact_plot   = u_star_np.reshape(Ngrid, Ngrid)
U_pred_plot    = u_pred_np.reshape(Ngrid, Ngrid)
Error_plot     = abs_error.reshape(Ngrid, Ngrid)

# -------------------------------
# 3. Plot Exact / Prediction / Error
# -------------------------------
plt.figure(figsize=(18, 5.5))

# Exact solution
plt.subplot(1, 3, 1)
im1 = plt.contourf(X_plot, Y_plot, U_exact_plot, levels=60, cmap="RdYlBu_r")
plt.colorbar(im1, fraction=0.046, pad=0.04)
plt.contour(X_plot, Y_plot, U_exact_plot, levels=15, colors='k', alpha=0.3, linewidths=0.5)
plt.title("Exact solution\nu(x,y) = cos(3πx) sin(3πy)", fontsize=14)
plt.xlabel("x"); plt.ylabel("y")
plt.axis("equal")

# PINN prediction
plt.subplot(1, 3, 2)
im2 = plt.contourf(X_plot, Y_plot, U_pred_plot, levels=60, cmap="RdYlBu_r")
plt.colorbar(im2, fraction=0.046, pad=0.04)
plt.contour(X_plot, Y_plot, U_pred_plot, levels=15, colors='k', alpha=0.3, linewidths=0.5)
plt.title(f"PINN Prediction\nRel. L2 = {rel_l2_error:.2e}", fontsize=14)
plt.xlabel("x"); plt.ylabel("y")
plt.axis("equal")

# Absolute error
plt.subplot(1, 3, 3)
im3 = plt.contourf(X_plot, Y_plot, Error_plot, levels=60, cmap="viridis")
plt.colorbar(im3, fraction=0.046, pad=0.04)
plt.contour(X_plot, Y_plot, Error_plot, levels=15, colors='k', alpha=0.2, linewidths=0.4)
plt.title(f"Absolute Error\nL∞ = {linf_error:.2e}", fontsize=14)
plt.xlabel("x"); plt.ylabel("y")
plt.axis("equal")

plt.tight_layout()
plt.savefig("helmholtz_solution_comparison.png", dpi=300, bbox_inches='tight')
plt.show()



# ================================================================
# === FFT COMPARISON: EXACT vs PRED (Midline X and Midline Y) ====
# ================================================================

import numpy.fft as fft

def fft_slice(u_vals, N, axis=1):
    """
    Takes a midline slice of u(x,y) and computes |FFT|.
    axis = 1 → slice along x-midline
    axis = 0 → slice along y-midline
    """
    u_vals = np.array(u_vals).reshape(N, N)

    if axis == 1:
        line = u_vals[N//2, :]     # horizontal slice (along x)
    else:
        line = u_vals[:, N//2]     # vertical slice (along y)

    spectrum = fft.fftshift(np.abs(fft.fft(line)))
    return spectrum


# === Extract midline FFTs ===
fft_x_pred = fft_slice(U_pred_plot,  Ngrid, axis=1)
fft_y_pred = fft_slice(U_pred_plot,  Ngrid, axis=0)

fft_x_true = fft_slice(U_exact_plot, Ngrid, axis=1)
fft_y_true = fft_slice(U_exact_plot, Ngrid, axis=0)


# === Plot ===
plt.figure(figsize=(7,5))
plt.semilogy(fft_x_true, 'k',  linewidth=1.8, label='Exact FFT x-midline')
plt.semilogy(fft_x_pred, 'g:', linewidth=2.0, label='Predicted FFT x-midline')
plt.semilogy(fft_y_true, 'b',  linewidth=1.8, label='Exact FFT y-midline')
plt.semilogy(fft_y_pred, 'r--', linewidth=2.0, label='Predicted FFT y-midline')


plt.title("FFT Comparison: Exact vs Prediction")
plt.xlabel("Frequency index")
plt.ylabel("|F(k)|")
plt.grid(True, which='both', ls='--', alpha=0.6)
plt.legend()

plt.savefig("fft_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

print("✓ FFT comparison figure saved as fft_comparison.png")



# -------------------------------
# Gradient functions
# -------------------------------
def u_exact(x, y):
    return U_fn(x, y)

def u_pred_fn(x, y):
    return pinn(x, y)


# Gradients
grad_exact = jax.grad(u_exact, argnums=(0,1))
grad_pred  = jax.grad(u_pred_fn, argnums=(0,1))


# Evaluate gradients on grid
ux_e, uy_e = jax.vmap(grad_exact)(xy_star[:,0], xy_star[:,1])
ux_p, uy_p = jax.vmap(grad_pred )(xy_star[:,0], xy_star[:,1])

grad_exact_vec = jnp.stack([ux_e, uy_e], axis=1)
grad_pred_vec  = jnp.stack([ux_p, uy_p], axis=1)

rel_grad_l2 = (
    jnp.linalg.norm(grad_pred_vec - grad_exact_vec)
    / jnp.linalg.norm(grad_exact_vec)
)

print(f"Relative ∇u L2 error gradient : {rel_grad_l2:.3e}")


#____________________________________________________
def lap_exact(x, y):
    uxx = jax.grad(jax.grad(u_exact, 0), 0)(x, y)
    uyy = jax.grad(jax.grad(u_exact, 1), 1)(x, y)
    return uxx + uyy

def lap_pred(x, y):
    uxx = jax.grad(jax.grad(u_pred_fn, 0), 0)(x, y)
    uyy = jax.grad(jax.grad(u_pred_fn, 1), 1)(x, y)
    return uxx + uyy

lap_e = jax.vmap(lap_exact)(xy_star[:,0], xy_star[:,1])
lap_p = jax.vmap(lap_pred )(xy_star[:,0], xy_star[:,1])

rel_lap_l2 = jnp.linalg.norm(lap_p - lap_e) / jnp.linalg.norm(lap_e)

print(f"Relative Δu L2 error Laplacian : {rel_lap_l2:.3e}")
#____________________________________________________

def barron_norm_2d(u_vals, X, Y):
    """
    u_vals: (N,N) grid values
    X,Y   : meshgrid coordinates
    """
    U_hat = np.fft.fftshift(np.fft.fftn(u_vals))
    kx = np.fft.fftshift(np.fft.fftfreq(u_vals.shape[0]))
    ky = np.fft.fftshift(np.fft.fftfreq(u_vals.shape[1]))
    KX, KY = np.meshgrid(kx, ky)
    omega_norm = np.sqrt(KX**2 + KY**2)
    return np.sum(omega_norm * np.abs(U_hat)) / u_vals.size
BN_exact = barron_norm_2d(U_exact_plot, X_plot, Y_plot)
BN_pred  = barron_norm_2d(U_pred_plot,  X_plot, Y_plot)

barron_error = abs(BN_pred - BN_exact) / BN_exact
print(f"Relative Barron norm error : {barron_error:.3e}")
#____________________________________________________________
from scipy.stats import skew, kurtosis

def moments(u):
    u = u.flatten()
    return {
        "mean":     np.mean(u),
        "variance": np.var(u),
        "skewness": skew(u),
        "kurtosis": kurtosis(u, fisher=False)
    }

mom_exact = moments(U_exact_plot)
mom_pred  = moments(U_pred_plot)

for k in mom_exact:
    print(f"{k}: exact={mom_exact[k]:.4e}, pred={mom_pred[k]:.4e}")
#____________________________________________________
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.titlesize": 14,
    "font.size": 12,
})

# =====================================================
#  PAPER-STYLE FIELD COMPARISON (Helmholtz, Equinox PINN)
# =====================================================

# --- Rename to paper notation ---
f_GT   = U_exact_plot
f_pred = U_pred_plot
f_err  = f_pred - f_GT

# --- Gradient magnitudes ---
grad_GT_mag = np.linalg.norm(np.array(grad_exact_vec), axis=1).reshape(Ngrid, Ngrid)
grad_err    = np.linalg.norm(
    np.array(grad_pred_vec) - np.array(grad_exact_vec),
    axis=1
).reshape(Ngrid, Ngrid)

# --- Laplacians ---
lap_GT  = np.array(lap_e).reshape(Ngrid, Ngrid)
lap_err = np.array(lap_p - lap_e).reshape(Ngrid, Ngrid)

# -------------------------------
# Figure layout (2 × 3)
# -------------------------------
fig, axes = plt.subplots(
    nrows=2, ncols=3,
    figsize=(14, 6),
    constrained_layout=True
)

# ===== TITLES (TOP ROW ONLY) =====
axes[0,0].set_title(r"$f_{\mathrm{GT}}(x,y)$")
axes[0,1].set_title(r"$\nabla f_{\mathrm{GT}}(x,y)$")
axes[0,2].set_title(r"$\Delta f_{\mathrm{GT}}(x,y)$")

# ===== TOP ROW: GT =====
im = axes[0,0].imshow(
    f_GT, origin="lower", extent=[-1,1,-1,1],
    cmap="viridis"
)
plt.colorbar(im, ax=axes[0,0], fraction=0.046)

im = axes[0,1].imshow(
    grad_GT_mag, origin="lower", extent=[-1,1,-1,1],
    cmap="viridis"
)
plt.colorbar(im, ax=axes[0,1], fraction=0.046)

im = axes[0,2].imshow(
    lap_GT, origin="lower", extent=[-1,1,-1,1],
    cmap="viridis"
)
plt.colorbar(im, ax=axes[0,2], fraction=0.046)

# ===== BOTTOM ROW: ERRORS =====
axes[1,0].set_title(r"$f_{\mathrm{pred}} - f_{\mathrm{GT}}$")
axes[1,1].set_title(r"$\nabla f_{\mathrm{pred}} - \nabla f_{\mathrm{GT}}$")
axes[1,2].set_title(r"$\Delta f_{\mathrm{pred}} - \Delta f_{\mathrm{GT}}$")

im = axes[1,0].imshow(
    f_err, origin="lower", extent=[-1,1,-1,1],
    cmap="RdBu_r"
)
plt.colorbar(im, ax=axes[1,0], fraction=0.046)

im = axes[1,1].imshow(
    grad_err, origin="lower", extent=[-1,1,-1,1],
    cmap="RdBu_r"
)
plt.colorbar(im, ax=axes[1,1], fraction=0.046)

im = axes[1,2].imshow(
    lap_err, origin="lower", extent=[-1,1,-1,1],
    cmap="RdBu_r"
)
plt.colorbar(im, ax=axes[1,2], fraction=0.046)

# --- Minimal axes (paper style) ---
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.savefig("helmholtz_paper_style_fields_eqx.png", dpi=300)
plt.close()

print("✓ Saved paper-style field figure: helmholtz_paper_style_fields_eqx.png")


# =====================================================
#  CMAME UNIFIED SPECTRAL ERRORS
# =====================================================

def spectral_error_metric(u_exact, u_pred, p, L=2.0):
    """
    Compute spectral error metric in Fourier space.
    
    Parameters:
    -----------
    u_exact : array of shape (N, N)
        Exact solution on uniform grid
    u_pred : array of shape (N, N)
        Predicted solution on uniform grid
    p : int
        Order of derivative (0 for L2, 2 for gradient, 4 for Laplacian)
    L : float
        Domain size (default 2.0 for [-1,1]^2)
    
    Returns:
    --------
    float : Spectral error weighted by frequency^p
    """
    N = u_exact.shape[0]
    Ue = np.fft.fftn(u_exact)
    Up = np.fft.fftn(u_pred)
    E  = Ue - Up
    
    k = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
    kx, ky = np.meshgrid(k, k, indexing='ij')
    k2 = kx**2 + ky**2
    
    weight = np.ones_like(k2) if p == 0 else k2**(p/2)
    return np.sum(weight * np.abs(E)**2) / (N**2)

# Compute spectral errors for p = 0, 2, 4
E_L2   = spectral_error_metric(U_exact_plot, U_pred_plot, p=0)
E_grad = spectral_error_metric(U_exact_plot, U_pred_plot, p=2)
E_lap  = spectral_error_metric(U_exact_plot, U_pred_plot, p=4)


import numpy as np

print("="*70)
print("CMAME Unified Spectral Errors")
print(f"p = 0  (L2)        : {E_L2:.6e}   | log10 = {np.log10(E_L2):.3f}")
print(f"p = 2  (Gradient)  : {E_grad:.6e} | log10 = {np.log10(E_grad):.3f}")
print(f"p = 4  (Laplacian) : {E_lap:.6e}  | log10 = {np.log10(E_lap):.3f}")
print("="*70)


# =====================================================
#  PAPER-STYLE FIELD COMPARISON FIGURE
# =====================================================

import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

# Compute gradient and Laplacian fields for visualization
grad_GT = np.linalg.norm(np.array(grad_exact_vec), axis=1).reshape(Ngrid, Ngrid)
grad_pred_norm = np.linalg.norm(np.array(grad_pred_vec), axis=1).reshape(Ngrid, Ngrid)
grad_err = grad_pred_norm - grad_GT

lap_GT = np.array(lap_e).reshape(Ngrid, Ngrid)
lap_pred_grid = np.array(lap_p).reshape(Ngrid, Ngrid)
lap_err = lap_pred_grid - lap_GT

f_GT = U_exact_plot
f_pred = U_pred_plot
f_err = f_pred - f_GT

# -------------------------
# Figure layout (2 rows × 3 columns)
# -------------------------
fig, axes = plt.subplots(
    nrows=2, ncols=3,
    figsize=(14, 6),
    constrained_layout=True
)

# ===== TOP ROW: Ground Truth =====
axes[0,0].set_title(r"$f_{\mathrm{GT}}(x,y)$")
axes[0,1].set_title(r"$\|\nabla f_{\mathrm{GT}}(x,y)\|$")
axes[0,2].set_title(r"$\Delta f_{\mathrm{GT}}(x,y)$")

im = axes[0,0].imshow(
    f_GT, origin="lower", extent=[-1,1,-1,1],
    cmap="viridis", aspect="auto"
)
plt.colorbar(im, ax=axes[0,0], fraction=0.046)

im = axes[0,1].imshow(
    grad_GT, origin="lower", extent=[-1,1,-1,1],
    cmap="viridis", aspect="auto"
)
plt.colorbar(im, ax=axes[0,1], fraction=0.046)

im = axes[0,2].imshow(
    lap_GT, origin="lower", extent=[-1,1,-1,1],
    cmap="viridis", aspect="auto"
)
plt.colorbar(im, ax=axes[0,2], fraction=0.046)

# ===== BOTTOM ROW: Errors =====
axes[1,0].set_title(r"$f_{\mathrm{pred}} - f_{\mathrm{GT}}$")
axes[1,1].set_title(r"$\|\nabla f_{\mathrm{pred}}\| - \|\nabla f_{\mathrm{GT}}\|$")
axes[1,2].set_title(r"$\Delta f_{\mathrm{pred}} - \Delta f_{\mathrm{GT}}$")

im = axes[1,0].imshow(
    f_err, origin="lower", extent=[-1,1,-1,1],
    cmap="RdBu_r", aspect="auto"
)
plt.colorbar(im, ax=axes[1,0], fraction=0.046)

im = axes[1,1].imshow(
    grad_err, origin="lower", extent=[-1,1,-1,1],
    cmap="RdBu_r", aspect="auto"
)
plt.colorbar(im, ax=axes[1,1], fraction=0.046)

im = axes[1,2].imshow(
    lap_err, origin="lower", extent=[-1,1,-1,1],
    cmap="RdBu_r", aspect="auto"
)
plt.colorbar(im, ax=axes[1,2], fraction=0.046)

# Paper-style minimal axes
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.savefig("paper_style_fields.png", dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved paper-style field comparison: paper_style_fields.png")



import matplotlib as mpl

# --- Match paper-style rendering ---
mpl.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

# -------------------------
# Figure layout (2 rows × 3 columns)
# -------------------------
fig, axes = plt.subplots(
    nrows=2, ncols=3,
    figsize=(14, 6),
    constrained_layout=True
)

# ===== TOP ROW: Ground Truth =====
axes[0,0].set_title(r"$f_{\mathrm{GT}}(x,y)$")
axes[0,1].set_title(r"$\|\nabla f_{\mathrm{GT}}(x,y)\|$")
axes[0,2].set_title(r"$\Delta f_{\mathrm{GT}}(x,y)$")

im = axes[0,0].imshow(
    f_GT,
    origin="lower",
    extent=[-1,1,-1,1],
    cmap="viridis",
    aspect="auto",
    interpolation="bilinear"   # 🔑 smooth
)
plt.colorbar(im, ax=axes[0,0], fraction=0.046)

im = axes[0,1].imshow(
    grad_GT,
    origin="lower",
    extent=[-1,1,-1,1],
    cmap="viridis",
    aspect="auto",
    interpolation="bilinear"
)
plt.colorbar(im, ax=axes[0,1], fraction=0.046)

im = axes[0,2].imshow(
    lap_GT,
    origin="lower",
    extent=[-1,1,-1,1],
    cmap="viridis",
    aspect="auto",
    interpolation="bilinear"
)
plt.colorbar(im, ax=axes[0,2], fraction=0.046)

# ===== BOTTOM ROW: Errors =====
axes[1,0].set_title(r"$f_{\mathrm{pred}} - f_{\mathrm{GT}}$")
axes[1,1].set_title(r"$\|\nabla f_{\mathrm{pred}}\| - \|\nabla f_{\mathrm{GT}}\|$")
axes[1,2].set_title(r"$\Delta f_{\mathrm{pred}} - \Delta f_{\mathrm{GT}}$")

im = axes[1,0].imshow(
    f_err,
    origin="lower",
    extent=[-1,1,-1,1],
    cmap="RdBu_r",
    aspect="auto",
    interpolation="bilinear"
)
plt.colorbar(im, ax=axes[1,0], fraction=0.046)

im = axes[1,1].imshow(
    grad_err,
    origin="lower",
    extent=[-1,1,-1,1],
    cmap="RdBu_r",
    aspect="auto",
    interpolation="bilinear"
)
plt.colorbar(im, ax=axes[1,1], fraction=0.046)

im = axes[1,2].imshow(
    lap_err,
    origin="lower",
    extent=[-1,1,-1,1],
    cmap="RdBu_r",
    aspect="auto",
    interpolation="bilinear"
)
plt.colorbar(im, ax=axes[1,2], fraction=0.046)

# Paper-style minimal axes
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.savefig("paper_style_fields_smooth.png", dpi=300, bbox_inches="tight")
plt.close()

print("✓ Saved smooth paper-style field comparison")

