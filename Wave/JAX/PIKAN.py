import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import grad, vmap, jacobian, jacfwd
import optax
import numpy as np
import time
import os
import json
from Model.KAN_C_JAX import KAN
from Model.KAN_C_JAX import init_params

from Model.save_model import save_checkpoint_jax
from config_KAN import get_config, update_config_from_cli, save_config
import argparse
from flax.core import freeze
from jax.tree_util import tree_map
from functools import partial
from jax.tree_util import tree_leaves
from jax.flatten_util import ravel_pytree
from Optimizers.minimize import  minimize

L_t = 1
L = 1
c = 1

## Load config
config = get_config()
config = update_config_from_cli(config)

device = config["device"]
model_cfg = config["model"]
train_cfg = config["train"]
data_cfg = config["data"]
weights_cfg = config["weights"]

save_dir = "checkpoints_" + config['train']['checkpoint_name']
save_config(config, save_dir)
slope_R_w = config["train"]["slope_R_w"]

print(jax.devices())

# This global config enables 64-bit precision for all subsequent JAX calculations.
if train_cfg["double_precision"]:
    jax.config.update("jax_enable_x64", True)
    # Print a confirmation for the user
    print("JAX running in DOUBLE precision (float64).")
else:
    jax.config.update("jax_enable_x64", False)
    print("JAX running in SINGLE precision (float32).")

DTYPE = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
print(f"Default DTYPE set to: {DTYPE}")

# 1. Instantiate the Module and Define the Model Function
# Initialize a main PRNG key
key = jax.random.PRNGKey(42)
key, model_init_key = jax.random.split(key)

layers = [model_cfg["in_c"]] + model_cfg["num_layer"]*[model_cfg["width_layer"]]+ [model_cfg["out_c"]]

# Create the KAN module (no params argument)
nn_module = KAN(
    layers=layers,
    degree=model_cfg["degree"]
)

## Define the model functional
def model(params, X):
    variables = freeze({'params': params})
    return nn_module.apply(variables, X)

def cast_to_dtype(tree, dtype):
    """Recursively casts all floating-point JAX arrays in a PyTree to the target dtype."""
    return tree_map(lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x, tree)

# 2. Initialize Parameters (Equivalent to Xavier/Weight Initialization)
# Create a dummy input for shape inference. Input shape is (2,) for a single (x, t) point.
dummy_input = jnp.zeros(model_cfg["in_c"])
initial_variables = nn_module.init(model_init_key, dummy_input)
initial_params = initial_variables['params']

## Cast the parameters into the correct DTYPE
initial_params = cast_to_dtype(initial_params, DTYPE)

num_params = sum(x.size for x in jax.tree_util.tree_leaves(initial_params))
print(f"Total parameters: {num_params:,}")

def w_schedule(w1_org,w2_org, w3_org, w4_org, w5_org, epoch):
    if epoch <=2000:
        w1, w2, w3, w4, w5 = w1_org, w2_org, w3_org, w4_org, w5_org
    if epoch > 2000 and epoch <=4000:
         w1, w2, w3, w4, w5 = w1_org, w2_org/2, w3_org/2, w4_org/2, w5_org/2
    if epoch > 4000 and epoch <= 6000:
        w1, w2, w3, w4, w5 = w1_org, w2_org/10, w3_org/10, w4_org/10, w5_org/10
    if epoch > 6000:
        w1, w2, w3, w4, w5 = w1_org, w2_org/25, w3_org/25, w4_org/25, w4_org/25
    return w1,w2,w3,w4,w5

def resample_points_jax(key, residuals, num_samples, epoch, epoch_resample):
    # 1. Prepare residuals and compute temperature
    residuals = jnp.abs(residuals.squeeze())
    
    # JAX equivalent of max(0.1, 1.0 * (0.75 ** (epoch - epoch_resample)))
    temp_decay = 0.75 ** (epoch - epoch_resample)
    temperature = jnp.clip(1.0 * temp_decay, a_min=0.1)
    alpha = 0.95
    N = residuals.size # Total number of points
    
    # 2. Compute residual-based probabilities (softmax)
    # Using jnp.exp and jnp.sum for softmax
    scaled_residuals = residuals / temperature
    exp_scaled_res = jnp.exp(scaled_residuals - jnp.max(scaled_residuals)) # Stabilized exp
    residual_probs = exp_scaled_res / jnp.sum(exp_scaled_res)

    # 3. Compute uniform probabilities
    uniform_probs = jnp.ones_like(residual_probs) / N
    
    # 4. Mix and normalize probabilities
    probabilities = alpha * residual_probs + (1 - alpha) * uniform_probs
    probabilities = probabilities / jnp.sum(probabilities) # Final normalization
    
    # 5. Sample indices using jax.random.choice with calculated probabilities
    # We sample indices from jnp.arange(N)
    sampled_indices = jax.random.choice(
        key,                  # PRNG key
        N,                    # Max index value (up to N-1)
        shape=(num_samples,), # Output shape
        replace=False,        # Sample without replacement
        p=probabilities       # Probability distribution
    )
    
    return sampled_indices

def random_sampling(data, ratio):
    num_samples = int(data.shape[0]*ratio)
    indices = np.random.choice(data.shape[0], size=num_samples, replace=False)
    return data[indices]

@partial(jax.jit, static_argnums=(1))
def compute_grad_norm(params, individual_loss_fn):
    """
    Computes the L2 norm of the gradient of a single loss component w.r.t. params.
    individual_loss_fn is a closure that only takes params as input.
    """
    # 1. Compute the gradient of the loss w.r.t. params
    grads = jax.grad(individual_loss_fn)(params)
    
    # 2. Compute the squared L2 norm: sum(g**2) for all gradient leaves
    squared_norm = sum(jnp.sum(g**2) for g in tree_leaves(grads))
    
    # 3. Return the L2 norm
    return jnp.sqrt(squared_norm)

def make_pde_residual_fn(model_fn):
    def u_scalar(params, X_i):
        return model_fn(params, X_i).squeeze()

    # Optimized: Forward-mode over Reverse-mode
    # grad(u_scalar) is Reverse-mode (fast for first deriv)
    # jacfwd(...) is Forward-mode (fast for small input Hessians)
    hessian_fn = jacfwd(grad(u_scalar, argnums=1), argnums=1)

    def pde_residual_scalar(params, X_i):
        # This still computes the full Hessian, but much faster than before
        H = hessian_fn(params, X_i) 
        return H[1, 1] - H[0, 0] # u_tt - u_xx

    vmap_pde = vmap(pde_residual_scalar, in_axes=(None, 0))
    return lambda params, X_batch: vmap_pde(params, X_batch).reshape(-1, 1)

Pde_residual_fn = make_pde_residual_fn(model)

def make_initial_v_fn(model_fn):
    """ Creates a vectorized function for the initial velocity u_t using direct grad/vmap."""
    
    # 1. Define u_scalar(params, X_i) where X_i is the full input [x, t]
    def u_scalar(params, X_i):
        return model_fn(params, X_i).squeeze()

    # 2. Get the gradient w.r.t input (u_x, u_t)
    # This returns a vector [du/dx, du/dt]
    u_grad_input_fn = grad(u_scalar, argnums=1) # argnums=1 means differentiate w.r.t X_i
    
    # 3. Define the scalar residual extractor: returns u_t
    def initial_v_scalar(params, X_i):
        grad_vec = u_grad_input_fn(params, X_i) 
        # u_t is the second element (index 1) of the gradient vector [du/dx, du/dt]
        return grad_vec[1] 
    
    # 4. Vectorize over the batch
    vmap_initial_v = vmap(initial_v_scalar, in_axes=(None, 0))

    return lambda params, X_batch: vmap_initial_v(params, X_batch).reshape(-1,1)
Initial_v_fn = make_initial_v_fn(model)

def compute_all_residuals(params, data_dict, model):
    ## A. PDE residual (u_tt - u_xx)
    domain_input = data_dict['domain_input']
    domain_residual = Pde_residual_fn(params, domain_input)

    ## B. Initial Condition (Magnitude): u(x, 0) = u_exact
    init_input = data_dict['init_input']
    init_target = data_dict['init_target']
    init_pred = vmap(model, in_axes=(None, 0))(params, init_input).squeeze()
    init_residual = init_pred.reshape(-1) - init_target.reshape(-1)

    ## C. Initial Condition (Velocity): u(x, 0) = 0
    init_v_pred = Initial_v_fn(params, init_input)
    init_v_residual = init_v_pred.squeeze()

    ## D. Left Boundary: u(0, t) = u_exact
    left_b_input = data_dict['left_b_input']
    left_b_target = data_dict['left_b_target']
    left_b_pred = vmap(model, in_axes=(None, 0))(params, left_b_input)
    left_b_residual = left_b_pred.squeeze() - left_b_target.squeeze()

    ## E. Right Boundary: u(X_L, t) = u_exact
    right_b_input = data_dict['right_b_input']
    right_b_target = data_dict['right_b_target']
    right_b_pred = vmap(model, in_axes=(None, 0))(params, right_b_input).squeeze()
    right_b_residual = right_b_pred.squeeze() - right_b_target.squeeze()

    return (domain_residual, init_residual, init_v_residual, left_b_residual, right_b_residual)

def loss_fn(params, data_dict, w_current, lambda_avg):
    residuals = compute_all_residuals(params, data_dict, model)
    (domain_res, init_res, init_v_res, left_res, right_res) = residuals

    ## Let's assume L2 loss (to be completed further)
    domain_loss = jnp.mean(domain_res**2)
    init_loss = jnp.mean(init_res**2)
    init_v_loss = jnp.mean(init_v_res**2)
    left_loss = jnp.mean(left_res**2)
    right_loss = jnp.mean(right_res**2)

    # Combine weighted losses
    w1, w2, w3, w4, w5 = w_current
    lambda_domain, lambda_init, lambda_init_v, lambda_left, lambda_right = lambda_avg
    total_loss = (w1*lambda_domain*domain_loss + w2*lambda_init*init_loss + \
        w3*lambda_init_v*init_v_loss + w4*lambda_left*left_loss + w5*lambda_right*right_loss)

    aux_losses = (domain_loss, init_loss, init_v_loss, left_loss, right_loss)
    return total_loss, aux_losses

@partial(jax.jit, static_argnums=(5,))  # Only opt_update_fn is static
def jax_train_step(params, opt_state, data_dict, w_current, lambda_avg, opt_update_fn):
    (loss, aux_data), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, data_dict, w_current, lambda_avg)
    
    updates, new_opt_state = opt_update_fn(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, loss, aux_data


@jax.jit
def jitted_calculate_relative_error(params, X, U):
    """Calculates the L2 relative error entirely on the device."""
    
    # 1. Prediction (already functional)
    U_pred = model(params, X).squeeze() # shape (N,)

    # 2. Calculate Error Norms
    numerator = jnp.linalg.norm(U_pred - U)
    denominator = jnp.linalg.norm(U)

    # 3. Calculate Relative Error
    rel_error = jnp.where(denominator == 0, 0.0, numerator / denominator)
    
    # Return the JAX array result
    return rel_error

def make_individual_loss_closures(params, data_dict, dummy_w, lambda_avg, loss_fn):
    def get_individual_loss(index):
        return lambda p: loss_fn(p, data_dict, dummy_w, lambda_avg)[1][index]
    return [get_individual_loss(i) for i in range(5)]

# ============= ADAM TRAINING STEP =============
@partial(jax.jit, static_argnums=(5,))
def adam_train_step(params, opt_state, data_dict, w_current, lambda_avg, opt_update_fn):
    (loss, aux_data), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, data_dict, w_current, lambda_avg)
    
    updates, new_opt_state = opt_update_fn(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, loss, aux_data

# ============= BFGS-RELATED FUNCTIONS =============
def flatten_params(params):
    """Flatten PyTree params to 1D vector for BFGS."""
    flat, _ = ravel_pytree(params)
    return flat

def get_unflatten_func(params):
    """Get the unraveling function for this params structure."""
    _, unravel_fn = ravel_pytree(params)
    return unravel_fn

def loss_and_gradient_bfgs(flat_weights, unflatten_func, data_dict, w_current, lambda_avg):
    """Loss function for BFGS that takes flattened weights."""
    params = unflatten_func(flat_weights)
    loss_val, aux = loss_fn(params, data_dict, w_current, lambda_avg)
    
    # Compute gradient and flatten it
    grad_pytree = jax.grad(lambda p: loss_fn(p, data_dict, w_current, lambda_avg)[0])(params)
    grad_flat, _ = ravel_pytree(grad_pytree)
    
    return loss_val, grad_flat

def loss_only_bfgs(flat_weights, unflatten_func, data_dict, w_current, lambda_avg):
    """Loss function for BFGS that takes flattened weights and returns only the loss."""
    params = unflatten_func(flat_weights)
    loss_val, aux = loss_fn(params, data_dict, w_current, lambda_avg)
    return loss_val

@partial(jax.jit, static_argnames=("unflatten_func", "static_options_tuple"))
def bfgs_step(
    flat_weights, H0,
    data_dict, w_current, lambda_avg,
    unflatten_func, static_options_tuple
):
    """
    Performs one BFGS optimization step.
    Note: key is handled outside this function to avoid JIT issues.
    """
    # Convert static options tuple back to dict and add current Hessian
    current_options = dict(static_options_tuple)
    current_options['initial_H'] = H0

    # Run BFGS optimization
    # Use loss_only_bfgs since minimize will compute gradients with jax.value_and_grad
    result = minimize(
        fun=loss_only_bfgs,
        x0=flat_weights,
        args=(unflatten_func, data_dict, w_current, lambda_avg),
        method='BFGS',
        options=current_options
    )

    # Recycle and symmetrize Hessian
    new_H0 = result.hess_inv
    new_H0 = (new_H0 + jnp.transpose(new_H0)) / 2
    
    try:
        L = jnp.linalg.cholesky(new_H0)
        is_failed = jnp.any(jnp.isnan(L))
        final_H0 = jax.lax.cond(
            is_failed,
            lambda op: jnp.eye(op.shape[0], dtype=op.dtype),
            lambda op: op,
            operand=new_H0
        )
    except:
        final_H0 = jnp.eye(new_H0.shape[0], dtype=new_H0.dtype)

    return result.x, final_H0, result.fun, result.nit

# ============= MAIN TRAINING FUNCTION =============
def train(key, initial_params, total_epochs, learning_rate, full_data, epoch_resample, 
          switch_epoch=5000, bfgs_options=None, test_data=None, double=True, checkpoint_interval=10):
    """
    Train with Adam first, then switch to BFGS optimizer.
    
    Args:
        switch_epoch: Epoch at which to switch from Adam to BFGS
        bfgs_options: Dictionary of options for BFGS (maxiter, maxfun, etc.)
        test_data: Dictionary with 'X' and 'U' for computing relative error
    """
    domain_loss_list = []
    init_loss_list = []
    init_v_loss_list =[]
    left_loss_list = []
    right_loss_list = []
    relative_error_list = []
    all_iterations = []  # Track actual iterations

    lambda_pde_avg = train_cfg["lambda_pde_avg"]
    lambda_init_avg = train_cfg["lambda_init_avg"]
    lambda_init_v_avg = train_cfg["lambda_init_v_avg"]
    lambda_left_avg = train_cfg["lambda_left_avg"]
    lambda_right_avg = train_cfg["lambda_right_avg"]
    grad_norm_interval = train_cfg["grad_norm_interval"]
    beta = train_cfg["beta"]

    lambda_avg = jnp.asarray([lambda_pde_avg, lambda_init_avg, lambda_init_v_avg, 
                               lambda_left_avg, lambda_right_avg], dtype=DTYPE)

    # Setup BFGS options
    if bfgs_options is None:
        Nchange = 100
        bfgs_options = {
            'maxiter': Nchange,
            'gtol': 1e-9,
            'update_method': "ssbroyden2",
            'initial_scale': True,
            'ls_normal_c1': 1e-4, 'ls_normal_c2': 0.9, 'ls_normal_maxiter': 15,
            'ls_fb_c1_try1': 1e-4, 'ls_fb_c2_try1': 0.8, 'ls_fb_maxiter_try1': 10,
            'ls_fb_c1_try2': 1e-4, 'ls_fb_c2_try2': 0.5, 'ls_fb_maxiter_try2': 25
        }
            
    # Convert to immutable tuple for JIT
    static_options_tuple = tuple(bfgs_options.items())

    # Adam optimizer setup
    scheduler = optax.exponential_decay(
        init_value=learning_rate, 
        transition_steps=2000, 
        decay_rate=0.9, 
        end_value=5e-6, 
        staircase=True
    )
    adam_optimizer = optax.adam(learning_rate=scheduler)
    opt_init_fn, opt_update_fn = adam_optimizer.init, adam_optimizer.update

    # Data preparation
    domain_data_org = full_data['domain_data']
    N_domain_total = domain_data_org.shape[0]

    data_dict = {
        'domain_input': jnp.asarray(full_data['domain_data'][:, [0,1]], dtype=DTYPE),
        'domain_target': jnp.asarray(full_data['domain_data'][:,-1], dtype=DTYPE),
        'left_b_input': jnp.asarray(full_data['left_b'][:, [0,1]], dtype=DTYPE),
        'left_b_target': jnp.asarray(full_data['left_b'][:, -1], dtype=DTYPE),
        'right_b_input': jnp.asarray(full_data['right_b'][:, [0,1]], dtype=DTYPE),
        'right_b_target': jnp.asarray(full_data['right_b'][:, -1], dtype=DTYPE),
        'init_input': jnp.asarray(full_data['init'][:,[0,1]], dtype=DTYPE),
        'init_target': jnp.asarray(full_data['init'][:,-1], dtype=DTYPE)
    }    

    w1_org, w2_org, w3_org, w4_org, w5_org = weights_cfg["w1_org"], weights_cfg["w2_org"], \
        weights_cfg["w3_org"], weights_cfg["w4_org"], weights_cfg["w5_org"]

    # Initialization
    params = initial_params
    opt_state = opt_init_fn(params)
    
    # BFGS-specific initialization using ravel_pytree
    flat_weights, unflatten_func = ravel_pytree(params)
    H0 = None  # Will be initialized when switching to BFGS
    
    # Track effective iterations
    adam_steps = 0
    effective_steps = 0

    # Main training loop
    print(f"Starting training: Adam for {switch_epoch} epochs, then BFGS")
    print("="*60)
    
    for epoch in range(total_epochs):
        start_time = time.time()
        key, subkey_sample = jax.random.split(key)

        # Get current weights
        w1, w2, w3, w4, w5 = w_schedule(w1_org, w2_org, w3_org, w4_org, w5_org, epoch)
        w_current = jnp.array([w1, w2, w3, w4, w5], dtype=DTYPE)

        # Data resampling - only for Adam phase
        if epoch < switch_epoch:
            # Adam phase: use resampling
            if epoch < epoch_resample:
                data_domain_current = random_sampling(full_data['domain_data'], ratio=0.25)
                X_domain_current = jnp.asarray(data_domain_current[:,[0,1]], dtype=DTYPE)
                U_domain_current = jnp.asarray(data_domain_current[:,-1], dtype=DTYPE)
            else:
                temp_data_dict = {
                    'domain_input': data_dict['domain_input'],
                    'init_input': data_dict['init_input'],
                    'init_target': data_dict['init_target'],
                    'left_b_input': data_dict['left_b_input'],
                    'right_b_input': data_dict['right_b_input'],
                    'left_b_target': data_dict['left_b_target'],
                    'right_b_target': data_dict['right_b_target']
                }

                (domain_res, _, _, _, _) = compute_all_residuals(params, temp_data_dict, model)
                sampled_idx = resample_points_jax(subkey_sample, jnp.abs(domain_res), 
                                                 N_domain_total//10, epoch, epoch_resample)

                X_domain_current = data_dict['domain_input'][sampled_idx]
                U_domain_current = data_dict['domain_target'][sampled_idx]
        else:
            checkpoint_interval = 10
            # BFGS phase: use FULL batch
            X_domain_current = data_dict['domain_input']
            U_domain_current = data_dict['domain_target']

        # Assemble current data dictionary
        data_dict_current = {
            'domain_input': X_domain_current,
            'domain_target': U_domain_current,
            'left_b_input': data_dict['left_b_input'],
            'left_b_target': data_dict['left_b_target'],
            'right_b_input': data_dict['right_b_input'],
            'right_b_target': data_dict['right_b_target'],
            'init_input': data_dict['init_input'],
            'init_target': data_dict['init_target']
        }

        # Update lambda weights periodically
        if epoch % grad_norm_interval == 0 and epoch > 0:
            loss_closures = make_individual_loss_closures(params, data_dict_current, 
                                                         w_current, lambda_avg, loss_fn) 
            grad_norms = jnp.array([compute_grad_norm(params, closure) for closure in loss_closures])
            
            G_total = jnp.sum(grad_norms)
            new_lambdas = G_total / jnp.maximum(grad_norms, 1e-6)
            lambda_avg = beta * lambda_avg + (1.0 - beta) * new_lambdas

        # ========== OPTIMIZER SWITCH ==========
        if epoch < switch_epoch:
            # Use Adam optimizer
            params, opt_state, loss, aux_losses = adam_train_step(
                params, opt_state, data_dict_current, w_current, lambda_avg, opt_update_fn
            )
            adam_steps += 1
            effective_steps = adam_steps
            nit = 1
            
            if epoch == switch_epoch - 1:
                print(f"\n{'='*60}")
                print(f"Switching from Adam to BFGS at epoch {switch_epoch}")
                print(f"Total Adam steps: {adam_steps}")
                print(f"Initializing Hessian matrix...")
                print(f"{'='*60}\n")
                # Prepare for BFGS using ravel_pytree
                flat_weights, unflatten_func = ravel_pytree(params)
                H0 = jnp.eye(len(flat_weights), dtype=DTYPE)
                print(f"Hessian shape: {H0.shape}, Memory: {H0.nbytes / 1e9:.2f} GB")
        else:
            # Use BFGS optimizer
            flat_weights, H0, loss, nit = bfgs_step(
                flat_weights, H0,
                data_dict_current, w_current, lambda_avg,
                unflatten_func, static_options_tuple
            )
            
            # Unflatten for evaluation and next iteration
            params = unflatten_func(flat_weights)
            effective_steps += nit
            
            # Get auxiliary losses for logging
            _, aux_losses = loss_fn(params, data_dict_current, w_current, lambda_avg)

        # Logging
        domain_loss_list.append(float(aux_losses[0]))
        init_loss_list.append(float(aux_losses[1]))
        init_v_loss_list.append(float(aux_losses[2]))
        left_loss_list.append(float(aux_losses[3]))
        right_loss_list.append(float(aux_losses[4]))
        all_iterations.append(effective_steps)
        
        # Compute relative error if test data is provided
        rel_err = jitted_calculate_relative_error(params, X_domain_current, U_domain_current)
        relative_error_list.append(float(rel_err))

        elapsed = time.time() - start_time
        
        optimizer_name = "Adam" if epoch < switch_epoch else "BFGS"
        if epoch % 10 == 0 or epoch == switch_epoch:
            print(f"[{optimizer_name}] Epoch {epoch}/{total_epochs} | "
                  f"Eff.Steps: {effective_steps} | "
                  f"BFGS nit: {nit} | "
                  f"Loss: {float(loss):.6e} | "
                  f"PDE: {float(aux_losses[0]):.6e} | "
                  f"Init: {float(aux_losses[1]):.6e} | "
                  f"Init_v: {float(aux_losses[2]):.6f} | "
                  f"left_b: {float(aux_losses[3]):.9f} | "
                  f"right_b: {float(aux_losses[4]):.9f} | "
                  f"Rel error: {float(rel_err):.6f} | "
                  f"Time: {elapsed:.3f}s")

        if (epoch+1)% checkpoint_interval == 0:
            temp_data_dict = {
                'domain_input': data_dict['domain_input'],
                'init_input': data_dict['init_input'],
                'init_target': data_dict['init_target'],
                'left_b_input': data_dict['left_b_input'],
                'right_b_input': data_dict['right_b_input'],
                'left_b_target': data_dict['left_b_target'],
                'right_b_target': data_dict['right_b_target']
            }

            (domain_res, _, _, _, _) = compute_all_residuals(params, temp_data_dict, model)
            best_loss = float('inf')
            best_loss = save_checkpoint_jax(params, epoch, loss, best_loss, checkpoint_name=checkpoint_name)
            # if epoch > epoch_resample:
            np.save(f"{checkpoint_name}_PDE_residual.npy",domain_res)

        if (epoch+1)% 1 ==0:
            with open(f'domain_loss_{checkpoint_name}.json','w') as file:
                json.dump(domain_loss_list, file)
            with open(f'Init_{checkpoint_name}.json', 'w') as file:
                json.dump(init_loss_list, file)
            with open(f'Init_v_{checkpoint_name}.json', 'w') as file:
                json.dump(init_v_loss_list, file)               
            with open(f'Left_{checkpoint_name}.json', 'w') as file:
                json.dump(left_loss_list, file)
            with open(f'Right_{checkpoint_name}.json', 'w') as file:
                json.dump(right_loss_list, file)
            with open(f'Rel_error_{checkpoint_name}.json', 'w') as file:
                json.dump(relative_error_list, file)

    return params, {
        'domain_loss': domain_loss_list,
        'init_loss': init_loss_list,
        'init_v_loss': init_v_loss_list,
        'left_loss': left_loss_list,
        'right_loss': right_loss_list,
        'relative_error': relative_error_list,
        'iterations': all_iterations
    }

# Load data from config
wave_data_dir = data_cfg["wave_data_dir"]
domain_data = np.load(os.path.join(wave_data_dir, data_cfg["domain_data_file"]))
init_data = np.load(os.path.join(wave_data_dir, data_cfg["init_data_file"]))
left_data = np.load(os.path.join(wave_data_dir + data_cfg["left_b_data_file"]))
right_data = np.load(os.path.join(wave_data_dir, data_cfg["right_b_data_file"]))

print("domain_data.shape:",domain_data.shape)
print("init_data.shape:",init_data.shape)
print("left_data.shape:",left_data.shape)
print("right_data.shape:",right_data.shape)

display_every = train_cfg["display_every"]
checkpoint_interval = train_cfg["checkpoint_interval"]
checkpoint_name = train_cfg["checkpoint_name"]

domain_data = jnp.asarray(domain_data, dtype=DTYPE)
init_data = jnp.asarray(init_data, dtype=DTYPE)
left_data = jnp.asarray(left_data, dtype=DTYPE)
right_data = jnp.asarray(right_data, dtype=DTYPE)
full_data = {"domain_data": domain_data, "init": init_data, "left_b": left_data, "right_b": right_data}

## Call train
final_params, final_opt_state = train(
    key,  # The remaining PRNG key
    initial_params = initial_params,
    total_epochs = train_cfg["adam_epochs"],
    learning_rate = train_cfg["learning_rate"],
    full_data= full_data,
    epoch_resample = train_cfg["epoch_resample"],
    switch_epoch= train_cfg["switch_epoch"],
    double = train_cfg["double_precision"],
    checkpoint_interval = checkpoint_interval
)