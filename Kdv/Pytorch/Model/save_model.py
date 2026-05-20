import torch
import os
from flax.serialization import to_state_dict, from_state_dict
import numpy as np

device = torch.device('cuda')

def save_checkpoint(model,epoch,loss,best_loss,checkpoint_name):
    checkpoint_dir = f"checkpoints_{checkpoint_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    if loss < best_loss:
        print(f"Loss has decreased ({best_loss}-->{loss}). Saving the checkpoint at epoch {epoch+1}")
        torch.save(model.state_dict(),os.path.join(checkpoint_dir, f"model_epoch{epoch+1}.pth"))
        return loss
    return best_loss

def save_checkpoint_jax(params, epoch, loss, best_loss, checkpoint_name):
    checkpoint_dir = f"checkpoints_{checkpoint_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    if loss < best_loss:
        print(f"Loss decreased ({best_loss} → {loss}). Saving checkpoint at epoch {epoch+1}")
        path = os.path.join(checkpoint_dir, f"model_epoch{epoch+1}.npz")

        state_dict = to_state_dict(params)
        np.savez(path, **state_dict)

        return loss

    return best_loss