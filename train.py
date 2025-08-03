import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import random
import os
from datetime import datetime

from model import (
    config, gol_step, count_neighbors_3x3, get_non_trivial_mask,
    create_model, get_device, setup_seed, ensure_checkpoint_dir,
    get_checkpoint_path, checkpoint_exists, to_device, print_progress_bar
)

# -------------------------------
# Dataset
# -------------------------------

class GoLReverseDataset(Dataset):
    """
    Each sample is made as:
        prev_state ~ Bernoulli(density)
        current_state = gol_step(prev_state)

    __getitem__ returns (current_state, prev_state) as float tensors in {0, 1}.
    """

    def __init__(self, n_samples: int, H: int, W: int, density: float = 0.15, warmup_steps: int = 0, seed: int = 0):
        self.n = n_samples
        self.H = H
        self.W = W
        self.density = density
        self.warmup_steps = warmup_steps
        self.seed = seed
        setup_seed(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Start from random binary field
        prev = torch.rand(1, self.H, self.W)
        prev = (prev < self.density).float()

        # Optional warmup steps to create more natural patterns
        for _ in range(self.warmup_steps):
            prev = gol_step(prev.unsqueeze(0)).squeeze(0)

        curr = gol_step(prev.unsqueeze(0)).squeeze(0)
        return curr, prev


# -------------------------------
# Iterative refinement loop
# -------------------------------

@dataclass
class RefineOutput:
    logits_per_iter: list  # Only for deep supervision training
    probs_per_iter: list   # Only for deep supervision training
    bin_per_iter: list     # Only for deep supervision training
    errmask_per_iter: list # Only for deep supervision training
    final_logits: torch.Tensor = None  # For evaluation and non-deep supervision
    final_probs: torch.Tensor = None   # For evaluation
    final_pred: torch.Tensor = None    # For evaluation


def _refinement_step(model: nn.Module, current_bin: torch.Tensor, pred_prev: torch.Tensor, 
                    err: torch.Tensor, latent: torch.Tensor) -> tuple:
    """
    Perform a single refinement step.
    
    Args:
        model: The refinement model
        current_bin: Current state tensor [B, 1, H, W]
        pred_prev: Previous prediction tensor [B, 1, H, W]
        err: Error mask tensor [B, 1, H, W]
        latent: Latent variables tensor [B, latent_dim, H, W]
        
    Returns:
        tuple: (logits, probs, pred_prev, err, latent) - updated values after one step
    """
    # Concatenate: current_bin (1) + pred_prev (1) + err (1) + latent (latent_dim)
    inp = torch.cat([current_bin, pred_prev, err, latent], dim=1)
    output = model(inp)
    
    # Split output into logits for previous state and latent variables
    logits = output[:, :1]  # First channel for previous state
    latent = output[:, 1:]  # Remaining channels for latent variables
    
    probs = torch.sigmoid(logits)
    pred_prev = (probs > 0.5).float()

    # Forward check and error dilation
    next_from_pred = gol_step(pred_prev)
    mismatch = (next_from_pred != current_bin).float()
    err = count_neighbors_3x3(mismatch)
    
    return logits, probs, pred_prev, err, latent


def refine(model: nn.Module, current_bin: torch.Tensor, steps: int, steps_training: int = None, deep_supervision: bool = True) -> RefineOutput:
    """
    Perform iterative prediction and error-driven refinement.

    Args:
        model: RefinementCNN
        current_bin: Tensor [B, 1, H, W] in {0, 1}
        steps: number of refinement iterations
        steps_training: number of steps with gradients enabled during training (if None, use all steps)
        deep_supervision: whether to store history for deep supervision

    Returns:
        RefineOutput with lists across iterations.
    """
    assert current_bin.dim() == 4 and current_bin.size(1) == 1, f"expected [B,1,H,W], got {tuple(current_bin.shape)}"
    pred_prev = current_bin
    next_from_pred = gol_step(pred_prev)
    mismatch = (next_from_pred != current_bin).float()
    err = count_neighbors_3x3(mismatch)
    
    # Initialize latent variables
    latent = torch.zeros(current_bin.size(0), model.latent_dim, current_bin.size(2), current_bin.size(3), 
                        device=current_bin.device, dtype=current_bin.dtype)

    logits_hist, probs_hist, bin_hist, err_hist = [], [], [], []
    final_logits, final_probs, final_pred = None, None, None

    # Determine gradient behavior based on training mode and steps_training parameter
    if model.training and steps_training is not None:
        # Training mode with specified steps_training
        import random
        max_no_grad = steps - steps_training
        steps_no_grad = random.randint(0, max_no_grad) if max_no_grad > 0 else 0
        steps_with_grad = steps - steps_no_grad
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                # First do steps without gradients
                for _ in range(steps_no_grad):
                    logits, probs, pred_prev, err, latent = _refinement_step(
                        model, current_bin, pred_prev, err, latent
                    )
                    # Don't store history for no-gradient steps - they can't be used for training!
                    # These tensors have requires_grad=False and can't be used in deep supervision
        
        # Then do steps with gradients
        for _ in range(steps_with_grad):
            logits, probs, pred_prev, err, latent = _refinement_step(
                model, current_bin, pred_prev, err, latent
            )
            # Store history only if deep supervision is enabled
            if deep_supervision:
                logits_hist.append(logits)
                probs_hist.append(probs)
                bin_hist.append(pred_prev)
                err_hist.append(err)
        
        # Store final values for evaluation
        final_logits = logits
        final_probs = probs
        final_pred = pred_prev
    else:
        # Testing mode or training without steps_training specified - use all steps with model's training state
        with torch.no_grad():
            for _ in range(steps):
                logits, probs, pred_prev, err, latent = _refinement_step(
                    model, current_bin, pred_prev, err, latent
                )
                # Store history only if deep supervision is enabled
                if deep_supervision:
                    logits_hist.append(logits)
                    probs_hist.append(probs)
                    bin_hist.append(pred_prev)
                    err_hist.append(err)
            
            # Store final values for evaluation
            final_logits = logits
            final_probs = probs
            final_pred = pred_prev

    return RefineOutput(logits_hist, probs_hist, bin_hist, err_hist, final_logits, final_probs, final_pred)


# -------------------------------
# Training and evaluation
# -------------------------------

def train_epoch(model: nn.Module,
                loader: DataLoader,
                opt: torch.optim.Optimizer,
                device: str,
                refine_steps: int = 3,
                steps_training: int = 1,
                deep_supervision: bool = True) -> float:
    """
    Train for one epoch.

    Loss:
        BCEWithLogitsLoss toward the true previous state.
        If deep_supervision is True, average loss across all refinement steps.
    """
    model.train()
    bce = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_samples = 0

    # Add progress bar for training
    total_batches = len(loader)
    print_progress_bar(0, total_batches, prefix='Training:', suffix='Complete', length=50)

    for batch_idx, (curr, prev) in enumerate(loader):
        curr = to_device(curr, device)
        prev = to_device(prev, device)  # [B, 1, H, W]
        opt.zero_grad()

        out = refine(model, curr, refine_steps, steps_training=steps_training, deep_supervision=deep_supervision)
        
        if deep_supervision and len(out.logits_per_iter) > 0:
            losses = []
            for logits in out.logits_per_iter:
                # Calculate loss on all pixels, not just non-trivial ones
                losses.append(bce(logits, prev))
            loss = sum(losses) / len(losses)
        elif deep_supervision and len(out.logits_per_iter) == 0:
            # Deep supervision enabled but no gradient steps were performed
            # Fall back to final step only
            loss = bce(out.final_logits, prev)
        else:
            # Calculate loss on all pixels, not just non-trivial ones
            # Use final logits directly instead of accessing list
            loss = bce(out.final_logits, prev)

        loss.backward()
        opt.step()

        # Count all samples for loss averaging
        batch_size = curr.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Update progress bar
        print_progress_bar(batch_idx + 1, total_batches, prefix='Training:', 
                          suffix=f'Loss: {loss.item():.4f}', length=50)

    return total_loss / max(1, total_samples)


@torch.no_grad()
def eval_metrics(model: nn.Module,
                 loader: DataLoader,
                 device: str,
                 refine_steps: int = 3):
    """
    Evaluate metrics (using non-trivial masks for evaluation):
        - Average BCE loss (excluding trivial cases)
        - Bit accuracy on previous state (excluding trivial cases)
        - Fraction of samples where forward reconstruction exactly matches current (excluding trivial cases)
    """
    model.eval()
    bce = nn.BCEWithLogitsLoss()  # Use same reduction as training

    total_bce = 0.0
    total_bce_samples = 0
    total_pixels = 0
    correct_prev_bits = 0
    recon_exact_count = 0
    total_samples = 0

    # Add progress bar for evaluation
    total_batches = len(loader)
    print_progress_bar(0, total_batches, prefix='Evaluating:', suffix='Complete', length=50)

    for batch_idx, (curr, prev) in enumerate(loader):
        curr = to_device(curr, device)
        prev = to_device(prev, device)

        out = refine(model, curr, refine_steps)
        # Use final values directly instead of accessing list
        logits = out.final_logits
        probs = out.final_probs
        pred_prev = out.final_pred

        # Get non-trivial mask for current state
        non_trivial_mask = get_non_trivial_mask(curr, device)
        
        # Create mask for non-trivial predictions (current state has neighbors AND predicted prev is not empty)
        non_trivial_pred_mask = non_trivial_mask & (pred_prev > 0.5)
        
        # BCE loss only for non-trivial cases
        if non_trivial_mask.sum() > 0:
            # Apply mask to logits and prev for BCE calculation
            masked_logits = logits[non_trivial_mask]
            masked_prev = prev[non_trivial_mask]
            if masked_logits.numel() > 0:
                total_bce += bce(masked_logits, masked_prev).item() * masked_logits.numel()
                total_bce_samples += masked_logits.numel()

        # Bit accuracy on previous state (excluding trivial cases)
        # Only count accuracy for non-trivial cases
        if non_trivial_mask.sum() > 0:
            correct_prev_bits += ((pred_prev == prev) & non_trivial_mask).sum().item()
            total_pixels += non_trivial_mask.sum().item()

        # Exact forward reconstruction check per sample (excluding trivial cases)
        # Only count samples that have non-trivial cases
        sample_has_nontrivial = non_trivial_mask.view(curr.size(0), -1).any(dim=1)
        if sample_has_nontrivial.sum() > 0:
            recon = gol_step(pred_prev)
            # Compare per-sample equality only for non-trivial samples
            batch_equal = (recon == curr).view(recon.size(0), -1).all(dim=1)
            recon_exact_count += (batch_equal & sample_has_nontrivial).sum().item()
            total_samples += sample_has_nontrivial.sum().item()

        # Update progress bar
        print_progress_bar(batch_idx + 1, total_batches, prefix='Evaluating:', 
                          suffix=f'BCE: {total_bce/max(1,total_bce_samples):.4f}', length=50)

    avg_bce = total_bce / max(1, total_bce_samples)
    bit_acc = correct_prev_bits / max(1, total_pixels)
    recon_ok_ratio = recon_exact_count / max(1, total_samples)
    return avg_bce, bit_acc, recon_ok_ratio


def log_best_model_results(model: nn.Module, epoch: int, train_loss: float, val_bce: float, 
                          val_bit_acc: float, val_recon_ok: float, 
                          base: int = 128, latent_dim: int = 8,
                          refine_steps: int = 3, grid_size: str = "64x64"):
    """
    Log the best model results to results.txt file.
    
    Args:
        model: The trained model object
        epoch: Training epoch number
        train_loss: Training loss
        val_bce: Validation BCE loss
        val_bit_acc: Validation bit accuracy
        val_recon_ok: Validation reconstruction accuracy
        base: Base number of channels
        latent_dim: Number of latent dimensions
        refine_steps: Number of refinement steps
        grid_size: Grid size as string (e.g., "64x64")
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract model name from the model object
    model_name = model.__class__.__name__
    
    # Create results.txt if it doesn't exist and add header
    if not os.path.exists('results.txt'):
        with open('results.txt', 'w') as f:
            f.write("Date,Time,Model_Name,Base_Channels,Latent_Dim,Grid_Size,Refine_Steps,Epoch,Train_Loss,Val_BCE,Val_Bit_Accuracy,Val_Recon_Accuracy\n")
    
    # Append the new best result
    with open('results.txt', 'a') as f:
        f.write(f"{timestamp.split()[0]},{timestamp.split()[1]},{model_name},{base},{latent_dim},{grid_size},{refine_steps},{epoch},{train_loss:.6f},{val_bce:.6f},{val_bit_acc:.6f},{val_recon_ok:.6f}\n")
    
    print(f"Logged best model results to results.txt: Epoch {epoch}, Bit Acc: {val_bit_acc:.4f}, Recon: {val_recon_ok:.4f}")


def train_model():
    device = get_device()
    H, W = config.grid_size, config.grid_size

    train_ds = GoLReverseDataset(n_samples=config.train_samples, H=H, W=W, 
                                density=config.density, warmup_steps=config.warmup_steps, seed=1)
    val_ds = GoLReverseDataset(n_samples=config.val_samples, H=H, W=W, 
                              density=config.density, warmup_steps=config.warmup_steps, seed=2)

    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = create_model(base=config.base_channels, latent_dim=config.latent_dim, model_type=config.model_type, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Track best model
    best_bit_acc = 0.0
    best_recon_ok = 0.0
    best_epoch = 0
    no_improve_count = 0

    print(f"Starting training for {config.epochs} epochs...")
    print_progress_bar(0, config.epochs, prefix='Overall Progress:', suffix='Complete', length=50)
    
    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        print("=" * 50)
        
        tr_loss = train_epoch(model, train_dl, opt, device, config.refine_steps, config.steps_training, deep_supervision=config.deep_supervision)
        val_bce, bit_acc, recon_ok = eval_metrics(model, val_dl, device, config.refine_steps)
        print(f"\nEpoch {epoch:02d} Results:")
        print(f"  Train BCE: {tr_loss:.4f}")
        print(f"  Val BCE:   {val_bce:.4f}")
        print(f"  Bit Acc:   {bit_acc:.4f}")
        print(f"  Recon OK:  {recon_ok:.4f}")
        
        # Update overall progress bar
        print_progress_bar(epoch, config.epochs, prefix='Overall Progress:', 
                          suffix=f'Best Acc: {best_bit_acc:.4f}', length=50)
        
        # Save model checkpoint after each epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'train_loss': tr_loss,
            'val_bce': val_bce,
            'val_bit_acc': bit_acc,
            'val_recon_ok': recon_ok
        }
        torch.save(checkpoint, get_checkpoint_path(epoch))
        print(f"Saved checkpoint for epoch {epoch}")
        
        # Save best model based on bit accuracy
        if bit_acc > best_bit_acc:
            best_bit_acc = bit_acc
            best_epoch = epoch
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'train_loss': tr_loss,
                'val_bce': val_bce,
                'val_bit_acc': bit_acc,
                'val_recon_ok': recon_ok,
                'best_bit_acc': best_bit_acc
            }
            torch.save(best_checkpoint, get_checkpoint_path())
            print(f"New best model saved! Bit accuracy: {bit_acc:.4f}")
            no_improve_count = 0  # Reset counter when we find a better model
        else:
            no_improve_count += 1
            if no_improve_count >= config.patience:
                print(f"Early stopping at epoch {epoch} - no improvement for {config.patience} epochs")
                break

    print(f"\nTraining completed!")
    print(f"Best model was from epoch {best_epoch} with bit accuracy: {best_bit_acc:.4f}")
    
    # Log the best model results to results.txt after training is complete
    if best_epoch > 0:  # Only log if we found a best model
        log_best_model_results(
            model=model,
            epoch=best_epoch,
            train_loss=best_checkpoint['train_loss'],
            val_bce=best_checkpoint['val_bce'],
            val_bit_acc=best_checkpoint['val_bit_acc'],
            val_recon_ok=best_checkpoint['val_recon_ok'],
            base=config.base_channels,
            latent_dim=config.latent_dim,
            refine_steps=config.refine_steps,
            grid_size=f"{H}x{W}"
        )

    # Show one sample check
    with torch.no_grad():
        curr, prev = val_ds[0]
        curr = to_device(curr, device).unsqueeze(0)
        prev = to_device(prev, device).unsqueeze(0)
        out = refine(model, curr, steps=config.refine_steps)
        pred_prev = out.final_pred
        recon = gol_step(pred_prev)
        forward_ok = bool((recon == curr).all().item())
        
        # Calculate bit accuracy excluding trivial cases
        non_trivial_mask = get_non_trivial_mask(curr, device)
        
        if non_trivial_mask.sum() > 0:
            prev_bit_acc = float(((pred_prev == prev) & non_trivial_mask).float().sum().item() / non_trivial_mask.sum().item())
        else:
            prev_bit_acc = 0.0
            
        print("forward check ok:", forward_ok)
        print("prev bit accuracy (non-trivial):", prev_bit_acc)
    
    # Example of how to use the visualization function
    print("\nTo visualize the reverse process, run:")
    print(f"python test.py --checkpoint {get_checkpoint_path()}")


if __name__ == "__main__":
    ensure_checkpoint_dir()
    train_model() 