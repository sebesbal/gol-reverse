import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import random
import os
from datetime import datetime

# -------------------------------
# Configuration
# -------------------------------

@dataclass
class Config:
    """Centralized configuration for all parameters."""
    # Model parameters
    base_channels: int = 128
    latent_dim: int = 8
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 10
    patience: int = 5  # Early stopping patience
    
    # Dataset parameters
    grid_size: int = 64
    density: float = 0.15
    warmup_steps: int = 10
    train_samples: int = 8000
    val_samples: int = 1000
    
    # Refinement parameters
    refine_steps: int = 10
    steps_training: int = 3  # Number of steps with gradients enabled during training
    
    # Visualization parameters
    forward_steps: int = 50
    reverse_steps: int = 5  # Number of CA steps to go backwards
    seed: int = 42
    
    # DataLoader parameters
    num_workers: int = 16

# Global configuration instance
config = Config()

# -------------------------------
# Game of Life forward simulator
# -------------------------------

def gol_step(x_bin: torch.Tensor) -> torch.Tensor:
    """
    Compute one Conway's Game of Life step.

    Args:
        x_bin: Tensor of shape [B, 1, H, W] with values in {0, 1} (float).

    Returns:
        next_state: Tensor of shape [B, 1, H, W] in {0, 1}.
    """
    device = x_bin.device
    kernel = torch.tensor(
        [[1, 1, 1],
         [1, 0, 1],
         [1, 1, 1]],
        dtype=x_bin.dtype,
        device=device
    ).view(1, 1, 3, 3)

    neighbors = F.conv2d(x_bin, kernel, padding=1)

    alive = x_bin > 0.5
    born = (~alive) & (neighbors == 3)
    survive = alive & ((neighbors == 2) | (neighbors == 3))

    next_state = torch.where(born | survive, torch.ones_like(x_bin), torch.zeros_like(x_bin))
    return next_state


def count_neighbors_3x3(mask_bin: torch.Tensor) -> torch.Tensor:
    """
    Count the number of mismatching neighbors in a 3x3 neighborhood for each cell.

    Args:
        mask_bin: Tensor [B, 1, H, W] in {0, 1}.

    Returns:
        Tensor of same shape, where each value is the count of mismatching neighbors (0-9).
    """
    kernel = torch.ones((1, 1, 3, 3), dtype=mask_bin.dtype, device=mask_bin.device)
    # Sum over 3x3 neighborhood (including center)
    neighbor_count = F.conv2d(mask_bin, kernel, padding=1)
    return neighbor_count

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
        random.seed(seed)

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
# Model
# -------------------------------

class RefinementCNN(nn.Module):
    """
    Input channels:
        c0 = current state
        c1 = previous prediction (initially zeros)
        c2 = error mask (initially zeros)
        c3-c6 = latent variables (initially zeros)

    Output:
        logits for the previous state (before sigmoid)
        latent variables for next iteration
    """

    def __init__(self, base: int = 32, latent_dim: int = 4):
        super().__init__()
        in_ch = latent_dim + 3
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),  # Add dropout for regularization
            nn.Conv2d(base, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),  # Add dropout for regularization
            nn.Conv2d(base, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),  # Add dropout for regularization
            nn.Conv2d(base, 1 + latent_dim, 1)  # 1 for previous state + latent_dim for latent variables
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RefinementCNN2(nn.Module):
    """
    Larger model with residual connections, batch normalization, and more capacity.
    
    Input channels:
        c0 = current state
        c1 = previous prediction (initially zeros)
        c2 = error mask (initially zeros)
        c3-c6 = latent variables (initially zeros)

    Output:
        logits for the previous state (before sigmoid)
        latent variables for next iteration
    """

    def __init__(self, base: int = 128, latent_dim: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        in_ch = latent_dim + 3  # Always use this relationship

        # Initial feature extraction
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.res_block1 = self._make_residual_block(base, base)
        self.res_block2 = self._make_residual_block(base, base)
        self.res_block3 = self._make_residual_block(base, base)
        self.res_block4 = self._make_residual_block(base, base)
        
        # Additional processing layers
        self.mid_conv = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),  # Add dropout for regularization
            nn.Conv2d(base, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True)
        )
        
        # Final output layers
        self.output_conv = nn.Sequential(
            nn.Conv2d(base, base // 2, 3, padding=1),
            nn.BatchNorm2d(base // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),  # Add dropout for regularization
            nn.Conv2d(base // 2, 1 + latent_dim, 1)  # 1 for previous state + latent_dim for latent variables
        )
        
    def _make_residual_block(self, in_channels, out_channels):
        """Create a residual block with batch normalization."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial feature extraction
        out = self.input_conv(x)
        
        # Residual blocks with skip connections
        residual = out
        out = self.res_block1(out)
        out = out + residual  # Skip connection
        out = F.relu(out)
        
        residual = out
        out = self.res_block2(out)
        out = out + residual  # Skip connection
        out = F.relu(out)
        
        residual = out
        out = self.res_block3(out)
        out = out + residual  # Skip connection
        out = F.relu(out)
        
        residual = out
        out = self.res_block4(out)
        out = out + residual  # Skip connection
        out = F.relu(out)
        
        # Additional processing
        out = self.mid_conv(out)
        
        # Final output
        out = self.output_conv(out)
        
        return out


# -------------------------------
# Iterative refinement loop
# -------------------------------

@dataclass
class RefineOutput:
    logits_per_iter: list
    probs_per_iter: list
    bin_per_iter: list
    errmask_per_iter: list


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


def refine(model: nn.Module, current_bin: torch.Tensor, steps: int, steps_training: int = None) -> RefineOutput:
    """
    Perform iterative prediction and error-driven refinement.

    Args:
        model: RefinementCNN
        current_bin: Tensor [B, 1, H, W] in {0, 1}
        steps: number of refinement iterations
        steps_training: number of steps with gradients enabled during training (if None, use all steps)

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

    # Determine gradient behavior based on training mode and steps_training parameter
    if model.training and steps_training is not None:
        # Training mode with specified steps_training
        import random
        max_no_grad = steps - steps_training
        steps_no_grad = random.randint(0, max_no_grad) if max_no_grad > 0 else 0
        steps_with_grad = steps - steps_no_grad
        
        # First do steps without gradients
        for _ in range(steps_no_grad):
            with torch.set_grad_enabled(False):
                logits, probs, pred_prev, err, latent = _refinement_step(
                    model, current_bin, pred_prev, err, latent
                )
                logits_hist.append(logits)
                probs_hist.append(probs)
                bin_hist.append(pred_prev)
                err_hist.append(err)
        
        # Then do steps with gradients
        for _ in range(steps_with_grad):
            with torch.set_grad_enabled(True):
                logits, probs, pred_prev, err, latent = _refinement_step(
                    model, current_bin, pred_prev, err, latent
                )
                logits_hist.append(logits)
                probs_hist.append(probs)
                bin_hist.append(pred_prev)
                err_hist.append(err)
    else:
        # Testing mode or training without steps_training specified - use all steps with model's training state
        with torch.set_grad_enabled(model.training):
            for _ in range(steps):
                logits, probs, pred_prev, err, latent = _refinement_step(
                    model, current_bin, pred_prev, err, latent
                )
                logits_hist.append(logits)
                probs_hist.append(probs)
                bin_hist.append(pred_prev)
                err_hist.append(err)

    return RefineOutput(logits_hist, probs_hist, bin_hist, err_hist)


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
        BCEWithLogitsLoss toward the true previous state (excluding trivial cases).
        If deep_supervision is True, average loss across all refinement steps.
    """
    model.train()
    bce = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_samples = 0

    for curr, prev in loader:
        curr = curr.to(device)
        prev = prev.to(device)  # [B, 1, H, W]
        opt.zero_grad()

        out = refine(model, curr, refine_steps, steps_training=steps_training)
        
        # Create neighbor count mask for current state to identify non-trivial cases
        kernel = torch.tensor(
            [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]],
            dtype=curr.dtype,
            device=device
        ).view(1, 1, 3, 3)
        
        neighbor_counts = F.conv2d(curr, kernel, padding=1)
        non_trivial_mask = neighbor_counts > 0
        
        if deep_supervision:
            losses = []
            for logits in out.logits_per_iter:
                if non_trivial_mask.sum() > 0:
                    # Apply mask to logits and prev for BCE calculation
                    masked_logits = logits[non_trivial_mask]
                    masked_prev = prev[non_trivial_mask]
                    if masked_logits.numel() > 0:
                        losses.append(bce(masked_logits, masked_prev))
                    else:
                        losses.append(torch.tensor(0.0, device=device))
                else:
                    losses.append(torch.tensor(0.0, device=device))
            loss = sum(losses) / len(losses)
        else:
            if non_trivial_mask.sum() > 0:
                # Apply mask to logits and prev for BCE calculation
                masked_logits = out.logits_per_iter[-1][non_trivial_mask]
                masked_prev = prev[non_trivial_mask]
                if masked_logits.numel() > 0:
                    loss = bce(masked_logits, masked_prev)
                else:
                    loss = torch.tensor(0.0, device=device)
            else:
                loss = torch.tensor(0.0, device=device)

        loss.backward()
        opt.step()

        # Count non-trivial samples for loss averaging
        sample_has_nontrivial = non_trivial_mask.view(curr.size(0), -1).any(dim=1)
        batch_size = sample_has_nontrivial.sum().item()
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(1, total_samples)


@torch.no_grad()
def eval_metrics(model: nn.Module,
                 loader: DataLoader,
                 device: str,
                 refine_steps: int = 3):
    """
    Evaluate:
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

    for curr, prev in loader:
        curr = curr.to(device)
        prev = prev.to(device)

        out = refine(model, curr, refine_steps)
        logits = out.logits_per_iter[-1]
        probs = torch.sigmoid(logits)
        pred_prev = (probs > 0.5).float()

        # Create neighbor count mask for current state
        kernel = torch.tensor(
            [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]],
            dtype=curr.dtype,
            device=device
        ).view(1, 1, 3, 3)
        
        neighbor_counts = F.conv2d(curr, kernel, padding=1)
        # Mask for non-trivial cases (where there are neighbors)
        non_trivial_mask = neighbor_counts > 0
        
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

    avg_bce = total_bce / max(1, total_bce_samples)
    bit_acc = correct_prev_bits / max(1, total_pixels)
    recon_ok_ratio = recon_exact_count / max(1, total_samples)
    return avg_bce, bit_acc, recon_ok_ratio


# -------------------------------
# Visualization
# -------------------------------

def visualize_reverse_gol(checkpoint_path: str, 
                         H: int = None, W: int = None, 
                         forward_steps: int = None, 
                         refine_steps: int = None,
                         density: float = None,
                         seed: int = None):
    """
    Visualize the reverse Game of Life process using a saved model.
    
    Args:
        checkpoint_path: Path to saved model checkpoint
        H, W: Grid dimensions (uses config.grid_size if None)
        forward_steps: Number of forward Game of Life steps to simulate (uses config.forward_steps if None)
        refine_steps: Number of refinement iterations for reverse process (uses config.refine_steps if None)
        density: Initial density for random state generation (uses config.density if None)
        seed: Random seed for reproducibility (uses config.seed if None)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import ListedColormap
    
    # Use config values if parameters are None
    H = H or config.grid_size
    W = W or config.grid_size
    forward_steps = forward_steps or config.forward_steps
    refine_steps = refine_steps or config.refine_steps
    density = density or config.density
    seed = seed or config.seed
    
    # Set random seed
    random.seed(seed)
    torch.manual_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_model(base=config.base_channels, latent_dim=config.latent_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Model validation metrics: BCE={checkpoint['val_bce']:.4f}, "
          f"Bit Acc={checkpoint['val_bit_acc']:.4f}, Recon={checkpoint['val_recon_ok']:.4f}")
    
    # Generate initial random state
    initial_state = torch.rand(1, H, W)
    initial_state = (initial_state < density).float()
    
    # Forward simulation
    print(f"\nForward simulation ({forward_steps} steps):")
    states_forward = [initial_state]
    current = initial_state
    
    for step in range(forward_steps):
        current = gol_step(current.unsqueeze(0)).squeeze(0)
        states_forward.append(current)
        print(f"  Step {step+1}: {current.sum().item():.0f} live cells")
    
    final_state = states_forward[-1]
    
    # Reverse reconstruction - multiple CA steps backwards
    print(f"\nReverse reconstruction ({config.reverse_steps} CA steps backwards):")
    reconstructed_states = []
    current_state = final_state
    
    for step in range(config.reverse_steps):
        # Use refine() to get the previous state (one CA step backwards)
        out = refine(model, current_state.unsqueeze(0).to(device), refine_steps)
        # Take the final prediction from the refinement process
        prev_state = (torch.sigmoid(out.logits_per_iter[-1]) > 0.5).float().squeeze(0)
        reconstructed_states.append(prev_state)
        print(f"  Reverse step {step+1}: {prev_state.sum().item():.0f} live cells")
        current_state = prev_state
    
    # Create visualization showing multiple reverse CA steps
    n_steps = config.reverse_steps + 1
    fig, axes = plt.subplots(2, n_steps, figsize=(15, 6))
    # Compute accuracy for this specific experiment (excluding trivial cases)
    experiment_accuracy = 0.0
    if len(reconstructed_states) > 0:
        total_acc = 0.0
        valid_steps = 0
        for i, recon_state in enumerate(reconstructed_states):
            gt_idx = len(states_forward) - 2 - i  # Corresponding ground truth
            if gt_idx >= 0:
                gt_state = states_forward[gt_idx].to(device)
                
                                # Create neighbor count mask for current state to identify non-trivial cases
                kernel = torch.tensor(
                    [[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]],
                    dtype=final_state.dtype,
                    device=device
                ).view(1, 1, 3, 3)
                
                # Use the state that was input to this reverse step
                if i == 0:
                    input_state = final_state.to(device)
                else:
                    input_state = reconstructed_states[i-1].to(device)
                
                neighbor_counts = F.conv2d(input_state.unsqueeze(0), kernel, padding=1).squeeze()
                non_trivial_mask = neighbor_counts > 0
                
                if non_trivial_mask.sum() > 0:
                    # Calculate accuracy only for non-trivial cases
                    correct_pixels = ((recon_state > 0.5) == (gt_state > 0.5)) & non_trivial_mask
                    acc = correct_pixels.float().sum().item() / non_trivial_mask.sum().item()
                    total_acc += acc
                    valid_steps += 1
        if valid_steps > 0:
            experiment_accuracy = total_acc / valid_steps
    
    fig.suptitle(f'Game of Life Reverse Process Visualization\n'
                 f'Model: Epoch {checkpoint["epoch"]}, '
                 f'Experiment Accuracy: {experiment_accuracy:.3f}', fontsize=14)
    
    # Custom colormap: white (dead), black (alive), red (difference)
    colors = ['white', 'black', 'red']
    cmap = ListedColormap(colors)
    
    # Column headers
    for i in range(n_steps):
        if i == 0:
            axes[0, i].set_title('Final State', fontweight='bold', fontsize=10)
        else:
            axes[0, i].set_title(f'Reverse Step {i}', fontweight='bold', fontsize=10)
    
    # Row labels
    axes[0, 0].text(-0.3, 0.5, 'Ground Truth', transform=axes[0, 0].transAxes, 
                     ha='center', va='center', fontweight='bold', fontsize=12, rotation=90)
    axes[1, 0].text(-0.3, 0.5, 'Model\nPrediction', transform=axes[1, 0].transAxes, 
                     ha='center', va='center', fontweight='bold', fontsize=12, rotation=90)
    
    # Row 1: Ground truth (forward simulation states in reverse order)
    # First column: final state from forward simulation
    axes[0, 0].imshow(final_state.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
    axes[0, 0].axis('off')
    axes[0, 0].text(0.5, -0.1, f'{final_state.sum().item():.0f} cells', 
                    ha='center', transform=axes[0, 0].transAxes, fontsize=8)
    
    # Subsequent columns: ground truth states going backwards
    for i in range(config.reverse_steps):
        gt_idx = len(states_forward) - 2 - i  # Go backwards from final state
        if gt_idx >= 0:
            gt_state = states_forward[gt_idx]
            axes[0, i+1].imshow(gt_state.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
            axes[0, i+1].axis('off')
            axes[0, i+1].text(0.5, -0.1, f'{gt_state.sum().item():.0f} cells', 
                            ha='center', transform=axes[0, i+1].transAxes, fontsize=8)
        else:
            axes[0, i+1].axis('off')
            axes[0, i+1].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[0, i+1].transAxes)
    
    # Row 2: Model predictions
    # First column: final state (same as ground truth)
    axes[1, 0].imshow(final_state.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
    axes[1, 0].axis('off')
    axes[1, 0].text(0.5, -0.1, f'{final_state.sum().item():.0f} cells', 
                    ha='center', transform=axes[1, 0].transAxes, fontsize=8)
    
    # Subsequent columns: model predictions for each reverse step
    for i in range(len(reconstructed_states)):
        recon_state = reconstructed_states[i]
        gt_idx = len(states_forward) - 2 - i  # Corresponding ground truth
        
        if gt_idx >= 0:
            gt_state = states_forward[gt_idx].to(device)
            
            # 0 = no difference, 1 = model alive but GT dead, 2 = model dead but GT alive
            diff_mask = torch.zeros_like(recon_state)
            diff_mask = torch.where((recon_state > 0.5) & (gt_state < 0.5), 
                                  torch.ones_like(diff_mask), diff_mask)  # Model alive, GT dead
            diff_mask = torch.where((recon_state < 0.5) & (gt_state > 0.5), 
                                  torch.full_like(diff_mask, 2), diff_mask)  # Model dead, GT alive
            
            # Combine states for visualization
            vis_state = recon_state.clone()
            vis_state = torch.where(diff_mask > 0, diff_mask, vis_state)
            
            axes[1, i+1].imshow(vis_state.squeeze().cpu(), cmap=cmap, vmin=0, vmax=2)
            axes[1, i+1].axis('off')
            
            # Calculate accuracy (excluding trivial cases)
            # Use the state that was input to this reverse step
            if i == 0:
                input_state = final_state.to(device)
            else:
                input_state = reconstructed_states[i-1].to(device)
            
            neighbor_counts = F.conv2d(input_state.unsqueeze(0), kernel, padding=1).squeeze()
            non_trivial_mask = neighbor_counts > 0
            
            if non_trivial_mask.sum() > 0:
                correct_pixels = ((recon_state > 0.5) == (gt_state > 0.5)) & non_trivial_mask
                accuracy = correct_pixels.float().sum().item() / non_trivial_mask.sum().item()
            else:
                accuracy = 0.0
            axes[1, i+1].text(0.5, -0.1, f'{recon_state.sum().item():.0f} cells\n{accuracy:.3f} acc', 
                            ha='center', transform=axes[1, i+1].transAxes, fontsize=8)
        else:
            axes[1, i+1].axis('off')
            axes[1, i+1].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[1, i+1].transAxes)
    
    # Hide unused subplots
    for i in range(len(reconstructed_states) + 1, n_steps):
        axes[1, i].axis('off')
    
    # Add legend for difference visualization
    legend_elements = [
        patches.Patch(color='white', label='Dead (Correct)'),
        patches.Patch(color='black', label='Alive (Correct)'),
        patches.Patch(color='red', label='Difference')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"Initial state: {initial_state.sum().item():.0f} live cells")
    print(f"Final state: {final_state.sum().item():.0f} live cells")
    
    if len(reconstructed_states) > 0:
        # Calculate accuracy for each reverse step
        total_accuracy = 0.0
        valid_steps = 0
        
        for i, recon_state in enumerate(reconstructed_states):
            gt_idx = len(states_forward) - 2 - i  # Corresponding ground truth
            if gt_idx >= 0:
                gt_state = states_forward[gt_idx].to(device)
                
                # Use the state that was input to this reverse step
                if i == 0:
                    input_state = final_state.to(device)
                else:
                    input_state = reconstructed_states[i-1].to(device)
                
                neighbor_counts = F.conv2d(input_state.unsqueeze(0), kernel, padding=1).squeeze()
                non_trivial_mask = neighbor_counts > 0
                
                if non_trivial_mask.sum() > 0:
                    correct_pixels = ((recon_state > 0.5) == (gt_state > 0.5)) & non_trivial_mask
                    accuracy = correct_pixels.float().sum().item() / non_trivial_mask.sum().item()
                    total_accuracy += accuracy
                    valid_steps += 1
                    print(f"Reverse step {i+1} accuracy: {accuracy:.4f}")
                else:
                    print(f"Reverse step {i+1}: No non-trivial cases")
        
        if valid_steps > 0:
            avg_accuracy = total_accuracy / valid_steps
            print(f"Average reverse accuracy: {avg_accuracy:.4f}")
        
        # Check if forward reconstruction matches for the first reverse step
        first_recon = reconstructed_states[0]
        recon_forward = gol_step(first_recon.unsqueeze(0)).squeeze(0)
        forward_match = torch.allclose(recon_forward.to(device), final_state.to(device), atol=1e-6)
        print(f"Forward reconstruction matches: {forward_match}")
    
    return model, states_forward, reconstructed_states

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

# -------------------------------
# Main
# -------------------------------

def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, W = config.grid_size, config.grid_size

    train_ds = GoLReverseDataset(n_samples=config.train_samples, H=H, W=W, 
                                density=config.density, warmup_steps=config.warmup_steps, seed=1)
    val_ds = GoLReverseDataset(n_samples=config.val_samples, H=H, W=W, 
                              density=config.density, warmup_steps=config.warmup_steps, seed=2)

    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = create_model(base=config.base_channels, latent_dim=config.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Track best model
    best_bit_acc = 0.0
    best_recon_ok = 0.0
    best_epoch = 0
    no_improve_count = 0

    for epoch in range(1, config.epochs + 1):
        tr_loss = train_epoch(model, train_dl, opt, device, config.refine_steps, config.steps_training, deep_supervision=True)
        val_bce, bit_acc, recon_ok = eval_metrics(model, val_dl, device, config.refine_steps)
        print(f"epoch {epoch:02d}  train_bce {tr_loss:.4f}  val_bce {val_bce:.4f}  bit_acc {bit_acc:.4f}  recon_ok {recon_ok:.4f}")
        
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
        torch.save(checkpoint, f'checkpoints/checkpoint_epoch_{epoch:02d}.pth')
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
            torch.save(best_checkpoint, 'checkpoints/best_model.pth')
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
        curr = curr.to(device).unsqueeze(0)
        prev = prev.to(device).unsqueeze(0)
        out = refine(model, curr, steps=config.refine_steps)
        pred_prev = (torch.sigmoid(out.logits_per_iter[-1]) > 0.5).float()
        recon = gol_step(pred_prev)
        forward_ok = bool((recon == curr).all().item())
        
        # Calculate bit accuracy excluding trivial cases
        kernel = torch.tensor(
            [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]],
            dtype=curr.dtype,
            device=device
        ).view(1, 1, 3, 3)
        
        neighbor_counts = F.conv2d(curr, kernel, padding=1)
        non_trivial_mask = neighbor_counts > 0
        
        if non_trivial_mask.sum() > 0:
            prev_bit_acc = float(((pred_prev == prev) & non_trivial_mask).float().sum().item() / non_trivial_mask.sum().item())
        else:
            prev_bit_acc = 0.0
            
        print("forward check ok:", forward_ok)
        print("prev bit accuracy (non-trivial):", prev_bit_acc)
    
    # Example of how to use the visualization function
    print("\nTo visualize the reverse process, run:")
    print(f"visualize_reverse_gol('checkpoints/best_model.pth', forward_steps={config.forward_steps}, refine_steps={config.refine_steps})")
    print("or")
    print(f"visualize_reverse_gol('checkpoints/checkpoint_epoch_10.pth', forward_steps={config.forward_steps}, refine_steps={config.refine_steps})")


def create_model(base: int = None, latent_dim: int = None):
    """Create a model with the specified parameters or use config defaults."""
    base = base or config.base_channels
    latent_dim = latent_dim or config.latent_dim
    return RefinementCNN(base=base, latent_dim=latent_dim)


if __name__ == "__main__":
    train = False
    os.makedirs('checkpoints', exist_ok=True)
    if train:
        train_model()
        print("\nTraining completed! Now running visualization...")
        visualize_reverse_gol('checkpoints/best_model.pth')
    else:
        if os.path.exists('checkpoints/best_model.pth'):
            print("Found best model, running visualization...")
            visualize_reverse_gol('checkpoints/best_model.pth')
        elif os.path.exists('checkpoints/checkpoint_epoch_10.pth'):
            print("Found epoch 10 checkpoint, running visualization...")
            visualize_reverse_gol('checkpoints/checkpoint_epoch_10.pth')

