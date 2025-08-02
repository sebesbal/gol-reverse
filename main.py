import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import random

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


def dilate_3x3(mask_bin: torch.Tensor) -> torch.Tensor:
    """
    Dilate a binary mask with a 3x3 neighborhood.

    Args:
        mask_bin: Tensor [B, 1, H, W] in {0, 1}.

    Returns:
        dilated mask of the same shape.
    """
    return F.max_pool2d(mask_bin, kernel_size=3, stride=1, padding=1)


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

    def __init__(self, n_samples: int, H: int, W: int, density: float = 0.15, mix_steps: int = 0, seed: int = 0):
        self.n = n_samples
        self.H = H
        self.W = W
        self.density = density
        self.mix_steps = mix_steps
        self.seed = seed
        random.seed(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Start from random binary field
        prev = torch.rand(1, self.H, self.W)
        prev = (prev < self.density).float()

        # Optional mixing to create more natural patterns
        for _ in range(self.mix_steps):
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

    Output:
        logits for the previous state (before sigmoid)
    """

    def __init__(self, in_ch: int = 3, base: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------------
# Iterative refinement loop
# -------------------------------

@dataclass
class RefineOutput:
    logits_per_iter: list
    probs_per_iter: list
    bin_per_iter: list
    errmask_per_iter: list


def refine(model: nn.Module, current_bin: torch.Tensor, steps: int) -> RefineOutput:
    """
    Perform iterative prediction and error-driven refinement.

    Args:
        model: RefinementCNN
        current_bin: Tensor [B, 1, H, W] in {0, 1}
        steps: number of refinement iterations

    Returns:
        RefineOutput with lists across iterations.
    """
    assert current_bin.dim() == 4 and current_bin.size(1) == 1, f"expected [B,1,H,W], got {tuple(current_bin.shape)}"
    pred_prev = torch.zeros_like(current_bin)
    err = torch.zeros_like(current_bin)

    logits_hist, probs_hist, bin_hist, err_hist = [], [], [], []

    # Allow grads during training, disable in eval automatically
    with torch.set_grad_enabled(model.training):
        for _ in range(steps):
            inp = torch.cat([current_bin, pred_prev, err], dim=1)
            logits = model(inp)
            probs = torch.sigmoid(logits)
            pred_prev = (probs > 0.5).float()

            # Forward check and error dilation
            next_from_pred = gol_step(pred_prev)
            mismatch = (next_from_pred != current_bin).float()
            err = dilate_3x3(mismatch)

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

    for curr, prev in loader:
        curr = curr.to(device)
        prev = prev.to(device)  # [B, 1, H, W]
        opt.zero_grad()

        out = refine(model, curr, refine_steps)
        if deep_supervision:
            losses = [bce(logits, prev) for logits in out.logits_per_iter]
            loss = sum(losses) / len(losses)
        else:
            loss = bce(out.logits_per_iter[-1], prev)

        loss.backward()
        opt.step()

        batch_size = curr.size(0)
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
        - Average BCE loss
        - Bit accuracy on previous state
        - Fraction of samples where forward reconstruction exactly matches current
    """
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="sum")

    total_bce = 0.0
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

        # Supervised BCE loss
        total_bce += bce(logits, prev).item()

        # Bit accuracy on previous state
        correct_prev_bits += (pred_prev == prev).sum().item()
        total_pixels += prev.numel()

        # Exact forward reconstruction check per sample
        recon = gol_step(pred_prev)
        # Compare per-sample equality
        batch_equal = (recon == curr).view(recon.size(0), -1).all(dim=1)
        recon_exact_count += batch_equal.sum().item()

        total_samples += curr.size(0)

    avg_bce = total_bce / max(1, total_samples)
    bit_acc = correct_prev_bits / max(1, total_pixels)
    recon_ok_ratio = recon_exact_count / max(1, total_samples)
    return avg_bce, bit_acc, recon_ok_ratio


# -------------------------------
# Visualization
# -------------------------------

def visualize_reverse_gol(checkpoint_path: str, 
                         H: int = 64, W: int = 64, 
                         forward_steps: int = 5, 
                         refine_steps: int = 3,
                         density: float = 0.15,
                         seed: int = 42):
    """
    Visualize the reverse Game of Life process using a saved model.
    
    Args:
        checkpoint_path: Path to saved model checkpoint
        H, W: Grid dimensions
        forward_steps: Number of forward Game of Life steps to simulate
        refine_steps: Number of refinement iterations for reverse process
        density: Initial density for random state generation
        seed: Random seed for reproducibility
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import ListedColormap
    
    # Set random seed
    random.seed(seed)
    torch.manual_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = RefinementCNN(in_ch=3, base=48).to(device)
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
    
    # Reverse reconstruction
    print(f"\nReverse reconstruction ({refine_steps} refinement steps):")
    out = refine(model, final_state.unsqueeze(0).to(device), refine_steps)
    reconstructed_states = []
    
    for i, pred_prev in enumerate(out.bin_per_iter):
        reconstructed_states.append(pred_prev.squeeze(0))
        print(f"  Refinement {i+1}: {pred_prev.sum().item():.0f} live cells")
    
    # Prepare ground truth states (reverse of forward simulation)
    ground_truth_states = list(reversed(states_forward[:-1]))  # Exclude final state
    
    # Create visualization with only 2 rows (ground truth and model reconstruction)
    # Show only refine_steps + 1 columns (input state + refinement steps)
    n_steps = refine_steps + 1
    fig, axes = plt.subplots(2, n_steps, figsize=(15, 6))
    fig.suptitle(f'Game of Life Reverse Process Visualization\n'
                 f'Model: Epoch {checkpoint["epoch"]}, '
                 f'Bit Acc: {checkpoint["val_bit_acc"]:.3f}, '
                 f'Recon: {checkpoint["val_recon_ok"]:.3f}', fontsize=14)
    
    # Custom colormap: white (dead), black (alive), red (difference)
    colors = ['white', 'black', 'red']
    cmap = ListedColormap(colors)
    
    # Column headers
    for i in range(n_steps):
        if i == 0:
            axes[0, i].set_title('Input State', fontweight='bold', fontsize=10)
        else:
            axes[0, i].set_title(f'Refinement {i}', fontweight='bold', fontsize=10)
    
    # Row labels
    axes[0, 0].text(-0.3, 0.5, 'Ground Truth\n(Reverse)', transform=axes[0, 0].transAxes, 
                     ha='center', va='center', fontweight='bold', fontsize=12, rotation=90)
    axes[1, 0].text(-0.3, 0.5, 'Model\nReconstruction', transform=axes[1, 0].transAxes, 
                     ha='center', va='center', fontweight='bold', fontsize=12, rotation=90)
    
    # Row 1: Ground truth (reverse of forward simulation)
    # First column: input state (final state from forward simulation)
    axes[0, 0].imshow(final_state.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
    axes[0, 0].axis('off')
    axes[0, 0].text(0.5, -0.1, f'{final_state.sum().item():.0f} cells (input)', 
                    ha='center', transform=axes[0, 0].transAxes, fontsize=8)
    
    # Subsequent columns: ground truth reverse states (show only up to refine_steps)
    for i in range(min(len(ground_truth_states), refine_steps)):
        state = ground_truth_states[i]
        axes[0, i+1].imshow(state.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
        axes[0, i+1].axis('off')
        axes[0, i+1].text(0.5, -0.1, f'{state.sum().item():.0f} cells', 
                        ha='center', transform=axes[0, i+1].transAxes, fontsize=8)
    
    # Hide unused subplots in first row
    for i in range(min(len(ground_truth_states), refine_steps) + 1, n_steps):
        axes[0, i].axis('off')
    
    # Row 2: Model reconstruction
    # First column: input state (same as ground truth)
    axes[1, 0].imshow(final_state.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
    axes[1, 0].axis('off')
    axes[1, 0].text(0.5, -0.1, f'{final_state.sum().item():.0f} cells (input)', 
                    ha='center', transform=axes[1, 0].transAxes, fontsize=8)
    
    # Subsequent columns: model reconstruction iterations (show only up to refine_steps)
    for i in range(min(len(reconstructed_states), refine_steps)):
        recon_state = reconstructed_states[i]
        if i < len(ground_truth_states):
            gt_state = ground_truth_states[i].to(device)
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
            
            # Calculate accuracy
            accuracy = ((recon_state > 0.5) == (gt_state > 0.5)).float().mean().item()
            axes[1, i+1].text(0.5, -0.1, f'{recon_state.sum().item():.0f} cells\n{accuracy:.3f} acc', 
                            ha='center', transform=axes[1, i+1].transAxes, fontsize=8)
        else:
            axes[1, i+1].imshow(recon_state.squeeze().cpu(), cmap='gray', vmin=0, vmax=1)
            axes[1, i+1].axis('off')
            axes[1, i+1].text(0.5, -0.1, f'{recon_state.sum().item():.0f} cells', 
                            ha='center', transform=axes[1, i+1].transAxes, fontsize=8)
    
    # Hide unused subplots in second row
    for i in range(min(len(reconstructed_states), refine_steps) + 1, n_steps):
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
    
    if len(reconstructed_states) > 0 and len(ground_truth_states) > 0:
        final_recon = reconstructed_states[-1]
        final_gt = ground_truth_states[0].to(device)  # First step in reverse
        final_accuracy = ((final_recon > 0.5) == (final_gt > 0.5)).float().mean().item()
        print(f"Final reconstruction accuracy: {final_accuracy:.4f}")
        
        # Check if forward reconstruction matches
        recon_forward = gol_step(final_recon.unsqueeze(0)).squeeze(0)
        # Move both tensors to the same device for comparison
        forward_match = torch.allclose(recon_forward.to(device), final_state.to(device), atol=1e-6)
        print(f"Forward reconstruction matches: {forward_match}")
    
    return model, states_forward, reconstructed_states, ground_truth_states


# -------------------------------
# Main
# -------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, W = 64, 64

    train_ds = GoLReverseDataset(n_samples=8000, H=H, W=W, density=0.15, mix_steps=2, seed=1)
    val_ds = GoLReverseDataset(n_samples=1000, H=H, W=W, density=0.15, mix_steps=2, seed=2)

    # Set num_workers=0 for portability across OSes
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model = RefinementCNN(in_ch=3, base=48).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    refine_steps = 3
    epochs = 10
    
    # Track best model
    best_bit_acc = 0.0
    best_recon_ok = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_dl, opt, device, refine_steps, deep_supervision=True)
        val_bce, bit_acc, recon_ok = eval_metrics(model, val_dl, device, refine_steps)
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

    print(f"\nTraining completed!")
    print(f"Best model was from epoch {best_epoch} with bit accuracy: {best_bit_acc:.4f}")

    # Show one sample check
    with torch.no_grad():
        curr, prev = val_ds[0]
        curr = curr.to(device).unsqueeze(0)
        prev = prev.to(device).unsqueeze(0)
        out = refine(model, curr, steps=refine_steps)
        pred_prev = (torch.sigmoid(out.logits_per_iter[-1]) > 0.5).float()
        recon = gol_step(pred_prev)
        forward_ok = bool((recon == curr).all().item())
        prev_bit_acc = float((pred_prev == prev).float().mean().item())
        print("forward check ok:", forward_ok)
        print("prev bit accuracy:", prev_bit_acc)
    
    # Example of how to use the visualization function
    print("\nTo visualize the reverse process, run:")
    print("visualize_reverse_gol('checkpoints/best_model.pth', forward_steps=5, refine_steps=3)")
    print("or")
    print("visualize_reverse_gol('checkpoints/checkpoint_epoch_10.pth', forward_steps=5, refine_steps=3)")


if __name__ == "__main__":
    # Check if we have a trained model to visualize
    import os
    os.makedirs('checkpoints', exist_ok=True)
    forward_steps=15
    refine_steps=3
    if os.path.exists('checkpoints/best_model.pth'):
        print("Found best model, running visualization...")
        visualize_reverse_gol('checkpoints/best_model.pth', forward_steps=forward_steps, refine_steps=refine_steps)
    elif os.path.exists('checkpoints/checkpoint_epoch_10.pth'):
        print("Found epoch 10 checkpoint, running visualization...")
        visualize_reverse_gol('checkpoints/checkpoint_epoch_10.pth', forward_steps=forward_steps, refine_steps=refine_steps)
    else:
        print("No trained model found. Running training first...")
        main()
        print("\nTraining completed! Now running visualization...")
        visualize_reverse_gol('checkpoints/best_model.pth', forward_steps=forward_steps, refine_steps=refine_steps)
