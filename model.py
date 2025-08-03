import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import os

# -------------------------------
# Constants and Paths
# -------------------------------

CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
CHECKPOINT_TEMPLATE = os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_{:02d}.pth")

# -------------------------------
# Configuration
# -------------------------------

@dataclass
class Config:
    """Centralized configuration for all parameters."""
    # Model parameters
    model_type: str = "RefinementCNN"  # Options: "RefinementCNN", "RefinementCNN2"
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
    deep_supervision: bool = False  # Whether to use deep supervision (requires history storage)
    
    # Visualization parameters
    forward_steps: int = 50
    reverse_steps: int = 5  # Number of CA steps to go backwards
    seed: int = 42
    
    # DataLoader parameters
    num_workers: int = 16

# Global configuration instance
config = Config()

# -------------------------------
# Utility Functions
# -------------------------------

def get_device():
    """Get the appropriate device (CUDA if available, else CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def setup_seed(seed=None):
    """Set up random seeds for reproducibility."""
    if seed is None:
        seed = config.seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)

def ensure_checkpoint_dir():
    """Ensure the checkpoint directory exists."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def get_checkpoint_path(epoch=None):
    """Get checkpoint path for a specific epoch or best model."""
    if epoch is None:
        return BEST_MODEL_PATH
    return CHECKPOINT_TEMPLATE.format(epoch)

def checkpoint_exists(epoch=None):
    """Check if a checkpoint exists."""
    return os.path.exists(get_checkpoint_path(epoch))

def to_device(tensor, device=None):
    """Move tensor to device with fallback to default device."""
    if device is None:
        device = get_device()
    return tensor.to(device)

# -------------------------------
# Helper functions
# -------------------------------

def create_3x3_kernel(dtype, device):
    """Create a 3x3 kernel for counting neighbors (including center)."""
    return torch.tensor(
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]],
        dtype=dtype,
        device=device
    ).view(1, 1, 3, 3)

def get_non_trivial_mask(state, device):
    """Calculate non-trivial mask for a state tensor."""
    kernel = create_3x3_kernel(state.dtype, device)
    neighbor_counts = F.conv2d(state, kernel, padding=1)
    return neighbor_counts > 0

def calculate_accuracy_excluding_trivial(pred_state, gt_state, input_state, device):
    """Calculate accuracy excluding trivial cases."""
    non_trivial_mask = get_non_trivial_mask(input_state, device)
    
    if non_trivial_mask.sum() > 0:
        correct_pixels = ((pred_state > 0.5) == (gt_state > 0.5)) & non_trivial_mask
        accuracy = correct_pixels.float().sum().item() / non_trivial_mask.sum().item()
    else:
        accuracy = 0.0
    
    return accuracy, non_trivial_mask

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


def create_model(base: int = None, latent_dim: int = None, model_type: str = None, device: str = None):
    """Create a model with the specified parameters or use config defaults."""
    base = base or config.base_channels
    latent_dim = latent_dim or config.latent_dim
    model_type = model_type or config.model_type
    
    if model_type == "RefinementCNN":
        model = RefinementCNN(base=base, latent_dim=latent_dim)
    elif model_type == "RefinementCNN2":
        model = RefinementCNN2(base=base, latent_dim=latent_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Available options: RefinementCNN, RefinementCNN2")
    
    if device is not None:
        model = model.to(device)
    return model 