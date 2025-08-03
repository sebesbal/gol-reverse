import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

from model import (
    config, gol_step, count_neighbors_3x3, get_non_trivial_mask,
    calculate_accuracy_excluding_trivial, create_model
)
from train import refine

# -------------------------------
# Visualization
# -------------------------------

def visualize_reverse_gol(checkpoint_path: str):
    """
    Visualize the reverse Game of Life process using a saved model.
    
    Args:
        checkpoint_path: Path to saved model checkpoint
    """
    # Set random seed
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_model(base=config.base_channels, latent_dim=config.latent_dim, model_type=config.model_type).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Model validation metrics: BCE={checkpoint['val_bce']:.4f}, "
          f"Bit Acc={checkpoint['val_bit_acc']:.4f}, Recon={checkpoint['val_recon_ok']:.4f}")
    
    # Generate initial random state
    initial_state = torch.rand(1, config.grid_size, config.grid_size)
    initial_state = (initial_state < config.density).float()
    
    # Forward simulation
    print(f"\nForward simulation ({config.forward_steps} steps):")
    states_forward = [initial_state]
    current = initial_state
    
    for step in range(config.forward_steps):
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
        out = refine(model, current_state.unsqueeze(0).to(device), config.refine_steps)
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
                
                # Use the state that was input to this reverse step
                if i == 0:
                    input_state = final_state.to(device)
                else:
                    input_state = reconstructed_states[i-1].to(device)
                
                # Calculate accuracy excluding trivial cases
                acc, non_trivial_mask = calculate_accuracy_excluding_trivial(
                    recon_state, gt_state, input_state, device
                )
                if non_trivial_mask.sum() > 0:
                    total_acc += acc
                    valid_steps += 1
        if valid_steps > 0:
            experiment_accuracy = total_acc / valid_steps
    
    fig.suptitle(f'Game of Life Reverse Process Visualization\n'
                 f'Model: Epoch {checkpoint["epoch"]}, '
                 f'Experiment Accuracy: {experiment_accuracy:.3f}', fontsize=14)
    
    # Custom colormap: black (dead), white (alive), red (extra), blue (missing)
    colors = ['black', 'white', 'red', 'blue']
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
            
            # 0 = no difference, 2 = extra (model alive but GT dead), 3 = missing (model dead but GT alive)
            diff_mask = torch.zeros_like(recon_state)
            diff_mask = torch.where((recon_state > 0.5) & (gt_state < 0.5), 
                                  torch.full_like(diff_mask, 2), diff_mask)  # Extra cells
            diff_mask = torch.where((recon_state < 0.5) & (gt_state > 0.5), 
                                  torch.full_like(diff_mask, 3), diff_mask)  # Missing cells
            
            # Debug diff_mask
            extra_cells = ((recon_state > 0.5) & (gt_state < 0.5)).sum().item()
            missing_cells = ((recon_state < 0.5) & (gt_state > 0.5)).sum().item()
            print(f"Step {i+1} Diff Debug: Extra={extra_cells}, Missing={missing_cells}")
            
            # Combine states for visualization
            # 0 = dead (white), 1 = alive (black), 2 = extra (red), 3 = missing (blue)
            vis_state = torch.zeros_like(recon_state)
            vis_state = torch.where(recon_state > 0.5, torch.ones_like(vis_state), vis_state)  # Alive cells
            
            # Only show differences for non-trivial cells
            non_trivial_mask = get_non_trivial_mask(input_state, device)
            diff_mask_non_trivial = diff_mask * non_trivial_mask.float()
            vis_state = torch.where(diff_mask_non_trivial > 0, diff_mask_non_trivial, vis_state)  # Override with differences
            
            axes[1, i+1].imshow(vis_state.squeeze().cpu(), cmap=cmap, vmin=0, vmax=3)
            axes[1, i+1].axis('off')
            
            # Calculate accuracy (excluding trivial cases)
            # Use the state that was input to this reverse step
            if i == 0:
                input_state = final_state.to(device)
            else:
                input_state = reconstructed_states[i-1].to(device)
            
            accuracy, non_trivial_mask = calculate_accuracy_excluding_trivial(
                recon_state, gt_state, input_state, device
            )
            
            # Debug information
            total_cells = recon_state.numel()
            non_trivial_cells = non_trivial_mask.sum().item()
            correct_cells = ((recon_state > 0.5) == (gt_state > 0.5)).sum().item()
            correct_non_trivial = ((recon_state > 0.5) == (gt_state > 0.5) & non_trivial_mask).sum().item()
            
            # Simple overall accuracy for comparison
            overall_accuracy = correct_cells / total_cells
            
            print(f"Step {i+1} Debug: Total={total_cells}, Non-trivial={non_trivial_cells}, "
                  f"Correct={correct_cells}, Correct_non_trivial={correct_non_trivial}, "
                  f"Overall_acc={overall_accuracy:.3f}, Non_trivial_acc={accuracy:.3f}")
            
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
        patches.Patch(color='red', label='Extra (Model Alive, GT Dead)'),
        patches.Patch(color='blue', label='Missing (Model Dead, GT Alive)')
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
                
                accuracy, non_trivial_mask = calculate_accuracy_excluding_trivial(
                    recon_state, gt_state, input_state, device
                )
                if non_trivial_mask.sum() > 0:
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


if __name__ == "__main__":
    # Check if checkpoint exists
    checkpoint_path = 'checkpoints/best_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        if os.path.exists('checkpoints'):
            for file in os.listdir('checkpoints'):
                if file.endswith('.pth'):
                    print(f"  checkpoints/{file}")
    else:
        print(f"Testing model from checkpoint: {checkpoint_path}")
        visualize_reverse_gol(checkpoint_path) 