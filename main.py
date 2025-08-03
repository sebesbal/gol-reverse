import os
from train import train_model
from test import visualize_reverse_gol
from model import get_checkpoint_path, checkpoint_exists, ensure_checkpoint_dir

def main():
    # Check if we should train or test
    if checkpoint_exists():
        print("Found best model, running visualization...")
        visualize_reverse_gol(get_checkpoint_path())
    elif checkpoint_exists(10):
        print("Found epoch 10 checkpoint, running visualization...")
        visualize_reverse_gol(get_checkpoint_path(10))
    else:
        print("No checkpoints found, starting training...")
        ensure_checkpoint_dir()
        train_model()
        print("\nTraining completed! Now running visualization...")
        visualize_reverse_gol(get_checkpoint_path())

if __name__ == "__main__":
    main()

