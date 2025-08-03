import os
from train import train_model
from test import visualize_reverse_gol

def main():
    # Check if we should train or test
    if os.path.exists('checkpoints/best_model.pth'):
        print("Found best model, running visualization...")
        visualize_reverse_gol('checkpoints/best_model.pth')
    elif os.path.exists('checkpoints/checkpoint_epoch_10.pth'):
        print("Found epoch 10 checkpoint, running visualization...")
        visualize_reverse_gol('checkpoints/checkpoint_epoch_10.pth')
    else:
        print("No checkpoints found, starting training...")
        os.makedirs('checkpoints', exist_ok=True)
        train_model()
        print("\nTraining completed! Now running visualization...")
        visualize_reverse_gol('checkpoints/best_model.pth')

if __name__ == "__main__":
    main()

