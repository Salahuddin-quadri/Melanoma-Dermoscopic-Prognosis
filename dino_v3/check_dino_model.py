import sys
from pathlib import Path
import torch

# Add the current directory to the Python path to enable local imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from robust_dino_model import RobustDINOv3Config, RobustDINOv3Model

def inspect_checkpoint_arch(checkpoint_path: str):
    """
    Loads a DINOv3 model from a checkpoint file and identifies its architecture.

    Args:
        checkpoint_path (str): The path to the .pt checkpoint file.
    """
    try:
        # 1. Define the device to load the model onto
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # 2. Load the checkpoint
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 3. Check if 'config' is in the checkpoint
        if 'config' not in checkpoint:
            print("Error: 'config' key not found in the checkpoint.")
            print("Cannot determine the model architecture.")
            return

        # 4. Extract the configuration dictionary
        loaded_config_dict = checkpoint['config']
        
        # 5. Create a config object from the loaded dictionary
        config = RobustDINOv3Config(**loaded_config_dict)
        
        # 6. Instantiate the model with the loaded configuration
        model = RobustDINOv3Model(cfg=config)
        
        # 7. Load the model's weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval() # Set the model to evaluation mode

        # 8. Identify and print the architecture
        used_arch = model.cfg.arch
        
        print("\n" + "="*50)
        print("Model loaded successfully!")
        print(f"The checkpoint was trained with the '{used_arch}' Vision Transformer architecture.")
        print("="*50 + "\n")

    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at '{checkpoint_path}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # The path to the checkpoint file you want to inspect
    model_file_path = r"G:\Melanoma-Dermoscopic-Prognosis\dino_v3\outputs_dino\checkpoints\best.pt"
    
    inspect_checkpoint_arch(checkpoint_path=model_file_path)