"""
Robust DINOv3 Training Script with comprehensive error handling.
This script trains DINOv3 on images in the bin/ folder with proper freezing and error handling.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
from sklearn.preprocessing import StandardScaler

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from robust_dino_model import (
    create_robust_dino_v3, 
    RobustDINOv3Config, 
    RobustDINOLoss
)
from robust_augmentation import create_robust_augmentation

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dino_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class RobustImageDataset(Dataset):
    """Robust dataset for DINOv3 training with proper error handling."""
    
    def __init__(self, image_paths: List[str], 
                 global_size: Tuple[int, int] = (224, 224), 
                 local_size: Tuple[int, int] = (96, 96), 
                 n_global: int = 2, 
                 n_local: int = 6):
        self.image_paths = image_paths
        self.global_size = global_size
        self.local_size = local_size
        self.n_global = n_global
        self.n_local = n_local
        
        # Create robust augmentation
        self.augmentation = create_robust_augmentation(global_size, local_size)
        
        # Validate paths
        self.valid_paths = self._validate_paths()
        logger.info(f"Valid images: {len(self.valid_paths)}/{len(self.image_paths)}")

    def _validate_paths(self) -> List[str]:
        """Validate image paths and return only valid ones."""
        valid_paths = []
        for path in self.image_paths:
            if os.path.exists(path) and self._is_valid_image(path):
                valid_paths.append(path)
            else:
                logger.warning(f"Invalid or corrupted image: {path}")
        return valid_paths

    def _is_valid_image(self, path: str) -> bool:
        """Check if image is valid and readable."""
        try:
            img = cv2.imread(path)
            return img is not None
        except:
            return False

    def _load_and_preprocess(self, path: str, size: Tuple[int, int]) -> torch.Tensor:
        """Load and preprocess image with error handling."""
        return self.augmentation.preprocess_image(path, size)

    def __len__(self) -> int:
        return len(self.valid_paths)

    def __getitem__(self, idx: int) -> dict:
        """Get item with error handling."""
        try:
            path = self.valid_paths[idx]
            
            # Load and augment global crops
            global_crops = []
            for _ in range(self.n_global):
                img = self._load_and_preprocess(path, self.global_size)
                img = self.augmentation.augment_image(img, self.global_size)
                global_crops.append(img)
            
            # Load and augment local crops
            local_crops = []
            for _ in range(self.n_local):
                img = self._load_and_preprocess(path, self.local_size)
                img = self.augmentation.augment_image(img, self.local_size)
                local_crops.append(img)
            
            return {
                "global": global_crops,
                "local": local_crops,
                "path": path
            }
            
        except Exception as e:
            logger.error(f"Error getting item {idx}: {e}")
            # Return dummy data to avoid breaking the batch
            dummy_global = [torch.zeros(3, self.global_size[0], self.global_size[1]) for _ in range(self.n_global)]
            dummy_local = [torch.zeros(3, self.local_size[0], self.local_size[1]) for _ in range(self.n_local)]
            return {
                "global": dummy_global,
                "local": dummy_local,
                "path": "dummy"
            }


def robust_collate_fn(batch):
    """Robust collate function with error handling."""
    try:
        globals_all, locals_all = [], []
        
        for item in batch:
            if "global" in item and "local" in item:
                globals_all.extend(item["global"])
                locals_all.extend(item["local"])
            else:
                logger.warning(f"Invalid batch item: {item.keys()}")
        
        if not globals_all or not locals_all:
            raise ValueError("Empty batch after collation")
        
        return {
            "global": torch.stack(globals_all),
            "local": torch.stack(locals_all),
        }
        
    except Exception as e:
        logger.error(f"Error in collate function: {e}")
        # Return dummy batch
        dummy_global = torch.zeros(2, 3, 224, 224)
        dummy_local = torch.zeros(6, 3, 96, 96)
        return {
            "global": dummy_global,
            "local": dummy_local,
        }


def load_images_from_bin(bin_folder: str) -> List[str]:
    """Load all images from the bin folder."""
    bin_path = Path(bin_folder)
    if not bin_path.exists():
        raise ValueError(f"Bin folder does not exist: {bin_folder}")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(list(bin_path.glob(f"*{ext}")))
        image_paths.extend(list(bin_path.glob(f"*{ext.upper()}")))
    
    image_paths = [str(p) for p in image_paths]
    logger.info(f"Found {len(image_paths)} images in {bin_folder}")
    
    if not image_paths:
        raise ValueError(f"No images found in {bin_folder}")
    
    return image_paths


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Robust DINOv3 Training")
    
    # Data arguments
    parser.add_argument("--image_folder", type=str, default="./bin", 
                       help="Path to folder containing images")
    parser.add_argument("--output_dir", type=str, default="./outputs_dino", 
                       help="Output directory for checkpoints and logs")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50, 
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, 
                       help="Batch size (reduced for memory efficiency)")
    parser.add_argument("--lr", type=float, default=1e-4, 
                       help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.996, 
                       help="EMA momentum for teacher update")
    
    # Model arguments
    parser.add_argument("--arch", type=str, default="vit_b_32", 
                       choices=["vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32"],
                       help="ViT architecture (vit_b_32 for 224x224, vit_b_16 for 384x384)")
    parser.add_argument("--proj_out_dim", type=int, default=1024, 
                       help="Projection output dimension")
    parser.add_argument("--freeze_layers", type=int, default=8, 
                       help="Number of layers to freeze")
    
    # Data arguments
    parser.add_argument("--global_size", type=int, nargs=2, default=[224, 224], 
                       help="Global crop size")
    parser.add_argument("--local_size", type=int, nargs=2, default=[224, 224], 
                       help="Local crop size")
    parser.add_argument("--n_global", type=int, default=2, 
                       help="Number of global crops")
    parser.add_argument("--n_local", type=int, default=6, 
                       help="Number of local crops")
    
    # System arguments
    parser.add_argument("--device", type=str, default="auto", 
                       help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--num_workers", type=int, default=0, 
                       help="Number of data loading workers")
    parser.add_argument("--pin_memory", action="store_true", 
                       help="Use pinned memory")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from (latest.pt or best.pt)")
    
    # Loss arguments
    parser.add_argument("--teacher_temp", type=float, default=0.04, 
                       help="Teacher temperature")
    parser.add_argument("--student_temp", type=float, default=0.1, 
                       help="Student temperature")
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup device with error handling."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
    else:
        device = torch.device(device_arg)
        logger.info(f"Using specified device: {device}")
    
    return device


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch with error handling."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        try:
            # Move data to device
            global_imgs = batch["global"].to(device, non_blocking=True)
            local_imgs = batch["local"].to(device, non_blocking=True)
            
            # Validate tensor shapes
            if global_imgs.dim() != 4 or local_imgs.dim() != 4:
                logger.warning(f"Invalid tensor dimensions: global={global_imgs.shape}, local={local_imgs.shape}")
                continue
            
            # Forward pass - student
            student_global = model.forward_student(global_imgs)
            student_local = model.forward_student(local_imgs)
            
            # Forward pass - teacher (no gradients)
            with torch.no_grad():
                teacher_global = model.forward_teacher(global_imgs)
            
            # Compute loss
            loss = criterion(student_global, teacher_global)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update teacher with EMA
            model.update_teacher()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })
            
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            continue
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, output_dir, is_best=False):
    """Save model checkpoint."""
    try:
        checkpoint_dir = Path(output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': model.cfg.__dict__
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, checkpoint_dir / "latest.pt")
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, checkpoint_dir / "best.pt")
            logger.info(f"Saved best checkpoint at epoch {epoch}")
            
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup device
    device = setup_device(args.device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file
    log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    logger.info("Starting DINOv3 training...")
    logger.info(f"Arguments: {args}")
    logger.info(f"Log file: {log_file}")
    
    try:
        # Load images
        logger.info(f"Loading images from {args.image_folder}")
        image_paths = load_images_from_bin(args.image_folder)
        
        # Create dataset
        dataset = RobustImageDataset(
            image_paths=image_paths,
            global_size=tuple(args.global_size),
            local_size=tuple(args.local_size),
            n_global=args.n_global,
            n_local=args.n_local
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=robust_collate_fn,
            drop_last=True  # Avoid issues with incomplete batches
        )
        
        logger.info(f"Dataset size: {len(dataset)}")
        logger.info(f"Number of batches: {len(dataloader)}")
        
        # Create model
        config = RobustDINOv3Config(
            arch=args.arch,
            proj_out_dim=args.proj_out_dim,
            freeze_layers=args.freeze_layers
        )
        
        model = create_robust_dino_v3(config).to(device)
        
        # Resume from checkpoint if specified
        start_epoch = 1
        if args.resume:
            if Path(args.resume).exists():
                logger.info(f"Resuming from checkpoint: {args.resume}")
                checkpoint = torch.load(args.resume, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                logger.info(f"Resumed from epoch {checkpoint['epoch']}, starting from epoch {start_epoch}")
            else:
                logger.warning(f"Checkpoint not found: {args.resume}, starting from scratch")
        
        # Count parameters
        total_params, trainable_params = model.count_parameters()
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Create loss and optimizer
        criterion = RobustDINOLoss(
            out_dim=config.proj_out_dim,
            teacher_temp=args.teacher_temp,
            student_temp=args.student_temp
        ).to(device)
        
        optimizer = torch.optim.AdamW(
            model.get_trainable_parameters(),
            lr=args.lr,
            weight_decay=0.04
        )
        
        # Training loop
        best_loss = float('inf')
        
        for epoch in range(start_epoch, args.epochs + 1):
            logger.info(f"Starting epoch {epoch}/{args.epochs}")
            
            # Train epoch
            avg_loss = train_epoch(
                model, dataloader, criterion, optimizer, device, epoch
            )
            
            logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
            
            # Save checkpoint
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
                
            save_checkpoint(model, optimizer, epoch, avg_loss, output_dir, is_best)
        
        logger.info("Training completed successfully!")
        logger.info(f"Best loss: {best_loss:.4f}")
        logger.info(f"Model saved to: {output_dir}")
        logger.info(f"Log file saved to: {log_file}")
        logger.info(f"Checkpoints saved to: {output_dir}/checkpoints/")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()


