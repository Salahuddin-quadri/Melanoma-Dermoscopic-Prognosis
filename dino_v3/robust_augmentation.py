"""
Robust augmentation utilities for DINOv3 training.
This module provides augmentation functions that avoid negative stride issues.
"""

import numpy as np
import torch
import torchvision.transforms as transforms
from typing import Tuple


class RobustAugmentation:
    """Robust augmentation class that avoids negative stride issues."""
    
    def __init__(self, global_size: Tuple[int, int] = (224, 224), 
                 local_size: Tuple[int, int] = (96, 96)):
        self.global_size = global_size
        self.local_size = local_size
        
        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def augment_image(self, img: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """Apply robust augmentation to image tensor."""
        try:
            # Convert to numpy for augmentation
            img_np = img.permute(1, 2, 0).numpy()
            
            # Random horizontal flip (using safe method)
            if np.random.rand() < 0.5:
                img_np = self._safe_flip(img_np)
            
            # Random brightness/contrast
            if np.random.rand() < 0.5:
                img_np = self._apply_brightness_contrast(img_np)
            
            # Ensure contiguous array
            img_np = np.ascontiguousarray(img_np)
            
            # Convert back to tensor
            return torch.from_numpy(np.transpose(img_np, (2, 0, 1)))
            
        except Exception as e:
            print(f"Warning: Augmentation failed: {e}")
            return img
    
    def _safe_flip(self, img: np.ndarray) -> np.ndarray:
        """Safely flip image horizontally without negative strides."""
        # Use np.flip with axis=1 instead of np.fliplr to avoid negative strides
        return np.flip(img, axis=1).copy()
    
    def _apply_brightness_contrast(self, img: np.ndarray) -> np.ndarray:
        """Apply brightness and contrast augmentation."""
        alpha = 1.0 + (np.random.rand() - 0.5) * 0.2  # contrast
        beta = (np.random.rand() - 0.5) * 0.1        # brightness
        return np.clip(img * alpha + beta, 0.0, 1.0)
    
    def preprocess_image(self, image_path: str, size: Tuple[int, int]) -> torch.Tensor:
        """Load and preprocess image with error handling."""
        try:
            import cv2
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            
            # Convert to float and normalize
            img = img.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization
            img = (img - self.mean) / self.std
            
            # Convert to tensor and transpose
            img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1)))
            
            return img_tensor
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # Return a dummy tensor to avoid breaking the batch
            return torch.zeros(3, size[0], size[1])


def create_robust_augmentation(global_size: Tuple[int, int] = (224, 224), 
                              local_size: Tuple[int, int] = (96, 96)) -> RobustAugmentation:
    """Create a robust augmentation instance."""
    return RobustAugmentation(global_size, local_size)


