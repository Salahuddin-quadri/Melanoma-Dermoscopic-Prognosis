"""
Model Inference Module

Handles model loading and prediction for the web application.
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from typing import Dict, Tuple

# Add parent directory to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.models.dino_hybrid import create_dino_hybrid_model
    from src.utils.preprocess import IMAGENET_MEAN, IMAGENET_STD
except ImportError:
    try:
        # Try alternative import path
        sys.path.insert(0, str(project_root / "src"))
        from models.dino_hybrid import create_dino_hybrid_model
        from utils.preprocess import IMAGENET_MEAN, IMAGENET_STD
    except ImportError:
        print("Warning: Could not import from src. Using local implementations.")
        # Fallback ImageNet stats
        IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # We'll need to define a minimal model creation function
        def create_dino_hybrid_model(**kwargs):
            raise ImportError("Cannot import model creation function. Please ensure src/ directory is accessible.")


class ModelInference:
    """Handles model loading and inference."""
    
    def __init__(self, model_path: str = None, dino_checkpoint: str = None):
        """
        Initialize the model for inference.
        
        Args:
            model_path: Path to the trained model checkpoint (model.pt)
            dino_checkpoint: Path to DINO pretrained checkpoint
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Default paths
        web_dir = Path(__file__).resolve().parent
        if model_path is None:
            model_path = web_dir / "model.pt"
        
        if dino_checkpoint is None:
            # Try to find DINO checkpoint in parent directory
            dino_checkpoint = project_root / "dino_v3" / "outputs_dino" / "checkpoints" / "best.pt"
            if not dino_checkpoint.exists():
                dino_checkpoint = None
        
        # Model configuration (from error.txt)
        self.config = {
            "model_type": "dino",
            "task": "classification",
            "multitask": True,
            "arch": "vit_b_32",  # Default from main.py
            "pretrained": True,
            "dino_checkpoint": str(dino_checkpoint) if dino_checkpoint and dino_checkpoint.exists() else None,
            "hidden_dim": 256,
            "dropout": 0.1,
            "freeze_backbone_layers": 8,
            "image_size": (224, 224),
        }
        
        # Create model
        self.model = self._create_model()
        
        # Load trained weights
        if model_path.exists():
            self._load_checkpoint(model_path)
        else:
            print(f"Warning: Model checkpoint not found at {model_path}")
            print("Using untrained model (ImageNet weights only)")
        
        self.model.eval()
        print("✓ Model ready for inference")
    
    def _create_model(self):
        """Create the model architecture."""
        try:
            model = create_dino_hybrid_model(
                task=self.config["task"],
                multitask=self.config["multitask"],
                arch=self.config["arch"],
                pretrained=self.config["pretrained"],
                dino_checkpoint=self.config["dino_checkpoint"],
                hidden_dim=self.config["hidden_dim"],
                dropout=self.config["dropout"],
                freeze_backbone_layers=self.config["freeze_backbone_layers"],
            )
            model = model.to(self.device)
            return model
        except Exception as e:
            print(f"Error creating model: {e}")
            raise
    
    def _load_checkpoint(self, checkpoint_path: Path):
        """Load trained model weights."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if "model_state" in checkpoint:
                    state_dict = checkpoint["model_state"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    # Assume the dict itself is the state dict
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Load state dict with error handling
            try:
                self.model.load_state_dict(state_dict, strict=True)
                print(f"✓ Loaded checkpoint from {checkpoint_path} (strict mode)")
            except RuntimeError as e:
                # Try with strict=False if strict loading fails
                print(f"Warning: Strict loading failed, trying flexible loading: {e}")
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    print(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
                if unexpected_keys:
                    print(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
                print(f"✓ Loaded checkpoint from {checkpoint_path} (flexible mode)")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            print("Continuing with pretrained weights only")
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess PIL image for model input.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed tensor of shape (1, 3, H, W)
        """
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV (if needed)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Resize
        size = self.config["image_size"]
        img_array = cv2.resize(img_array, size, interpolation=cv2.INTER_AREA)
        
        # Convert BGR back to RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0
        
        # Standardize using ImageNet stats
        img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD
        
        # Convert to tensor: (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1))
        
        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor.to(self.device)
    
    def predict(self, image: Image.Image) -> Dict:
        """
        Perform prediction on an image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with classification and regression results
        """
        # Preprocess
        img_tensor = self._preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
        
        # Process outputs
        if isinstance(outputs, dict):
            # Multitask model
            cls_logit = outputs.get("cls")
            reg_value = outputs.get("reg")
            
            # Apply sigmoid to classification if not already applied
            if cls_logit is not None:
                # Check if already in [0,1] range (sigmoid already applied)
                if cls_logit.min() >= 0 and cls_logit.max() <= 1:
                    cls_prob = cls_logit.item()
                else:
                    cls_prob = torch.sigmoid(cls_logit).item()
            else:
                cls_prob = None
            
            if reg_value is not None:
                reg_value = reg_value.item()
        else:
            # Single task model
            if self.config["task"] == "classification":
                # Check if already in [0,1] range
                if outputs.min() >= 0 and outputs.max() <= 1:
                    cls_prob = outputs.item()
                else:
                    cls_prob = torch.sigmoid(outputs).item()
                reg_value = None
            else:
                cls_prob = None
                reg_value = outputs.item()
        
        # Format results
        results = {
            "classification": {
                "probability": float(cls_prob) if cls_prob is not None else None,
                "prediction": "Melanoma" if (cls_prob is not None and cls_prob >= 0.5) else "Benign",
                "confidence": float(abs(cls_prob - 0.5) * 2) if cls_prob is not None else None,  # 0-1 scale
            },
            "regression": {
                "thickness_mm": float(reg_value) if reg_value is not None else None,
                "thickness_category": self._categorize_thickness(reg_value) if reg_value is not None else None,
            }
        }
        
        return results
    
    def _categorize_thickness(self, thickness: float) -> str:
        """Categorize Breslow thickness into clinical stages."""
        if thickness is None:
            return "Unknown"
        elif thickness < 1.0:
            return "T1 (≤1.0mm)"
        elif thickness < 2.0:
            return "T2 (1.01-2.0mm)"
        elif thickness < 4.0:
            return "T3 (2.01-4.0mm)"
        else:
            return "T4 (>4.0mm)"

