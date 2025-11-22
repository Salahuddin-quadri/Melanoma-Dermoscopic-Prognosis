# from __future__ import annotations

# from pathlib import Path
# from typing import Iterable, Tuple

# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import roc_auc_score, mean_absolute_error
# from tqdm import tqdm

# # Support running as module or script
# try:
# 	from .utils import preprocess_image, build_image_augment
# 	from .utils.losses import WeightedBCELoss, FocalLoss, compute_class_weights
# except ImportError:
# 	from utils import preprocess_image, build_image_augment
# 	from utils.losses import WeightedBCELoss, FocalLoss, compute_class_weights


# class FrameDataset(Dataset):
# 	"""
# 	Dataset for image + clinical features with class-aware augmentation.
	
# 	Supports strong augmentation for minority class (MEL) to address imbalance.
# 	"""
# 	def __init__(
# 		self,
# 		df,
# 		feature_cols: Iterable[str],
# 		image_size: Tuple[int, int] = (224, 224),
# 		augment: bool = False,
# 		class_aware_augment: bool = False,
# 	):
# 		self.image_paths = df["image_path"].tolist()
# 		self.structured = df[list(feature_cols)].to_numpy(dtype=np.float32)
# 		# Targets for multitask: expect both columns if present
# 		self.y_cls = df["label"].to_numpy(dtype=np.float32) if "label" in df.columns else None
# 		self.y_reg = df["thickness"].to_numpy(dtype=np.float32) if "thickness" in df.columns else None
# 		# Unified target for single-task
# 		self.y_single = df["target"].to_numpy(dtype=np.float32) if "target" in df.columns else None
# 		self.image_size = image_size
# 		self.class_aware_augment = class_aware_augment
		
# 		# Setup augmentation: class-aware means strong aug for minority class
# 		if augment:
# 			if class_aware_augment:
# 				# We'll apply strong aug conditionally in __getitem__
# 				self.augment_light = build_image_augment(strong=False)
# 				self.augment_strong = build_image_augment(strong=True)
# 			else:
# 				self.augment_light = build_image_augment(strong=False)
# 				self.augment_strong = None
# 		else:
# 			self.augment_light = None
# 			self.augment_strong = None

# 	def __len__(self):
# 		return len(self.image_paths)

# 	def __getitem__(self, idx):
# 		img = preprocess_image(self.image_paths[idx], size=self.image_size)
		
# 		# Apply augmentation if enabled
# 		if self.augment_light is not None:
# 			if self.class_aware_augment and self.y_cls is not None:
# 				# Strong augmentation for minority class (label=1.0 = MEL)
# 				if self.y_cls[idx] == 1.0:
# 					img = self.augment_strong(img)
# 				else:
# 					img = self.augment_light(img)
# 			else:
# 				# Standard augmentation for all
# 				img = self.augment_light(img)
		
# 		# to CHW tensor
# 		img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
# 		tab = torch.from_numpy(self.structured[idx])
# 		out = {"img": img, "tab": tab}
# 		if self.y_single is not None:
# 			out["target"] = torch.tensor(self.y_single[idx], dtype=torch.float32)
# 		if self.y_cls is not None:
# 			# Labels should already be normalized to [0,1] by data loader
# 			val = float(self.y_cls[idx])
# 			val = 1.0 if val > 1 else (0.0 if val < 0 else val)
# 			out["label"] = torch.tensor(val, dtype=torch.float32)
# 		if self.y_reg is not None:
# 			out["thickness"] = torch.tensor(self.y_reg[idx], dtype=torch.float32)
# 		return out


# def train_model(
# 	model: torch.nn.Module,
# 	train_df,
# 	val_df,
# 	feature_cols: Iterable[str],
# 	batch_size: int = 32,
# 	epochs: int = 20,
# 	image_size: Tuple[int, int] = (224, 224),
# 	output_dir: str = "outputs",
# 	device: str | None = None,
# 	task: str = "classification",
# 	multitask: bool = False,
# 	loss_alpha: float = 0.5,
# 	cls_loss_type: str = "bce",
# 	class_aware_augment: bool = False,
# 	focal_gamma: float = 2.0,
# ):
# 	"""
# 	Train the hybrid model with optional loss weighting and class imbalance handling.
	
# 	Args:
# 		model: Model to train
# 		train_df: Training dataframe
# 		val_df: Validation dataframe
# 		feature_cols: List of clinical feature column names
# 		batch_size: Batch size for training
# 		epochs: Number of training epochs
# 		image_size: Image dimensions (height, width)
# 		output_dir: Directory to save checkpoints
# 		device: Device to use ('cuda', 'cpu', or None for auto)
# 		task: Task type ('classification' or 'regression')
# 		multitask: If True, train dual heads (classification + regression)
# 		loss_alpha: Weight for classification loss in multitask setting (0-1).
# 			Total loss = alpha * loss_cls + (1 - alpha) * loss_reg
# 			Default 0.5 means equal weighting.
# 		cls_loss_type: Classification loss type: 'bce', 'weighted_bce', or 'focal'
# 		class_aware_augment: If True, apply strong augmentation only to minority class (MEL)
# 		focal_gamma: Gamma parameter for Focal Loss (default: 2.0)
# 	"""
# 	Path(output_dir, "checkpoints").mkdir(parents=True, exist_ok=True)
# 	ckpt_path = str(Path(output_dir, "checkpoints", "best.pt"))

# 	device = device or ("cuda" if torch.cuda.is_available() else "cpu")
# 	model = model.to(device)

# 	# Setup datasets with class-aware augmentation if requested
# 	train_ds = FrameDataset(
# 		train_df, 
# 		feature_cols, 
# 		image_size, 
# 		augment=True,
# 		class_aware_augment=class_aware_augment
# 	)
# 	val_ds = FrameDataset(val_df, feature_cols, image_size, augment=False)

# 	def collate(batch):
# 		imgs = torch.stack([b["img"] for b in batch])
# 		tabs = torch.stack([b["tab"] for b in batch])
# 		out = {"imgs": imgs, "tabs": tabs}
# 		if batch[0].get("target") is not None:
# 			out["target"] = torch.stack([b["target"] for b in batch])
# 		if batch[0].get("label") is not None:
# 			out["label"] = torch.stack([b["label"] for b in batch])
# 		if batch[0].get("thickness") is not None:
# 			out["thickness"] = torch.stack([b["thickness"] for b in batch])
# 		return out

# 	# Windows multiprocessing can fail to pickle local callables; keep workers=0 for safety
# 	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, collate_fn=collate)
# 	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate)

# 	# Setup classification loss based on type
# 	if task == "classification" or multitask:
# 		# Validate that we have labels in training data
# 		if "label" not in train_df.columns:
# 			raise ValueError("'label' column not found in training dataframe!")
		
# 		# Check label distribution before computing weights
# 		label_counts = train_df["label"].value_counts()
# 		print(f"\n📊 Training set label distribution:")
# 		for label_val, count in label_counts.items():
# 			print(f"   Label {label_val}: {count:,} samples")
		
# 		if cls_loss_type == "weighted_bce":
# 			# Compute class weights from training data
# 			class_weights = compute_class_weights(train_df, label_col="label")
# 			print(f"\n📊 Class imbalance info:")
# 			print(f"   BEN (class 0): {class_weights['n_negative']:,} samples")
# 			print(f"   MEL (class 1): {class_weights['n_positive']:,} samples")
# 			print(f"   Imbalance ratio: {class_weights['imbalance_ratio']:.2f}:1")
# 			print(f"   Positive weight: {class_weights['pos_weight']:.3f}")
			
# 			criterion_cls = WeightedBCELoss.from_class_counts(
# 				n_positive=class_weights['n_positive'],
# 				n_negative=class_weights['n_negative'],
# 				device=device
# 			).to(device)
# 			print(f"   ✓ Using Weighted BCE Loss")
			
# 		elif cls_loss_type == "focal":
# 			# Compute class weights for Focal Loss alpha
# 			class_weights = compute_class_weights(train_df, label_col="label")
# 			print(f"\n📊 Class imbalance info:")
# 			print(f"   BEN (class 0): {class_weights['n_negative']:,} samples")
# 			print(f"   MEL (class 1): {class_weights['n_positive']:,} samples")
# 			print(f"   Imbalance ratio: {class_weights['imbalance_ratio']:.2f}:1")
# 			print(f"   Focal gamma: {focal_gamma}")
			
# 			criterion_cls = FocalLoss.from_class_counts(
# 				n_positive=class_weights['n_positive'],
# 				n_negative=class_weights['n_negative'],
# 				gamma=focal_gamma,
# 				device=device
# 			).to(device)
# 			print(f"   ✓ Using Focal Loss (gamma={focal_gamma})")
			
# 		else:  # "bce"
# 			criterion_cls = torch.nn.BCELoss()
# 			print(f"   ✓ Using standard BCE Loss")
# 	else:
# 		criterion_cls = torch.nn.BCELoss()
	
# 	criterion_reg = torch.nn.L1Loss()
# 	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
	
# 	if class_aware_augment:
# 		print(f"   ✓ Class-aware augmentation enabled (strong aug for MEL)")
	
# 	best_val_auc = -1.0
# 	patience, bad_epochs = 5, 0

# 	for epoch in range(1, epochs + 1):
# 		model.train()
# 		running_loss = 0.0
# 		for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
# 			imgs = batch["imgs"].to(device)
# 			tabs = batch["tabs"].to(device)
# 			optimizer.zero_grad()
# 			outputs = model(imgs, tabs)
# 			if multitask:
# 				loss_cls = criterion_cls(outputs["cls"], batch["label"].to(device))
# 				loss_reg = criterion_reg(outputs["reg"], batch["thickness"].to(device))
# 				# Weighted combination of classification and regression losses
# 				# This allows balancing the two objectives during training
# 				loss = loss_alpha * loss_cls + (1.0 - loss_alpha) * loss_reg
# 			else:
# 				targets = batch["target"].to(device)
# 				if task == "classification":
# 					loss = criterion_cls(outputs, targets)
# 				else:
# 					loss = criterion_reg(outputs, targets)
# 			loss.backward()
# 			optimizer.step()
# 			running_loss += loss.item() * imgs.size(0)
# 		train_loss = running_loss / len(train_loader.dataset)

# 		# Validation
# 		model.eval()
# 		val_outputs, val_targets = [], []
# 		with torch.no_grad():
# 			for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
# 				imgs = batch["imgs"].to(device)
# 				tabs = batch["tabs"].to(device)
# 				outputs = model(imgs, tabs)
# 				if multitask:
# 					val_outputs.append({
# 						"cls": outputs["cls"].cpu().numpy(),
# 						"reg": outputs["reg"].cpu().numpy(),
# 					})
# 					val_targets.append({
# 						"cls": batch["label"].numpy(),
# 						"reg": batch["thickness"].numpy(),
# 					})
# 				else:
# 					val_outputs.append(outputs.cpu().numpy())
# 					val_targets.append(batch["target"].numpy())
# 		if multitask:
# 			probs = np.concatenate([o["cls"] for o in val_outputs])
# 			y_cls = np.concatenate([t["cls"] for t in val_targets])
# 			pred_reg = np.concatenate([o["reg"] for o in val_outputs])
# 			y_reg = np.concatenate([t["reg"] for t in val_targets])
# 			# AUC may be undefined if only one class in y_true
# 			try:
# 				val_auc = roc_auc_score(y_cls, probs)
# 			except Exception:
# 				preds = (probs >= 0.5).astype(int)
# 				val_auc = (preds == y_cls.astype(int)).mean()
# 			val_mae = mean_absolute_error(y_reg, pred_reg)
# 			# Combined metric weighted by loss_alpha for consistency
# 			# Normalize MAE by typical range (e.g., 0-10mm) to make it comparable to AUC (0-1)
# 			mae_normalized = val_mae / 10.0  # Assume thickness range 0-10mm
# 			val_metric = loss_alpha * val_auc + (1.0 - loss_alpha) * (1.0 - mae_normalized)
# 			print({
# 				"epoch": epoch,
# 				"train_loss": train_loss,
# 				"val_auc": float(val_auc),
# 				"val_mae": float(val_mae),
# 				"loss_cls_weight": float(loss_alpha),
# 				"loss_reg_weight": float(1.0 - loss_alpha),
# 			})
# 			is_better = (best_val_auc < 0) or (val_metric > best_val_auc)
# 		else:
# 			val_outputs = np.concatenate(val_outputs)
# 			val_targets = np.concatenate(val_targets)
# 			if task == "classification":
# 				try:
# 					val_metric = roc_auc_score(val_targets, val_outputs)
# 				except Exception:
# 					preds = (val_outputs >= 0.5).astype(int)
# 					val_metric = (preds == val_targets.astype(int)).mean()
# 				print({"epoch": epoch, "train_loss": train_loss, "val_auc_or_acc": float(val_metric)})
# 				is_better = val_metric > best_val_auc
# 			else:
# 				val_metric = mean_absolute_error(val_targets, val_outputs)
# 				print({"epoch": epoch, "train_loss": train_loss, "val_mae": float(val_metric)})
# 				is_better = (best_val_auc < 0) or (val_metric < best_val_auc)

# 		# Early stopping and checkpoint saving
# 		if is_better:
# 			best_val_auc = val_metric
# 			bad_epochs = 0
# 			torch.save({"model_state": model.state_dict()}, ckpt_path)
# 			print(f"✓ Saved new best checkpoint (metric={val_metric:.4f}) to {ckpt_path}")
# 		else:
# 			bad_epochs += 1
# 			if bad_epochs >= patience:
# 				print(f"Early stopping triggered after {patience} epochs without improvement")
# 				break

# 	return {"best_val_auc": float(best_val_auc)}, ckpt_path


"""
Training Pipeline for Skin Lesion Classification

Provides PyTorch-based training with:
    - Class-aware augmentation for imbalanced datasets
    - Multiple loss functions (BCE, Weighted BCE, Focal)
    - Multi-task learning support (classification + regression)
    - Early stopping and checkpointing
    - Multi-worker DataLoader support

References:
    Focal Loss: Lin et al., "Focal Loss for Dense Object Detection" (2017)
    Class Imbalance: Cui et al., "Class-Balanced Loss" (2019)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import json

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# Support running as module or script
try:
    from .utils import build_image_augment, preprocess_image
    from .utils.losses import FocalLoss, WeightedBCELoss, compute_class_weights
except ImportError:
    from utils import build_image_augment, preprocess_image
    from utils.losses import FocalLoss, WeightedBCELoss, compute_class_weights


# ============================================================================
# Dataset Classes
# ============================================================================

class FrameDataset(Dataset):
    """
    Dataset for dermoscopic images only.
    
    Supports class-aware augmentation where minority class (melanoma) receives
    stronger augmentation to address dataset imbalance.
    
    Attributes:
        image_paths: List of paths to dermoscopic images
        y_cls: Classification labels (0=benign, 1=melanoma)
        y_reg: Regression targets (e.g., Breslow thickness)
        y_single: Unified target for single-task learning
        image_size: Target image dimensions (width, height)
        class_aware_augment: Whether to apply strong aug to minority class
        augment_light: Light augmentation function
        augment_strong: Strong augmentation function (minority class only)
    """

    def __init__(
        self,
        df,
        image_size: Tuple[int, int] = (384, 384),
        augment: bool = False,
        class_aware_augment: bool = False,
    ):
        """
        Initialize dataset from pandas DataFrame.
        
        Args:
            df: DataFrame with columns:
                - 'image_path': Path to image file
                - 'label': Classification target (optional)
                - 'thickness': Regression target (optional)
                - 'target': Unified target for single-task (optional)
            image_size: Target (width, height) for preprocessing
            augment: Whether to apply data augmentation
            class_aware_augment: If True, apply strong aug to minority class
        """
        # Extract paths
        self.image_paths = df["image_path"].tolist()
        
        # Extract targets (may be None if not present)
        self.y_cls = (
            df["label"].to_numpy(dtype=np.float32)
            if "label" in df.columns
            else None
        )
        self.y_reg = (
            df["thickness"].to_numpy(dtype=np.float32)
            if "thickness" in df.columns
            else None
        )
        self.y_single = (
            df["target"].to_numpy(dtype=np.float32)
            if "target" in df.columns
            else None
        )
        
        self.image_size = image_size
        self.class_aware_augment = class_aware_augment

        # Setup augmentation functions
        if augment:
            self.augment_light = build_image_augment(strong=False)
            if class_aware_augment:
                self.augment_strong = build_image_augment(strong=True)
            else:
                self.augment_strong = None
        else:
            self.augment_light = None
            self.augment_strong = None

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and preprocess a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - 'img': Image tensor, shape (3, H, W)
                - 'label': Classification target (if available)
                - 'thickness': Regression target (if available)
                - 'target': Unified target (if available)
        """
        # Load and preprocess image
        img = preprocess_image(self.image_paths[idx], size=self.image_size)

        # Apply augmentation if enabled
        if self.augment_light is not None:
            if self.class_aware_augment and self.y_cls is not None:
                # Strong augmentation for minority class (melanoma = 1.0)
                if self.y_cls[idx] == 1.0:
                    img = self.augment_strong(img)
                else:
                    img = self.augment_light(img)
            else:
                # Standard augmentation for all samples
                img = self.augment_light(img)

        # Convert to PyTorch tensor: (H, W, C) -> (C, H, W)
        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))

        # Build output dictionary
        out = {"img": img}

        # Add targets if available
        if self.y_single is not None:
            out["target"] = torch.tensor(self.y_single[idx], dtype=torch.float32)

        if self.y_cls is not None:
            # Ensure label is in [0, 1] range
            val = float(self.y_cls[idx])
            val = max(0.0, min(1.0, val))
            out["label"] = torch.tensor(val, dtype=torch.float32)

        if self.y_reg is not None:
            out["thickness"] = torch.tensor(self.y_reg[idx], dtype=torch.float32)

        return out


# ============================================================================
# Collate Function (Must be at module level for multiprocessing)
# ============================================================================

def custom_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader with image data.
    
    This function must be defined at module level (not nested) to support
    multi-worker DataLoaders on Windows.
    
    Args:
        batch: List of sample dictionaries from FrameDataset
        
    Returns:
        Batched dictionary with stacked tensors
    """
    # Stack image data
    imgs = torch.stack([b["img"] for b in batch])
    out = {"imgs": imgs}

    # Stack optional targets if present
    if "target" in batch[0]:
        out["target"] = torch.stack([b["target"] for b in batch])
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    if "thickness" in batch[0]:
        out["thickness"] = torch.stack([b["thickness"] for b in batch])

    return out


# ============================================================================
# Training Functions
# ============================================================================

def train_model(
    model: torch.nn.Module,
    train_df,
    val_df,
    batch_size: int = 32,
    epochs: int = 20,
    image_size: Tuple[int, int] = (384, 384),
    output_dir: str = "outputs",
    device: Optional[str] = None,
    task: str = "classification",
    multitask: bool = False,
    loss_alpha: float = 0.5,
    cls_loss_type: str = "bce",
    class_aware_augment: bool = False,
    focal_gamma: float = 2.0,
    num_workers: int = 4,
    learning_rate: float = 1e-3,
    patience: int = 15,
    lr_scheduler: Optional[str] = "cosine",
) -> Tuple[Dict[str, float], str]:
    """
    Train image-only model with advanced loss handling.
    
    Args:
        model: PyTorch model to train (image-only input, dual-head output)
        train_df: Training DataFrame with required columns
        val_df: Validation DataFrame with required columns
        batch_size: Mini-batch size for training
        epochs: Maximum number of training epochs
        image_size: Target image dimensions (width, height)
        output_dir: Directory for saving checkpoints
        device: Device for training ('cuda', 'cpu', or None for auto)
        task: Task type ('classification' or 'regression')
        multitask: If True, train dual-head model (cls + reg)
        loss_alpha: Weight for classification loss in multitask.
            Total loss = alpha * L_cls + (1 - alpha) * L_reg
            Range: [0, 1]. Default: 0.5 (equal weighting)
        cls_loss_type: Classification loss type:
            - 'bce': Standard binary cross-entropy
            - 'weighted_bce': Class-weighted BCE for imbalance
            - 'focal': Focal loss for hard examples
        class_aware_augment: Apply strong augmentation to minority class
        focal_gamma: Focusing parameter for Focal Loss (default: 2.0)
        num_workers: Number of DataLoader worker processes
        learning_rate: Initial learning rate for Adam optimizer
        patience: Early stopping patience (epochs without improvement)
        lr_scheduler: Learning rate scheduler type ('cosine', 'step', or None)
        
    Returns:
        Tuple of (metrics_dict, checkpoint_path):
            - metrics_dict: Best validation metrics
            - checkpoint_path: Path to saved best model
            
    Raises:
        ValueError: If required columns missing from DataFrames
        
    Example:
        >>> results, ckpt_path = train_model(
        ...     model=model,
        ...     train_df=train_df,
        ...     val_df=val_df,
        ...     batch_size=32,
        ...     epochs=50,
        ...     image_size=(384, 384),
        ...     cls_loss_type='focal',
        ...     class_aware_augment=True,
        ...     num_workers=4,
        ...     lr_scheduler='cosine'
        ... )
    """
    # Setup YOLO-style output directory (train1, train2, etc.)
    base_output_dir = Path(output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    train_dirs = [d for d in base_output_dir.iterdir() if d.is_dir() and d.name.startswith("train")]
    train_numbers = []
    for d in train_dirs:
        try:
            num = int(d.name.replace("train", ""))
            train_numbers.append(num)
        except ValueError:
            pass
    next_train_num = max(train_numbers) + 1 if train_numbers else 1
    output_dir = base_output_dir / f"train{next_train_num}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup subdirectories
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(checkpoint_dir / "best.pt")
    
    # Create log file path
    log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"\n🖥️  Using device: {device}")

    # Create datasets
    train_ds = FrameDataset(
        train_df,
        image_size,
        augment=True,
        class_aware_augment=class_aware_augment,
    )
    val_ds = FrameDataset(
        val_df, 
        image_size, 
        augment=False
    )

    # Create data loaders with multi-worker support
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=custom_collate_fn,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=custom_collate_fn,
        persistent_workers=(num_workers > 0),
    )
    
    print(f"📦 DataLoader: {num_workers} workers, batch_size={batch_size}")

    # Setup classification loss
    criterion_cls = _setup_classification_loss(
        task=task,
        multitask=multitask,
        cls_loss_type=cls_loss_type,
        focal_gamma=focal_gamma,
        train_df=train_df,
        device=device,
    )

    # Setup regression loss
    criterion_reg = torch.nn.L1Loss()

    # Setup optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )
    
    # Setup learning rate scheduler
    scheduler = None
    if lr_scheduler and lr_scheduler != "none":
        if lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=learning_rate * 0.01
            )
            print(f"🎯 Optimizer: Adam (lr={learning_rate}) with CosineAnnealingLR")
        elif lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=epochs // 3, gamma=0.1
            )
            print(f"🎯 Optimizer: Adam (lr={learning_rate}) with StepLR")
    else:
        print(f"🎯 Optimizer: Adam (lr={learning_rate})")

    if class_aware_augment:
        print(f"✅ Class-aware augmentation enabled (strong aug for melanoma)")

    # Training loop
    best_val_metric = -1.0
    bad_epochs = 0
    
    # Training history for logging and visualization
    training_history = {
        "epoch": [],
        "train_loss": [],
        "val_metric": [],
        "learning_rate": [],
    }
    if multitask:
        training_history["val_auc"] = []
        training_history["val_mae"] = []
    elif task == "classification":
        training_history["val_auc"] = []
    else:
        training_history["val_mae"] = []

    for epoch in range(1, epochs + 1):
        # Training phase
        train_loss = _train_epoch(
            model=model,
            train_loader=train_loader,
            criterion_cls=criterion_cls,
            criterion_reg=criterion_reg,
            optimizer=optimizer,
            device=device,
            task=task,
            multitask=multitask,
            loss_alpha=loss_alpha,
            epoch=epoch,
            epochs=epochs,
        )

        # Validation phase
        val_metric, val_stats = _validate_epoch(
            model=model,
            val_loader=val_loader,
            device=device,
            task=task,
            multitask=multitask,
            loss_alpha=loss_alpha,
            epoch=epoch,
            epochs=epochs,
        )

        # Update learning rate scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()
        
        # Record training history
        training_history["epoch"].append(epoch)
        training_history["train_loss"].append(train_loss)
        training_history["val_metric"].append(val_metric)
        training_history["learning_rate"].append(current_lr)
        for key, value in val_stats.items():
            if key not in training_history:
                training_history[key] = []
            training_history[key].append(value)
        
        # Print epoch summary
        _print_epoch_summary(
            epoch=epoch,
            train_loss=train_loss,
            val_stats=val_stats,
            task=task,
            multitask=multitask,
            loss_alpha=loss_alpha,
            current_lr=current_lr,
        )
        
        # Save training log
        _save_training_log(training_history, log_file, epoch)

        # Check for improvement
        is_better = (best_val_metric < 0) or (val_metric > best_val_metric)

        if is_better:
            best_val_metric = val_metric
            bad_epochs = 0
            torch.save({"model_state": model.state_dict()}, checkpoint_path)
            print(f"✅ Saved best checkpoint (metric={val_metric:.4f}) → {checkpoint_path}\n")
        else:
            bad_epochs += 1
            print(f"❌ No improvement ({bad_epochs}/{patience})\n")
            if bad_epochs >= patience:
                print(f"🛑 Early stopping triggered after {patience} epochs without improvement")
                break
    
    # Save final training plots
    _save_training_plots(training_history, output_dir)
    
    print(f"\n📁 Training outputs saved to: {output_dir}")
    print(f"📝 Training log: {log_file}")

    return {"best_val_metric": float(best_val_metric)}, checkpoint_path


# ============================================================================
# Helper Functions
# ============================================================================

def _setup_classification_loss(
    task: str,
    multitask: bool,
    cls_loss_type: str,
    focal_gamma: float,
    train_df,
    device: str,
) -> torch.nn.Module:
    """Setup classification loss function based on configuration."""
    if task != "classification" and not multitask:
        return torch.nn.BCELoss()

    # Validate that labels exist
    if "label" not in train_df.columns:
        raise ValueError("'label' column not found in training DataFrame!")

    # Print label distribution
    label_counts = train_df["label"].value_counts()
    print(f"\n📊 Training set label distribution:")
    for label_val, count in label_counts.items():
        print(f"   Label {label_val}: {count:,} samples")

    # Setup loss based on type
    if cls_loss_type == "weighted_bce":
        class_weights = compute_class_weights(train_df, label_col="label")
        _print_class_weights(class_weights)
        criterion = WeightedBCELoss.from_class_counts(
            n_positive=class_weights["n_positive"],
            n_negative=class_weights["n_negative"],
            device=device,
        )
        print(f"   ✅ Using Weighted BCE Loss")

    elif cls_loss_type == "focal":
        class_weights = compute_class_weights(train_df, label_col="label")
        _print_class_weights(class_weights)
        print(f"   Focal gamma: {focal_gamma}")
        criterion = FocalLoss.from_class_counts(
            n_positive=class_weights["n_positive"],
            n_negative=class_weights["n_negative"],
            gamma=focal_gamma,
            device=device,
        )
        print(f"   ✅ Using Focal Loss (gamma={focal_gamma})")

    else:  # "bce"
        criterion = torch.nn.BCELoss()
        print(f"   ✅ Using standard BCE Loss")

    return criterion


def _print_class_weights(class_weights: Dict) -> None:
    """Print class imbalance statistics."""
    print(f"\n📊 Class imbalance info:")
    print(f"   BEN (class 0): {class_weights['n_negative']:,} samples")
    print(f"   MEL (class 1): {class_weights['n_positive']:,} samples")
    print(f"   Imbalance ratio: {class_weights['imbalance_ratio']:.2f}:1")
    print(f"   Positive weight: {class_weights['pos_weight']:.3f}")


def _train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion_cls: torch.nn.Module,
    criterion_reg: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    task: str,
    multitask: bool,
    loss_alpha: float,
    epoch: int,
    epochs: int,
) -> float:
    """Execute one training epoch."""
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{epochs} [train]",
        leave=False,
    )

    for batch in progress_bar:
        imgs = batch["imgs"].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(imgs)

        # Compute loss
        if multitask:
            loss_cls = criterion_cls(outputs["cls"], batch["label"].to(device))
            loss_reg = criterion_reg(outputs["reg"], batch["thickness"].to(device))
            loss = loss_alpha * loss_cls + (1.0 - loss_alpha) * loss_reg
        else:
            targets = batch["target"].to(device)
            if task == "classification":
                loss = criterion_cls(outputs, targets)
            else:
                loss = criterion_reg(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track loss
        running_loss += loss.item() * imgs.size(0)
        progress_bar.set_postfix({"loss": loss.item()})

    return running_loss / len(train_loader.dataset)


def _validate_epoch(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: str,
    task: str,
    multitask: bool,
    loss_alpha: float,
    epoch: int,
    epochs: int,
) -> Tuple[float, Dict]:
    """Execute one validation epoch."""
    model.eval()
    val_outputs, val_targets = [], []

    progress_bar = tqdm(
        val_loader,
        desc=f"Epoch {epoch}/{epochs} [val]",
        leave=False,
    )

    with torch.no_grad():
        for batch in progress_bar:
            imgs = batch["imgs"].to(device)
            outputs = model(imgs)

            if multitask:
                val_outputs.append({
                    "cls": outputs["cls"].cpu().numpy(),
                    "reg": outputs["reg"].cpu().numpy(),
                })
                val_targets.append({
                    "cls": batch["label"].numpy(),
                    "reg": batch["thickness"].numpy(),
                })
            else:
                val_outputs.append(outputs.cpu().numpy())
                val_targets.append(batch["target"].numpy())

    # Compute metrics
    if multitask:
        probs = np.concatenate([o["cls"] for o in val_outputs])
        y_cls = np.concatenate([t["cls"] for t in val_targets])
        pred_reg = np.concatenate([o["reg"] for o in val_outputs])
        y_reg = np.concatenate([t["reg"] for t in val_targets])

        # Compute AUC (handle edge cases)
        try:
            val_auc = roc_auc_score(y_cls, probs)
        except Exception:
            preds = (probs >= 0.5).astype(int)
            val_auc = (preds == y_cls.astype(int)).mean()

        val_mae = mean_absolute_error(y_reg, pred_reg)

        # Combined metric (normalize MAE to [0, 1] scale)
        mae_normalized = min(val_mae / 10.0, 1.0)  # Assume thickness range 0-10mm
        val_metric = loss_alpha * val_auc + (1.0 - loss_alpha) * (1.0 - mae_normalized)

        val_stats = {
            "val_auc": float(val_auc),
            "val_mae": float(val_mae),
            "val_metric": float(val_metric),
        }

    else:
        val_outputs = np.concatenate(val_outputs)
        val_targets = np.concatenate(val_targets)

        if task == "classification":
            try:
                val_metric = roc_auc_score(val_targets, val_outputs)
                metric_name = "val_auc"
            except Exception:
                preds = (val_outputs >= 0.5).astype(int)
                val_metric = (preds == val_targets.astype(int)).mean()
                metric_name = "val_acc"

            val_stats = {metric_name: float(val_metric)}

        else:  # regression
            val_metric = mean_absolute_error(val_targets, val_outputs)
            val_stats = {"val_mae": float(val_metric)}
            # For regression, lower is better, so negate for comparison
            val_metric = -val_metric

    return val_metric, val_stats


def _print_epoch_summary(
    epoch: int,
    train_loss: float,
    val_stats: Dict,
    task: str,
    multitask: bool,
    loss_alpha: float,
    current_lr: float = None,
) -> None:
    """Print formatted epoch summary."""
    summary = {"epoch": epoch, "train_loss": f"{train_loss:.4f}"}
    summary.update({k: f"{v:.4f}" for k, v in val_stats.items()})
    
    if current_lr is not None:
        summary["lr"] = f"{current_lr:.6f}"

    if multitask:
        summary["loss_cls_weight"] = f"{loss_alpha:.2f}"
        summary["loss_reg_weight"] = f"{1.0 - loss_alpha:.2f}"

    # Format as aligned key-value pairs
    print("\n" + "=" * 60)
    for key, value in summary.items():
        print(f"   {key:20s}: {value}")
    print("=" * 60)


def _save_training_log(history: Dict, log_file: Path, epoch: int) -> None:
    """Save training log to file (YOLO-style)."""
    with open(log_file, "a") as f:
        if epoch == 1:
            # Write header
            header = "epoch"
            for key in history.keys():
                if key != "epoch":
                    header += f",{key}"
            f.write(header + "\n")
        
        # Write current epoch data
        row = str(history["epoch"][-1])
        for key in history.keys():
            if key != "epoch":
                row += f",{history[key][-1]:.6f}"
        f.write(row + "\n")


def _save_training_plots(history: Dict, output_dir: Path) -> None:
    """Save training plots (YOLO-style visuals)."""
    try:
        epochs = history["epoch"]
        
        # Create figure with subplots
        n_plots = 2
        if "val_auc" in history:
            n_plots += 1
        if "val_mae" in history:
            n_plots += 1
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Plot 1: Training Loss
        axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot 2: Learning Rate
        if "learning_rate" in history:
            axes[1].plot(epochs, history["learning_rate"], "g-", label="LR", linewidth=2)
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Learning Rate")
            axes[1].set_title("Learning Rate Schedule")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
            axes[1].set_yscale("log")
        
        # Plot 3: Validation Metrics
        plot_idx = 2
        if "val_auc" in history:
            axes[plot_idx].plot(epochs, history["val_auc"], "r-", label="Val AUC", linewidth=2)
            axes[plot_idx].set_xlabel("Epoch")
            axes[plot_idx].set_ylabel("AUC")
            axes[plot_idx].set_title("Validation AUC")
            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].legend()
            plot_idx += 1
        
        if "val_mae" in history:
            axes[plot_idx].plot(epochs, history["val_mae"], "m-", label="Val MAE", linewidth=2)
            axes[plot_idx].set_xlabel("Epoch")
            axes[plot_idx].set_ylabel("MAE")
            axes[plot_idx].set_title("Validation MAE")
            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].legend()
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].axis("off")
        
        plt.tight_layout()
        plot_path = output_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"📊 Training plots saved to: {plot_path}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not save training plots: {e}")