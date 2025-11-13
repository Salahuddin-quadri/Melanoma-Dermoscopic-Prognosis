# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Iterable, Tuple

# import numpy as np
# import cv2
# from sklearn.preprocessing import StandardScaler
# import torch

# # Torch image normalization for ResNet/ImageNet
# IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
# IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# def preprocess_image(image_path: str, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
# 	"""Load, resize, normalize to [0,1], then standardize using ImageNet stats."""
# 	img = cv2.imread(image_path)
# 	if img is None:
# 		raise FileNotFoundError(f"Failed to read image: {image_path}")
# 	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 	img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
# 	img = img.astype(np.float32) / 255.0
# 	img = (img - IMAGENET_MEAN) / IMAGENET_STD
# 	return img


# def build_image_augment(strong: bool = False):
# 	"""
# 	Return a callable that applies augmentation using OpenCV-like ops.
	
# 	Args:
# 		strong: If True, applies stronger augmentation (useful for minority class).
# 			Strong augmentation includes:
# 			- More aggressive color jitter
# 			- Random rotation
# 			- Random scaling/cropping
# 			- Gaussian blur
# 			- Cutout-like effects
# 	"""
# 	def _augment_light(img: np.ndarray) -> np.ndarray:
# 		"""Light augmentation: horizontal flip + mild color jitter."""
# 		# Horizontal flip with 50% chance
# 		if np.random.rand() < 0.5:
# 			img = np.ascontiguousarray(img[:, ::-1, :])
# 		# Small brightness/contrast jitter
# 		if np.random.rand() < 0.5:
# 			alpha = 1.0 + (np.random.rand() - 0.5) * 0.2  # contrast
# 			beta = (np.random.rand() - 0.5) * 0.1        # brightness
# 			img = np.clip(img * alpha + beta, 0.0, 1.0)
# 		return img
	
# 	def _augment_strong(img: np.ndarray) -> np.ndarray:
# 		"""Strong augmentation for minority class to improve generalization."""
# 		# Horizontal flip (50%)
# 		if np.random.rand() < 0.5:
# 			img = np.ascontiguousarray(img[:, ::-1, :])
		
# 		# Vertical flip (30% chance, less common but useful for medical images)
# 		if np.random.rand() < 0.3:
# 			img = np.ascontiguousarray(img[::-1, :, :])
		
# 		# Strong color jitter (brightness, contrast, saturation)
# 		if np.random.rand() < 0.7:
# 			# Brightness
# 			brightness = 1.0 + (np.random.rand() - 0.5) * 0.4  # ±20%
# 			# Contrast
# 			contrast = 1.0 + (np.random.rand() - 0.5) * 0.4    # ±20%
# 			# Apply
# 			img = np.clip(img * contrast * brightness, 0.0, 1.0)
		
# 		# Random rotation (small angles, ±15 degrees)
# 		if np.random.rand() < 0.5:
# 			angle = np.random.uniform(-15, 15)
# 			h, w = img.shape[:2]
# 			center = (w // 2, h // 2)
# 			M = cv2.getRotationMatrix2D(center, angle, 1.0)
# 			img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
		
# 		# Random scaling + crop (zoom in/out)
# 		if np.random.rand() < 0.5:
# 			scale = np.random.uniform(0.9, 1.1)
# 			h, w = img.shape[:2]
# 			new_h, new_w = int(h * scale), int(w * scale)
# 			img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
# 			# Crop back to original size
# 			if scale > 1.0:
# 				# Crop center
# 				start_y = (new_h - h) // 2
# 				start_x = (new_w - w) // 2
# 				img = img_resized[start_y:start_y+h, start_x:start_x+w]
# 			else:
# 				# Pad
# 				pad_y = (h - new_h) // 2
# 				pad_x = (w - new_w) // 2
# 				img = np.pad(img_resized, ((pad_y, h-new_h-pad_y), (pad_x, w-new_w-pad_x), (0, 0)), 
# 							mode='reflect')
# 				img = img[:h, :w]
		
# 		# Gaussian blur (20% chance, slight)
# 		if np.random.rand() < 0.2:
# 			kernel_size = 3
# 			sigma = np.random.uniform(0.5, 1.5)
# 			img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
# 			img = np.clip(img, 0.0, 1.0)
		
# 		# Random noise (10% chance)
# 		if np.random.rand() < 0.1:
# 			noise = np.random.normal(0, 0.02, img.shape)
# 			img = np.clip(img + noise, 0.0, 1.0)
		
# 		return img
	
# 	return _augment_strong if strong else _augment_light


# @dataclass
# class StructuredPreprocessor:
# 	feature_names: Iterable[str]
# 	_scaler: StandardScaler | None = None

# 	def fit(self, X: np.ndarray) -> "StructuredPreprocessor":
# 		self._scaler = StandardScaler()
# 		self._scaler.fit(X)
# 		return self

# 	def transform(self, X: np.ndarray) -> np.ndarray:
# 		if self._scaler is None:
# 			raise RuntimeError("StructuredPreprocessor must be fit before transform.")
# 		return self._scaler.transform(X)

# 	def fit_transform(self, X: np.ndarray) -> np.ndarray:
# 		return self.fit(X).transform(X)

"""
Medical Image Preprocessing Pipeline for Skin Lesion Classification

This module provides preprocessing and augmentation utilities optimized for
dermatological image analysis with imbalanced datasets.

Key Features:
    - ImageNet-normalized preprocessing for transfer learning
    - Conservative augmentations preserving diagnostic features
    - Class-balanced augmentation strategy (strong/light)
    - PyTorch integration with proper tensor conversion
    - Structured data preprocessing with standardization

References:
    ImageNet normalization: He et al., "Deep Residual Learning" (2016)
    Augmentation strategy: Tschandl et al., "HAM10000 Dataset" (2018)
    
Requirements:
    - OpenCV (cv2)
    - NumPy
    - scikit-learn
    - PyTorch (optional, for tensor conversion)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple, Union

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ImageNet normalization constants for transfer learning (RGB order)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Default image size for model input
# IMPORTANT: Must match your model's expected input size
# For ViT models: usually 224, 384, or 512
DEFAULT_IMAGE_SIZE = (384, 384)  # (width, height)


def preprocess_image(
    image_path: str, 
    size: Tuple[int, int] = DEFAULT_IMAGE_SIZE
) -> np.ndarray:
    """
    Load and preprocess image for neural network input.
    
    Pipeline:
        1. Load image from disk
        2. Convert BGR (OpenCV) to RGB
        3. Resize to target dimensions
        4. Normalize to [0, 1] range
        5. Standardize using ImageNet statistics
    
    Args:
        image_path: Path to input image file
        size: Target (width, height) for resizing. Default: (384, 384)
    
    Returns:
        Preprocessed image as float32 array with shape (H, W, 3)
        
    Raises:
        FileNotFoundError: If image cannot be read from path
        
    Note:
        Uses INTER_AREA interpolation for downsampling to reduce aliasing.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    
    # Convert color space: BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to target dimensions
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1] range
    img = img.astype(np.float32) / 255.0
    
    # Standardize using ImageNet statistics
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    
    return img


def preprocess_image_for_pytorch(
    image_path: str,
    size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    to_tensor: bool = True
) -> np.ndarray | torch.Tensor:
    """
    Load and preprocess image for PyTorch models.
    
    This function wraps preprocess_image() and optionally converts to PyTorch tensor
    with proper channel ordering (C, H, W).
    
    Args:
        image_path: Path to input image file
        size: Target (width, height) for resizing. Default: (384, 384)
        to_tensor: If True, converts to PyTorch tensor with shape (C, H, W)
        
    Returns:
        If to_tensor=True: torch.Tensor with shape (3, H, W)
        If to_tensor=False: np.ndarray with shape (H, W, 3)
        
    Example:
        >>> # For single image inference
        >>> img_tensor = preprocess_image_for_pytorch('lesion.jpg')
        >>> img_batch = img_tensor.unsqueeze(0)  # Add batch dimension -> (1, 3, 384, 384)
        >>> output = model(img_batch)
    """
    import torch
    
    # Preprocess using standard pipeline
    img = preprocess_image(image_path, size)
    
    if to_tensor:
        # Convert to PyTorch tensor: (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        return img_tensor
    
    return img


def create_pytorch_transform(
    size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    augment: bool = False,
    strong_augment: bool = False
) -> Callable:
    """
    Create a transform function compatible with PyTorch Dataset classes.
    
    Args:
        size: Target (width, height) for resizing
        augment: If True, applies augmentation
        strong_augment: If True, uses strong augmentation (for minority class)
        
    Returns:
        Transform function that takes image path and returns PyTorch tensor
        
    Example:
        >>> from torch.utils.data import Dataset
        >>> 
        >>> class SkinLesionDataset(Dataset):
        ...     def __init__(self, image_paths, labels, transform=None):
        ...         self.image_paths = image_paths
        ...         self.labels = labels
        ...         self.transform = transform
        ...     
        ...     def __getitem__(self, idx):
        ...         img = self.transform(self.image_paths[idx])
        ...         label = self.labels[idx]
        ...         return img, label
        >>> 
        >>> # For training with strong augmentation
        >>> train_transform = create_pytorch_transform(
        ...     size=(384, 384), 
        ...     augment=True, 
        ...     strong_augment=True
        ... )
        >>> train_dataset = SkinLesionDataset(train_paths, train_labels, train_transform)
    """
    import torch
    
    augment_fn = None
    if augment:
        augment_fn = build_image_augment(strong=strong_augment)
    
    def transform(image_path: str) -> torch.Tensor:
        # Load and preprocess
        img = preprocess_image(image_path, size)
        
        # Apply augmentation if specified
        if augment_fn is not None:
            img = augment_fn(img)
        
        # Convert to PyTorch tensor: (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        return img_tensor
    
    return transform


def augment_light(img: np.ndarray) -> np.ndarray:
    """
    Light augmentation for majority class samples.

    Operations:
        - Horizontal flip (p=0.5)
        - Mild brightness/contrast jitter (p=0.5, ±10%)

    Args:
        img: Input image in [0, 1] range, shape (H, W, 3)

    Returns:
        Augmented image with the same shape and range as input.
    """
    if np.random.rand() < 0.5:
        img = np.ascontiguousarray(img[:, ::-1, :])

    if np.random.rand() < 0.5:
        contrast = 1.0 + (np.random.rand() - 0.5) * 0.2  # ±10%
        brightness = (np.random.rand() - 0.5) * 0.2      # ±10%
        img = np.clip(img * contrast + brightness, 0.0, 1.0)

    return img


def augment_strong(img: np.ndarray) -> np.ndarray:
    """
    Strong augmentation for minority class samples.

    Enhanced augmentation sequence to improve generalization on rare classes:
        - Horizontal flip (p=0.5)
        - Small rotation ±15° (p=0.5)
        - Brightness & contrast jitter ±20% (p=0.7)
        - Random zoom/scaling ±10% (p=0.3)
        - Gaussian blur σ∈[0.5,1.5] (p=0.2)
        - Gaussian noise σ=0.02 (p=0.1)

    Args:
        img: Input image in [0, 1] range, shape (H, W, 3)

    Returns:
        Augmented image with the same shape and range as input.
    """
    h, w = img.shape[:2]

    if np.random.rand() < 0.5:
        img = np.ascontiguousarray(img[:, ::-1, :])

    if np.random.rand() < 0.5:
        angle = np.random.uniform(-15, 15)
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        img = cv2.warpAffine(
            img,
            rotation_matrix,
            (w, h),
            borderMode=cv2.BORDER_REFLECT_101,
        )

    if np.random.rand() < 0.7:
        contrast = 1.0 + (np.random.rand() - 0.5) * 0.4  # ±20%
        brightness = (np.random.rand() - 0.5) * 0.4      # ±20%
        img = np.clip(img * contrast + brightness, 0.0, 1.0)

    if np.random.rand() < 0.3:
        scale = np.random.uniform(0.9, 1.1)
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(
            img,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
        )

        if scale > 1.0:
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            img = img_resized[start_y:start_y + h, start_x:start_x + w]
        else:
            pad_y = (h - new_h) // 2
            pad_x = (w - new_w) // 2
            img = np.pad(
                img_resized,
                ((pad_y, h - new_h - pad_y), (pad_x, w - new_w - pad_x), (0, 0)),
                mode="reflect",
            )
            img = img[:h, :w]

    if np.random.rand() < 0.2:
        sigma = np.random.uniform(0.5, 1.5)
        img = cv2.GaussianBlur(img, (3, 3), sigma)
        img = np.clip(img, 0.0, 1.0)

    if np.random.rand() < 0.1:
        noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)
        img = np.clip(img + noise, 0.0, 1.0)

    return img


def build_image_augment(strong: bool = False) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create augmentation function for training data.
    
    Strategy:
        - Light augmentation: For majority class (minimal distortion)
        - Strong augmentation: For minority class (enhanced diversity)
    
    Conservative approach ensures diagnostic features are preserved while
    providing sufficient variation to prevent overfitting.
    
    Args:
        strong: If True, applies stronger augmentation suite for minority class
        
    Returns:
        Augmentation function that transforms input images in-place
        
    Note:
        All augmentations preserve lesion diagnostic characteristics:
        - No hue/saturation changes (preserve pigmentation)
        - Limited rotation angles (≤15°)
        - Mild brightness/contrast adjustments (≤20%)
    """
    return augment_strong if strong else augment_light


@dataclass
class StructuredPreprocessor:
    """
    Preprocessor for structured/tabular features.
    
    Applies z-score normalization (standardization) to ensure all features
    contribute equally to distance-based models and gradient descent.
    
    Attributes:
        feature_names: Names of features for documentation/debugging
        _scaler: Internal StandardScaler instance (fitted during training)
    
    Example:
        >>> preprocessor = StructuredPreprocessor(feature_names=['age', 'size'])
        >>> X_train_scaled = preprocessor.fit_transform(X_train)
        >>> X_test_scaled = preprocessor.transform(X_test)
    """
    
    feature_names: Iterable[str]
    _scaler: StandardScaler | None = None

    def fit(self, X: np.ndarray) -> "StructuredPreprocessor":
        """
        Fit scaler to training data.
        
        Computes mean and standard deviation for each feature.
        
        Args:
            X: Training data with shape (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        self._scaler = StandardScaler()
        self._scaler.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply standardization to data.
        
        Transforms features to zero mean and unit variance using
        statistics computed during fit().
        
        Args:
            X: Data to transform with shape (n_samples, n_features)
            
        Returns:
            Standardized data with same shape as input
            
        Raises:
            RuntimeError: If called before fit()
        """
        if self._scaler is None:
            raise RuntimeError(
                "StructuredPreprocessor must be fit before transform."
            )
        return self._scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit scaler and transform data in one step.
        
        Convenience method for training data preprocessing.
        
        Args:
            X: Training data with shape (n_samples, n_features)
            
        Returns:
            Standardized training data
        """
        return self.fit(X).transform(X)