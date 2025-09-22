from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
import torch

# Torch image normalization for ResNet/ImageNet
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image_path: str, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
	"""Load, resize, normalize to [0,1], then standardize using ImageNet stats."""
	img = cv2.imread(image_path)
	if img is None:
		raise FileNotFoundError(f"Failed to read image: {image_path}")
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
	img = img.astype(np.float32) / 255.0
	img = (img - IMAGENET_MEAN) / IMAGENET_STD
	return img


def build_image_augment():
	"""Return a callable that applies light augmentation using OpenCV-like ops."""
	def _augment(img: np.ndarray) -> np.ndarray:
		# Horizontal flip with 50% chance
		if np.random.rand() < 0.5:
			img = np.ascontiguousarray(img[:, ::-1, :])
		# Small brightness/contrast jitter
		if np.random.rand() < 0.5:
			alpha = 1.0 + (np.random.rand() - 0.5) * 0.2  # contrast
			beta = (np.random.rand() - 0.5) * 0.1        # brightness
			img = np.clip(img * alpha + beta, 0.0, 1.0)
		return img
	return _augment


@dataclass
class StructuredPreprocessor:
	feature_names: Iterable[str]
	_scaler: StandardScaler | None = None

	def fit(self, X: np.ndarray) -> "StructuredPreprocessor":
		self._scaler = StandardScaler()
		self._scaler.fit(X)
		return self

	def transform(self, X: np.ndarray) -> np.ndarray:
		if self._scaler is None:
			raise RuntimeError("StructuredPreprocessor must be fit before transform.")
		return self._scaler.transform(X)

	def fit_transform(self, X: np.ndarray) -> np.ndarray:
		return self.fit(X).transform(X)


