"""
Loss Functions for Handling Class Imbalance

This module implements advanced loss functions to address severe class imbalance
in binary classification tasks (88% BEN, 12% MEL in this case).

Functions:
- WeightedBCELoss: Binary Cross-Entropy with class weights (inverse frequency)
- FocalLoss: Focal Loss for addressing hard examples and class imbalance
"""

from __future__ import annotations

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
	"""
	Weighted Binary Cross-Entropy Loss.
	
	Addresses class imbalance by weighting the loss inversely proportional to
	class frequency. The minority class gets higher weight.
	
	Formula: loss = -w_pos * y * log(p) - w_neg * (1-y) * log(1-p)
	
	Where weights are typically computed as:
	- w_pos = n_total / (2 * n_positive)
	- w_neg = n_total / (2 * n_negative)
	
	Args:
		pos_weight: Weight for positive class (default: auto-computed)
		reduction: 'mean', 'sum', or 'none'
	"""
	
	def __init__(self, pos_weight: float | torch.Tensor | None = None, reduction: str = "mean"):
		super().__init__()
		if pos_weight is not None:
			if isinstance(pos_weight, float):
				self.pos_weight = torch.tensor(pos_weight)
			else:
				self.pos_weight = pos_weight
		else:
			self.pos_weight = None
		self.reduction = reduction
	
	def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		"""
		Compute weighted BCE loss.
		
		Args:
			predictions: Predicted probabilities, shape (N,) or (N, 1)
			targets: True binary labels, shape (N,)
		
		Returns:
			Weighted BCE loss
		"""
		# Ensure predictions are in [0, 1] range
		if predictions.dim() > 1:
			predictions = predictions.squeeze(1)
		
		# Clamp to avoid numerical issues
		predictions = torch.clamp(predictions, min=1e-7, max=1.0 - 1e-7)
		
		# Standard BCE loss
		loss = F.binary_cross_entropy(
			predictions, 
			targets, 
			weight=None,  # We'll apply pos_weight separately
			pos_weight=self.pos_weight,
			reduction='none'
		)
		
		if self.reduction == 'mean':
			return loss.mean()
		elif self.reduction == 'sum':
			return loss.sum()
		else:
			return loss
	
	@classmethod
	def from_class_counts(cls, n_positive: int, n_negative: int, device: str = "cpu") -> "WeightedBCELoss":
		"""
		Create WeightedBCELoss with automatic weight calculation from class counts.
		
		Args:
			n_positive: Number of positive class samples (MEL)
			n_negative: Number of negative class samples (BEN)
			device: Device to place weights on
		
		Returns:
			WeightedBCELoss instance
		"""
		if n_positive == 0:
			raise ValueError(
				f"Cannot create WeightedBCELoss: No positive class samples found! "
				f"This usually means the train split has no MEL samples. "
				f"Check your data split or use a different random seed."
			)
		if n_negative == 0:
			raise ValueError(
				f"Cannot create WeightedBCELoss: No negative class samples found!"
			)
		
		n_total = n_positive + n_negative
		# Standard weighting: inverse frequency
		pos_weight = torch.tensor(n_total / (2.0 * n_positive), device=device)
		return cls(pos_weight=pos_weight)


class FocalLoss(nn.Module):
	"""
	Focal Loss for addressing class imbalance and hard examples.
	
	Focal Loss down-weights easy examples and focuses training on hard examples.
	This is particularly effective for imbalanced datasets.
	
	Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
	
	Where:
	- p_t = p if y=1, else 1-p
	- alpha: Balancing factor for classes
	- gamma: Focusing parameter (higher = more focus on hard examples)
	
	Args:
		alpha: Balancing factor (can be scalar or [alpha_neg, alpha_pos])
		gamma: Focusing parameter (default: 2.0, common range: 1.0-3.0)
		reduction: 'mean', 'sum', or 'none'
	"""
	
	def __init__(
		self,
		alpha: float | list[float] | None = None,
		gamma: float = 2.0,
		reduction: str = "mean",
	):
		super().__init__()
		if alpha is None:
			self.alpha = None
		elif isinstance(alpha, (list, tuple)):
			self.alpha = torch.tensor(alpha)
		else:
			self.alpha = torch.tensor([1.0 - alpha, alpha])  # [alpha_neg, alpha_pos]
		self.gamma = gamma
		self.reduction = reduction
	
	def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		"""
		Compute Focal Loss.
		
		Args:
			predictions: Predicted probabilities, shape (N,) or (N, 1)
			targets: True binary labels, shape (N,)
		
		Returns:
			Focal loss
		"""
		# Ensure predictions are in [0, 1] range
		if predictions.dim() > 1:
			predictions = predictions.squeeze(1)
		
		# Clamp to avoid numerical issues
		predictions = torch.clamp(predictions, min=1e-7, max=1.0 - 1e-7)
		
		# Compute p_t
		p_t = predictions * targets + (1 - predictions) * (1 - targets)
		
		# Compute focal weight: (1 - p_t)^gamma
		focal_weight = (1.0 - p_t) ** self.gamma
		
		# Apply alpha weighting if provided
		if self.alpha is not None:
			if self.alpha.device != predictions.device:
				self.alpha = self.alpha.to(predictions.device)
			alpha_t = self.alpha[0] * (1 - targets) + self.alpha[1] * targets
			focal_weight = alpha_t * focal_weight
		
		# Compute BCE loss
		bce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')
		
		# Apply focal weighting
		focal_loss = focal_weight * bce_loss
		
		if self.reduction == 'mean':
			return focal_loss.mean()
		elif self.reduction == 'sum':
			return focal_loss.sum()
		else:
			return focal_loss
	
	@classmethod
	def from_class_counts(
		cls,
		n_positive: int,
		n_negative: int,
		gamma: float = 2.0,
		device: str = "cpu",
	) -> "FocalLoss":
		"""
		Create FocalLoss with automatic alpha calculation from class counts.
		
		Args:
			n_positive: Number of positive class samples (MEL)
			n_negative: Number of negative class samples (BEN)
			gamma: Focusing parameter
			device: Device to place weights on
		
		Returns:
			FocalLoss instance
		"""
		if n_positive == 0:
			raise ValueError(
				f"Cannot create FocalLoss: No positive class samples found! "
				f"This usually means the train split has no MEL samples. "
				f"Check your data split or use a different random seed."
			)
		if n_negative == 0:
			raise ValueError(
				f"Cannot create FocalLoss: No negative class samples found!"
			)
		
		n_total = n_positive + n_negative
		alpha_neg = n_total / (2.0 * n_negative)  # Weight for negative class
		alpha_pos = n_total / (2.0 * n_positive)  # Weight for positive class
		return cls(alpha=[alpha_neg, alpha_pos], gamma=gamma)


def compute_class_weights(
	train_df,
	label_col: str = "label",
) -> dict[str, float]:
	"""
	Compute class weights from training dataframe.
	
	Returns weights that can be used for weighted sampling or loss weighting.
	
	Args:
		train_df: Training dataframe
		label_col: Column name for labels
	
	Returns:
		Dictionary with 'pos_weight' and 'class_counts' (n_neg, n_pos)
	"""
	if label_col not in train_df.columns:
		raise ValueError(f"Label column '{label_col}' not found in dataframe")
	
	labels = pd.to_numeric(train_df[label_col], errors="coerce")
	
	# Count classes: 0.0 or 0 = negative (BEN), 1.0 or 1 = positive (MEL)
	n_positive = ((labels == 1.0) | (labels == 1)).sum()
	n_negative = ((labels == 0.0) | (labels == 0)).sum()
	
	# Debug: show what we're counting
	unique_labels = sorted(labels.dropna().unique())
	print(f"   Debug: Unique labels in train: {unique_labels}")
	print(f"   Debug: Label value counts: {dict(labels.value_counts())}")
	
	n_total = n_positive + n_negative
	
	# Inverse frequency weighting
	pos_weight = n_total / (2.0 * n_positive) if n_positive > 0 else 1.0
	neg_weight = n_total / (2.0 * n_negative) if n_negative > 0 else 1.0
	
	return {
		"pos_weight": float(pos_weight),
		"neg_weight": float(neg_weight),
		"n_positive": int(n_positive),
		"n_negative": int(n_negative),
		"n_total": int(n_total),
		"imbalance_ratio": float(n_negative / n_positive) if n_positive > 0 else float('inf'),
	}

