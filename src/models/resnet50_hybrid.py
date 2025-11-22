"""
ResNet50-based Model for Melanoma Prognosis

This module implements a dual-head model using ResNet50 backbone for image-only input.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Model(nn.Module):
	"""
	Dual-head model using ResNet50 backbone for image-only input.
	
	Architecture:
	1. Image backbone: ResNet50 (ImageNet pretrained)
	2. Feature projection: MLP to project ResNet features to hidden dimension
	3. Heads: Classification (binary) and Regression (thickness)
	"""
	
	def __init__(
		self,
		hidden_dim: int = 256,
		dropout_rate: float = 0.3,
		pretrained: bool = True,
		task: str = "classification",  # classification or regression
		multitask: bool = False,
	):
		super().__init__()
		# Image backbone
		backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
		for param in backbone.parameters():
			param.requires_grad = False
		self.cnn_backbone = nn.Sequential(*list(backbone.children())[:-1])  # output: (B, 2048, 1, 1)
		cnn_out_dim = 2048

		# Feature projection: ResNet features -> hidden dimension
		self.feature_proj = nn.Sequential(
			nn.Linear(cnn_out_dim, hidden_dim),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout_rate),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout_rate),
		)
		
		self.task = task
		self.multitask = multitask
		
		# Heads
		self.cls_head = nn.Linear(hidden_dim, 1)
		self.reg_head = nn.Linear(hidden_dim, 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, image_tensor: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
		"""
		Forward pass through the model.
		
		Args:
			image_tensor: Batch of images, shape (B, C, H, W)
		
		Returns:
			- If multitask=True: dict with "cls" and "reg" keys
			- If multitask=False and task="classification": (B,) binary logits
			- If multitask=False and task="regression": (B,) regression predictions
		"""
		# Extract features from ResNet backbone
		x = self.cnn_backbone(image_tensor)  # (B, 2048, 1, 1)
		x = torch.flatten(x, 1)  # (B, 2048)
		
		# Project features to hidden dimension
		projected = self.feature_proj(x)  # (B, hidden_dim)
		
		# Apply prediction heads
		if self.multitask:
			cls = self.sigmoid(self.cls_head(projected)).squeeze(1)
			reg = self.reg_head(projected).squeeze(1)
			return {"cls": cls, "reg": reg}
		else:
			if self.task == "classification":
				return self.sigmoid(self.cls_head(projected)).squeeze(1)
			else:
				return self.reg_head(projected).squeeze(1)


def create_hybrid_model(
	hidden_dim: int = 256,
	dropout_rate: float = 0.3,
	pretrained: bool = True,
	task: str = "classification",
	multitask: bool = False,
) -> nn.Module:
	"""
	Factory function to create a ResNet50 model.
	
	Note: The function name is kept as "create_hybrid_model" for backward
	compatibility, but the model no longer uses clinical features.
	"""
	return ResNet50Model(
		hidden_dim=hidden_dim,
		dropout_rate=dropout_rate,
		pretrained=pretrained,
		task=task,
		multitask=multitask,
	)
