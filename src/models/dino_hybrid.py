"""
DINO-based Hybrid Model for Melanoma Prognosis

This module implements a dual-head model combining:
1. Domain-specific DINO ViT backbone (pre-trained on 14K dermoscopic images)
2. Clinical feature encoder (MLP over structured data)
3. Cross-attention fusion mechanism
4. Dual prediction heads: classification (malignant/benign) and prognosis (Breslow thickness)

The model is designed for fine-tuning on a smaller labeled dataset (1000 images)
while leveraging the rich representations learned from domain-specific SSL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torchvision.models as tvm

from .fusion import CrossAttentionFusion, SimpleConcatFusion


def _get_vit_embed_dim(arch: str) -> int:
	"""Get embedding dimension for ViT architecture."""
	arch = arch.lower()
	if arch in {"vit_b_16", "vit_b_32"}:
		return 768
	elif arch in {"vit_l_16", "vit_l_32"}:
		return 1024
	elif arch == "vit_h_14":
		return 1280
	else:
		raise ValueError(f"Unsupported ViT architecture: {arch}")


def _build_vit_feature_extractor(
	arch: str = "vit_b_16",
	pretrained: bool = True,
	return_tokens: bool = False,
) -> nn.Module:
	"""
	Build a ViT feature extractor compatible with torchvision models.
	
	This extractor can return either:
	- CLS token only (standard): shape (B, D)
	- Full token sequence: shape (B, 1+N_patches, D)
	
	The full token option enables experiments with mean pooling vs CLS token.
	
	Args:
		arch: ViT architecture name (e.g., 'vit_b_16')
		pretrained: Whether to use ImageNet pretrained weights
		return_tokens: If True, return all tokens; if False, return CLS token only
	
	Returns:
		ViT feature extractor module
	"""
	# Select pretrained weights based on architecture
	weights = None
	if pretrained:
		arch_lower = arch.lower()
		if arch_lower == "vit_b_16":
			weights = tvm.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
		elif arch_lower == "vit_b_32":
			weights = tvm.ViT_B_32_Weights.IMAGENET1K_V1
		elif arch_lower == "vit_l_16":
			weights = tvm.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1
		elif arch_lower == "vit_l_32":
			weights = tvm.ViT_L_32_Weights.IMAGENET1K_V1
		elif arch_lower == "vit_h_14":
			weights = tvm.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
	
	# Build ViT model
	model_fn = getattr(tvm, arch)
	vit_model = model_fn(weights=weights)
	
	class VitFeatureExtractor(nn.Module):
		"""Wrapper to extract features from ViT backbone."""
		
		def __init__(self, vit: nn.Module, return_tokens: bool):
			super().__init__()
			self.vit = vit
			self.return_tokens = return_tokens
		
		def forward(self, x: torch.Tensor) -> torch.Tensor:
			"""
			Extract features from ViT.
			
			Args:
				x: Input images, shape (B, C, H, W)
			
			Returns:
				Feature tensor:
				- If return_tokens=False: (B, D) - CLS token only
				- If return_tokens=True: (B, 1+N_patches, D) - CLS + patch tokens
			"""
			# Process input: patch embedding + positional encoding
			x = self.vit._process_input(x)  # (B, N_patches, D)
			
			# Add CLS token
			batch_size = x.shape[0]
			cls_token = self.vit.class_token.expand(batch_size, -1, -1)  # (B, 1, D)
			x = torch.cat([cls_token, x], dim=1)  # (B, 1+N_patches, D)
			
			# Pass through encoder
			x = self.vit.encoder(x)  # (B, 1+N_patches, D)
			
			if self.return_tokens:
				return x  # Return all tokens for flexible pooling
			else:
				return x[:, 0]  # Return CLS token only
	
	return VitFeatureExtractor(vit_model, return_tokens)


def _load_dino_checkpoint(
	checkpoint_path: str,
	vit_model: nn.Module,
	device: torch.device,
) -> bool:
	"""
	Load domain-specific DINO weights into ViT backbone.
	
	This function extracts the student backbone weights from a DINO checkpoint
	and loads them into the ViT feature extractor. It handles key name mismatches
	by mapping DINO checkpoint keys to torchvision ViT keys.
	
	Args:
		checkpoint_path: Path to DINO checkpoint (.pt file)
		vit_model: ViT feature extractor to load weights into
		device: Device to load checkpoint on
	
	Returns:
		True if loading succeeded, False otherwise
	"""
	checkpoint_path = Path(checkpoint_path)
	if not checkpoint_path.exists():
		return False
	
	try:
		checkpoint = torch.load(checkpoint_path, map_location=device)
		
		# Handle different checkpoint formats
		if isinstance(checkpoint, dict):
			# Try to find model state dict in common keys
			if "model_state_dict" in checkpoint:
				state_dict = checkpoint["model_state_dict"]
			elif "state_dict" in checkpoint:
				state_dict = checkpoint["state_dict"]
			elif "model" in checkpoint:
				state_dict = checkpoint["model"]
			else:
				state_dict = checkpoint
		else:
			state_dict = checkpoint
		
		# Extract student backbone weights
		# DINO checkpoints typically prefix with 'student_backbone.'
		vit_state = {}
		for key, value in state_dict.items():
			# Map DINO checkpoint keys to torchvision ViT keys
			if key.startswith("student_backbone.vit."):
				# Remove 'student_backbone.' prefix
				new_key = key.replace("student_backbone.vit.", "")
				vit_state[new_key] = value
			elif key.startswith("student_backbone.") and not "head" in key:
				# Handle alternative naming
				new_key = key.replace("student_backbone.", "")
				if new_key.startswith("vit."):
					new_key = new_key.replace("vit.", "")
					vit_state[new_key] = value
		
		# If no matching keys found, try direct loading (might work if keys align)
		if not vit_state:
			# Try to match keys that contain 'vit' or encoder-related terms
			for key, value in state_dict.items():
				if any(term in key.lower() for term in ["vit", "encoder", "patch_embed", "class_token"]):
					if "head" not in key.lower() and "teacher" not in key.lower():
						# Try to clean the key
						cleaned_key = key
						for prefix in ["student_backbone.", "student.", "module."]:
							if cleaned_key.startswith(prefix):
								cleaned_key = cleaned_key[len(prefix):]
						if cleaned_key.startswith("vit."):
							cleaned_key = cleaned_key[4:]
						vit_state[cleaned_key] = value
		
		if vit_state:
			# Load into ViT model
			missing_keys, unexpected_keys = vit_model.vit.load_state_dict(vit_state, strict=False)
			if missing_keys:
				print(f"Warning: {len(missing_keys)} keys not found in checkpoint")
			if unexpected_keys:
				print(f"Warning: {len(unexpected_keys)} unexpected keys in checkpoint")
			return True
		else:
			print("Warning: No matching ViT weights found in checkpoint")
			return False
			
	except Exception as e:
		print(f"Error loading DINO checkpoint: {e}")
		return False


class DinoHybrid(nn.Module):
	"""
	Dual-head hybrid model combining DINO ViT features with clinical data.
	
	Architecture:
	1. Image branch: DINO ViT backbone (domain-specific or ImageNet pretrained)
	2. Clinical branch: MLP encoder for structured features
	3. Fusion: Cross-attention mechanism (or simple concatenation)
	4. Heads: Classification (binary) and Regression (thickness)
	
	The model supports fine-tuning from domain-specific DINO checkpoints.
	"""
	
	def __init__(
		self,
		num_structured_features: int,
		task: str = "classification",
		multitask: bool = False,
		arch: str = "vit_b_16",
		pretrained: bool = True,
		dino_checkpoint: Optional[str] = None,
		use_tokens: bool = False,
		fusion_type: str = "cross_attention",
		fusion_hidden: int = 256,
		num_clin_tokens: int = 4,
		dropout: float = 0.1,
		freeze_backbone_layers: int = 0,
	):
		"""
		Initialize DINO hybrid model.
		
		Args:
			num_structured_features: Number of clinical/structured features
			task: "classification" or "regression" (ignored if multitask=True)
			multitask: If True, use dual heads (classification + regression)
			arch: ViT architecture name
			pretrained: Use ImageNet pretrained weights (if dino_checkpoint not provided)
			dino_checkpoint: Path to domain-specific DINO checkpoint
			use_tokens: If True, return full token sequence; if False, CLS only
			fusion_type: "cross_attention" or "concat"
			fusion_hidden: Hidden dimension for fusion module
			num_clin_tokens: Number of learned clinical tokens for cross-attention
			dropout: Dropout rate
			freeze_backbone_layers: Number of encoder layers to freeze (0 = all trainable)
		"""
		super().__init__()
		self.task = task
		self.multitask = multitask
		self.arch = arch
		embed_dim = _get_vit_embed_dim(arch)
		
		# Build ViT feature extractor
		self.backbone = _build_vit_feature_extractor(
			arch=arch,
			pretrained=pretrained,
			return_tokens=use_tokens,
		)
		
		# Load domain-specific DINO weights if provided
		if dino_checkpoint:
			device = next(self.backbone.parameters()).device
			loaded = _load_dino_checkpoint(dino_checkpoint, self.backbone, device)
			if loaded:
				print(f"✓ Loaded domain-specific DINO weights from {dino_checkpoint}")
			else:
				print(f"⚠ Failed to load DINO checkpoint, using pretrained weights")
		
		# Freeze specified encoder layers
		if freeze_backbone_layers > 0:
			self._freeze_layers(freeze_backbone_layers)
		
		# Clinical feature encoder (simple MLP)
		self.clinical_encoder = nn.Sequential(
			nn.Linear(num_structured_features, 64),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(64, 32),
		)
		clin_dim = 32
		
		# Fusion module
		if fusion_type == "cross_attention":
			self.fusion = CrossAttentionFusion(
				img_dim=embed_dim,
				clin_dim=clin_dim,
				num_clin_tokens=num_clin_tokens,
				hidden=fusion_hidden,
				dropout=dropout,
			)
		elif fusion_type == "concat":
			self.fusion = SimpleConcatFusion(
				img_dim=embed_dim,
				clin_dim=clin_dim,
				hidden=fusion_hidden,
				dropout=dropout,
			)
		else:
			raise ValueError(f"Unknown fusion_type: {fusion_type}")
		
		# Prediction heads
		self.cls_head = nn.Sequential(
			nn.Linear(fusion_hidden, 1),
			nn.Sigmoid(),  # For binary classification
		)
		self.reg_head = nn.Linear(fusion_hidden, 1)  # For regression (Breslow thickness)
	
	def _freeze_layers(self, num_layers: int):
		"""Freeze the first N encoder layers for transfer learning."""
		if hasattr(self.backbone, "vit") and hasattr(self.backbone.vit, "encoder"):
			encoder_layers = self.backbone.vit.encoder.layers
			num_to_freeze = min(num_layers, len(encoder_layers))
			
			for i in range(num_to_freeze):
				for param in encoder_layers[i].parameters():
					param.requires_grad = False
			print(f"Froze {num_to_freeze} encoder layers")
	
	def forward(
		self,
		image_tensor: torch.Tensor,
		structured_tensor: torch.Tensor,
	) -> torch.Tensor | dict[str, torch.Tensor]:
		"""
		Forward pass through the model.
		
		Args:
			image_tensor: Batch of images, shape (B, C, H, W)
			structured_tensor: Batch of clinical features, shape (B, F)
		
		Returns:
			- If multitask=True: dict with "cls" and "reg" keys
			- If multitask=False and task="classification": (B,) binary logits
			- If multitask=False and task="regression": (B,) regression predictions
		"""
		# Extract image features from DINO backbone
		img_features = self.backbone(image_tensor)  # (B, D) or (B, N, D)
		
		# Encode clinical features
		clin_features = self.clinical_encoder(structured_tensor)  # (B, clin_dim)
		
		# Fuse image and clinical features
		fused_features = self.fusion(img_features, clin_features)  # (B, fusion_hidden)
		
		# Apply prediction heads
		if self.multitask:
			cls_pred = self.cls_head(fused_features).squeeze(1)  # (B,)
			reg_pred = self.reg_head(fused_features).squeeze(1)  # (B,)
			return {"cls": cls_pred, "reg": reg_pred}
		elif self.task == "classification":
			return self.cls_head(fused_features).squeeze(1)  # (B,)
		else:  # regression
			return self.reg_head(fused_features).squeeze(1)  # (B,)


def create_dino_hybrid_model(
	num_structured_features: int,
	task: str = "classification",
	multitask: bool = False,
	arch: str = "vit_b_16",
	pretrained: bool = True,
	dino_checkpoint: Optional[str] = None,
	use_tokens: bool = False,
	fusion_type: str = "cross_attention",
	fusion_hidden: int = 256,
	num_clin_tokens: int = 4,
	dropout: float = 0.1,
	freeze_backbone_layers: int = 0,
) -> nn.Module:
	"""
	Factory function to create a DINO hybrid model.
	
	This is the main entry point for creating the model with default or custom
	configuration. Used by the training pipeline.
	"""
	return DinoHybrid(
		num_structured_features=num_structured_features,
		task=task,
		multitask=multitask,
		arch=arch,
		pretrained=pretrained,
		dino_checkpoint=dino_checkpoint,
		use_tokens=use_tokens,
		fusion_type=fusion_type,
		fusion_hidden=fusion_hidden,
		num_clin_tokens=num_clin_tokens,
		dropout=dropout,
		freeze_backbone_layers=freeze_backbone_layers,
	)

