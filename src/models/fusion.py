"""
Cross-Attention Fusion Module for Multi-Modal Feature Integration

This module implements a novel cross-attention mechanism where image features
(extracted from DINO backbone) act as queries, and clinical feature embeddings
act as keys and values. This explicitly models the conditional relationship
between visual patterns in dermoscopic images and clinical prognostic factors.

Technical Details:
- Single-head cross-attention for computational efficiency
- Clinical features are projected into multiple learned tokens to provide
  rich context for attention
- Residual connection from original image features preserves visual semantics
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
	"""
	Cross-attention fusion module that conditions image features on clinical data.
	
	The architecture follows:
	1. Image features (from DINO ViT) -> Query
	2. Clinical features -> projected to multiple tokens -> Keys & Values
	3. Cross-attention computes context vector
	4. Concatenated fusion with residual from image features
	
	Args:
		img_dim: Dimension of image feature vector (e.g., 768 for ViT-B/16)
		clin_dim: Dimension of clinical feature vector
		num_clin_tokens: Number of learned tokens to project clinical features into
			(provides multiple attention "anchors" for richer context)
		hidden: Hidden dimension for attention projections
		dropout: Dropout rate for regularization
	"""
	
	def __init__(
		self,
		img_dim: int,
		clin_dim: int,
		num_clin_tokens: int = 4,
		hidden: int = 256,
		dropout: float = 0.1,
	):
		super().__init__()
		self.num_clin_tokens = num_clin_tokens
		
		# Query projection from image features
		self.img_query = nn.Linear(img_dim, hidden, bias=False)
		
		# Project clinical features into multiple learned tokens
		# This creates a richer representation space for attention
		self.clin_proj = nn.Sequential(
			nn.Linear(clin_dim, hidden * num_clin_tokens),
			nn.GELU(),
			nn.Dropout(dropout),
		)
		
		# Key and value projections from clinical tokens
		self.clin_key = nn.Linear(hidden, hidden, bias=False)
		self.clin_value = nn.Linear(hidden, hidden, bias=False)
		
		# Output projection that fuses attended context with original image features
		self.output_proj = nn.Sequential(
			nn.Linear(hidden + img_dim, hidden),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden, hidden),
		)
		
		# Layer normalization for stability
		self.norm = nn.LayerNorm(hidden)
	
	def forward(self, img_feat: torch.Tensor, clin_feat: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass through cross-attention fusion.
		
		Args:
			img_feat: Image features from backbone
				- Shape: (B, D_img) for CLS token, or (B, N_tokens, D_img) for patch tokens
			clin_feat: Clinical feature vector
				- Shape: (B, D_clin)
		
		Returns:
			Fused feature vector of shape (B, hidden)
		"""
		batch_size = img_feat.size(0)
		
		# Handle both CLS-only and full token outputs from ViT
		# If we have full tokens, we'll use mean pooling (alternative: learnable attention pooling)
		if img_feat.dim() == 3:
			# Full token sequence: mean pool over spatial/token dimension
			img_feat = img_feat.mean(dim=1)  # (B, D_img)
		
		# Project image features to query space
		query = self.img_query(img_feat).unsqueeze(1)  # (B, 1, hidden)
		
		# Project clinical features into multiple tokens
		# This allows the model to attend to different aspects of clinical data
		clin_tokens_flat = self.clin_proj(clin_feat)  # (B, hidden * num_clin_tokens)
		clin_tokens = clin_tokens_flat.view(batch_size, self.num_clin_tokens, -1)  # (B, num_tokens, hidden)
		
		# Compute keys and values from clinical tokens
		keys = self.clin_key(clin_tokens)    # (B, num_tokens, hidden)
		values = self.clin_value(clin_tokens)  # (B, num_tokens, hidden)
		
		# Scaled dot-product attention
		# Q @ K^T / sqrt(d_k) -> softmax -> @ V
		scale = (keys.size(-1)) ** -0.5
		attn_scores = torch.matmul(query, keys.transpose(1, 2)) * scale  # (B, 1, num_tokens)
		attn_weights = F.softmax(attn_scores, dim=-1)  # (B, 1, num_tokens)
		
		# Compute attended context vector
		attended_context = torch.matmul(attn_weights, values)  # (B, 1, hidden)
		attended_context = attended_context.squeeze(1)  # (B, hidden)
		attended_context = self.norm(attended_context)
		
		# Fuse attended context with original image features
		# This preserves the raw visual semantics while incorporating clinical context
		concatenated = torch.cat([img_feat, attended_context], dim=-1)  # (B, img_dim + hidden)
		fused = self.output_proj(concatenated)  # (B, hidden)
		
		return fused


class SimpleConcatFusion(nn.Module):
	"""
	Simple concatenation-based fusion (baseline for ablation studies).
	
	This is a straightforward concatenation followed by MLP, useful for
	comparison against cross-attention fusion.
	"""
	
	def __init__(
		self,
		img_dim: int,
		clin_dim: int,
		hidden: int = 256,
		dropout: float = 0.1,
	):
		super().__init__()
		self.fusion_mlp = nn.Sequential(
			nn.Linear(img_dim + clin_dim, hidden),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden, hidden),
		)
	
	def forward(self, img_feat: torch.Tensor, clin_feat: torch.Tensor) -> torch.Tensor:
		"""Simple concatenation and MLP projection."""
		if img_feat.dim() == 3:
			img_feat = img_feat.mean(dim=1)
		concatenated = torch.cat([img_feat, clin_feat], dim=-1)
		return self.fusion_mlp(concatenated)

