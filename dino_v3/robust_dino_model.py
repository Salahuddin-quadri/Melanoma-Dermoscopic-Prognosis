"""
Robust DINOv3 Model Implementation with proper error handling and freezing.
This implementation fixes common singleton tensor errors and size mismatches.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from dataclasses import dataclass
from typing import Optional, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_vit_embed_dim(arch: str) -> int:
    """Return embedding dimension for selected ViT backbone from torchvision."""
    arch = arch.lower()
    if arch in {"vit_b_16", "vit_b_32"}:
        return 768
    if arch in {"vit_l_16", "vit_l_32"}:
        return 1024
    if arch in {"vit_h_14"}:
        return 1280
    raise ValueError(f"Unsupported ViT arch: {arch}")


def _build_vit_backbone(arch: str = "vit_b_16", pretrained: bool = True) -> nn.Module:
    """Build a ViT backbone from torchvision and return a feature extractor."""
    try:
        weights = None
        arch_lower = arch.lower()
        if pretrained:
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

        model_fn = getattr(tvm, arch)
        vit = model_fn(weights=weights)
        
        class VitEncoder(nn.Module):
            def __init__(self, vit_model: nn.Module):
                super().__init__()
                self.vit = vit_model
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                try:
                    x = self.vit._process_input(x)
                    n = x.shape[0]
                    cls_token = self.vit.class_token.expand(n, -1, -1)
                    x = torch.cat([cls_token, x], dim=1)
                    x = self.vit.encoder(x)
                    return x[:, 0]  # Take CLS token
                except Exception as e:
                    logger.error(f"Error in VitEncoder forward: {e}")
                    raise
                    
        return VitEncoder(vit)
    except Exception as e:
        logger.error(f"Error building ViT backbone: {e}")
        raise


class RobustDINOHead(nn.Module):
    """Robust DINO projection head with proper error handling."""
    
    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 256, 
                 nlayers: int = 3, bottleneck_dim: int = 256):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        layers = []
        dim = in_dim
        
        for i in range(nlayers - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            dim = hidden_dim
            
        layers.append(nn.Linear(dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        
        # Use weight normalization for stability
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1.0)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            # Ensure input is 2D
            if x.dim() != 2:
                x = x.view(x.size(0), -1)
                
            x = self.mlp(x)
            x = F.normalize(x, dim=-1, p=2)
            return self.last_layer(x)
        except Exception as e:
            logger.error(f"Error in DINOHead forward: {e}, input shape: {x.shape}")
            raise


class RobustDINOLoss(nn.Module):
    """Robust DINO loss with proper tensor handling."""
    
    def __init__(self, out_dim: int, teacher_temp: float = 0.04, 
                 student_temp: float = 0.1, center_momentum: float = 0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor) -> torch.Tensor:
        try:
            # Ensure inputs are 2D tensors
            if student_outputs.dim() != 2:
                student_outputs = student_outputs.view(student_outputs.size(0), -1)
            if teacher_outputs.dim() != 2:
                teacher_outputs = teacher_outputs.view(teacher_outputs.size(0), -1)
                
            # Check dimensions match
            if student_outputs.size(0) != teacher_outputs.size(0):
                logger.warning(f"Batch size mismatch: student={student_outputs.size(0)}, teacher={teacher_outputs.size(0)}")
                min_size = min(student_outputs.size(0), teacher_outputs.size(0))
                student_outputs = student_outputs[:min_size]
                teacher_outputs = teacher_outputs[:min_size]
            
            # Ensure all tensors are on the same device
            device = student_outputs.device
            teacher_outputs = teacher_outputs.to(device)
            self.center = self.center.to(device)
            
            # Apply temperature scaling
            student_out = F.log_softmax(student_outputs / self.student_temp, dim=-1)
            teacher_out = F.softmax((teacher_outputs - self.center) / self.teacher_temp, dim=-1)
            
            # Compute loss
            loss = torch.mean(torch.sum(-teacher_out * student_out, dim=-1))
            
            # Update center with proper error handling
            with torch.no_grad():
                batch_center = torch.mean(teacher_outputs, dim=0, keepdim=True)
                self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
                
            return loss
            
        except Exception as e:
            logger.error(f"Error in DINOLoss forward: {e}")
            logger.error(f"Student shape: {student_outputs.shape}, Teacher shape: {teacher_outputs.shape}")
            raise


@dataclass
class RobustDINOv3Config:
    arch: str = "vit_b_16"
    pretrained: bool = True
    proj_hidden_dim: int = 2048
    proj_out_dim: int = 1024
    proj_layers: int = 3
    bottleneck_dim: int = 256
    freeze_layers: int = 8  # Number of layers to freeze


class RobustDINOv3Model(nn.Module):
    """Robust DINOv3 model with proper error handling and freezing."""
    
    def __init__(self, cfg: Optional[RobustDINOv3Config] = None):
        super().__init__()
        self.cfg = cfg or RobustDINOv3Config()
        self.embed_dim = _get_vit_embed_dim(self.cfg.arch)
        
        try:
            # Build backbones
            self.student_backbone = _build_vit_backbone(self.cfg.arch, pretrained=self.cfg.pretrained)
            self.teacher_backbone = _build_vit_backbone(self.cfg.arch, pretrained=self.cfg.pretrained)
            
            # Build heads
            self.student_head = RobustDINOHead(
                self.embed_dim, 
                hidden_dim=self.cfg.proj_hidden_dim, 
                out_dim=self.cfg.proj_out_dim, 
                nlayers=self.cfg.proj_layers, 
                bottleneck_dim=self.cfg.bottleneck_dim
            )
            self.teacher_head = RobustDINOHead(
                self.embed_dim, 
                hidden_dim=self.cfg.proj_hidden_dim, 
                out_dim=self.cfg.proj_out_dim, 
                nlayers=self.cfg.proj_layers, 
                bottleneck_dim=self.cfg.bottleneck_dim
            )
            
            # Freeze teacher parameters
            self._freeze_teacher()
            
            # Freeze specified layers in student
            self._freeze_student_layers()
            
        except Exception as e:
            logger.error(f"Error initializing RobustDINOv3Model: {e}")
            raise

    def _freeze_teacher(self):
        """Freeze all teacher parameters."""
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False
        for param in self.teacher_head.parameters():
            param.requires_grad = False

    def _freeze_student_layers(self):
        """Freeze specified number of layers in student backbone."""
        if hasattr(self.student_backbone, 'vit') and hasattr(self.student_backbone.vit, 'encoder'):
            encoder_blocks = self.student_backbone.vit.encoder.layers
            num_frozen = min(self.cfg.freeze_layers, len(encoder_blocks))
            
            for i, block in enumerate(encoder_blocks):
                if i < num_frozen:
                    for param in block.parameters():
                        param.requires_grad = False
                    logger.info(f"Froze encoder block {i}")
        else:
            logger.warning("Could not find encoder blocks to freeze")

    def forward_student(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through student network."""
        try:
            if images.dim() != 4:
                raise ValueError(f"Expected 4D input (B, C, H, W), got {images.dim()}D")
                
            x = self.student_backbone(images)
            return self.student_head(x)
        except Exception as e:
            logger.error(f"Error in forward_student: {e}, input shape: {images.shape}")
            raise

    @torch.no_grad()
    def forward_teacher(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through teacher network."""
        try:
            if images.dim() != 4:
                raise ValueError(f"Expected 4D input (B, C, H, W), got {images.dim()}D")
                
            x = self.teacher_backbone(images)
            return self.teacher_head(x)
        except Exception as e:
            logger.error(f"Error in forward_teacher: {e}, input shape: {images.shape}")
            raise

    @torch.no_grad()
    def update_teacher(self, momentum: float = 0.996):
        """Update teacher parameters using EMA."""
        try:
            # Update backbone parameters
            for student_param, teacher_param in zip(
                self.student_backbone.parameters(), 
                self.teacher_backbone.parameters()
            ):
                teacher_param.data = teacher_param.data * momentum + student_param.data * (1.0 - momentum)
                
            # Update head parameters
            for student_param, teacher_param in zip(
                self.student_head.parameters(), 
                self.teacher_head.parameters()
            ):
                teacher_param.data = teacher_param.data * momentum + student_param.data * (1.0 - momentum)
                
        except Exception as e:
            logger.error(f"Error updating teacher: {e}")
            raise

    def get_trainable_parameters(self):
        """Get list of trainable parameters."""
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params

    def count_parameters(self):
        """Count total and trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


def create_robust_dino_v3(cfg: Optional[RobustDINOv3Config] = None) -> RobustDINOv3Model:
    """Create a robust DINOv3 model."""
    return RobustDINOv3Model(cfg)


