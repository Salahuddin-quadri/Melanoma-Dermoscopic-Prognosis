from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models



class ResNet50Hybrid(nn.Module):
	def __init__(
		self,
		num_structured_features: int,
		structured_hidden_units: Tuple[int, ...] = (64, 32),
		fusion_hidden_units: Tuple[int, ...] = (128, 64),
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

		# Structured branch
		layers_struct: list[nn.Module] = []
		in_dim = num_structured_features
		for i, units in enumerate(structured_hidden_units):
			layers_struct.append(nn.Linear(in_dim, units))
			layers_struct.append(nn.ReLU(inplace=True))
			layers_struct.append(nn.Dropout(dropout_rate))
			in_dim = units
		self.structured_net = nn.Sequential(*layers_struct) if layers_struct else nn.Identity()
		struct_out_dim = in_dim

		# Fusion head
		fusion_layers: list[nn.Module] = []
		in_dim = cnn_out_dim + struct_out_dim
		for i, units in enumerate(fusion_hidden_units):
			fusion_layers.append(nn.Linear(in_dim, units))
			fusion_layers.append(nn.ReLU(inplace=True))
			fusion_layers.append(nn.Dropout(dropout_rate))
			in_dim = units
		self.fusion_body = nn.Sequential(*fusion_layers)
		self.task = task
		self.multitask = multitask
		# Heads
		self.cls_head = nn.Linear(in_dim, 1)
		self.reg_head = nn.Linear(in_dim, 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, image_tensor: torch.Tensor, structured_tensor: torch.Tensor) -> torch.Tensor:
		# image_tensor: (B, 3, H, W), structured_tensor: (B, F)
		x = self.cnn_backbone(image_tensor)  # (B, 2048, 1, 1)
		x = torch.flatten(x, 1)  # (B, 2048)
		z = self.structured_net(structured_tensor)
		fused = torch.cat([x, z], dim=1)
		fused = self.fusion_body(fused)
		if self.multitask:
			cls = self.sigmoid(self.cls_head(fused)).squeeze(1)
			reg = self.reg_head(fused).squeeze(1)
			return {"cls": cls, "reg": reg}
		else:
			if self.task == "classification":
				return self.sigmoid(self.cls_head(fused)).squeeze(1)
			else:
				return self.reg_head(fused).squeeze(1)


def create_hybrid_model(
	num_structured_features: int,
	structured_hidden_units: Tuple[int, ...] = (64, 32),
	fusion_hidden_units: Tuple[int, ...] = (128, 64),
	dropout_rate: float = 0.3,
	pretrained: bool = True,
	task: str = "classification",
	multitask: bool = False,
) -> nn.Module:
	return ResNet50Hybrid(
		num_structured_features=num_structured_features,
		structured_hidden_units=structured_hidden_units,
		fusion_hidden_units=fusion_hidden_units,
		dropout_rate=dropout_rate,
		pretrained=pretrained,
		task=task,
		multitask=multitask,
	)


