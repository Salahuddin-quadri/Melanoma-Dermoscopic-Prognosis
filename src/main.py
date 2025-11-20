from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import pandas as pd

# Allow running as a module (python -m src.main) or as a script (python src/main.py)
try:
	from .utils import load_emb_data, split_dataset, StructuredPreprocessor
	from .models import create_hybrid_model, create_dino_hybrid_model
	from .train import train_model
	from .evaluate import evaluate_model
except ImportError:  # running as script
	import os, sys
	CURRENT_DIR = Path(__file__).resolve().parent
	PARENT_DIR = CURRENT_DIR.parent
	if str(CURRENT_DIR) not in sys.path:
		sys.path.insert(0, str(CURRENT_DIR))
	if str(PARENT_DIR) not in sys.path:
		sys.path.insert(0, str(PARENT_DIR))
	from utils import load_emb_data, split_dataset, StructuredPreprocessor
	from models import create_hybrid_model, create_dino_hybrid_model
	from train import train_model
	from evaluate import evaluate_model


def parse_args():
	parser = argparse.ArgumentParser(
		description="Melanoma Dermoscopic Prognosis: Dual-Head Model with DINO Backbone",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	
	# Data arguments
	parser.add_argument("--metadata_path", type=str, default="data/merged_dataset.csv",
		help="Path to metadata CSV file")
	parser.add_argument("--image_dir", type=str, default="data/images",
		help="Directory containing dermoscopic images")
	parser.add_argument("--image_size", type=int, nargs=2, default=[384,384],
		help="Image dimensions [height, width]")
	
	# Model selection
	parser.add_argument("--model_type", type=str, choices=["resnet", "dino"], default="dino",
		help="Model architecture: 'resnet' for ResNet50 hybrid, 'dino' for DINO ViT hybrid")
	parser.add_argument("--dino_checkpoint", type=str, default="",
		help="Path to domain-specific DINO checkpoint (e.g., dino_v3/outputs_dino/checkpoints/best.pt). "
		     "If not provided, uses ImageNet pretrained ViT weights.")
	parser.add_argument("--vit_arch", type=str, default="vit_b_32",
		choices=["vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32"],
		help="ViT architecture for DINO model")
	parser.add_argument("--freeze_backbone_layers", type=int, default=7,
		help="Number of ViT encoder layers to freeze (0 = all trainable)")
	parser.add_argument("--use_tokens", action="store_true",
		help="Use full token sequence instead of CLS token only (enables mean/attention pooling)")
	parser.add_argument("--fusion_type", type=str, choices=["cross_attention", "concat"], default="cross_attention",
		help="Fusion mechanism: 'cross_attention' (novel) or 'concat' (baseline)")
	parser.add_argument("--num_clin_tokens", type=int, default=4,
		help="Number of learned clinical tokens for cross-attention fusion")
	
	# Training arguments
	parser.add_argument("--mode", type=str, choices=["train", "evaluate"], default="train")
	parser.add_argument("--epochs", type=int, default=30,
		help="Number of training epochs")
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--loss_alpha", type=float, default=0.5,
		help="Weight for classification loss in multitask (0-1). "
		     "Total loss = alpha * loss_cls + (1-alpha) * loss_reg")
	parser.add_argument("--cls_loss_type", type=str, choices=["bce", "weighted_bce", "focal"], default="weighted_bce",
		help="Classification loss type: 'bce' (standard), 'weighted_bce' (class weights), "
		     "'focal' (Focal Loss for imbalance). Default: weighted_bce")
	parser.add_argument("--focal_gamma", type=float, default=2.0,
		help="Gamma parameter for Focal Loss (higher = more focus on hard examples). Range: 1.0-3.0")
	parser.add_argument("--class_aware_augment", action="store_true",
		help="Enable class-aware augmentation (strong augmentation for minority class MEL)")
	parser.add_argument("--lr_scheduler", type=str, choices=["cosine", "step", "none"], default="cosine",
		help="Learning rate scheduler: 'cosine' (CosineAnnealingLR), 'step' (StepLR), or 'none'")
	parser.add_argument("--task", type=str, choices=["classification", "regression"], default="classification",
		help="Single-task mode: 'classification' or 'regression'")
	parser.add_argument("--multitask", action="store_true",
		help="Enable dual-head training (classification + regression)")
	parser.add_argument("--target_col", type=str, default="",
		help="Override target column name; defaults to 'label' or 'thickness' based on task")
	
	# Evaluation arguments
	parser.add_argument("--subgroup_cols", type=str, nargs="*", default=None,
		help="Column names for subgroup analysis (e.g., --subgroup_cols age_group skin_tone)")
	parser.add_argument("--no_bootstrap", action="store_true",
		help="Disable bootstrap confidence intervals (faster evaluation)")
	parser.add_argument("--no_ece", action="store_true",
		help="Disable Expected Calibration Error computation")
	
	# General arguments
	parser.add_argument("--output_dir", type=str, default="outputs",
		help="Directory for checkpoints and logs")
	parser.add_argument("--weights_path", type=str, default="",
		help="Path to checkpoint for evaluation (defaults to output_dir/checkpoints/best.pt)")
	parser.add_argument("--val_size", type=float, default=0.15,
		help="Validation set fraction")
	parser.add_argument("--test_size", type=float, default=0.15,
		help="Test set fraction")
	parser.add_argument("--device", type=str, default="auto",
		help="Device to use: 'cuda', 'cpu', or 'auto'")
	# parser.add_argument("--patience", type=int, default=5,
	# 	help="Early stopping patience (epochs without improvement)")                                              │                                                                   
	
	return parser.parse_args()


def main():
	args = parse_args()

	# Load and split data
	df = load_emb_data(args.metadata_path, args.image_dir)
	splits = split_dataset(df, val_size=args.val_size, test_size=args.test_size)

	# Determine target column
	target_col = args.target_col if args.target_col else ("label" if args.task == "classification" else "thickness")
	if target_col not in splits.train.columns:
		raise ValueError(f"Target column '{target_col}' not found in data columns: {list(splits.train.columns)}")
	# Build a unified 'target' column
	for part in [splits.train, splits.val, splits.test]:
		part.loc[:, "target"] = part[target_col]
		if args.task == "classification":
			part.loc[:, "target"] = part["target"].astype(float)

	# Determine structured feature columns (exclude non-numeric and helper cols)
	exclude = {"image_path", "image", "label", "target", "cathegory", "category"}
	feature_cols = [c for c in splits.train.columns if c not in exclude]
	# Keep only numeric columns
	feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(splits.train[c])]

	# Ensure feature columns are float dtypes to avoid bool/float assignment issues
	for part in [splits.train, splits.val, splits.test]:
		for col in feature_cols:
			part[col] = pd.to_numeric(part[col], errors="coerce").astype(float)

	# Fit scaler on training split (float arrays) and transform
	sp = StructuredPreprocessor(feature_names=feature_cols)
	train_struct = sp.fit_transform(splits.train[feature_cols].to_numpy())
	val_struct = sp.transform(splits.val[feature_cols].to_numpy())
	test_struct = sp.transform(splits.test[feature_cols].to_numpy())

	# Replace in dataframes for downstream dataset builders (ensure float dtype)
	for i, col in enumerate(feature_cols):
		splits.train[col] = train_struct[:, i].astype(float)
		splits.val[col] = val_struct[:, i].astype(float)
		splits.test[col] = test_struct[:, i].astype(float)

	# Build model based on selected architecture
	image_h, image_w = args.image_size
	
	print(f"\n{'='*60}")
	print(f"MODEL CONFIGURATION")
	print(f"{'='*60}")
	print(f"Model type: {args.model_type}")
	print(f"Task: {args.task}{' (multitask)' if args.multitask else ''}")
	print(f"Clinical features: {len(feature_cols)}")
	
	if args.model_type == "dino":
		# DINO hybrid model
		dino_ckpt = args.dino_checkpoint if args.dino_checkpoint else None
		if dino_ckpt:
			print(f"DINO checkpoint: {dino_ckpt}")
		else:
			print("DINO checkpoint: None (using ImageNet pretrained)")
		
		model = create_dino_hybrid_model(
			num_structured_features=len(feature_cols),
			task=args.task,
			multitask=args.multitask,
			arch=args.vit_arch,
			pretrained=True,
			dino_checkpoint=dino_ckpt,
			use_tokens=args.use_tokens,
			fusion_type=args.fusion_type,
			num_clin_tokens=args.num_clin_tokens,
			freeze_backbone_layers=args.freeze_backbone_layers,
		)
		print(f"Fusion: {args.fusion_type}")
		print(f"Frozen layers: {args.freeze_backbone_layers}")
	else:
		# ResNet50 hybrid model (baseline)
		model = create_hybrid_model(
			num_structured_features=len(feature_cols),
			task=args.task,
			multitask=args.multitask,
		)
	print(f"{'='*60}\n")
	
	# Training or evaluation
	if args.mode == "train":
		print(f"Starting training with loss_alpha={args.loss_alpha}")
		print(f"  Classification weight: {args.loss_alpha}")
		print(f"  Regression weight: {1.0 - args.loss_alpha}")
		print()
		
		_, ckpt_path = train_model(
			model,
			splits.train,
			splits.val,
			feature_cols=feature_cols,
			batch_size=args.batch_size,
			epochs=args.epochs,
			image_size=(image_h, image_w),
			output_dir=args.output_dir,
			device=(args.device if args.device != "auto" else None),
			task=args.task,
			multitask=args.multitask,
			loss_alpha=args.loss_alpha,
			cls_loss_type=args.cls_loss_type,
			class_aware_augment=args.class_aware_augment,
			focal_gamma=args.focal_gamma,
			lr_scheduler=args.lr_scheduler,
		)
		print(f"\n✓ Training complete! Best checkpoint: {ckpt_path}")
	else:
		# Evaluation mode
		weights = args.weights_path if args.weights_path else str(Path(args.output_dir, "checkpoints", "best.pt"))
		
		metrics = evaluate_model(
			model,
			splits.test,
			feature_cols=feature_cols,
			weights_path=weights,
			batch_size=args.batch_size,
			image_size=(image_h, image_w),
			device=(args.device if args.device != "auto" else None),
			multitask=args.multitask,
			subgroup_cols=args.subgroup_cols,
			compute_bootstrap=not args.no_bootstrap,
			compute_ece=not args.no_ece,
		)


if __name__ == "__main__":
	main()


