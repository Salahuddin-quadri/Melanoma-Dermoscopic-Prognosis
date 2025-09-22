from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import pandas as pd

# Allow running as a module (python -m src.main) or as a script (python src/main.py)
try:
	from .utils import load_emb_data, split_dataset, StructuredPreprocessor
	from .models import create_hybrid_model
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
	from models import create_hybrid_model
	from train import train_model
	from evaluate import evaluate_model


def parse_args():
	parser = argparse.ArgumentParser(description="EMB Melanoma Prognosis: Train/Evaluate Hybrid Model")
	parser.add_argument("--metadata_path", type=str, default="data/metadata_atlas.csv")
	parser.add_argument("--image_dir", type=str, default="data/dermoscopy_images")
	parser.add_argument("--mode", type=str, choices=["train", "evaluate"], default="train")
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--output_dir", type=str, default="outputs")
	parser.add_argument("--weights_path", type=str, default="")
	parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
	parser.add_argument("--val_size", type=float, default=0.15)
	parser.add_argument("--test_size", type=float, default=0.15)
	parser.add_argument("--device", type=str, default="cuda", help="cuda|cpu|auto")
	parser.add_argument("--task", type=str, choices=["classification", "regression"], default="classification", help="Target task: classification uses 'label'; regression uses 'thickness'")
	parser.add_argument("--multitask", action="store_true", help="Enable multi-head (classification + regression)")
	parser.add_argument("--target_col", type=str, default="", help="Override target column name; defaults to 'label' or 'thickness' based on task")
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
	exclude = {"image_path", "image_id", "label", "target", "cathegory", "category", "source"}
	feature_cols = [c for c in splits.train.columns if c not in exclude]
	# Keep only numeric columns
	feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(splits.train[c])]

	# Ensure feature columns are float dtypes to avoid bool/float assignment issues
	for part in [splits.train, splits.val, splits.test]:
		part.loc[:, feature_cols] = part[feature_cols].apply(pd.to_numeric, errors="coerce").astype(float)

	# Fit scaler on training split (float arrays) and transform
	sp = StructuredPreprocessor(feature_names=feature_cols)
	train_struct = sp.fit_transform(splits.train[feature_cols].to_numpy())
	val_struct = sp.transform(splits.val[feature_cols].to_numpy())
	test_struct = sp.transform(splits.test[feature_cols].to_numpy())

	# Replace in dataframes for downstream dataset builders
	splits.train.loc[:, feature_cols] = train_struct.astype(float)
	splits.val.loc[:, feature_cols] = val_struct.astype(float)
	splits.test.loc[:, feature_cols] = test_struct.astype(float)

	# Build model
	image_h, image_w = args.image_size
	model = create_hybrid_model(num_structured_features=len(feature_cols), task=args.task, multitask=args.multitask)

	if args.mode == "train":
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
		)
		print(f"Best checkpoint saved at: {ckpt_path}")
	else:
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
		)
		print(metrics)


if __name__ == "__main__":
	main()


