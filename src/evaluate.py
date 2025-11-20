from __future__ import annotations

from typing import Iterable, Tuple, Optional, List
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
	from .utils import preprocess_image
	from .utils.eval_metrics import (
		compute_classification_metrics,
		compute_regression_metrics,
		compute_subgroup_metrics,
	)
except ImportError:
	from utils import preprocess_image
	from utils.eval_metrics import (
		compute_classification_metrics,
		compute_regression_metrics,
		compute_subgroup_metrics,
	)


class EvalDataset(Dataset):
	def __init__(self, df, feature_cols: Iterable[str], image_size: Tuple[int, int] = (224, 224)):
		self.image_paths = df["image_path"].tolist()
		self.structured = df[list(feature_cols)].to_numpy(dtype=np.float32)
		self.has_label = "label" in df.columns
		self.has_thickness = "thickness" in df.columns
		self.labels = df["label"].to_numpy(dtype=np.float32) if self.has_label else None
		self.thickness = df["thickness"].to_numpy(dtype=np.float32) if self.has_thickness else None
		self.image_size = image_size

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		img = preprocess_image(self.image_paths[idx], size=self.image_size)
		img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
		tab = torch.from_numpy(self.structured[idx])
		out = {"img": img, "tab": tab}
		if self.has_label:
			out["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)
		if self.has_thickness:
			out["thickness"] = torch.tensor(self.thickness[idx], dtype=torch.float32)
		return out


def evaluate_model(
	model: torch.nn.Module,
	test_df,
	feature_cols: Iterable[str],
	weights_path: str | None = None,
	batch_size: int = 32,
	image_size: Tuple[int, int] = (224, 224),
	device: str | None = None,
	multitask: bool = False,
	subgroup_cols: Optional[List[str]] = None,
	compute_bootstrap: bool = True,
	compute_ece: bool = True,
):
	"""
	Comprehensive model evaluation with enhanced metrics.
	
	Computes:
	- Classification metrics: accuracy, precision, recall, F1, AUROC, ECE
	- Regression metrics: MAE, RMSE
	- Bootstrap confidence intervals for robustness
	- Subgroup analysis for fairness assessment (if subgroup columns provided)
	
	Args:
		model: Model to evaluate
		test_df: Test dataframe (must contain 'label' and/or 'thickness')
		feature_cols: List of clinical feature column names
		weights_path: Path to model checkpoint
		batch_size: Batch size for evaluation
		image_size: Image dimensions
		device: Device to use
		multitask: Whether model outputs dual heads
		subgroup_cols: Optional list of column names for subgroup analysis
			(e.g., ['age_group', 'skin_tone'])
		compute_bootstrap: Whether to compute bootstrap CIs (slower but more robust)
		compute_ece: Whether to compute Expected Calibration Error
	
	Returns:
		Dictionary of evaluation metrics
	"""
	device = device or ("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)
	
	# Load checkpoint if provided
	if weights_path:
		state = torch.load(weights_path, map_location=device)
		model.load_state_dict(
			state["model_state"] if isinstance(state, dict) and "model_state" in state else state
		)
		print(f"✓ Loaded weights from {weights_path}")

	# Prepare dataset and dataloader
	ds = EvalDataset(test_df, feature_cols, image_size)
	loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

	# Collect predictions
	model.eval()
	probs_all, labels_all = [], []
	reg_all, reg_targets = [], []
	
	with torch.no_grad():
		for batch in loader:
			imgs = batch["img"].to(device)
			tabs = batch["tab"].to(device)
			outputs = model(imgs, tabs)
			
			if isinstance(outputs, dict):
				# Multitask model
				if "label" in batch:
					probs_all.append(outputs["cls"].cpu().numpy())
					labels_all.append(batch["label"].numpy())
				if "thickness" in batch:
					reg_all.append(outputs["reg"].cpu().numpy())
					reg_targets.append(batch["thickness"].numpy())
			else:
				# Single-head model (assume classification)
				probs_all.append(outputs.cpu().numpy())
				if "label" in batch:
					labels_all.append(batch["label"].numpy())

	# Compute metrics
	metrics = {}
	
	# Classification metrics
	if labels_all:
		probs = np.concatenate(probs_all)
		labels = np.concatenate(labels_all)
		
		cls_metrics = compute_classification_metrics(
			labels,
			probs,
			compute_bootstrap=compute_bootstrap,
			compute_ece=compute_ece,
		)
		metrics["classification"] = cls_metrics
		
		# Subgroup analysis for classification
		if subgroup_cols:
			try:
				subgroup_metrics = compute_subgroup_metrics(
					test_df,
					probs,
					labels,
					subgroup_cols,
					task="classification",
					compute_bootstrap=compute_bootstrap,
				)
				metrics["subgroups"] = subgroup_metrics
			except Exception as e:
				print(f"Warning: Subgroup analysis failed: {e}")
	
	# Regression metrics
	if reg_all:
		pred_reg = np.concatenate(reg_all)
		y_reg = np.concatenate(reg_targets)
		
		reg_metrics = compute_regression_metrics(
			y_reg,
			pred_reg,
			compute_bootstrap=compute_bootstrap,
		)
		metrics["regression"] = reg_metrics
		
		# Subgroup analysis for regression
		if subgroup_cols:
			try:
				subgroup_metrics = compute_subgroup_metrics(
					test_df,
					pred_reg,
					y_reg,
					subgroup_cols,
					task="regression",
					compute_bootstrap=compute_bootstrap,
				)
				if "subgroups" not in metrics:
					metrics["subgroups"] = {}
				# Merge regression subgroup results
				for col, values in subgroup_metrics.items():
					if col not in metrics["subgroups"]:
						metrics["subgroups"][col] = {}
					metrics["subgroups"][col].update(values)
			except Exception as e:
				print(f"Warning: Regression subgroup analysis failed: {e}")
	
	# Print summary
	print("\n" + "="*60)
	print("EVALUATION METRICS SUMMARY")
	print("="*60)
	if "classification" in metrics:
		cls = metrics["classification"]
		print(f"\nClassification:")
		print(f"  Accuracy:  {cls['accuracy']:.4f}")
		print(f"  Precision: {cls['precision']:.4f}")
		print(f"  Recall:    {cls['recall']:.4f}")
		print(f"  F1 Score:  {cls['f1']:.4f}")
		print(f"  AUROC:     {cls['auroc']:.4f}")
		if "auroc_ci_lower" in cls:
			print(f"  AUROC CI:  [{cls['auroc_ci_lower']:.4f}, {cls['auroc_ci_upper']:.4f}]")
		if "ece" in cls:
			print(f"  ECE:       {cls['ece']:.4f}")
	
	if "regression" in metrics:
		reg = metrics["regression"]
		print(f"\nRegression:")
		print(f"  MAE:       {reg['mae']:.4f}")
		print(f"  RMSE:      {reg['rmse']:.4f}")
		print(f"  R²:        {reg['r2']:.4f}")
		if "mae_ci_lower" in reg:
			print(f"  MAE CI:    [{reg['mae_ci_lower']:.4f}, {reg['mae_ci_upper']:.4f}]")
		if "r2_ci_lower" in reg:
			print(f"  R² CI:     [{reg['r2_ci_lower']:.4f}, {reg['r2_ci_upper']:.4f}]")
	
	if "subgroups" in metrics:
		print(f"\nSubgroup Analysis:")
		for col, values in metrics["subgroups"].items():
			print(f"  {col}:")
			for value, sub_metrics in values.items():
				if "classification" in sub_metrics:
					print(f"    {value}: AUROC={sub_metrics.get('auroc', 'N/A'):.4f}, "
						  f"n={sub_metrics.get('n_samples', 'N/A')}")
				elif "regression" in sub_metrics or "mae" in sub_metrics:
					print(f"    {value}: MAE={sub_metrics.get('mae', 'N/A'):.4f}, "
						  f"R²={sub_metrics.get('r2', 'N/A'):.4f}, "
						  f"n={sub_metrics.get('n_samples', 'N/A')}")
	print("="*60 + "\n")
	
	return metrics


