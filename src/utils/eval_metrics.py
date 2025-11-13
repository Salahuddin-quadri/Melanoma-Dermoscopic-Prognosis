"""
Enhanced Evaluation Metrics for Model Assessment

This module provides comprehensive evaluation metrics including:
- Expected Calibration Error (ECE) for probability calibration assessment
- Bootstrap confidence intervals for robustness quantification
- Subgroup analysis for fairness evaluation
- Extended classification and regression metrics

These metrics are essential for rigorous model evaluation in clinical settings.
"""

from __future__ import annotations

from typing import Optional, Dict, List, Tuple, Any
import numpy as np
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error
import pandas as pd


def compute_ece(
	predictions: np.ndarray,
	labels: np.ndarray,
	n_bins: int = 15,
) -> float:
	"""
	Compute Expected Calibration Error (ECE).
	
	ECE measures how well-calibrated the predicted probabilities are.
	A perfectly calibrated model would have ECE=0, meaning predicted
	probabilities match the true frequencies.
	
	Algorithm:
	1. Partition predictions into n_bins by confidence
	2. For each bin, compute average confidence vs accuracy
	3. Weighted average of |confidence - accuracy| across bins
	
	Args:
		predictions: Predicted probabilities, shape (N,)
		labels: True binary labels, shape (N,)
		n_bins: Number of bins for calibration assessment
	
	Returns:
		Expected Calibration Error (float, 0-1 range)
	"""
	predictions = np.clip(predictions, 0.0, 1.0)
	bin_boundaries = np.linspace(0, 1, n_bins + 1)
	bin_lowers = bin_boundaries[:-1]
	bin_uppers = bin_boundaries[1:]
	
	ece = 0.0
	for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
		# Find samples in this bin
		in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
		prop_in_bin = in_bin.mean()
		
		if prop_in_bin > 0:
			# Average confidence in this bin
			accuracy_in_bin = labels[in_bin].mean()
			avg_confidence_in_bin = predictions[in_bin].mean()
			
			# Add to ECE
			ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
	
	return float(ece)


def bootstrap_confidence_interval(
	metric_fn,
	y_true: np.ndarray,
	y_pred: np.ndarray,
	n_bootstrap: int = 1000,
	confidence_level: float = 0.95,
	random_seed: Optional[int] = None,
) -> Tuple[float, float, float]:
	"""
	Compute bootstrap confidence interval for a metric.
	
	Bootstrap sampling provides robust uncertainty quantification
	especially important when test set size is limited.
	
	Args:
		metric_fn: Function that computes metric(y_true, y_pred) -> float
		y_true: True labels/values, shape (N,)
		y_pred: Predicted labels/values, shape (N,)
		n_bootstrap: Number of bootstrap samples
		confidence_level: Confidence level (e.g., 0.95 for 95% CI)
		random_seed: Random seed for reproducibility
	
	Returns:
		Tuple of (metric_value, lower_ci, upper_ci)
	"""
	rng = np.random.default_rng(random_seed)
	n_samples = len(y_true)
	
	# Compute metric on full data
	metric_value = metric_fn(y_true, y_pred)
	
	# Bootstrap sampling
	bootstrap_metrics = []
	for _ in range(n_bootstrap):
		# Sample with replacement
		indices = rng.integers(0, n_samples, n_samples)
		bootstrap_true = y_true[indices]
		bootstrap_pred = y_pred[indices]
		
		# Compute metric on bootstrap sample
		try:
			bootstrap_metric = metric_fn(bootstrap_true, bootstrap_pred)
			bootstrap_metrics.append(bootstrap_metric)
		except Exception:
			# Skip if metric computation fails (e.g., only one class)
			continue
	
	if not bootstrap_metrics:
		return float(metric_value), float(metric_value), float(metric_value)
	
	bootstrap_metrics = np.array(bootstrap_metrics)
	
	# Compute confidence interval
	alpha = 1 - confidence_level
	lower_percentile = (alpha / 2) * 100
	upper_percentile = (1 - alpha / 2) * 100
	
	lower_ci = np.percentile(bootstrap_metrics, lower_percentile)
	upper_ci = np.percentile(bootstrap_metrics, upper_percentile)
	
	return float(metric_value), float(lower_ci), float(upper_ci)


def compute_classification_metrics(
	y_true: np.ndarray,
	y_pred_probs: np.ndarray,
	threshold: float = 0.5,
	compute_bootstrap: bool = True,
	compute_ece: bool = True,
) -> Dict[str, Any]:
	"""
	Compute comprehensive classification metrics.
	
	Includes: accuracy, precision, recall, F1, AUROC, ECE, and bootstrap CIs.
	
	Args:
		y_true: True binary labels, shape (N,)
		y_pred_probs: Predicted probabilities, shape (N,)
		threshold: Decision threshold for binary predictions
		compute_bootstrap: Whether to compute bootstrap CIs
		compute_ece: Whether to compute Expected Calibration Error
	
	Returns:
		Dictionary of metrics
	"""
	y_pred_binary = (y_pred_probs >= threshold).astype(int)
	y_true_int = y_true.astype(int)
	
	# Basic metrics
	accuracy = float(np.mean(y_pred_binary == y_true_int))
	precision = float(np.sum((y_pred_binary == 1) & (y_true_int == 1)) / max(np.sum(y_pred_binary == 1), 1))
	recall = float(np.sum((y_pred_binary == 1) & (y_true_int == 1)) / max(np.sum(y_true_int == 1), 1))
	f1 = float(2 * precision * recall / max(precision + recall, 1e-10))
	
	# AUROC
	try:
		auroc = float(roc_auc_score(y_true, y_pred_probs))
		auroc_available = True
	except Exception:
		auroc = accuracy  # Fallback if only one class
		auroc_available = False
	
	metrics = {
		"accuracy": accuracy,
		"precision": precision,
		"recall": recall,
		"f1": f1,
		"auroc": auroc,
		"auroc_computed": auroc_available,
	}
	
	# Bootstrap CIs
	if compute_bootstrap and auroc_available:
		try:
			_, auroc_lower, auroc_upper = bootstrap_confidence_interval(
				roc_auc_score, y_true, y_pred_probs
			)
			metrics.update({
				"auroc_ci_lower": auroc_lower,
				"auroc_ci_upper": auroc_upper,
			})
		except Exception:
			pass
	
	# ECE
	if compute_ece:
		ece = compute_ece(y_pred_probs, y_true_int)
		metrics["ece"] = ece
	
	return metrics


def compute_regression_metrics(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	compute_bootstrap: bool = True,
) -> Dict[str, Any]:
	"""
	Compute comprehensive regression metrics.
	
	Includes: MAE, RMSE, and bootstrap CIs.
	
	Args:
		y_true: True regression values, shape (N,)
		y_pred: Predicted regression values, shape (N,)
		compute_bootstrap: Whether to compute bootstrap CIs
	
	Returns:
		Dictionary of metrics
	"""
	mae = float(mean_absolute_error(y_true, y_pred))
	rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
	
	metrics = {
		"mae": mae,
		"rmse": rmse,
	}
	
	# Bootstrap CIs
	if compute_bootstrap:
		try:
			_, mae_lower, mae_upper = bootstrap_confidence_interval(
				mean_absolute_error, y_true, y_pred
			)
			metrics.update({
				"mae_ci_lower": mae_lower,
				"mae_ci_upper": mae_upper,
			})
		except Exception:
			pass
	
	return metrics


def compute_subgroup_metrics(
	df: pd.DataFrame,
	predictions: np.ndarray,
	labels: np.ndarray,
	subgroup_cols: List[str],
	task: str = "classification",
	compute_bootstrap: bool = False,
) -> Dict[str, Dict[str, Any]]:
	"""
	Compute metrics stratified by subgroup variables.
	
	Subgroup analysis is crucial for identifying performance disparities
	across patient demographics or clinical characteristics.
	
	Args:
		df: DataFrame containing subgroup columns
		predictions: Model predictions (probabilities or values)
		labels: True labels/values
		subgroup_cols: List of column names to stratify by
		task: "classification" or "regression"
		compute_bootstrap: Whether to compute bootstrap CIs (can be slow)
	
	Returns:
		Nested dictionary: {subgroup_col: {value: {metric: value}}}
	"""
	subgroup_results = {}
	
	for col in subgroup_cols:
		if col not in df.columns:
			continue
		
		subgroup_results[col] = {}
		unique_values = df[col].unique()
		
		for value in unique_values:
			mask = df[col] == value
			if mask.sum() < 5:  # Skip subgroups with too few samples
				continue
			
			subgroup_preds = predictions[mask]
			subgroup_labels = labels[mask]
			
			if task == "classification":
				metrics = compute_classification_metrics(
					subgroup_labels,
					subgroup_preds,
					compute_bootstrap=compute_bootstrap,
					compute_ece=False,  # Skip ECE for subgroups (too few samples per bin)
				)
			else:  # regression
				metrics = compute_regression_metrics(
					subgroup_labels,
					subgroup_preds,
					compute_bootstrap=compute_bootstrap,
				)
			
			metrics["n_samples"] = int(mask.sum())
			subgroup_results[col][str(value)] = metrics
	
	return subgroup_results

