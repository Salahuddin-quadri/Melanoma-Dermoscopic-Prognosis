from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class EMBDataSplits:
	train: pd.DataFrame
	val: pd.DataFrame
	test: pd.DataFrame


def _resolve_image_path(row: pd.Series, image_dir: Path, image_col_candidates: List[str]) -> Optional[str]:
	for col in image_col_candidates:
		if col in row and pd.notna(row[col]):
			name = str(row[col])
			# If extension present, try directly
			path = image_dir / name
			if path.exists():
				return str(path)
			# Try appending common extensions
			stem = Path(name).stem
			for ext in IMAGE_EXTS:
				cand = image_dir / f"{stem}{ext}"
				if cand.exists():
					return str(cand)
	return None


def load_emb_data(metadata_path: str, image_dir: str) -> pd.DataFrame:
	"""Load EMB metadata and join with resolved image paths.

	Required columns (or equivalents):
	- image identifier column (e.g., image_id, image_name, filename)
	- label (0/1)
	- thickness (numeric)
	- stage_ajcc (numeric preferred)
	- type (categorical; e.g., "dermoscopic", "clinical")
	"""
	md_path = Path(metadata_path)
	img_dir = Path(image_dir)
	assert md_path.exists(), f"Metadata not found: {metadata_path}"
	assert img_dir.exists() and img_dir.is_dir(), f"Image directory not found or not a directory: {image_dir}"

	df = pd.read_csv(md_path)

	# Identify possible columns
	image_cols = [
		"image_id",
		"image_name",
		"filename",
		"image",
		"file",
	]
	label_cols = ["label", "target", "y"]
	thickness_cols = ["thickness", "breslow_thickness"]
	stage_cols = ["stage_ajcc", "ajcc_stage", "stage"]
	type_cols = ["type", "image_type"]

	def _find_col(cands: List[str]) -> Optional[str]:
		for c in cands:
			if c in df.columns:
				return c
		return None

	image_col = _find_col(image_cols)
	label_col = _find_col(label_cols)
	thickness_col = _find_col(thickness_cols)
	stage_col = _find_col(stage_cols)
	type_col = _find_col(type_cols)

	missing = [
		name for name, v in {
			"image_col": image_col,
			"label_col": label_col,
			"thickness_col": thickness_col,
			"stage_col": stage_col,
			"type_col": type_col,
		}.items() if v is None
	]
	if missing:
		raise ValueError(f"Missing required columns (or aliases) in metadata: {missing}")

	# Resolve image paths
	df["image_path"] = df.apply(
		lambda r: _resolve_image_path(r, img_dir, [image_col] + [c for c in image_cols if c != image_col]), axis=1
	)
	df = df[df["image_path"].notna()].copy()

	# Select and rename standardized columns
	# include optional category column if present for cleaner label mapping
	opt_cols = [c for c in ["cathegory", "category"] if c in df.columns]
	keep_cols = [image_col, label_col, thickness_col, stage_col, type_col, "image_path"] + opt_cols
	df = df[keep_cols].rename(
		columns={
			image_col: "image_id",
			label_col: "label",
			thickness_col: "thickness",
			stage_col: "stage_ajcc",
			type_col: "type",
		}
	)

	# One-hot for type
	type_dummies = pd.get_dummies(df["type"], prefix="type").astype(float)
	df = pd.concat([df.drop(columns=["type"]), type_dummies], axis=1)

	# Ensure numeric types
	df["thickness"] = pd.to_numeric(df["thickness"], errors="coerce")
	df["stage_ajcc"] = pd.to_numeric(df["stage_ajcc"], errors="coerce")
	df = df.dropna(subset=["thickness", "stage_ajcc"]).copy()
	# Label mapping: ensure binary {0,1}
	if "label" in df.columns:
		lab = pd.to_numeric(df["label"], errors="coerce")
		# If values are not in {0,1}, try to derive from cathegory/category
		if not set(lab.dropna().unique()).issubset({0, 1}):
			cat_col = "cathegory" if "cathegory" in df.columns else ("category" if "category" in df.columns else None)
			if cat_col:
				cat_series = df[cat_col]
				# If numeric-like, use >0 mapping; else string contains 'MEL'
				cat_num = pd.to_numeric(cat_series, errors="coerce")
				if cat_num.notna().any():
					lab = (cat_num.fillna(0) > 0).astype(int).astype(float)
				else:
					lab = cat_series.astype(str).str.upper().map(lambda x: 1 if "MEL" in x else 0).astype(float)
			else:
				lab = lab.fillna(0).astype(float)
				lab = (lab > 0).astype(int).astype(float)
		else:
			lab = lab.astype(float)
		df["label"] = lab

	return df


def split_dataset(
	df: pd.DataFrame,
	val_size: float = 0.15,
	test_size: float = 0.15,
	random_state: int = 42,
	stratify: bool = True,
) -> EMBDataSplits:
	"""Split the dataset into train/val/test DataFrames with optional stratification by label.

	Note: If 'label' is missing (e.g., regression task), stratification is disabled.
	"""
	assert 0 < val_size < 0.5 and 0 < test_size < 0.5 and val_size + test_size < 1
	use_strat = stratify and ("label" in df.columns) and df["label"].notna().all()
	strat = df["label"] if use_strat else None
	train_df, temp_df = train_test_split(
		df, test_size=val_size + test_size, random_state=random_state, stratify=strat
	)
	# Compute val proportion of temp
	val_prop = val_size / (val_size + test_size)
	strat_temp = temp_df["label"] if use_strat else None
	val_df, test_df = train_test_split(
		temp_df, test_size=1 - val_prop, random_state=random_state, stratify=strat_temp
	)
	return EMBDataSplits(train=train_df.reset_index(drop=True), val=val_df.reset_index(drop=True), test=test_df.reset_index(drop=True))


