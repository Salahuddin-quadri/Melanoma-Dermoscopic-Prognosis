"""
data_loader.py

Data loading utilities for melanoma imaging datasets.

This module provides functions to load, preprocess, and split melanoma imaging
datasets with clinical annotations including Breslow thickness and AJCC staging.

@author: Syed Salahuddin Quadri
@date: November 2025
"""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class EMBDataSplits:
    """Container for train/validation/test dataset splits."""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _resolve_image_path(
    row: pd.Series, 
    image_dir: Path, 
    image_col_candidates: List[str]
) -> Optional[str]:
    """
    Locate the actual image file path from metadata row.
    
    Searches candidate columns for filenames and attempts to match them
    against files in the image directory, with or without extensions.
    
    Args:
        row: Metadata row containing potential filename columns
        image_dir: Root directory containing image files
        image_col_candidates: Ordered list of column names to check
        
    Returns:
        Absolute path to image file if found, None otherwise
    """
    for col in image_col_candidates:
        if col in row and pd.notna(row[col]):
            name = str(row[col])
            
            # Direct path match
            path = image_dir / name
            if path.exists():
                return str(path)
            
            # Try appending standard image extensions
            stem = Path(name).stem
            for ext in IMAGE_EXTS:
                cand = image_dir / f"{stem}{ext}"
                if cand.exists():
                    return str(cand)
    
    return None


def load_emb_data(metadata_path: str, image_dir: str) -> pd.DataFrame:
    """
    Load and prepare melanoma imaging dataset with clinical annotations.
    
    This function reads metadata CSV files and matches them with corresponding
    image files. It handles various column naming conventions and performs
    necessary preprocessing including label encoding and feature engineering.
    
    Expected metadata columns (flexible naming):
        - Image identifier: image_id, image_name, filename, image, or file
        - Category encoding: category_encoded or `label` (binary: 0=benign, 1=melanoma)
    Clinical fields such as thickness or AJCC stage are optional and will be ignored
    when preparing an image-only dataset for training.
    
    Args:
        metadata_path: Path to CSV file containing clinical annotations
        image_dir: Root directory containing image files
        
    Returns:
        DataFrame with standardized columns including image_path and label
        
    Raises:
        ValueError: If required columns are missing or label distribution is invalid
        AssertionError: If paths do not exist
    """
    md_path = Path(metadata_path)
    img_dir = Path(image_dir)
    assert md_path.exists(), f"Metadata not found: {metadata_path}"
    assert img_dir.exists() and img_dir.is_dir(), f"Invalid image directory: {image_dir}"

    df = pd.read_csv(md_path)
    
    # Define flexible column name mappings
    image_cols = ["image_id", "image_name", "filename", "image", "file"]
    thickness_cols = ["thickness", "breslow_thickness"]
    stage_cols = ["stage_ajcc", "ajcc_stage", "stage"]
    type_cols = ["type", "image_type"]
    label_cols = ["category_encoded", "label"]

    def _find_col(candidates: List[str]) -> Optional[str]:
        """Return first matching column name from candidates."""
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # Identify actual column names in dataset
    image_col = _find_col(image_cols)
    thickness_col = _find_col(thickness_cols)
    stage_col = _find_col(stage_cols)
    type_col = _find_col(type_cols)
    label_col = _find_col(label_cols)

    # Validate presence of required columns: at minimum we need an image id and a label
    missing = [
        name for name, v in {"image_col": image_col, "label_col": label_col}.items() if v is None
    ]
    if missing:
        raise ValueError(f"Missing required columns in metadata: {missing}")

    # Match metadata entries to actual image files
    df["image_path"] = df.apply(
        lambda r: _resolve_image_path(
            r, img_dir, [image_col] + [c for c in image_cols if c != image_col]
        ), 
        axis=1
    )
    
    # Remove entries without corresponding image files
    initial_count = len(df)
    df = df[df["image_path"].notna()].copy()
    removed = initial_count - len(df)
    if removed > 0:
        print(f"Note: {removed} entries removed due to missing image files")

    # Standardize column names
    source_cols = ["source"] if "source" in df.columns else []
    category_cols = ["category_encoded"] if "category_encoded" in df.columns else []
    
    # Build list of columns to keep depending on what's available. Prioritise image id and label.
    keep_cols = [c for c in [image_col, label_col, "image_path"] if c is not None]
    if type_col is not None:
        keep_cols.append(type_col)
    if thickness_col is not None:
        keep_cols.append(thickness_col)
    if stage_col is not None:
        keep_cols.append(stage_col)
    if "source" in df.columns:
        keep_cols.append("source")

    # Select and rename present columns to standardized names
    rename_map = {image_col: "image_id"}
    if label_col is not None:
        rename_map[label_col] = "category_encoded" if label_col == "category_encoded" else "label"
    if type_col is not None:
        rename_map[type_col] = "type"
    if thickness_col is not None:
        rename_map[thickness_col] = "thickness"
    if stage_col is not None:
        rename_map[stage_col] = "stage_ajcc"

    df = df[keep_cols].rename(columns=rename_map)

    # One-hot encode image type (dermoscopic vs clinical) if present
    if "type" in df.columns:
        type_dummies = pd.get_dummies(df["type"], prefix="type").astype(float)
        df = pd.concat([df.drop(columns=["type"]), type_dummies], axis=1)

    # If thickness exists, keep it numeric, but do not require it for image-only workflows
    if "thickness" in df.columns:
        df["thickness"] = pd.to_numeric(df["thickness"], errors="coerce")

    # Process binary labels from category encoding or label column
    if "category_encoded" in df.columns:
        df["category_encoded"] = pd.to_numeric(df["category_encoded"], errors="coerce")
        unique_encoded = sorted(df["category_encoded"].dropna().unique())
        if len(unique_encoded) != 2:
            raise ValueError(
                f"Expected binary classification (2 classes), found {len(unique_encoded)} classes"
            )
        if set(unique_encoded) == {0.0, 1.0}:
            df["label"] = df["category_encoded"].astype(float)
        else:
            label_mapping = {unique_encoded[0]: 0.0, unique_encoded[1]: 1.0}
            df["label"] = df["category_encoded"].map(label_mapping).astype(float)
    elif "label" in df.columns:
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        unique_labels = sorted(df["label"].dropna().unique())
        if len(unique_labels) == 2 and set(unique_labels) != {0.0, 1.0}:
            label_mapping = {unique_labels[0]: 0.0, unique_labels[1]: 1.0}
            df["label"] = df["label"].map(label_mapping).astype(float)
    else:
        raise ValueError("No valid label column found in metadata")

    # Final validation
    final_labels = sorted(df["label"].dropna().unique())
    if set(final_labels) != {0.0, 1.0}:
        raise ValueError(f"Invalid label values: {final_labels}. Expected [0.0, 1.0]")

    # By default, perform a small downsampling of the majority class to reduce extreme
    # imbalance when present; preserve index ordering.
    pos = df[df.label == 1.0]
    neg = df[df.label == 0.0]
    if len(neg) > 0 and len(pos) > 0 and len(neg) > 2 * len(pos):
        neg = neg.sample(frac=0.5, random_state=42)

    df = pd.concat([pos, neg]).reset_index(drop=True)
    # TEMPORARY: remove class 0 entirely (keep only melanoma)
    df = df[df.label == 1.0].reset_index(drop=True)

    return df


def _create_thickness_strata(df: pd.DataFrame, n_bins: int = 5) -> pd.Series:
    """
    Create thickness-based stratification groups.
    
    Bins thickness values into quantile-based groups for balanced splitting.
    Handles missing values by assigning them to a separate group.
    
    Args:
        df: DataFrame with 'thickness' column
        n_bins: Number of bins to create
        
    Returns:
        Series with stratum labels for each row
    """
    if "thickness" not in df.columns:
        return None
    
    thickness = pd.to_numeric(df["thickness"], errors="coerce")
    
    # Handle missing values
    has_thickness = thickness.notna()
    
    if has_thickness.sum() == 0:
        return None
    
    # Create bins for non-missing values using quantiles
    valid_thickness = thickness[has_thickness]
    
    # Use quantile-based binning for balanced distribution
    try:
        bins = pd.qcut(valid_thickness, q=n_bins, duplicates='drop', labels=False)
    except ValueError:
        # If quantile binning fails (e.g., too many duplicates), use equal-width bins
        bins = pd.cut(valid_thickness, bins=n_bins, labels=False, duplicates='drop')
    
    # Create stratum labels
    strata = pd.Series(index=df.index, dtype=object)
    strata[has_thickness] = bins.astype(str)
    strata[~has_thickness] = "missing"
    
    return strata


def split_dataset(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True,
    stratify_by_thickness: bool = True,
) -> EMBDataSplits:
    """
    Split dataset into train/validation/test subsets with optional stratification.
    
    Performs stratified splitting to maintain class distribution across splits.
    When thickness is available, can also stratify by thickness bins to ensure
    balanced distribution of thickness values across splits.
    
    Args:
        df: Input dataset with 'label' column for stratification
        val_size: Proportion for validation set (0-0.5)
        test_size: Proportion for test set (0-0.5)
        random_state: Random seed for reproducibility
        stratify: Whether to maintain class distribution in splits
        stratify_by_thickness: Whether to also stratify by thickness bins (if available)
        
    Returns:
        EMBDataSplits object containing train, validation, and test DataFrames
        
    Raises:
        AssertionError: If split proportions are invalid
    """
    assert 0 < val_size < 0.5 and 0 < test_size < 0.5 and val_size + test_size < 1, \
        "Invalid split proportions"
    
    # Prepare stratification
    strat = None
    strat_label = None
    
    if stratify and "label" in df.columns:
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        
        # Remove samples with missing labels
        valid_mask = df["label"].notna()
        if not valid_mask.all():
            n_removed = (~valid_mask).sum()
            print(f"Warning: Removing {n_removed} samples with missing labels")
            df = df[valid_mask].copy()
        
        # Check class distribution
        unique_labels = sorted(df["label"].unique())
        label_counts = df["label"].value_counts()
        
        if len(unique_labels) >= 2:
            strat_label = df["label"].astype(int)
            print(f"\nStratified splitting enabled:")
            print(f"  Total samples: {len(df):,}")
            print(f"  Class distribution: {dict(label_counts)}")
        else:
            print(f"Warning: Only one class present. Stratification disabled.")
    
    # Create combined stratification if thickness is available
    if stratify_by_thickness and "thickness" in df.columns:
        thickness_strata = _create_thickness_strata(df, n_bins=5)
        
        if thickness_strata is not None:
            # Combine label and thickness stratification
            if strat_label is not None:
                # Create combined strata: label_thickness_bin
                combined_strata = (
                    strat_label.astype(str) + "_" + thickness_strata.astype(str)
                )
                
                # Check if we have enough samples per stratum
                stratum_counts = combined_strata.value_counts()
                min_samples = stratum_counts.min()
                
                if min_samples < 2:
                    print(f"Warning: Some strata have < 2 samples. Using label-only stratification.")
                    strat = strat_label
                else:
                    strat = combined_strata
                    print(f"  Thickness stratification enabled: {len(stratum_counts)} combined strata")
                    print(f"  Thickness distribution across strata:")
                    for s, count in stratum_counts.head(10).items():
                        print(f"    {s}: {count} samples")
                    if len(stratum_counts) > 10:
                        print(f"    ... and {len(stratum_counts) - 10} more strata")
            else:
                # Only thickness stratification
                strat = thickness_strata
                print(f"  Thickness-only stratification enabled")
        else:
            strat = strat_label
    else:
        strat = strat_label
    
    # Two-step splitting: (train) vs (val+test), then (val) vs (test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=val_size + test_size, 
        random_state=random_state, 
        stratify=strat
    )
    
    val_proportion = val_size / (val_size + test_size)
    
    # Create stratification for temp split
    if strat is not None:
        if isinstance(strat, pd.Series):
            strat_temp = strat.loc[temp_df.index]
        else:
            strat_temp = temp_df["label"].astype(int) if "label" in temp_df.columns else None
    else:
        strat_temp = None
    
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=1 - val_proportion, 
        random_state=random_state, 
        stratify=strat_temp
    )
    
    # Report split statistics
    if "label" in df.columns:
        print(f"\nFinal split statistics:")
        for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            counts = split_df["label"].value_counts()
            print(f"  {name}: {len(split_df):,} samples {dict(counts)}")
            
            # Report thickness statistics if available
            if "thickness" in split_df.columns:
                thickness_valid = pd.to_numeric(split_df["thickness"], errors="coerce").dropna()
                if len(thickness_valid) > 0:
                    print(f"    Thickness: mean={thickness_valid.mean():.3f}mm, "
                          f"median={thickness_valid.median():.3f}mm, "
                          f"range=[{thickness_valid.min():.3f}, {thickness_valid.max():.3f}]mm")
    
    return EMBDataSplits(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True)
    )


def load_unlabeled_images_for_dino(image_root: str) -> pd.DataFrame:
    """
    Collect unlabeled images for self-supervised pretraining.
    
    Recursively scans directory for image files and infers type
    (dermoscopic vs clinical) from parent directory names.
    Suitable for DINO or other self-supervised learning approaches.
    
    Args:
        image_root: Root directory to scan for images
        
    Returns:
        DataFrame with columns: image_id, image_path, type
        
    Raises:
        ValueError: If no images found
        AssertionError: If directory path invalid
    """
    root = Path(image_root)
    assert root.exists() and root.is_dir(), f"Invalid image directory: {image_root}"

    image_records: List[Dict[str, str]] = []
    
    for ext in IMAGE_EXTS:
        for p in root.rglob(f"*{ext}"):
            parent_name = p.parent.name.lower()
            
            # Infer image type from directory structure
            img_type: Optional[str] = None
            if "dermoscopic" in parent_name:
                img_type = "dermoscopic"
            elif "clinical" in parent_name:
                img_type = "clinical"
            
            image_records.append({
                "image_id": p.stem,
                "image_path": str(p),
                "type": img_type if img_type else "",
            })

    if not image_records:
        raise ValueError(f"No images found in {image_root}")

    print(f"Collected {len(image_records):,} images for self-supervised learning")
    return pd.DataFrame(image_records)
