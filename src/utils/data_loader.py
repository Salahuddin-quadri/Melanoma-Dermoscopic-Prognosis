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
        - Breslow thickness: thickness or breslow_thickness (mm)
        - AJCC stage: stage_ajcc, ajcc_stage, or stage
        - Image type: type or image_type (dermoscopic/clinical)
        - Category encoding: category_encoded (binary: 0=benign, 1=melanoma)
    
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

    # Validate presence of required columns
    missing = [
        name for name, v in {
            "image_col": image_col,
            "thickness_col": thickness_col,
            "stage_col": stage_col,
            "type_col": type_col,
        }.items() if v is None
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
    
    keep_cols = [
        image_col, thickness_col, stage_col, type_col, "image_path"
    ] + category_cols + source_cols
    
    df = df[keep_cols].rename(columns={
        image_col: "image_id",
        thickness_col: "thickness",
        stage_col: "stage_ajcc",
        type_col: "type",
    })

    # One-hot encode image type (dermoscopic vs clinical)
    type_dummies = pd.get_dummies(df["type"], prefix="type").astype(float)
    df = pd.concat([df.drop(columns=["type"]), type_dummies], axis=1)

    # Ensure proper numeric types
    df["thickness"] = pd.to_numeric(df["thickness"], errors="coerce")
    
    # Remove samples with invalid thickness measurements
    pre_filter = len(df)
    df = df.dropna(subset=["thickness"]).copy()
    if len(df) < pre_filter:
        print(f"Filtered {pre_filter - len(df)} samples with invalid thickness values")

    # Process binary labels from category encoding
    if "category_encoded" in df.columns:
        df["category_encoded"] = pd.to_numeric(df["category_encoded"], errors="coerce")
        
        unique_encoded = sorted(df["category_encoded"].dropna().unique())
        
        if len(unique_encoded) != 2:
            raise ValueError(
                f"Expected binary classification (2 classes), found {len(unique_encoded)} classes"
            )
        
        # Normalize to standard binary encoding (0.0, 1.0)
        if set(unique_encoded) == {0.0, 1.0}:
            df["label"] = df["category_encoded"].astype(float)
        else:
            # Map arbitrary values to 0.0 and 1.0
            label_mapping = {unique_encoded[0]: 0.0, unique_encoded[1]: 1.0}
            df["label"] = df["category_encoded"].map(label_mapping).astype(float)
        
        # Verify class balance
        label_counts = df["label"].value_counts()
        print(f"\nDataset composition:")
        print(f"  Class 0 (benign): {label_counts.get(0.0, 0):,} samples")
        print(f"  Class 1 (malignant): {label_counts.get(1.0, 0):,} samples")
        
        # Ensure both classes are represented
        if label_counts.get(0.0, 0) == 0 or label_counts.get(1.0, 0) == 0:
            raise ValueError("Dataset must contain both benign and malignant samples")
            
    elif "label" in df.columns:
        # Fallback for pre-encoded labels
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

    df = pd.concat([df[df.label == 1.0], df[df.label == 0.0].sample(frac=0.5, random_state=42)]) #to keep 50% of ben samples


    return df


def split_dataset(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True,
) -> EMBDataSplits:
    """
    Split dataset into train/validation/test subsets with optional stratification.
    
    Performs stratified splitting to maintain class distribution across splits.
    This is particularly important for imbalanced medical imaging datasets.
    
    Args:
        df: Input dataset with 'label' column for stratification
        val_size: Proportion for validation set (0-0.5)
        test_size: Proportion for test set (0-0.5)
        random_state: Random seed for reproducibility
        stratify: Whether to maintain class distribution in splits
        
    Returns:
        EMBDataSplits object containing train, validation, and test DataFrames
        
    Raises:
        AssertionError: If split proportions are invalid
    """
    assert 0 < val_size < 0.5 and 0 < test_size < 0.5 and val_size + test_size < 1, \
        "Invalid split proportions"
    
    strat = None
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
            strat = df["label"].astype(int)
            print(f"\nStratified splitting enabled:")
            print(f"  Total samples: {len(df):,}")
            print(f"  Class distribution: {dict(label_counts)}")
        else:
            print(f"Warning: Only one class present. Stratification disabled.")
    
    # Two-step splitting: (train) vs (val+test), then (val) vs (test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=val_size + test_size, 
        random_state=random_state, 
        stratify=strat
    )
    
    val_proportion = val_size / (val_size + test_size)
    strat_temp = temp_df["label"].astype(int) if strat is not None else None
    
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=1 - val_proportion, 
        random_state=random_state, 
        stratify=strat_temp
    )
    
    # Report split statistics
    if "label" in df.columns and strat is not None:
        print(f"\nFinal split statistics:")
        for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            counts = split_df["label"].value_counts()
            print(f"  {name}: {len(split_df):,} samples {dict(counts)}")
    
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