# Class Imbalance Solutions Implementation

## Overview

This document describes the implemented solutions for handling severe class imbalance (88% BEN, 12% MEL) in the melanoma dermoscopic prognosis dataset.

## Implemented Solutions

### 1. Label Normalization
- **Issue**: Labels were 0 (BEN) and 6 (MEL) instead of standard 0/1
- **Solution**: Automatic normalization in `data_loader.py` that maps labels to [0, 1]
- **Location**: `src/utils/data_loader.py` lines 128-140

### 2. Weighted Binary Cross-Entropy Loss
- **Implementation**: `src/utils/losses.py` - `WeightedBCELoss` class
- **How it works**: 
  - Automatically computes class weights inversely proportional to frequency
  - Positive class (MEL) gets weight = `n_total / (2 * n_positive)`
  - Negative class (BEN) gets weight = `n_total / (2 * n_negative)`
- **Formula**: `loss = -w_pos * y * log(p) - w_neg * (1-y) * log(1-p)`
- **Expected weight for MEL**: ~4.15 (since ratio is 7:1)

### 3. Focal Loss
- **Implementation**: `src/utils/losses.py` - `FocalLoss` class
- **How it works**:
  - Focuses training on hard examples
  - Down-weights easy examples: `FL = -alpha * (1 - p_t)^gamma * log(p_t)`
  - Combines class balancing (alpha) with hard example focusing (gamma)
- **Default gamma**: 2.0 (can be tuned: 1.0-3.0)
- **Benefits**: 
  - Addresses both class imbalance and hard examples
  - Particularly effective for medical imaging

### 4. Class-Aware Strong Augmentation
- **Implementation**: `src/utils/preprocess.py` - `build_image_augment(strong=True)`
- **How it works**:

  - Minority class (MEL) gets strong augmentation
  - Majority class (BEN) gets light augmentation
  - Applied automatically in dataset based on label
- **Strong augmentation includes**:
  - Horizontal + vertical flips
  - Aggressive color jitter (±20% brightness/contrast)
  - Random rotation (±15 degrees)
  - Random scaling/cropping (0.9x - 1.1x zoom)
  - Gaussian blur (20% chance)
  - Random noise (10% chance)

## Usage Examples

### Training with Weighted BCE Loss (Recommended)
```bash
python src/main.py \
  --mode train \
  --model_type dino \
  --dino_checkpoint dino_v3/outputs_dino/checkpoints/best.pt \
  --multitask \
  --cls_loss_type weighted_bce \
  --class_aware_augment \
  --epochs 30 \
  --batch_size 16
```

### Training with Focal Loss
```bash
python src/main.py \
  --mode train \
  --model_type dino \
  --dino_checkpoint dino_v3/outputs_dino/checkpoints/best.pt \
  --multitask \
  --cls_loss_type focal \
  --focal_gamma 2.0 \
  --class_aware_augment \
  --epochs 30
```

### Training with Standard BCE (Baseline)
```bash
python src/main.py \
  --mode train \
  --model_type dino \
  --cls_loss_type bce \
  --epochs 30
```

## Class Imbalance Statistics

From your dataset (`data/merged_dataset.csv`):
- **Total samples**: 9,165
- **BEN (class 0)**: 8,061 (88.0%)
- **MEL (class 1)**: 1,104 (12.0%)
- **Imbalance ratio**: ~7.3:1

## Expected Behavior

When using weighted BCE or Focal Loss, you should see:
1. Printout of class imbalance statistics during training
2. Automatic weight calculation
3. Better recall for minority class (MEL)
4. More balanced precision-recall curve

## Recommendations

1. **Start with Weighted BCE**: Easiest to tune, well-established
2. **Try Focal Loss**: Better for hard examples, good for medical imaging
3. **Always use class-aware augmentation**: Improves generalization for minority class
4. **Monitor both AUROC and Precision-Recall**: AUROC can be misleading with imbalance
5. **Use stratified splits**: Already implemented in `split_dataset()` function

## Files Modified/Created

1. `src/utils/data_loader.py` - Label normalization
2. `src/utils/losses.py` - NEW: WeightedBCELoss, FocalLoss
3. `src/utils/preprocess.py` - Strong augmentation
4. `src/train.py` - Loss selection and class-aware augmentation
5. `src/main.py` - CLI arguments for loss selection

## Testing

To verify label normalization works:
```python
from src.utils.data_loader import load_emb_data
df = load_emb_data('data/merged_dataset.csv', 'data/images')
print('Labels:', sorted(df['label'].unique()))  # Should be [0.0, 1.0]
```

To test loss functions:
```python
from src.utils.losses import compute_class_weights
import pandas as pd

df = pd.read_csv('data/merged_dataset.csv')
weights = compute_class_weights(df, label_col='label')
print(weights)
```

## Notes

- OpenCV (`cv2`) is required for strong augmentation features
- Ensure labels are properly normalized before training
- Class weights are computed automatically from training data
- Augmentation is only applied during training (not validation/test)

