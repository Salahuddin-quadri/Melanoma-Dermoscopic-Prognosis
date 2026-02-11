# Metrics Formulas - Melanoma Dermoscopic Prognosis

This document details all the mathematical formulas used to compute evaluation metrics in the training and validation pipeline.

---

## **Classification Metrics**

### 1. **Accuracy**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- **TP**: True Positives
- **TN**: True Negatives
- **FP**: False Positives
- **FN**: False Negatives

**Implementation:**
```python
accuracy = (y_pred_binary == y_true_int).mean()
```
Where `y_pred_binary = (y_pred_probs >= 0.5).astype(int)`

---

### 2. **Precision**
```
Precision = TP / (TP + FP)
```
Measures the proportion of predicted positives that are actually positive (true positive rate among predictions).

**Implementation:**
```python
precision = np.sum((y_pred_binary == 1) & (y_true_int == 1)) / max(np.sum(y_pred_binary == 1), 1)
```

---

### 3. **Recall (Sensitivity/TPR)**
```
Recall = TP / (TP + FN)
```
Measures the proportion of actual positives that were correctly identified (true positive rate among ground truth).

**Implementation:**
```python
recall = np.sum((y_pred_binary == 1) & (y_true_int == 1)) / max(np.sum(y_true_int == 1), 1)
```

---

### 4. **F1 Score**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
Harmonic mean of precision and recall; balanced metric for imbalanced datasets.

**Implementation:**
```python
f1 = 2 * precision * recall / max(precision + recall, 1e-10)
```

---

### 5. **AUROC (Area Under ROC Curve)**
```
AUROC = ∫ TPR(FPR) dFPR
```
Area under the Receiver Operating Characteristic curve, ranging from 0 to 1.
- 0.5 = Random classifier
- 1.0 = Perfect classifier

**Implementation:**
```python
auroc = roc_auc_score(y_true, y_pred_probs)
```
(Uses `sklearn.metrics.roc_auc_score`)

---

### 6. **Expected Calibration Error (ECE)**

ECE measures how well-calibrated predicted probabilities are:

```
ECE = Σ_bin [ n_bin / N × |accuracy_bin - confidence_bin| ]
```

Where for each bin:
- **n_bin**: Number of samples in bin
- **N**: Total number of samples
- **accuracy_bin**: Actual accuracy (proportion of correct predictions) in bin
- **confidence_bin**: Average predicted probability in bin

The model is well-calibrated when `confidence ≈ accuracy` in each bin.

**Algorithm:**
1. Partition predictions into n_bins (default: 15) by confidence level
2. For each bin, compute average confidence and actual accuracy
3. Weighted average of absolute differences

**Implementation:**
```python
def compute_ece(predictions, labels, n_bins=15):
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = predictions[in_bin].mean()
            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece
```

---

## **Regression Metrics**

### 1. **Mean Absolute Error (MAE)**
```
MAE = (1/n) × Σ |y_true_i - y_pred_i|
```
Average absolute difference between predicted and actual values.

**Units**: Same as target variable (mm for thickness)

**Implementation:**
```python
mae = mean_absolute_error(y_true, y_pred)
# From sklearn.metrics
```

---

### 2. **Root Mean Squared Error (RMSE)**
```
RMSE = √[ (1/n) × Σ (y_true_i - y_pred_i)² ]
```
Square root of average squared error; penalizes larger errors more heavily.

**Units**: Same as target variable (mm for thickness)

**Implementation:**
```python
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```

---

### 3. **R² Score (Coefficient of Determination)**
```
R² = 1 - (SS_res / SS_tot)
```
Where:
- **SS_res**: Sum of squared residuals = Σ (y_true_i - y_pred_i)²
- **SS_tot**: Total sum of squares = Σ (y_true_i - ȳ)²
- **ȳ**: Mean of true values

Interpretation:
- R² = 1: Perfect predictions
- R² = 0: Model predicts mean value
- R² < 0: Model worse than predicting mean

**Implementation:**
```python
r2 = r2_score(y_true, y_pred)
# From sklearn.metrics
```

---

## **Multitask Learning Loss & Metrics**

### Combined Training Loss
```
Total_Loss = α × Loss_cls + (1 - α) × Loss_reg
```

Where:
- **α** (`loss_alpha`): Weight for classification loss (0 to 1)
- **Loss_cls**: Classification loss (BCE, Weighted BCE, or Focal Loss)
- **Loss_reg**: Regression loss (L1 Loss)

**Default**: α = 0.5 (equal weighting)

**Implementation:**
```python
loss = loss_alpha * loss_cls + (1.0 - loss_alpha) * loss_reg
```

### Combined Validation Metric (for early stopping)
```
val_metric = α × val_auc + (1 - α) × (1 - MAE_normalized)
```

Where:
- **val_auc**: Validation AUROC for classification
- **MAE_normalized**: val_mae / 10.0 (normalized to [0,1] range)

**Implementation:**
```python
mae_normalized = val_mae / 10.0  # Assumes thickness range 0-10mm
val_metric = loss_alpha * val_auc + (1.0 - loss_alpha) * (1.0 - mae_normalized)
```

---

## **Classification Loss Functions**

### 1. **Binary Cross-Entropy (BCE) Loss**
```
BCE = -[y × log(p) + (1 - y) × log(1 - p)]
```

Where:
- **y**: Ground truth (0 or 1)
- **p**: Predicted probability

Applied per sample, then averaged across batch.

**Implementation:**
```python
criterion_cls = torch.nn.BCELoss()
```

---

### 2. **Weighted BCE Loss**
```
BCE_weighted = -w₁ × [y × log(p)] - w₀ × [(1 - y) × log(1 - p)]
```

Where:
- **w₁**: Weight for positive class (MEL)
- **w₀**: Weight for negative class (BEN)
- Weights computed from class frequency: w_i = N / (2 × N_i)

**Class Weights Formula:**
```
w_pos = n_samples / (2 × n_positive)
w_neg = n_samples / (2 × n_negative)
pos_weight = w_pos / w_neg = n_negative / n_positive
```

For imbalanced data (more BEN than MEL), w_pos > w_neg to penalize misclassifying minority class.

**Implementation:**
```python
pos_weight = n_negative / n_positive
criterion_cls = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

---

### 3. **Focal Loss**
```
FL(p_t) = -(1 - p_t)^γ × log(p_t)
```

Where:
- **p_t**: Model's probability for correct class
- **γ** (gamma): Focusing parameter (typically 2.0)
  - When γ=0: reduces to BCE
  - When γ>0: down-weights easy examples, focuses on hard negatives
  
Weighted version:
```
Focal_Loss = -α_t × (1 - p_t)^γ × log(p_t)
```

**Effect**:
- Easy examples (high p_t): reduced by factor (1 - p_t)^γ → 0
- Hard examples (low p_t): reduced by small factor → large gradient

**Implementation:**
```python
criterion_cls = FocalLoss.from_class_counts(
    n_positive=n_positive,
    n_negative=n_negative,
    gamma=focal_gamma,  # Default: 2.0
    device=device
)
```

---

### 4. **L1 Loss (for Regression)**
```
L1 = (1/n) × Σ |y_true_i - y_pred_i|
```

Same as MAE but applied per batch during training.

**Implementation:**
```python
criterion_reg = torch.nn.L1Loss()
loss_reg = criterion_reg(outputs["reg"], batch["thickness"])
```

---

## **Bootstrap Confidence Intervals**

For robust uncertainty quantification, bootstrap CIs are computed:

```
Algorithm:
1. Compute metric on full dataset: metric_value
2. For i = 1 to n_bootstrap (default: 1000):
   a. Sample n_samples with replacement from dataset
   b. Compute metric on bootstrap sample
3. Compute percentiles: CI = [percentile(2.5), percentile(97.5)]
```

**For 95% CI:**
```
lower_ci = percentile(bootstrap_metrics, 2.5)
upper_ci = percentile(bootstrap_metrics, 97.5)
```

**Implementation:**
```python
def bootstrap_confidence_interval(metric_fn, y_true, y_pred, 
                                  n_bootstrap=1000, 
                                  confidence_level=0.95):
    rng = np.random.default_rng()
    bootstrap_metrics = []
    
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), len(y_true))
        bootstrap_metric = metric_fn(y_true[indices], y_pred[indices])
        bootstrap_metrics.append(bootstrap_metric)
    
    alpha = 1 - confidence_level
    lower_ci = np.percentile(bootstrap_metrics, (alpha/2)*100)
    upper_ci = np.percentile(bootstrap_metrics, (1-alpha/2)*100)
    
    return metric_value, lower_ci, upper_ci
```

---

## **Output Metrics Dictionary Structure**

### Classification Metrics
```python
{
    "accuracy": float,           # Overall accuracy
    "precision": float,          # Precision (PPV)
    "recall": float,             # Recall/Sensitivity (TPR)
    "f1": float,                 # F1 score
    "auroc": float,              # Area under ROC curve
    "auroc_computed": bool,      # Whether AUROC was computable
    "auroc_ci_lower": float,     # 95% CI lower bound
    "auroc_ci_upper": float,     # 95% CI upper bound
    "ece": float,                # Expected Calibration Error
}
```

### Regression Metrics
```python
{
    "mae": float,                # Mean Absolute Error
    "rmse": float,               # Root Mean Squared Error
    "r2": float,                 # R² score
    "mae_ci_lower": float,       # 95% CI lower bound
    "mae_ci_upper": float,       # 95% CI upper bound
    "r2_ci_lower": float,        # 95% CI lower bound
    "r2_ci_upper": float,        # 95% CI upper bound
}
```

### Multitask Metrics (combined)
```python
{
    "classification": {...},     # Classification metrics dict
    "regression": {...},         # Regression metrics dict
    "subgroups": {...}          # Optional subgroup analysis
}
```

---

## **Example: Computing Metrics from Predictions**

```python
from src.utils.eval_metrics import compute_classification_metrics, compute_regression_metrics

# Classification
y_true = np.array([0, 1, 1, 0, 1])
y_pred_probs = np.array([0.1, 0.9, 0.8, 0.2, 0.7])

cls_metrics = compute_classification_metrics(y_true, y_pred_probs)
# Output: {accuracy: 1.0, precision: 1.0, recall: 1.0, f1: 1.0, auroc: 1.0, ece: 0.05, ...}

# Regression
y_true_reg = np.array([1.5, 2.3, 3.1, 2.0, 1.8])
y_pred_reg = np.array([1.6, 2.1, 3.2, 2.1, 1.9])

reg_metrics = compute_regression_metrics(y_true_reg, y_pred_reg)
# Output: {mae: 0.14, rmse: 0.152, r2: 0.996, mae_ci_lower: ..., mae_ci_upper: ...}
```

---

## **Training Logging Output Example**

```python
{
    "epoch": 50,
    "train_loss": 0.406,              # α×loss_cls + (1-α)×loss_reg
    "val_auc": 0.895,                 # AUROC on validation set
    "val_mae": 0.47,                  # MAE on validation set
    "loss_cls_weight": 0.5,           # α parameter
    "loss_reg_weight": 0.5            # 1-α parameter
}
```

---

## **References**

- Fawcett, T. (2006). "An introduction to ROC analysis." Pattern Recognition Letters.
- Lin, T. Y., et al. (2017). "Focal Loss for Dense Object Detection." ICCV.
- Guo, C., & Pleiss, G. (2017). "On Calibration of Modern Neural Networks." ICML.
- Scikit-learn Documentation: https://scikit-learn.org/stable/modules/model_evaluation.html

