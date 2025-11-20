# Melanoma Dermoscopic Prognosis: Self-Supervised Vision Transformers with Clinical Integration

**A deep learning pipeline for melanoma thickness prediction and prognosis using dermoscopic images and clinical metadata**

## Overview

This repository provides a comprehensive, end-to-end deep learning framework for non-invasive **melanoma thickness prediction and prognosis classification** using dermoscopic images and structured clinical features. The system integrates state-of-the-art self-supervised Vision Transformers (DINO-ViT) with clinical metadata through a novel cross-attention fusion mechanism, addressing key challenges in medical AI: class imbalance, clinical interpretability, and robust generalization across diverse datasets.

### Key Contributions

- **Hybrid Multimodal Architecture**: Combines visual features from dermoscopic images (DINO-ViT backbone) with structured clinical features (Breslow thickness, AJCC stage) through cross-attention fusion
- **Self-Supervised Pretraining**: Leverages domain-specific DINO-v3 pretraining for improved feature extraction without relying solely on ImageNet weights
- **Robust Handling of Class Imbalance**: Implements Focal Loss, Weighted BCE, and class-aware augmentation to address the inherent BEN/MEL imbalance in dermoscopy datasets
- **Clinical Interpretability**: Enables explainable predictions through XAI methods (Grad-CAM, Integrated Gradients) for clinician trust and validation
- **Reproducible & Modular Pipeline**: Provides flexible architecture selection (DINO, ResNet50), task options (classification, regression, multitask), and comprehensive evaluation metrics

## Model Architecture

### Image Branch
- **Backbone**: Vision Transformer (ViT-B/32) pretrained with DINO-v3 or ImageNet
- **Pooling**: CLS token or learned clinical tokens with cross-attention mechanism
- **Fusion Options**: Cross-attention (novel), concatenation (baseline)

### Clinical Branch
- **Input**: Standardized structured features (thickness, AJCC stage, lesion type, etc.)
- **Processing**: MLP with Dense layers and dropout for feature learning

### Fusion Layer
- **Cross-Attention**: Learns rich interactions between visual and clinical modalities
- **Classification Head**: Binary output for BEN/MEL classification
- **Regression Head** (optional multitask): Thickness prediction in mm

## Getting Started

### Installation

```bash
# Clone repository
git clone https://github.com/Salahuddin-quadri/Melanoma-Dermoscopic-Prognosis.git
cd Melanoma-Dermoscopic-Prognosis

# Create virtual environment (Python 3.10+ recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

1. **CSV Structure**: Prepare a metadata CSV with the following columns:
   - `image_path` or `image_name`: Path/name of dermoscopic image file
   - `label`: Binary target (0=Benign, 1=Melanoma)
   - `thickness`: Breslow thickness in mm (for regression/multitask)
   - `stage_ajcc`: AJCC stage (numerical: 0-4)
   - Additional clinical features (optional): `age`, `sex`, `site`, etc.

2. **File Organization**:
   ```
   data/
   ├── meta_data.csv
   └── images/
       ├── img001.jpg
       ├── img002.png
       └── ...
   ```

3. **Supported Datasets**:
   - ISIC Archive (multiple versions)
   - HAM10000
   - Early Melanoma Benchmark (EMB)
   - PH2 (external validation)

### Quick Start

#### Training a DINO-ViT Hybrid Model

```bash
python -m src.main \
  --metadata_path data/meta_data.csv \
  --image_dir data/images \
  --model_type dino \
  --mode train \
  --epochs 200 \
  --batch_size 32 \
  --image_size 384 384 \
  --multitask \
  --fusion_type cross_attention \
  --cls_loss_type weighted_bce \
  --output_dir outputs
```

#### Evaluation on Test Set

```bash
python -m src.main \
  --metadata_path data/meta_data.csv \
  --image_dir data/images \
  --model_type dino \
  --mode evaluate \
  --batch_size 32 \
  --weights_path outputs/checkpoints/best.pt \
  --output_dir outputs
```

### Advanced Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_type` | `dino` | Model architecture: `dino` or `resnet` |
| `--dino_checkpoint` | `` | Path to domain-specific DINO checkpoint |
| `--fusion_type` | `cross_attention` | Fusion mechanism: `cross_attention` or `concat` |
| `--cls_loss_type` | `weighted_bce` | Loss: `bce`, `weighted_bce`, or `focal` |
| `--focal_gamma` | `2.0` | Gamma parameter for Focal Loss |
| `--task` | `classification` | Single-task: `classification` or `regression` |
| `--multitask` | `False` | Enable dual-head (classification + regression) |
| `--class_aware_augment` | `False` | Strong augmentation for minority class |
| `--freeze_backbone_layers` | `7` | Number of ViT layers to freeze (0=all trainable) |
| `--val_size` | `0.15` | Validation set fraction |
| `--test_size` | `0.15` | Test set fraction |

## Class Imbalance Solutions

The pipeline addresses severe class imbalance (typically ~90% BEN, ~10% MEL) through multiple strategies:

### 1. **Loss Functions**
- **Weighted BCE**: Automatically computes class weights as weight_mel = N_ben / N_mel
- **Focal Loss**: Reduces loss for easy examples, focuses on hard examples: FL(p_t) = -α_t(1-p_t)^γ log(p_t)

### 2. **Augmentation**
- **Class-Aware Augmentation**: Strong augmentations (color jitter, rotation, elastic deformation) for minority class MEL, light augmentations for BEN
- **Random Over/Under-sampling**: Balances minority class representation per batch

### 3. **Training Configuration**
```bash
python -m src.main \
  --multitask \
  --cls_loss_type focal \
  --focal_gamma 2.0 \
  --class_aware_augment \
  --loss_alpha 0.7  # 70% classification, 30% regression loss
```

## Evaluation Metrics

### Classification Metrics
- **Accuracy, Precision, Recall, F1-Score**
- **ROC-AUC, PR-AUC** (preferred for imbalanced data)
- **Sensitivity/Specificity** (clinical focus)
- **Balanced Accuracy** = (Sensitivity + Specificity) / 2

### Regression Metrics (Thickness)
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)

### Calibration & Reliability
- **Expected Calibration Error (ECE)**: Measures confidence calibration
- **Bootstrap Confidence Intervals**: 95% CI around metrics

### Subgroup Analysis
```bash
python -m src.main \
  --mode evaluate \
  --subgroup_cols age_group skin_tone anatomical_site
```

## ▶️ Run on Google Colab (Recommended for Cloud GPU)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Salahuddin-quadri/Melanoma-Dermoscopic-Prognosis/blob/main/notebooks/train_colab.ipynb)

The Colab notebook provides:
- Automatic dependency installation
- Google Drive integration for data storage
- GPU/TPU acceleration
- Pre-configured training pipeline
- Real-time monitoring and visualization

## Reproducibility

All experiments include:
- **Fixed random seeds** for deterministic results
- **Detailed hyperparameter logging** (config YAML per run)
- **Checkpoint management**: Saves best and latest models
- **Structured outputs**: Metrics, logs, and prediction files per experiment
- **Version control**: Git integration for tracking code changes

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{melanoma_prognosis_2025,
  title={Melanoma Dermoscopic Prognosis: Self-Supervised Vision Transformers with Clinical Integration},
  author={Salahuddin, Quadri},
  year={2025},
  url={https://github.com/Salahuddin-quadri/Melanoma-Dermoscopic-Prognosis}
}
```

## Dataset Attribution

This project utilizes publicly available datasets:
- **ISIC Archive**: https://isic-archive.com/
- **HAM10000**: Tschandl, P., Rosendahl, C., & Kittler, H. (2018)
- **PH2**: Mendonça et al. (2013), https://www.fc.up.pt/lts/ph2database/

## Requirements

- Python 3.10+
- PyTorch 2.3.1 with CUDA 11.8
- scikit-learn, pandas, numpy
- OpenCV, Pillow for image processing
- See `requirements.txt` for full dependencies

## Contact & Support

For questions, issues, or contributions, please:
- Open a GitHub issue
- Contact: [Your Contact Info]

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

**Note**: This framework is designed for research and educational purposes. Always consult with dermatologists and follow institutional review board (IRB) guidelines when using in clinical settings.
