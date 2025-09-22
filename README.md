Emb Project: End-to-End Melanoma Prognosis (EMB)

Overview

This repository provides a reproducible deep learning pipeline for melanoma prognosis using the Early Melanoma Benchmark (EMB) dataset. It implements a hybrid model that combines image features from dermoscopy images (via a ResNet50 backbone) with structured clinical features (e.g., thickness, stage_ajcc, type) for binary classification.

Project Structure

emb-project/
- data/
  - metadata_atlas.csv
  - dermoscopy_images/
- src/
  - models/
    - __init__.py
    - resnet50_hybrid.py
  - utils/
    - __init__.py
    - data_loader.py
    - preprocess.py
  - main.py
  - train.py
  - evaluate.py
- notebooks/
  - exploratory_data_analysis.ipynb
- requirements.txt
- README.md

Setup

1) Python 3.10 is recommended.
2) Create and activate a virtual environment.
3) Install dependencies:

```bash
pip install -r requirements.txt
```

Data Preparation

- Place the EMB metadata CSV at: data/metadata_atlas.csv.
- Place the dermoscopy images in: data/dermoscopy_images/.
- If your images are already in another folder (e.g., EMB_dataset/EMB_images), you can pass that path via the CLI flag --image_dir.

CSV Expectations

The project expects at least the following columns in metadata_atlas.csv:
- image_id or image_name: name of image file (with or without extension)
- label: binary target (0/1)
- thickness: numerical
- stage_ajcc: numerical (if categorical in your CSV, convert to numerical or provide a mapping)
- type: categorical (e.g., "dermoscopic" or "clinical")

Quick Start

1) Train:

```bash
python -m src.main \
  --metadata_path data/metadata_atlas.csv \
  --image_dir data/dermoscopy_images \
  --mode train \
  --epochs 20 \
  --batch_size 32 \
  --output_dir outputs
```

2) Evaluate:

```bash
python -m src.main \
  --metadata_path data/metadata_atlas.csv \
  --image_dir data/dermoscopy_images \
  --mode evaluate \
  --batch_size 32 \
  --output_dir outputs \
  --weights_path outputs/checkpoints/best.ckpt
```

Model

- Image branch: ResNet50 (ImageNet weights), frozen, followed by GlobalAveragePooling2D.
- Tabular branch: Dense layers over standardized features.
- Fusion: Concatenate branches, then Dense + Dropout, final sigmoid output.

Reproducibility

- Deterministic seeds set in training utilities where practical.
- Checkpoints and logs written under the provided --output_dir.

Notes

- Ensure that image file names in the CSV match files under --image_dir. If the CSV provides names without extensions, the loader tries common extensions (".jpg", ".jpeg", ".png").
- If stage_ajcc is not numeric in your CSV, update src/utils/data_loader.py to map your categories to numeric.


