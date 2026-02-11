# Data Flow Diagram (DFD) - Melanoma Dermoscopic Prognosis

This document outlines the data flow across the project's components, from raw data collection to web application results.

## Level 0: Global Context Diagram
The system takes dermoscopic images and clinical metadata as input and provides malignancy classifications and Breslow thickness predictions.

```mermaid
graph LR
    User([User/Clinician]) -- Upload Image --> WebApp[Melanoma Prognosis System]
    Metadata[(Clinical Metadata)] -- Input CSV --> WebApp
    WebApp -- Analysis Results --> User
```

---

## Level 1: System Process Detail
The system operates in three main phases: Pretraining, Training, and Inference.

### Phase 1: Self-Supervised Pretraining (DINO-v3)
The goal is to learn domain-specific visual features from unlabeled medical images.

```mermaid
graph TD
    RawData[(Raw Images bin/)] --> Loader[robust_dino_trainer.py]
    Loader --> Aug[robust_augmentation.py]
    Aug -- Global/Local Crops --> Model[robust_dino_model.py]
    Model -- Backprop Loss --> Model
    Model -- Save Weights --> Checkpoint[dino_v3/outputs_dino/checkpoints/best.pt]
```

### Phase 2: Hybrid Multitask Training
Combining learned visual features with clinical data to perform specific diagnostic tasks.

```mermaid
graph TD
    DINO_W[DINO best.pt] --> HybridModel[src/models/dino_hybrid.py]
    Metadata[(merged_dataset.csv)] --> Main[src/main.py]
    Images[(data/images/)] --> Main
    
    Main --> DataLoader[src/utils/data_loader.py]
    DataLoader --> Pre[(src/utils/preprocess.py)]
    
    Pre -- Normalization/Resizing --> Trainer[src/train.py]
    HybridModel -- Feature Extraction --> Trainer
    Trainer -- CrossAttentionFusion --> Trainer
    Trainer -- Joint Loss Optimization --> Trainer
    
    Trainer -- Final Best Model --> Final_W[outputs/trainX/checkpoints/best.pt]
```

### Phase 3: Web Application Inference
Real-time analysis of user-uploaded images.

```mermaid
graph LR
    Client[web_app/templates/index.html] -- POST /predict --> App[web_app/app.py]
    App -- Invoke --> Inference[web_app/model_inference.py]
    Final_W[(Trained Weights)] --> Inference
    
    Inference -- Preprocessing --> ViT[DINO ViT Backbone]
    ViT -- Feature Map --> Attn[Attention Mechanism]
    Attn -- Prediction --> Results[Results UI]
    Results -- JSON Response --> Client
```

---

## File Responsibilities in Data Flow

| Process | Primary Files | Data Input | Data Output |
| :--- | :--- | :--- | :--- |
| **Ingestion** | `data_loader.py`, `preprocess.py` | CSV, JPEG/PNG | Normalized Tensors |
| **Feature Extraction** | `dino_hybrid.py`, `robust_dino_model.py` | Image Tensors | Feature Vectors (768d/1024d) |
| **Knowledge Fusion** | `fusion.py` | Image + Clinical Features | Fused Context Vector |
| **Diagnostics** | `train.py`, `evaluate.py` | Optimized Weights | Malignancy Prob, Thickness (mm) |
| **Interface** | `app.py`, `app.js` | User File Stream | Predictions & Visualizations |
