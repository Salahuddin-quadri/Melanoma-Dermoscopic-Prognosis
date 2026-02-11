# Melanoma Dermoscopic Prognosis - Web Application

A beautiful, interactive web application for melanoma diagnosis using the DINO multitask model.

## Features

- 🎨 **Beautiful UI**: Modern, dark-themed interface with gradient accents
- 🧠 **Interactive Neural Network Visualization**: Animated network flow during prediction
- 🔬 **Dual-Head Results**: 
  - Classification: Melanoma vs Benign with confidence
  - Regression: Breslow thickness prediction with clinical staging
- 📱 **Responsive Design**: Works on desktop and mobile devices
- ⚡ **Fast Inference**: Optimized model loading and prediction

## Setup

1. **Place your model checkpoint**:
   - Copy your trained `model.pt` file to the `web_app/` directory
   - The model should be trained with:
     - `multitask=True`
     - `model_type="dino"`
     - `arch="vit_b_32"` (or as specified in your config)

2. **Install dependencies**:
   ```bash
   cd web_app
   pip install -r requirements.txt
   ```

3. **Optional: DINO checkpoint**:
   - If you have a DINO pretrained checkpoint, place it at:
     - `../dino_v3/outputs_dino/checkpoints/best.pt`
   - Or modify the path in `model_inference.py`

## Running the Application

```bash
cd web_app
python app.py
```

The application will be available at `http://localhost:5000`

## Model Configuration

The model is configured based on your training settings:
- **Architecture**: DINO ViT (vit_b_32)
- **Task**: Multitask (Classification + Regression)
- **Image Size**: 224x224
- **Hidden Dimension**: 256
- **Freeze Layers**: 8

These settings are in `model_inference.py` and can be adjusted if needed.

## Usage

1. Open the web application in your browser
2. Upload a dermoscopic image (drag & drop or click to browse)
3. Watch the neural network visualization as the model processes
4. View the results:
   - **Classification**: Probability of melanoma with confidence
   - **Regression**: Predicted Breslow thickness with T-stage category

## File Structure

```
web_app/
├── app.py                 # Flask backend
├── model_inference.py     # Model loading and inference
├── model.pt              # Your trained model (place here)
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Frontend HTML
└── static/
    ├── css/
    │   └── style.css     # Styling
    └── js/
        └── app.js        # Frontend logic and visualization
```

## Troubleshooting

- **Model not found**: Ensure `model.pt` is in the `web_app/` directory
- **Import errors**: Make sure you're running from the project root or have the `src/` directory in your Python path
- **CUDA errors**: The app will automatically fall back to CPU if CUDA is not available

## Notes

- This is a research tool and not a substitute for professional medical diagnosis
- The model expects RGB images
- Supported formats: JPEG, PNG, BMP, etc. (any format PIL can read)

