"""
Flask Web Application for Melanoma Dermoscopic Prognosis

This application provides a web interface for the DINO multitask model
that predicts both classification (melanoma vs benign) and regression (Breslow thickness).
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
from PIL import Image
import io
import base64

# Add parent directory to path to import project modules
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from model_inference import ModelInference

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Initialize model inference
model_inference = None

def init_model():
    """Initialize the model for inference."""
    global model_inference
    try:
        model_inference = ModelInference()
        print("✓ Model initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Error initializing model: {e}")
        return False

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image prediction request."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Perform inference
        if model_inference is None:
            return jsonify({'error': 'Model not initialized'}), 500
        
        results = model_inference.predict(image)
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_inference is not None
    })

if __name__ == '__main__':
    print("Initializing model...")
    if init_model():
        print("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize model. Please check model.pt file exists.")
        sys.exit(1)

