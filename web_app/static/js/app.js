// Neural Network Visualization and App Logic

let currentImage = null;
let networkAnimation = null;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    setupFileUpload();
    initializeNetworkVisualization();
});

// File Upload Setup
function setupFileUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const imageInput = document.getElementById('imageInput');
    const uploadSection = document.getElementById('uploadSection');
    
    // Click to upload
    uploadArea.addEventListener('click', () => {
        imageInput.click();
    });
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleImageFile(files[0]);
        }
    });
    
    // File input change
    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleImageFile(e.target.files[0]);
        }
    });
}

// Handle Image File
function handleImageFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = (e) => {
        currentImage = e.target.result;
        showImagePreview(currentImage);
        startPrediction(file);
    };
    reader.readAsDataURL(file);
}

// Show Image Preview
function showImagePreview(imageSrc) {
    const uploadArea = document.getElementById('uploadArea');
    const imagePreview = document.getElementById('imagePreview');
    const previewImage = document.getElementById('previewImage');
    
    uploadArea.style.display = 'none';
    previewImage.src = imageSrc;
    imagePreview.style.display = 'block';
}

// Start Prediction
function startPrediction(imageFile) {
    // Show network visualization
    document.getElementById('networkSection').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
    
    // Start network animation
    startNetworkAnimation();
    
    // Prepare form data
    const formData = new FormData();
    formData.append('image', imageFile);
    
    // Send prediction request
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            stopNetworkAnimation();
            // Add 0.3 second delay before displaying results
            setTimeout(() => {
                displayResults(data.results);
            }, 5000);
        } else {
            throw new Error(data.error || 'Prediction failed');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error: ' + error.message);
        stopNetworkAnimation();
        document.getElementById('networkSection').style.display = 'none';
    });
}

// Network Visualization
function initializeNetworkVisualization() {
    const svg = document.getElementById('networkSvg');
    const ns = "http://www.w3.org/2000/svg";
    
    // Define network structure
    const layers = [
        { name: 'Input', nodes: 1, x: 50 },
        { name: 'ViT Backbone', nodes: 6, x: 200 },
        { name: 'Feature Projection', nodes: 4, x: 400 },
        { name: 'Dual Heads', nodes: 2, x: 600 }
    ];
    
    const nodeRadius = 15;
    const layerHeight = 350;
    
    // Draw layers
    layers.forEach((layer, layerIdx) => {
        const nodeSpacing = layerHeight / (layer.nodes + 1);
        
        // Draw nodes
        for (let i = 0; i < layer.nodes; i++) {
            const y = nodeSpacing * (i + 1);
            const circle = document.createElementNS(ns, 'circle');
            circle.setAttribute('cx', layer.x);
            circle.setAttribute('cy', y);
            circle.setAttribute('r', nodeRadius);
            circle.setAttribute('class', 'network-node');
            circle.setAttribute('id', `node-${layerIdx}-${i}`);
            svg.appendChild(circle);
        }
        
        // Draw connections (to next layer)
        if (layerIdx < layers.length - 1) {
            const nextLayer = layers[layerIdx + 1];
            const nextNodeSpacing = layerHeight / (nextLayer.nodes + 1);
            
            for (let i = 0; i < layer.nodes; i++) {
                const y1 = nodeSpacing * (i + 1);
                for (let j = 0; j < nextLayer.nodes; j++) {
                    const y2 = nextNodeSpacing * (j + 1);
                    const line = document.createElementNS(ns, 'line');
                    line.setAttribute('x1', layer.x + nodeRadius);
                    line.setAttribute('y1', y1);
                    line.setAttribute('x2', nextLayer.x - nodeRadius);
                    line.setAttribute('y2', y2);
                    line.setAttribute('class', 'network-connection');
                    line.setAttribute('id', `conn-${layerIdx}-${i}-${layerIdx + 1}-${j}`);
                    svg.appendChild(line);
                }
            }
        }
    });
}

// Start Network Animation
function startNetworkAnimation() {
    const svg = document.getElementById('networkSvg');
    const nodes = svg.querySelectorAll('.network-node');
    const connections = svg.querySelectorAll('.network-connection');
    
    let currentLayer = 0;
    const layers = [0, 1, 2, 3]; // Layer indices
    
    function animateLayer(layerIdx) {
        // Activate nodes in current layer
        nodes.forEach((node, idx) => {
            const nodeLayer = parseInt(node.id.split('-')[1]);
            if (nodeLayer === layerIdx) {
                node.classList.add('active');
                setTimeout(() => {
                    node.classList.remove('active');
                }, 500);
            }
        });
        
        // Activate connections from current layer
        connections.forEach((conn) => {
            const connParts = conn.id.split('-');
            const fromLayer = parseInt(connParts[1]);
            if (fromLayer === layerIdx) {
                conn.classList.add('active');
                setTimeout(() => {
                    conn.classList.remove('active');
                }, 500);
            }
        });
        
        currentLayer++;
        if (currentLayer < layers.length) {
            setTimeout(() => animateLayer(currentLayer), 600);
        } else {
            // Loop animation
            currentLayer = 0;
            setTimeout(() => animateLayer(currentLayer), 300);
        }
    }
    
    networkAnimation = setInterval(() => {
        currentLayer = 0;
        animateLayer(currentLayer);
    }, 2500);
    
    // Start first animation
    animateLayer(0);
}

// Stop Network Animation
function stopNetworkAnimation() {
    if (networkAnimation) {
        clearInterval(networkAnimation);
        networkAnimation = null;
    }
    
    // Remove all active classes
    const svg = document.getElementById('networkSvg');
    svg.querySelectorAll('.active').forEach(el => el.classList.remove('active'));
}

// Display Results
function displayResults(results) {
    // Hide network section
    document.getElementById('networkSection').style.display = 'none';
    
    // Show results section
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
    
    // Classification Results
    if (results.classification) {
        const cls = results.classification;
        const clsPrediction = document.getElementById('clsPrediction');
        const clsProbabilityBar = document.getElementById('clsProbabilityBar');
        const clsProbabilityText = document.getElementById('clsProbabilityText');
        const clsConfidence = document.getElementById('clsConfidence');
        
        clsPrediction.textContent = cls.prediction;
        clsPrediction.className = 'prediction-display ' + 
            (cls.prediction === 'Melanoma' ? 'danger' : 'success');
        
        const probability = (cls.probability * 100).toFixed(1);
        clsProbabilityBar.style.width = probability + '%';
        clsProbabilityText.textContent = `Probability: ${probability}%`;
        
        if (cls.confidence) {
            clsConfidence.textContent = `Confidence: ${(cls.confidence * 100).toFixed(1)}%`;
        }
    }
    
    // Regression Results
    if (results.regression) {
        const reg = results.regression;
        const regThickness = document.getElementById('regThickness');
        const regCategory = document.getElementById('regCategory');
        const scaleIndicator = document.getElementById('scaleIndicator');
        
        if (reg.thickness_mm !== null) {
            regThickness.textContent = reg.thickness_mm.toFixed(2) + ' mm';
            regCategory.textContent = reg.thickness_category;
            
            // Update scale indicator (assuming max thickness of 8mm)
            const maxThickness = 8.0;
            const percentage = Math.min((reg.thickness_mm / maxThickness) * 100, 100);
            scaleIndicator.style.width = percentage + '%';
        }
    }
}

// Reset Upload
function resetUpload() {
    currentImage = null;
    document.getElementById('uploadArea').style.display = 'block';
    document.getElementById('imagePreview').style.display = 'none';
    document.getElementById('networkSection').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('imageInput').value = '';
    stopNetworkAnimation();
}

// Add CSS for network animation
const style = document.createElement('style');
style.textContent = `
    .network-node {
        fill: #334155;
        stroke: #64748b;
        stroke-width: 2;
        transition: all 0.3s ease;
    }
    
    .network-node.active {
        fill: #6366f1;
        stroke: #818cf8;
        filter: drop-shadow(0 0 10px #6366f1);
        animation: pulse 0.5s ease;
    }
    
    .network-connection {
        stroke: #334155;
        stroke-width: 1;
        opacity: 0.3;
        transition: all 0.3s ease;
    }
    
    .network-connection.active {
        stroke: #6366f1;
        stroke-width: 2;
        opacity: 1;
        animation: flow 0.5s ease;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.3); }
    }
    
    @keyframes flow {
        0% { stroke-dasharray: 0 100; }
        100% { stroke-dasharray: 100 0; }
    }
    
    .prediction-display.danger {
        color: #ef4444;
    }
    
    .prediction-display.success {
        color: #10b981;
    }
`;
document.head.appendChild(style);

