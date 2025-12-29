from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os
import sys
import json
import traceback
from datetime import datetime

# ============================
# Flask App Configuration
# ============================
app = Flask(__name__, 
            static_folder='static',
            static_url_path='/static',
            template_folder='templates')

# Enable CORS for all routes
CORS(app)

# Proxy fix for running behind reverse proxies
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH,
    SECRET_KEY=os.environ.get('SECRET_KEY', 'plant-disease-classifier-secret-key-2024'),
    JSON_SORT_KEYS=False,
    JSONIFY_PRETTYPRINT_REGULAR=True
)

# ============================
# Ensure Required Directories Exist
# ============================
def ensure_directories():
    """Create required directories if they don't exist"""
    directories = [
        'templates',
        'static/css',
        'static/js',
        UPLOAD_FOLDER,
        'logs'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")

ensure_directories()

# ============================
# Model Configuration - 38 CLASSES
# ============================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {DEVICE}")

# Model parameters
IMAGE_SIZE = 380  # EfficientNet B4 optimal size
NUM_CLASSES = 38  # UPDATED: Changed from 33 to 38

# Plant disease classes (38 classes) - COMPLETE LIST
CLASS_NAMES = [
    'Apple_Apple_scab',
    'Apple_Black_rot',
    'Apple_Cedar_apple_rust',
    'Apple_healthy',
    'Cherry_(including_sour)_healthy',
    'Cherry_(including_sour)_Powdery_mildew',
    'Corn_(maize)_Cercospora_leaf_spot_Gray_leaf_spot',
    'Corn_(maize)_Common_rust',
    'Corn_(maize)_healthy',
    'Corn_(maize)_Northern_Leaf_Blight',
    'Grape_Black_rot',
    'Grape_Esca_(Black_Measles)',
    'Grape_healthy',
    'Grape_Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Peach_Bacterial_spot',
    'Peach_healthy',
    'Pepper_bell_Bacterial_spot',
    'Pepper_bell_healthy',
    'Potato_Early_blight',
    'Potato_healthy',
    'Potato_Late_blight',
    'Strawberry_healthy',
    'Strawberry_Leaf_scorch',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_healthy',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two-spotted_spider_mite',
    'Tomato_Target_Spot',
    'Tomato_Tomato_mosaic_virus',
    'Tomato_Tomato_Yellow_Leaf_Curl_Virus',
    
    'Blueberry___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew'
]

PLANT_CATEGORIES = {
    'Apple': ['Apple_Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy'],
    'Cherry': ['Cherry_(including_sour)_healthy', 'Cherry_(including_sour)_Powdery_mildew'],
    'Corn': ['Corn_(maize)_Cercospora_leaf_spot_Gray_leaf_spot', 'Corn_(maize)_Common_rust',
             'Corn_(maize)_healthy', 'Corn_(maize)_Northern_Leaf_Blight'],
    'Grape': ['Grape_Black_rot', 'Grape_Esca_(Black_Measles)', 'Grape_healthy', 
              'Grape_Leaf_blight_(Isariopsis_Leaf_Spot)'],
    'Peach': ['Peach_Bacterial_spot', 'Peach_healthy'],
    'Pepper': ['Pepper_bell_Bacterial_spot', 'Pepper_bell_healthy'],
    'Potato': ['Potato_Early_blight', 'Potato_healthy', 'Potato_Late_blight'],
    'Strawberry': ['Strawberry_healthy', 'Strawberry_Leaf_scorch'],
    'Tomato': ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_healthy', 
               'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
               'Tomato_Spider_mites_Two-spotted_spider_mite', 'Tomato_Target_Spot',
               'Tomato_Tomato_mosaic_virus', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus'],

    'Blueberry': ['Blueberry___healthy'],
    'Orange': ['Orange___Haunglongbing_(Citrus_greening)'],
    'Raspberry': ['Raspberry___healthy'],
    'Soybean': ['Soybean___healthy'],
    'Squash': ['Squash___Powdery_mildew']
}

# ============================
# Model Functions
# ============================
def create_model():
    """Create and initialize EfficientNet B4 model with 38 classes"""
    print("🧠 Initializing EfficientNet B4 model...")
    
    # Load pretrained EfficientNet B4
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Get number of features in classifier
    num_ftrs = model.classifier[1].in_features
    
    # Replace classifier for our 38 classes
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, NUM_CLASSES)
    )
    
    return model

def load_model_weights(model, model_path="model_imbalanced.pth"):
    """Load trained model weights - FIXED VERSION"""
    print(f"🔍 Looking for model file: {model_path}")
    print(f"📂 Current directory: {os.getcwd()}")
    
    if os.path.exists(model_path):
        try:
            print(f"📥 Loading model weights from: {model_path}")
            file_size = os.path.getsize(model_path) / (1024*1024)
            print(f"📊 File size: {file_size:.2f} MB")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=DEVICE)
            print(f"✅ Checkpoint loaded. Type: {type(checkpoint)}")
            
            # Check if it's an OrderedDict (state_dict) or regular dict
            if isinstance(checkpoint, dict):
                print(f"📦 Checkpoint is a dict with {len(checkpoint)} keys")
                
                # If it has the classifier key, it's a state_dict
                if 'classifier.1.weight' in checkpoint or 'classifier.1.bias' in checkpoint:
                    print("📦 Loading as state_dict...")
                    try:
                        model.load_state_dict(checkpoint)
                        print("✅ Model loaded successfully from state_dict")
                        return True
                    except Exception as e:
                        print(f"❌ Error loading state_dict: {e}")
                        return False
                else:
                    # Might be a checkpoint with 'state_dict' key
                    if 'state_dict' in checkpoint:
                        print("📦 Found 'state_dict' key")
                        model.load_state_dict(checkpoint['state_dict'])
                        print("✅ Model loaded from 'state_dict'")
                        return True
                    elif 'model_state_dict' in checkpoint:
                        print("📦 Found 'model_state_dict' key")
                        model.load_state_dict(checkpoint['model_state_dict'])
                        print("✅ Model loaded from 'model_state_dict'")
                        return True
                    else:
                        print("❌ Unknown checkpoint format")
                        return False
            else:
                # Assume it's a state_dict (OrderedDict)
                print("📦 Loading as state_dict (OrderedDict)...")
                try:
                    model.load_state_dict(checkpoint)
                    print("✅ Model loaded successfully")
                    return True
                except Exception as e:
                    print(f"❌ Error loading: {e}")
                    return False
                    
        except Exception as e:
            print(f"❌ Error loading model weights: {e}")
            traceback.print_exc()
            return False
    else:
        print(f"❌ Model file not found: {model_path}")
        pth_files = [f for f in os.listdir('.') if f.endswith('.pth')]
        if pth_files:
            print(f"📂 Available .pth files: {pth_files}")
        return False

# Initialize model
MODEL = create_model()
MODEL_LOADED = load_model_weights(MODEL)
MODEL.to(DEVICE)
MODEL.eval()

print(f"\n✅ Model initialized: EfficientNet B4")
print(f"✅ Model loaded: {MODEL_LOADED}")
print(f"✅ Number of classes: {NUM_CLASSES}")

# ============================
# Image Transforms
# ============================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# ============================
# Helper Functions
# ============================
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_class_name(class_name):
    """Format class name for display"""
    # Replace underscores with spaces
    formatted = class_name.replace('_', ' ')
    
    # Fix specific cases
    formatted = formatted.replace('Com (maize)', 'Corn')
    formatted = formatted.replace('Haunglongbing', 'Huanglongbing')
    
    # Capitalize words
    words = formatted.split()
    formatted_words = []
    for word in words:
        if word.lower() in ['ii', 'iii', 'iv', 'virus', 'tv', 'a', 'an', 'the', 'of', 'in', 'to']:
            formatted_words.append(word.lower())
        else:
            formatted_words.append(word.capitalize())
    
    return ' '.join(formatted_words)

def get_plant_type(class_name):
    """Get plant type from class name"""
    for plant_type, classes in PLANT_CATEGORIES.items():
        if class_name in classes:
            return plant_type
    return "Unknown"

def predict_image(image):
    """Predict plant disease from image"""
    try:
        # Check if model is loaded
        if not MODEL_LOADED:
            raise Exception("Model not loaded. Please check the model file.")
        
        # Preprocess image
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Make prediction
        with torch.no_grad():
            outputs = MODEL(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Get prediction results
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_score = confidence.item()
        
        # Get top 5 predictions
        top5_probs, top5_indices = torch.topk(probabilities, 5)
        top_predictions = []
        
        for i, (prob, idx) in enumerate(zip(top5_probs[0], top5_indices[0])):
            class_name = CLASS_NAMES[idx.item()]
            plant_type = get_plant_type(class_name)
            
            top_predictions.append({
                'rank': i + 1,
                'class': class_name,
                'formatted_class': format_class_name(class_name),
                'plant_type': plant_type,
                'confidence': round(prob.item() * 100, 2)
            })
        
        # Get all probabilities for the predicted plant type
        predicted_plant = get_plant_type(predicted_class)
        plant_type_probs = {}
        
        for idx, class_name in enumerate(CLASS_NAMES):
            if get_plant_type(class_name) == predicted_plant:
                plant_type_probs[format_class_name(class_name)] = round(probabilities[0][idx].item() * 100, 2)
        
        # Sort plant type probabilities by confidence
        plant_type_probs = dict(sorted(plant_type_probs.items(), 
                                      key=lambda x: x[1], 
                                      reverse=True))
        
        return {
            'success': True,
            'predicted_class': predicted_class,
            'formatted_class': format_class_name(predicted_class),
            'plant_type': predicted_plant,
            'confidence': round(confidence_score * 100, 2),
            'top_predictions': top_predictions,
            'plant_type_probabilities': plant_type_probs,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def log_request(endpoint, data=None, status='success'):
    """Log API requests for debugging"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'endpoint': endpoint,
        'status': status,
        'data': data
    }
    
    log_file = os.path.join('logs', f"api_{datetime.now().strftime('%Y%m%d')}.log")
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

# ============================
# API Routes
# ============================
@app.route('/')
def home():
    """Serve the main web interface"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def api_health():
    """Health check endpoint"""
    log_request('health')
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Plant Disease Classifier API',
        'version': '1.1.0',  # Updated version
        'model': {
            'architecture': 'EfficientNet B4',
            'loaded': MODEL_LOADED,
            'num_classes': NUM_CLASSES,
            'device': str(DEVICE),
            'image_size': IMAGE_SIZE
        },
        'system': {
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'num_classes': NUM_CLASSES,
            'plant_types': list(PLANT_CATEGORIES.keys())
        }
    })

@app.route('/api/model/info', methods=['GET'])
def api_model_info():
    """Get model information"""
    log_request('model_info')
    
    total_params = sum(p.numel() for p in MODEL.parameters())
    trainable_params = sum(p.numel() for p in MODEL.parameters() if p.requires_grad)
    
    return jsonify({
        'success': True,
        'model': {
            'architecture': 'EfficientNet B4',
            'total_parameters': f"{total_params:,}",
            'trainable_parameters': f"{trainable_params:,}",
            'frozen': True,
            'dropout': 0.3,
            'input_size': f"{IMAGE_SIZE}x{IMAGE_SIZE}",
            'normalization': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        },
        'dataset': {
            'num_classes': NUM_CLASSES,
            'plant_types': list(PLANT_CATEGORIES.keys()),
            'plant_count': len(PLANT_CATEGORIES)
        }
    })

@app.route('/api/plants', methods=['GET'])
def api_plants():
    """Get all plant types with their diseases"""
    log_request('plants')
    
    plants_info = []
    for plant_type, classes in PLANT_CATEGORIES.items():
        healthy_classes = [c for c in classes if 'healthy' in c.lower()]
        disease_classes = [c for c in classes if 'healthy' not in c.lower()]
        
        plants_info.append({
            'plant_type': plant_type,
            'total_classes': len(classes),
            'diseases': len(disease_classes),
            'healthy': len(healthy_classes),
            'classes': [format_class_name(c) for c in classes]
        })
    
    return jsonify({
        'success': True,
        'plants': plants_info,
        'total_plants': len(plants_info),
        'total_classes': NUM_CLASSES
    })

@app.route('/api/classes', methods=['GET'])
def api_classes():
    """Get all disease classes"""
    log_request('classes')
    
    formatted_classes = {}
    for plant_type, classes in PLANT_CATEGORIES.items():
        formatted_classes[plant_type] = [format_class_name(c) for c in classes]
    
    return jsonify({
        'success': True,
        'classes': CLASS_NAMES,
        'formatted_classes': formatted_classes,
        'plant_categories': list(PLANT_CATEGORIES.keys()),
        'category_counts': {pt: len(cls) for pt, cls in PLANT_CATEGORIES.items()}
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Predict plant disease from uploaded image"""
    try:
        # Check if model is loaded
        if not MODEL_LOADED:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please check the model file.',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        # Check if file was uploaded
        if 'file' not in request.files:
            log_request('predict', status='error', data={'error': 'No file uploaded'})
            return jsonify({
                'success': False,
                'error': 'No file uploaded. Please provide an image file.',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        file = request.files['file']
        
        # Check if file has a name
        if file.filename == '':
            log_request('predict', status='error', data={'error': 'No file selected'})
            return jsonify({
                'success': False,
                'error': 'No file selected. Please choose an image file.',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            log_request('predict', status='error', data={'error': 'Invalid file type'})
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Read and process image
        try:
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
        except Exception as e:
            log_request('predict', status='error', data={'error': f'Image processing failed: {str(e)}'})
            return jsonify({
                'success': False,
                'error': f'Failed to process image: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Make prediction
        prediction_result = predict_image(image)
        
        # Log successful prediction
        log_request('predict', status='success', data={
            'predicted_class': prediction_result.get('predicted_class'),
            'confidence': prediction_result.get('confidence')
        })
        
        return jsonify(prediction_result)
        
    except Exception as e:
        error_msg = f'Server error: {str(e)}'
        print(f"❌ API error: {error_msg}")
        traceback.print_exc()
        
        log_request('predict', status='error', data={'error': error_msg})
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }), 500

# ============================
# Error Handlers
# ============================
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'timestamp': datetime.now().isoformat()
    }), 404

@app.errorhandler(405)
def method_not_allowed_error(error):
    return jsonify({
        'success': False,
        'error': 'Method not allowed',
        'timestamp': datetime.now().isoformat()
    }), 405

@app.errorhandler(413)
def request_too_large_error(error):
    return jsonify({
        'success': False,
        'error': f'File too large. Maximum size is {MAX_CONTENT_LENGTH // (1024*1024)}MB',
        'timestamp': datetime.now().isoformat()
    }), 413

@app.errorhandler(500)
def internal_error(error):
    print(f"❌ Internal server error: {error}")
    traceback.print_exc()
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500

# ============================
# Application Entry Point
# ============================
if __name__ == '__main__':
    print("\n" + "="*80)
    print("🌱 MULTI-PLANT DISEASE CLASSIFIER API - 38 CLASSES")
    print("="*80)
    print(f"📊 Model: EfficientNet B4")
    print(f"📊 Classes: {NUM_CLASSES} diseases across {len(PLANT_CATEGORIES)} plant types")
    print(f"⚙️  Device: {DEVICE}")
    print(f"🖼️  Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"✅ Model Loaded: {MODEL_LOADED}")
    print(f"📁 Upload Folder: {UPLOAD_FOLDER}")
    print("\n📡 API Endpoints:")
    print("  GET  /                    - Web interface")
    print("  GET  /api/health          - Health check")
    print("  GET  /api/model/info      - Model information")
    print("  GET  /api/plants          - List all plant types")
    print("  GET  /api/classes         - List all disease classes")
    print("  POST /api/predict         - Single image prediction")
    print("\n🌐 Starting server...")
    print(f"👉 Web Interface: http://localhost:5000")
    print(f"👉 API Base URL: http://localhost:5000/api")
    print("="*80)
    
    # Development settings
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )