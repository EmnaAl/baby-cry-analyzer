from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
from audio_processor import AudioProcessor
from model import BabyCryClassifier
import numpy as np
from datetime import datetime
import logging

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
audio_processor = None
classifier = None
model_loaded = False

# Categories and their descriptions
CATEGORIES = {
    'hungry': {
        'description': 'Baby needs feeding',
        'recommendations': [
            'Check if it\'s feeding time',
            'Prepare a bottle or breastfeed',
            'Try offering food if baby is old enough for solids'
        ]
    },
    'pain': {
        'description': 'Baby is in discomfort or pain',
        'recommendations': [
            'Check for visible injuries or discomfort',
            'Look for signs of illness (fever, etc.)',
            'Consider consulting a pediatrician if pain persists',
            'Try gentle comfort measures'
        ]
    },
    'need_to_change': {
        'description': 'Baby needs diaper change',
        'recommendations': [
            'Check and change the diaper',
            'Clean the baby thoroughly',
            'Apply diaper cream if needed'
        ]
    },
    'tired': {
        'description': 'Baby is sleepy or tired',
        'recommendations': [
            'Create a calm environment',
            'Try swaddling or gentle rocking',
            'Dim the lights',
            'Play soft, soothing music or white noise'
        ]
    }
}


def allowed_file(filename):
    """Check if uploaded file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def initialize_components():
    """Initialize audio processor and load trained model."""
    global audio_processor, classifier, model_loaded
    
    try:
        # Initialize audio processor
        audio_processor = AudioProcessor()
        logger.info("Audio processor initialized")
        
        # Load trained model
        model_path = "models/baby_cry_rf_model.pkl"
        if os.path.exists(model_path):
            classifier = BabyCryClassifier()
            classifier.load_model(model_path)
            model_loaded = True
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model file not found: {model_path}")
            # Try alternative model
            alt_model_path = "models/baby_cry_svm_model.pkl"
            if os.path.exists(alt_model_path):
                classifier = BabyCryClassifier()
                classifier.load_model(alt_model_path)
                model_loaded = True
                logger.info(f"Alternative model loaded from {alt_model_path}")
            else:
                logger.warning("No trained model found. Please train a model first.")
                model_loaded = False
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        model_loaded = False


# Initialize components when the app starts
initialize_components()

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def home():
    """Home page with API documentation."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Baby Cry Analyzer API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { color: #0066cc; font-weight: bold; }
            .status { padding: 5px 10px; border-radius: 3px; color: white; }
            .ready { background: #28a745; }
            .not-ready { background: #dc3545; }
            .categories { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }
            .category { background: #e3f2fd; padding: 15px; border-radius: 8px; border-left: 4px solid #2196f3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Baby Cry Analyzer API</h1>
            <p>Machine Learning-powered baby cry analysis system</p>
            
            <div class="status {{ status_class }}">
                Status: {{ status_text }}
            </div>
            
            <h2>API Endpoints</h2>
            
            <div class="endpoint">
                <h3><span class="method">POST</span> /predict</h3>
                <p>Upload a .wav audio file to analyze baby cry</p>
                <p><strong>Parameters:</strong> audio (file) - WAV audio file</p>
                <p><strong>Returns:</strong> JSON with prediction results</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">GET</span> /health</h3>
                <p>Check API health status</p>
                <p><strong>Returns:</strong> Health status information</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">GET</span> /categories</h3>
                <p>Get all available cry categories and their descriptions</p>
                <p><strong>Returns:</strong> List of categories with recommendations</p>
            </div>
            
            <h2>Supported Categories</h2>
            <div class="categories">
                {% for category, info in categories.items() %}
                <div class="category">
                    <h4>{{ category.title() }}</h4>
                    <p>{{ info.description }}</p>
                </div>
                {% endfor %}
            </div>
            
            <h2>Example Usage</h2>
            <pre><code>curl -X POST -F "audio=@baby_cry.wav" http://localhost:5000/predict</code></pre>
        </div>
    </body>
    </html>
    """
    
    status_class = "ready" if model_loaded else "not-ready"
    status_text = "Model Ready" if model_loaded else "Model Not Loaded"
    
    return render_template_string(html_template, 
                                 status_class=status_class, 
                                 status_text=status_text,
                                 categories=CATEGORIES)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


@app.route('/categories', methods=['GET'])
def get_categories():
    """Get all available categories."""
    return jsonify({
        'categories': CATEGORIES,
        'total_categories': len(CATEGORIES)
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint."""
    try:
        # Check if model is loaded
        if not model_loaded or classifier is None:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Please train a model first using train_model.py'
            }), 503
        
        # Check if file is present
        if 'audio' not in request.files:
            return jsonify({
                'error': 'No audio file provided',
                'message': 'Please upload a WAV audio file'
            }), 400
        
        file = request.files['audio']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'message': 'Please select a file to upload'
            }), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file format',
                'message': 'Only WAV files are supported'
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        try:
            # Process audio and extract features
            features = audio_processor.process_file(file_path)
            
            # Make prediction
            prediction_result = classifier.predict_single(features)
            
            # Get category information
            predicted_category = prediction_result['predicted_need']
            category_info = CATEGORIES.get(predicted_category, {})
            
            # Prepare response
            response = {
                'success': True,
                'prediction': {
                    'predicted_need': predicted_category,
                    'confidence': prediction_result['confidence'],
                    'description': category_info.get('description', 'Unknown category'),
                    'recommendations': category_info.get('recommendations', [])
                },
                'all_probabilities': prediction_result['all_probabilities'],
                'top_predictions': prediction_result['top_predictions'],
                'file_info': {
                    'filename': filename,
                    'processed_at': datetime.now().isoformat()
                }
            }
            
            # Log prediction
            logger.info(f"Prediction made: {predicted_category} (confidence: {prediction_result['confidence']:.3f})")
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error processing audio file: {str(e)}")
            return jsonify({
                'error': 'Audio processing failed',
                'message': str(e)
            }), 500
            
        finally:
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
    
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred'
        }), 500


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get information about the loaded model."""
    if not model_loaded or classifier is None:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'No model is currently loaded'
        }), 503
    
    try:
        info = classifier.get_model_info()
        return jsonify({
            'model_info': info,
            'categories': CATEGORIES
        })
    except Exception as e:
        return jsonify({
            'error': 'Error getting model info',
            'message': str(e)
        }), 500


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({
        'error': 'File too large',
        'message': f'File size exceeds {MAX_CONTENT_LENGTH // (1024*1024)}MB limit'
    }), 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


if __name__ == '__main__':
    print("Starting Baby Cry Analyzer API...")
    print(f"Model loaded: {model_loaded}")
    print("API Documentation: http://localhost:5000")
    print("-" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
