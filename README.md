# Baby Cry Analyzer

A machine learning-based application that analyzes baby cries to determine their needs (hungry, pain, need-to-change, tired, attention).

## Features

- RESTful API for audio file upload and analysis
- Machine learning model for cry classification
- Audio preprocessing and feature extraction
- Support for multiple baby cry categories
- Model training and evaluation capabilities

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create the dataset directory structure:
   ```
   dataset/
   ├── hungry/
   ├── pain/
   ├── need_to_change/
   ├── tired/
   └── attention/
   ```

4. Add your .wav audio files to the respective category folders

## Usage

### Training the Model
```bash
python train_model.py
```

### Running the API Server
```bash
python app.py
```

### API Endpoints

- `POST /predict` - Upload a .wav file for cry analysis
- `GET /health` - Health check endpoint
- `GET /categories` - Get available cry categories

### Example API Usage
```bash
curl -X POST -F "audio=@baby_cry.wav" http://localhost:5000/predict
```

## Project Structure

- `app.py` - Flask API server
- `model.py` - ML model implementation
- `audio_processor.py` - Audio preprocessing utilities
- `train_model.py` - Model training script
- `dataset/` - Training data organized by categories
- `models/` - Saved trained models
- `uploads/` - Temporary file storage

## Categories

- **Hungry**: Baby needs feeding
- **Pain**: Baby is in discomfort or pain
- **Need to Change**: Baby needs diaper change
- **Tired**: Baby is sleepy or tired
- **Attention**: Baby wants attention or comfort
