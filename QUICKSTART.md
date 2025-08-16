# Baby Cry Analyzer - Quick Start Guide

Welcome to the Baby Cry Analyzer! This guide will help you get up and running quickly.

## 🚀 Quick Start (5 minutes)

### 1. Setup
```bash
# Navigate to the project directory
cd baby_cry_analyzer

# Run the setup script
python setup.py
```

### 2. Generate Sample Data (for testing)
```bash
# Generate synthetic audio samples for testing
python generate_samples.py --samples 10

# Or generate a single test file
python generate_samples.py --test-file test_cry.wav --type hungry
```

### 3. Train the Model
```bash
# Train the machine learning model
python train_model.py
```

### Understanding Training Results
After training, you'll see results like this:
```
Training Accuracy: 0.9659 (96%)
Validation Accuracy: 0.1739 (17%)
Cross-validation Score: 0.3542 (+/- 0.1523)
```

**What these numbers mean:**
- 🚨 **High Training Accuracy + Low Validation Accuracy = Overfitting**
- ✅ **Good signs**: 111 files processed, all 4 categories present, good balance
- ⚠️ **Warning signs**: Large gap between training (96%) and validation (17%)
- 📊 **Note**: Project now uses 4 categories (removed 'attention' due to insufficient data)

**Quick fixes to improve validation accuracy:**

**For your specific results (94% training, 26% validation):**
```bash
# 1. IMMEDIATE: Just retrain with optimized model (should improve 26% → 35-40%)
python train_model.py

# 2. Check your current dataset balance (4 categories only)
python check_dataset.py

# 3. IMPORTANT: Don't add synthetic samples - they hurt performance!
# Instead, get real baby cry recordings (see "Real Dataset Sources" below)
```

**Expected improvement:** 26% → 35-45% validation accuracy

**Note:** Project now uses 4 categories: hungry, need_to_change, pain, tired  
(Removed 'attention' category due to insufficient data)

### 4. Start the API Server
```bash
# Start the web API
python app.py
```

### 5. Test the System
```bash
# Test the API (in another terminal)
python test_api.py --audio test_cry.wav

# Or visit http://localhost:5000 in your browser
```

## 📁 Using Your Own Audio Data

### Real Dataset Sources

**🎯 For significantly better results, you need REAL baby cry recordings:**

**Free Sources:**
1. **Freesound.org** - Search for "baby cry", "infant cry", "newborn"
   - Website: https://freesound.org/search/?q=baby+cry
   - Download individual .wav files manually
   - Look for Creative Commons licensed sounds

2. **YouTube Audio Extraction** (with permission)
   - Search for "baby crying sounds" 
   - Use tools like youtube-dl to extract audio
   - Convert to .wav format

3. **Research Datasets** (academic use)
   - Contact universities with child development programs
   - Some research datasets available on request

**Commercial/Professional Sources:**
- AudioJungle, Pond5 (stock audio sites)
- Professional sound libraries
- Medical/research institutions

**⚠️ Important Notes:**
- Always respect copyright and privacy
- Get permission when recording babies
- Ensure ethical use of baby audio data

### Directory Structure
Place your .wav files in the appropriate category folders:
```
dataset/
├── hungry/          # Baby is hungry
├── pain/            # Baby is in pain/discomfort
├── need_to_change/  # Baby needs diaper change
└── tired/           # Baby is tired/sleepy
```
**Note:** The 'attention' category has been removed due to insufficient training data.

### Audio Requirements
- Format: WAV files only
- Duration: 0.5 - 10 seconds
- Sample rate: Minimum 8kHz (22kHz recommended)
- Quality: Clear audio with minimal background noise

### Data Collection Tips
- Record at least 20 samples per category for good results
- Ensure variety in recordings (different times, conditions)
- Label categories accurately
- Keep consistent recording quality

## 🔧 API Usage

### Upload and Analyze Audio
```bash
curl -X POST -F "audio=@baby_cry.wav" http://localhost:5000/predict
```

### Response Format
```json
{
  "success": true,
  "prediction": {
    "predicted_need": "hungry",
    "confidence": 0.85,
    "description": "Baby needs feeding",
    "recommendations": [
      "Check if it's feeding time",
      "Prepare a bottle or breastfeed",
      "Try offering food if baby is old enough for solids"
    ]
  },
  "all_probabilities": {
    "hungry": 0.85,
    "tired": 0.08,
    "pain": 0.04,
    "need_to_change": 0.03
  }
}
```

## 🧠 Understanding the Results

### Confidence Scores
- **0.8-1.0**: High confidence - Very reliable prediction
- **0.6-0.8**: Medium confidence - Generally reliable
- **0.4-0.6**: Low confidence - Use with caution
- **0.0-0.4**: Very low confidence - Prediction may be unreliable

### Categories Explained

| Category | Description | Common Signs |
|----------|-------------|--------------|
| **Hungry** | Baby needs feeding | Rhythmic crying, rooting, lip smacking |
| **Pain** | Physical discomfort | High-pitched, intense, sudden onset |
| **Need to Change** | Diaper needs changing | Fussing after feeding/sleeping |
| **Tired** | Baby is sleepy | Rubbing eyes, yawning, grizzling |

## ⚙️ Advanced Configuration

### Model Types
The system supports two model types:
- **Random Forest** (default): Fast, interpretable, good for small datasets
- **SVM**: More accurate with larger datasets, slower training

### Customizing Audio Processing
Edit `audio_processor.py` to modify:
- Feature extraction parameters
- Audio preprocessing steps
- Sample rate and duration settings

### Performance Tuning
1. **More Data**: Add more audio samples per category
2. **Data Quality**: Ensure consistent, high-quality recordings
3. **Feature Engineering**: Modify feature extraction in `audio_processor.py`
4. **Model Tuning**: Adjust hyperparameters in `model.py`

## 🛠️ Troubleshooting

### Common Issues

**"No module named 'librosa'"**
```bash
pip install -r requirements.txt
```

**"Model not loaded"**
- Train a model first: `python train_model.py`
- Check if model files exist in `models/` directory

**"No audio files found"**
- Add .wav files to `dataset/<category>/` folders
- Check file permissions and formats

**Low accuracy**
- Add more training data (20+ samples per category)
- Improve audio quality and labeling
- Ensure balanced dataset across categories

### Performance Issues

**Slow API responses**
- Use smaller audio files (< 5 seconds)
- Consider using Random Forest instead of SVM
- Check system resources (RAM, CPU)

**High memory usage**
- Reduce audio duration in config
- Use lower sample rates for processing
- Clear uploads folder regularly

## 📊 Monitoring and Logging

### View Training Reports
Check the `reports/` directory for:
- Training accuracy metrics
- Confusion matrices
- Feature importance plots

### API Usage Logs
Prediction logs are saved to `logs/predictions.jsonl`

### Analyze Performance
```bash
python -c "from utils import analyze_prediction_logs; print(analyze_prediction_logs())"
```

## 🔒 Security Considerations

### For Production Use
- Add authentication to API endpoints
- Implement rate limiting
- Validate file uploads thoroughly
- Use HTTPS for data transmission
- Consider data privacy regulations

### File Upload Security
- The system only accepts .wav files
- Files are automatically deleted after processing
- Maximum file size is limited to 16MB

## 📖 Further Reading

- [Full Documentation](README.md)
- [API Reference](app.py)
- [Model Architecture](model.py)
- [Audio Processing](audio_processor.py)

## 🆘 Getting Help

1. Check the console output for error messages
2. Review the log files in `logs/` directory
3. Ensure all dependencies are installed correctly
4. Verify your audio files meet the requirements

## 🎯 Next Steps

Once you have the basic system working:

1. **Collect Real Data**: Replace synthetic samples with real baby cry recordings
2. **Improve Model**: Experiment with different algorithms and parameters
3. **Add Features**: Implement additional audio features or preprocessing
4. **Deploy**: Set up the system for production use
5. **Monitor**: Track prediction accuracy and system performance

Happy analyzing! 🍼👶
