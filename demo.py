"""
Sample usage script for the Baby Cry Analyzer.
This script demonstrates how to use the analyzer programmatically.
"""

import os
import sys
import numpy as np
from audio_processor import AudioProcessor
from model import BabyCryClassifier
from utils import get_dataset_statistics, validate_audio_file
import warnings
warnings.filterwarnings('ignore')


def demo_audio_processing():
    """Demonstrate audio processing capabilities."""
    print("=" * 50)
    print("Audio Processing Demo")
    print("=" * 50)
    
    processor = AudioProcessor()
    
    # Check if we have any sample audio files
    sample_files = []
    for category in ['hungry', 'pain', 'need_to_change', 'tired', 'attention']:
        category_path = os.path.join('dataset', category)
        if os.path.exists(category_path):
            files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
            if files:
                sample_files.append(os.path.join(category_path, files[0]))
    
    if not sample_files:
        print("No sample audio files found in dataset directories.")
        print("Please add .wav files to the dataset folders to run this demo.")
        return
    
    # Process first available sample
    sample_file = sample_files[0]
    print(f"Processing sample file: {sample_file}")
    
    try:
        # Validate file
        validation = validate_audio_file(sample_file)
        print(f"File validation: {'✓ Valid' if validation['is_valid'] else '✗ Invalid'}")
        
        if validation['is_valid']:
            props = validation['properties']
            print(f"  Duration: {props['duration']:.2f} seconds")
            print(f"  Sample rate: {props['sample_rate']} Hz")
            print(f"  File size: {props['file_size_mb']:.2f} MB")
        
        # Extract features
        features = processor.process_file(sample_file)
        print(f"Extracted {len(features)} features")
        print(f"Feature vector shape: {features.shape}")
        print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")


def demo_dataset_analysis():
    """Demonstrate dataset analysis."""
    print("\n" + "=" * 50)
    print("Dataset Analysis Demo")
    print("=" * 50)
    
    stats = get_dataset_statistics()
    
    print(f"Total files: {stats['total_files']}")
    print(f"Total duration: {stats['total_duration']:.1f} seconds")
    print(f"Average duration: {stats['average_duration']:.1f} seconds")
    
    print("\nCategory breakdown:")
    for category, category_stats in stats['categories'].items():
        print(f"  {category}: {category_stats['count']} files, {category_stats['duration']:.1f}s")
    
    if stats['sample_rates']:
        print(f"\nSample rates found: {list(stats['sample_rates'].keys())}")
    
    if stats['errors']:
        print(f"\nErrors encountered: {len(stats['errors'])}")
        for error in stats['errors'][:3]:  # Show first 3 errors
            print(f"  - {error}")


def demo_model_training():
    """Demonstrate model training (if data is available)."""
    print("\n" + "=" * 50)
    print("Model Training Demo")
    print("=" * 50)
    
    # Check if we have training data
    stats = get_dataset_statistics()
    if stats['total_files'] < 5:
        print("Insufficient training data. Need at least 5 audio files across categories.")
        print("Please add more .wav files to the dataset directories.")
        return
    
    print("Training data available. Starting training demo...")
    
    try:
        # Import training module
        from train_model import load_dataset, train_and_evaluate_model
        
        # Load dataset
        features, labels, file_paths = load_dataset()
        
        if len(features) > 0:
            print(f"Loaded {len(features)} samples for training")
            
            # Train a simple model
            classifier, results = train_and_evaluate_model(features, labels, 'random_forest')
            
            if classifier and results:
                print("Training completed successfully!")
                print(f"Validation accuracy: {results['validation_accuracy']:.3f}")
                
                # Save model
                model_path = "models/demo_model.pkl"
                classifier.save_model(model_path)
                print(f"Model saved to: {model_path}")
            else:
                print("Training failed.")
        else:
            print("No features extracted from dataset.")
            
    except ImportError as e:
        print(f"Could not import training modules: {str(e)}")
    except Exception as e:
        print(f"Error during training demo: {str(e)}")


def demo_prediction():
    """Demonstrate prediction with a trained model."""
    print("\n" + "=" * 50)
    print("Prediction Demo")
    print("=" * 50)
    
    # Check if model exists
    model_path = "models/baby_cry_rf_model.pkl"
    if not os.path.exists(model_path):
        print("No trained model found. Please train a model first.")
        return
    
    # Load model
    try:
        classifier = BabyCryClassifier()
        classifier.load_model(model_path)
        print("Model loaded successfully!")
        
        # Get model info
        info = classifier.get_model_info()
        print(f"Model type: {info['model_type']}")
        print(f"Classes: {info['classes']}")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Find sample audio file
    sample_files = []
    for category in ['hungry', 'pain', 'need_to_change', 'tired', 'attention']:
        category_path = os.path.join('dataset', category)
        if os.path.exists(category_path):
            files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
            if files:
                sample_files.append((os.path.join(category_path, files[0]), category))
    
    if not sample_files:
        print("No sample audio files found for prediction demo.")
        return
    
    # Make prediction
    sample_file, true_category = sample_files[0]
    print(f"Making prediction for: {sample_file}")
    print(f"True category: {true_category}")
    
    try:
        processor = AudioProcessor()
        features = processor.process_file(sample_file)
        
        result = classifier.predict_single(features)
        
        print(f"Predicted need: {result['predicted_need']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Correct: {'✓' if result['predicted_need'] == true_category else '✗'}")
        
        print("\nAll probabilities:")
        for category, prob in result['all_probabilities'].items():
            print(f"  {category}: {prob:.3f}")
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")


def main():
    """Run all demos."""
    print("Baby Cry Analyzer - Demo Script")
    print("This script demonstrates the key features of the analyzer.")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Run demos
    demo_dataset_analysis()
    demo_audio_processing()
    demo_model_training()
    demo_prediction()
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    print("To use the API server, run: python app.py")
    print("To train a full model, run: python train_model.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
