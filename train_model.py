import os
import glob
import numpy as np
from audio_processor import AudioProcessor
from model import BabyCryClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def create_sample_dataset():
    """Create sample dataset structure if it doesn't exist."""
    dataset_path = "dataset"
    categories = ['hungry', 'pain', 'need_to_change', 'tired']
    
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        print(f"Created dataset directory: {dataset_path}")
    
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)
            print(f"Created category directory: {category_path}")
    
    print("Dataset structure created. Please add .wav files to the respective category folders.")


def load_dataset(dataset_path: str = "dataset"):
    """
    Load audio files from dataset directory and extract features.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Tuple of (features, labels, file_paths)
    """
    if not os.path.exists(dataset_path):
        create_sample_dataset()
        return [], [], []
    
    processor = AudioProcessor()
    features = []
    labels = []
    file_paths = []
    
    categories = ['hungry', 'pain', 'need_to_change', 'tired']
    
    print("Loading dataset and extracting features...")
    
    total_files = 0
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        if os.path.exists(category_path):
            wav_files = glob.glob(os.path.join(category_path, "*.wav"))
            total_files += len(wav_files)
    
    if total_files == 0:
        print("No .wav files found in dataset directories.")
        print("Please add audio files to the following directories:")
        for category in categories:
            print(f"  - {os.path.join(dataset_path, category)}/")
        return [], [], []
    
    processed_files = 0
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        if os.path.exists(category_path):
            wav_files = glob.glob(os.path.join(category_path, "*.wav"))
            
            print(f"Processing {len(wav_files)} files from category: {category}")
            
            for file_path in wav_files:
                try:
                    # Extract features
                    file_features = processor.process_file(file_path)
                    features.append(file_features)
                    labels.append(category)
                    file_paths.append(file_path)
                    processed_files += 1
                    
                    if processed_files % 10 == 0:
                        print(f"Processed {processed_files}/{total_files} files...")
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    continue
    
    print(f"Successfully processed {len(features)} audio files")
    return features, labels, file_paths


def train_and_evaluate_model(features, labels, model_type='random_forest'):
    """
    Train and evaluate the model.
    
    Args:
        features: List of feature vectors
        labels: List of labels
        model_type: Type of model to train
        
    Returns:
        Trained model and evaluation results
    """
    if len(features) == 0:
        print("No features available for training")
        return None, None
    
    print(f"\nTraining {model_type} model...")
    
    # Create classifier
    classifier = BabyCryClassifier(model_type=model_type)
    
    # Prepare data
    X, y = classifier.prepare_data(features, labels)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for class_name, count in zip(unique, counts):
        print(f"  {class_name}: {count} samples")
    
    # Train model
    results = classifier.train(X, y)
    
    print("\nTraining Results:")
    print(f"Training Accuracy: {results['train_accuracy']:.4f}")
    print(f"Validation Accuracy: {results['validation_accuracy']:.4f}")
    print(f"Cross-validation Score: {results['cv_mean']:.4f} (+/- {results['cv_std']*2:.4f})")
    
    print("\nClassification Report:")
    print(results['classification_report'])
    
    return classifier, results


def save_training_report(classifier, results, model_type):
    """Save training report to file."""
    reports_dir = "reports"
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(reports_dir, f"training_report_{model_type}_{timestamp}.txt")
    
    with open(report_path, 'w') as f:
        f.write(f"Baby Cry Analyzer - Training Report\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Training Accuracy: {results['train_accuracy']:.4f}\n")
        f.write(f"Validation Accuracy: {results['validation_accuracy']:.4f}\n")
        f.write(f"Cross-validation Score: {results['cv_mean']:.4f} (+/- {results['cv_std']*2:.4f})\n\n")
        
        f.write("Classification Report:\n")
        f.write(results['classification_report'])
        f.write("\n")
        
        if results['feature_importance'] is not None:
            f.write("Feature Importance (top 10):\n")
            importance = results['feature_importance']
            indices = np.argsort(importance)[::-1][:10]
            for i, idx in enumerate(indices):
                f.write(f"{i+1}. Feature {idx}: {importance[idx]:.4f}\n")
    
    print(f"Training report saved to: {report_path}")


def plot_confusion_matrix(confusion_matrix, classes, model_type):
    """Plot and save confusion matrix."""
    reports_dir = "reports"
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_type.title()} Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(reports_dir, f"confusion_matrix_{model_type}_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {plot_path}")


def main():
    """Main training function."""
    print("Baby Cry Analyzer - Model Training")
    print("=" * 40)
    
    # Load dataset
    features, labels, file_paths = load_dataset()
    
    if len(features) == 0:
        print("No training data available. Please add .wav files to the dataset directories.")
        return
    
    # Train Random Forest model
    print("\n" + "="*40)
    print("Training Random Forest Model")
    print("="*40)
    
    rf_classifier, rf_results = train_and_evaluate_model(features, labels, 'random_forest')
    
    if rf_classifier is not None:
        # Save model
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        rf_model_path = os.path.join(models_dir, "baby_cry_rf_model.pkl")
        rf_classifier.save_model(rf_model_path)
        
        # Save reports
        save_training_report(rf_classifier, rf_results, 'random_forest')
        plot_confusion_matrix(rf_results['confusion_matrix'], 
                            rf_classifier.label_encoder.classes_, 'random_forest')
    
    # Train SVM model
    print("\n" + "="*40)
    print("Training SVM Model")
    print("="*40)
    
    svm_classifier, svm_results = train_and_evaluate_model(features, labels, 'svm')
    
    if svm_classifier is not None:
        svm_model_path = os.path.join(models_dir, "baby_cry_svm_model.pkl")
        svm_classifier.save_model(svm_model_path)
        
        # Save reports
        save_training_report(svm_classifier, svm_results, 'svm')
        plot_confusion_matrix(svm_results['confusion_matrix'], 
                            svm_classifier.label_encoder.classes_, 'svm')
    
    print("\n" + "="*40)
    print("Training completed!")
    print("Models saved in 'models/' directory")
    print("Reports saved in 'reports/' directory")
    print("="*40)


if __name__ == "__main__":
    main()
