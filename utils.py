"""
Utility functions for the Baby Cry Analyzer.
"""

import os
import numpy as np
import librosa
from typing import List, Dict, Any
import json
from datetime import datetime


def create_directory_structure():
    """Create the required directory structure for the project."""
    directories = [
        'dataset/hungry',
        'dataset/pain',
        'dataset/need_to_change',
        'dataset/tired',
        'dataset/attention',
        'models',
        'uploads',
        'reports',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def validate_audio_file(file_path: str) -> Dict[str, Any]:
    """
    Validate an audio file and return its properties.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Dictionary with file validation results
    """
    validation_result = {
        'is_valid': False,
        'error': None,
        'properties': {}
    }
    
    try:
        if not os.path.exists(file_path):
            validation_result['error'] = "File does not exist"
            return validation_result
        
        # Load audio file
        audio, sr = librosa.load(file_path, sr=None)
        
        # Get file properties
        duration = len(audio) / sr
        file_size = os.path.getsize(file_path)
        
        validation_result['properties'] = {
            'duration': duration,
            'sample_rate': sr,
            'num_samples': len(audio),
            'file_size_bytes': file_size,
            'file_size_mb': file_size / (1024 * 1024),
            'channels': 1 if audio.ndim == 1 else audio.shape[0]
        }
        
        # Validation checks
        if duration < 0.5:
            validation_result['error'] = "Audio too short (minimum 0.5 seconds)"
        elif duration > 10.0:
            validation_result['error'] = "Audio too long (maximum 10 seconds)"
        elif sr < 8000:
            validation_result['error'] = "Sample rate too low (minimum 8kHz)"
        else:
            validation_result['is_valid'] = True
            
    except Exception as e:
        validation_result['error'] = f"Error loading audio file: {str(e)}"
    
    return validation_result


def get_dataset_statistics(dataset_path: str = "dataset") -> Dict[str, Any]:
    """
    Get statistics about the dataset.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'total_files': 0,
        'categories': {},
        'total_duration': 0.0,
        'average_duration': 0.0,
        'sample_rates': {},
        'errors': []
    }
    
    categories = ['hungry', 'pain', 'need_to_change', 'tired', 'attention']
    
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        category_stats = {
            'count': 0,
            'duration': 0.0,
            'files': []
        }
        
        if os.path.exists(category_path):
            wav_files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
            category_stats['count'] = len(wav_files)
            
            for wav_file in wav_files:
                file_path = os.path.join(category_path, wav_file)
                try:
                    audio, sr = librosa.load(file_path, sr=None)
                    duration = len(audio) / sr
                    category_stats['duration'] += duration
                    category_stats['files'].append({
                        'name': wav_file,
                        'duration': duration,
                        'sample_rate': sr
                    })
                    
                    # Track sample rates
                    if sr not in stats['sample_rates']:
                        stats['sample_rates'][sr] = 0
                    stats['sample_rates'][sr] += 1
                    
                except Exception as e:
                    stats['errors'].append(f"Error processing {file_path}: {str(e)}")
        
        stats['categories'][category] = category_stats
        stats['total_files'] += category_stats['count']
        stats['total_duration'] += category_stats['duration']
    
    if stats['total_files'] > 0:
        stats['average_duration'] = stats['total_duration'] / stats['total_files']
    
    return stats


def save_prediction_log(prediction_result: Dict[str, Any], file_info: Dict[str, Any]):
    """
    Save prediction results to a log file.
    
    Args:
        prediction_result: The prediction results
        file_info: Information about the processed file
    """
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'prediction': prediction_result,
        'file_info': file_info
    }
    
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    log_file = os.path.join(logs_dir, "predictions.jsonl")
    
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


def analyze_prediction_logs(log_file: str = "logs/predictions.jsonl") -> Dict[str, Any]:
    """
    Analyze prediction logs to get insights.
    
    Args:
        log_file: Path to the log file
        
    Returns:
        Analysis results
    """
    if not os.path.exists(log_file):
        return {'error': 'No prediction logs found'}
    
    predictions = []
    
    with open(log_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                predictions.append(entry)
            except json.JSONDecodeError:
                continue
    
    if not predictions:
        return {'error': 'No valid predictions found in logs'}
    
    # Analyze predictions
    categories = {}
    confidence_scores = []
    
    for pred in predictions:
        category = pred['prediction']['predicted_need']
        confidence = pred['prediction']['confidence']
        
        if category not in categories:
            categories[category] = {'count': 0, 'avg_confidence': 0, 'confidences': []}
        
        categories[category]['count'] += 1
        categories[category]['confidences'].append(confidence)
        confidence_scores.append(confidence)
    
    # Calculate averages
    for category in categories:
        confidences = categories[category]['confidences']
        categories[category]['avg_confidence'] = np.mean(confidences)
        categories[category]['min_confidence'] = np.min(confidences)
        categories[category]['max_confidence'] = np.max(confidences)
        del categories[category]['confidences']  # Remove raw data
    
    analysis = {
        'total_predictions': len(predictions),
        'categories': categories,
        'overall_avg_confidence': np.mean(confidence_scores),
        'overall_min_confidence': np.min(confidence_scores),
        'overall_max_confidence': np.max(confidence_scores),
        'date_range': {
            'first': predictions[0]['timestamp'] if predictions else None,
            'last': predictions[-1]['timestamp'] if predictions else None
        }
    }
    
    return analysis


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"


def get_model_performance_summary(results: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary of model performance.
    
    Args:
        results: Training results dictionary
        
    Returns:
        Formatted performance summary
    """
    summary = f"""
Model Performance Summary:
=========================
Training Accuracy: {results['train_accuracy']:.1%}
Validation Accuracy: {results['validation_accuracy']:.1%}
Cross-Validation Score: {results['cv_mean']:.1%} (±{results['cv_std']*2:.1%})

Performance Interpretation:
"""
    
    val_acc = results['validation_accuracy']
    if val_acc >= 0.85:
        summary += "🟢 Excellent performance - Model is ready for production use\n"
    elif val_acc >= 0.75:
        summary += "🟡 Good performance - Model is suitable for general use\n"
    elif val_acc >= 0.65:
        summary += "🟠 Fair performance - Consider improving dataset or features\n"
    else:
        summary += "🔴 Poor performance - Significant improvements needed\n"
    
    summary += f"\nNote: Results based on {len(results.get('confusion_matrix', []))} categories"
    
    return summary
