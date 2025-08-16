"""
Sample data generator for testing the Baby Cry Analyzer.
This script generates synthetic audio samples for testing purposes.
WARNING: These are not real baby cries - for testing only!
"""

import numpy as np
import soundfile as sf
import os
from scipy import signal
import random


def generate_synthetic_cry(duration=3.0, sample_rate=22050, cry_type='hungry'):
    """
    Generate a synthetic baby cry sound for testing purposes.
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        cry_type: Type of cry to simulate
        
    Returns:
        Generated audio signal
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Base frequency patterns for different cry types
    base_patterns = {
        'hungry': {'freq_range': (300, 600), 'pattern': 'rising'},
        'pain': {'freq_range': (400, 800), 'pattern': 'sharp'},
        'tired': {'freq_range': (200, 400), 'pattern': 'falling'},
        'attention': {'freq_range': (250, 500), 'pattern': 'rhythmic'},
        'need_to_change': {'freq_range': (350, 550), 'pattern': 'irregular'}
    }
    
    pattern = base_patterns.get(cry_type, base_patterns['hungry'])
    freq_min, freq_max = pattern['freq_range']
    
    # Generate base frequency modulation
    if pattern['pattern'] == 'rising':
        freq = np.linspace(freq_min, freq_max, len(t))
    elif pattern['pattern'] == 'falling':
        freq = np.linspace(freq_max, freq_min, len(t))
    elif pattern['pattern'] == 'sharp':
        freq = np.full(len(t), freq_max)
        # Add sharp spikes
        spike_positions = np.random.choice(len(t), size=int(len(t) * 0.1), replace=False)
        freq[spike_positions] = freq[spike_positions] * 1.5
    elif pattern['pattern'] == 'rhythmic':
        freq = freq_min + (freq_max - freq_min) * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t))
    else:  # irregular
        freq = freq_min + (freq_max - freq_min) * np.random.random(len(t))
        freq = signal.savgol_filter(freq, min(len(freq)//10 + 1, 51), 3)  # Smooth
    
    # Generate the base signal
    phase = np.cumsum(2 * np.pi * freq / sample_rate)
    audio = np.sin(phase)
    
    # Add harmonics
    audio += 0.3 * np.sin(2 * phase)  # Second harmonic
    audio += 0.1 * np.sin(3 * phase)  # Third harmonic
    
    # Add formants (resonant frequencies)
    formant1 = 1000 + 200 * np.sin(2 * np.pi * 0.5 * t)
    formant2 = 2000 + 300 * np.sin(2 * np.pi * 0.3 * t)
    
    # Apply formant filtering (simplified)
    b1, a1 = signal.butter(4, formant1/(sample_rate/2), btype='band')
    b2, a2 = signal.butter(4, formant2/(sample_rate/2), btype='band')
    
    try:
        formant_signal1 = signal.filtfilt(b1, a1, audio)
        formant_signal2 = signal.filtfilt(b2, a2, audio)
        audio += 0.2 * formant_signal1 + 0.1 * formant_signal2
    except:
        pass  # Skip formant filtering if it fails
    
    # Add noise for realism
    noise = np.random.normal(0, 0.05, len(t))
    audio += noise
    
    # Apply amplitude envelope (cry pattern)
    if cry_type in ['pain', 'hungry']:
        # Sharp attack, sustained, gradual decay
        envelope = np.ones_like(t)
        attack_samples = int(0.1 * sample_rate)
        decay_samples = int(0.3 * sample_rate)
        
        # Attack
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        # Decay
        envelope[-decay_samples:] = np.linspace(1, 0.2, decay_samples)
    else:
        # Gradual build-up and fade
        envelope = np.exp(-((t - duration/2) / (duration/4))**2)
    
    audio *= envelope
    
    # Add breathing gaps (realistic pauses)
    gap_probability = 0.1
    gap_mask = np.random.random(len(t)) > gap_probability
    
    # Smooth the gaps
    gap_mask = signal.savgol_filter(gap_mask.astype(float), min(len(gap_mask)//20 + 1, 51), 3)
    audio *= gap_mask
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio


def generate_sample_dataset(samples_per_category=5):
    """
    Generate a sample dataset for testing.
    
    Args:
        samples_per_category: Number of samples to generate per category
    """
    print("Generating synthetic baby cry samples for testing...")
    print("WARNING: These are NOT real baby cries - for testing only!")
    
    categories = ['hungry', 'pain', 'need_to_change', 'tired', 'attention']
    
    for category in categories:
        category_path = f"dataset/{category}"
        os.makedirs(category_path, exist_ok=True)
        
        print(f"Generating {samples_per_category} samples for '{category}'...")
        
        for i in range(samples_per_category):
            # Vary duration slightly
            duration = random.uniform(2.5, 4.0)
            
            # Generate audio
            audio = generate_synthetic_cry(duration=duration, cry_type=category)
            
            # Save to file
            filename = f"synthetic_{category}_{i+1:02d}.wav"
            filepath = os.path.join(category_path, filename)
            
            sf.write(filepath, audio, 22050)
            print(f"  Created: {filename}")
    
    print(f"\nGenerated {len(categories) * samples_per_category} synthetic audio samples.")
    print("You can now train the model using: python train_model.py")


def generate_test_file(filename="test_cry.wav", cry_type="hungry"):
    """
    Generate a single test file.
    
    Args:
        filename: Output filename
        cry_type: Type of cry to generate
    """
    print(f"Generating test file: {filename} (type: {cry_type})")
    
    audio = generate_synthetic_cry(duration=3.0, cry_type=cry_type)
    sf.write(filename, audio, 22050)
    
    print(f"Test file created: {filename}")
    print(f"You can test it with: python test_api.py --audio {filename}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic baby cry samples for testing")
    parser.add_argument('--samples', type=int, default=5, help='Samples per category')
    parser.add_argument('--test-file', help='Generate a single test file')
    parser.add_argument('--type', default='hungry', choices=['hungry', 'pain', 'need_to_change', 'tired', 'attention'],
                       help='Type of cry for test file')
    
    args = parser.parse_args()
    
    if args.test_file:
        generate_test_file(args.test_file, args.type)
    else:
        generate_sample_dataset(args.samples)


if __name__ == "__main__":
    main()
