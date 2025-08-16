#!/usr/bin/env python3
"""
Debug Feature Extraction - Check for NaN values and audio issues
"""

import numpy as np
from pathlib import Path
from audio_processor import AudioProcessor


def debug_feature_extraction():
    """Debug feature extraction to find NaN sources"""
    print("Baby Cry Analyzer - Feature Extraction Debug")
    print("=" * 50)
    
    processor = AudioProcessor()
    dataset_path = Path("dataset")
    categories = ["attention", "hungry", "need_to_change", "pain", "tired"]
    
    problematic_files = []
    successful_files = []
    nan_count = 0
    
    for category in categories:
        category_path = dataset_path / category
        if category_path.exists():
            audio_files = list(category_path.glob("*.wav"))[:3]  # Test only first 3 files per category
            print(f"\nTesting {category} category ({len(audio_files)} files):")
            
            for audio_file in audio_files:
                try:
                    print(f"  Processing: {audio_file.name}")
                    
                    # Load audio first
                    audio, sr = processor.load_audio(str(audio_file))
                    print(f"    Audio shape: {audio.shape}, Sample rate: {sr}")
                    print(f"    Audio stats: min={audio.min():.4f}, max={audio.max():.4f}, mean={audio.mean():.4f}")
                    
                    # Check for audio issues
                    if len(audio) == 0:
                        print(f"    ERROR: Empty audio file")
                        problematic_files.append((audio_file, "Empty audio"))
                        continue
                    
                    if np.all(audio == 0):
                        print(f"    ERROR: Silent audio file")
                        problematic_files.append((audio_file, "Silent audio"))
                        continue
                    
                    # Extract features
                    features = processor.extract_all_features(audio)
                    print(f"    Features shape: {features.shape}")
                    print(f"    Features stats: min={features.min():.4f}, max={features.max():.4f}")
                    
                    # Check for NaN values
                    nan_mask = np.isnan(features)
                    if nan_mask.any():
                        nan_indices = np.where(nan_mask)[0]
                        print(f"    ERROR: Found {nan_mask.sum()} NaN values at indices: {nan_indices}")
                        problematic_files.append((audio_file, f"NaN at indices {nan_indices}"))
                        nan_count += 1
                    else:
                        print(f"    SUCCESS: No NaN values found")
                        successful_files.append(audio_file)
                    
                    # Check for infinite values
                    inf_mask = np.isinf(features)
                    if inf_mask.any():
                        inf_indices = np.where(inf_mask)[0]
                        print(f"    ERROR: Found {inf_mask.sum()} infinite values at indices: {inf_indices}")
                        problematic_files.append((audio_file, f"Inf at indices {inf_indices}"))
                    
                except Exception as e:
                    print(f"    ERROR: Exception during processing: {e}")
                    problematic_files.append((audio_file, f"Exception: {e}"))
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Successful files: {len(successful_files)}")
    print(f"Problematic files: {len(problematic_files)}")
    print(f"Files with NaN: {nan_count}")
    
    if problematic_files:
        print("\nProblematic Files:")
        for file_path, issue in problematic_files:
            print(f"  {file_path.name}: {issue}")
    
    if successful_files:
        print("\nSample successful feature extraction:")
        try:
            sample_file = successful_files[0]
            audio, _ = processor.load_audio(str(sample_file))
            features = processor.extract_all_features(audio)
            
            print(f"  File: {sample_file.name}")
            print(f"  Feature vector length: {len(features)}")
            print(f"  Feature ranges: [{features.min():.4f}, {features.max():.4f}]")
            print(f"  Feature sample: {features[:10]}")
        except Exception as e:
            print(f"  Error getting sample: {e}")
    
    return len(problematic_files) == 0


if __name__ == "__main__":
    success = debug_feature_extraction()
    if success:
        print("\n✅ All features extracted successfully!")
        print("You can now run: python train_optimized_simple.py")
    else:
        print("\n❌ Found issues with feature extraction")
        print("Need to fix audio processing before training")
