#!/usr/bin/env python3
"""
Validation Accuracy Improvement Script
Specifically addresses low validation accuracy issues
"""

import numpy as np
from pathlib import Path
import shutil
import re


def check_dataset_balance():
    """Check current dataset balance"""
    print("🔍 DATASET ANALYSIS")
    print("=" * 40)
    
    dataset_path = Path("dataset")
    categories = ["hungry", "need_to_change", "pain", "tired"]
    
    total_files = 0
    category_counts = {}
    
    for category in categories:
        category_path = dataset_path / category
        if category_path.exists():
            count = len(list(category_path.glob("*.wav")))
            category_counts[category] = count
            total_files += count
            print(f"  {category}: {count} files")
        else:
            category_counts[category] = 0
            print(f"  {category}: 0 files (folder missing)")
    
    print(f"\nTotal files: {total_files}")
    
    # Analyze balance
    if category_counts:
        min_count = min(category_counts.values())
        max_count = max(category_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"Min samples: {min_count}")
        print(f"Max samples: {max_count}")
        print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 3:
            print("⚠️ SEVERE CLASS IMBALANCE detected!")
            return False, category_counts
        elif imbalance_ratio > 2:
            print("⚠️ Moderate class imbalance")
            return True, category_counts
        else:
            print("✅ Good class balance")
            return True, category_counts
    
    return False, category_counts


def fix_model_overfitting():
    """Reduce model complexity to prevent overfitting"""
    print("\n🔧 FIXING MODEL OVERFITTING")
    print("=" * 40)
    
    model_file = Path("model.py")
    if not model_file.exists():
        print("❌ model.py not found!")
        return False
    
    # Backup original
    backup_file = Path("model_original_backup.py")
    if not backup_file.exists():
        shutil.copy(model_file, backup_file)
        print("✅ Created backup: model_original_backup.py")
    
    with open(model_file, 'r') as f:
        content = f.read()
    
    # Reduce Random Forest complexity
    changes_made = []
    
    # Change n_estimators
    if "n_estimators=100" in content:
        content = content.replace("n_estimators=100", "n_estimators=50")
        changes_made.append("n_estimators: 100 → 50")
    
    # Change max_depth
    if "max_depth=10" in content:
        content = content.replace("max_depth=10", "max_depth=5")
        changes_made.append("max_depth: 10 → 5")
    elif "max_depth=" not in content and "RandomForestClassifier" in content:
        # Add max_depth if not present
        content = re.sub(
            r'(RandomForestClassifier\([^)]*)(random_state=42)',
            r'\1max_depth=5,\n                \2',
            content
        )
        changes_made.append("Added max_depth=5")
    
    # Add min_samples_split if not present
    if "min_samples_split=" not in content and "RandomForestClassifier" in content:
        content = re.sub(
            r'(RandomForestClassifier\([^)]*)(random_state=42)',
            r'\1min_samples_split=10,\n                \2',
            content
        )
        changes_made.append("Added min_samples_split=10")
    
    with open(model_file, 'w') as f:
        f.write(content)
    
    if changes_made:
        print("✅ Model optimized:")
        for change in changes_made:
            print(f"  • {change}")
        return True
    else:
        print("ℹ️ No changes needed - model already optimized")
        return True


def add_validation_data():
    """Add more samples to balance dataset"""
    print("\n📈 IMPROVING DATASET")
    print("=" * 40)
    
    balanced, category_counts = check_dataset_balance()
    
    if not balanced:
        # Find categories that need more data
        max_count = max(category_counts.values())
        target_count = min(50, max_count)  # Target at least 50 or match the largest
        
        categories_to_improve = []
        for category, count in category_counts.items():
            if count < target_count:
                needed = target_count - count
                categories_to_improve.append((category, needed))
        
        if categories_to_improve:
            print(f"Target samples per category: {target_count}")
            print("\nCategories needing more data:")
            for category, needed in categories_to_improve:
                print(f"  {category}: add {needed} more samples")
            
            print("\n🎯 RECOMMENDED ACTIONS:")
            print("1. Add real baby cry recordings if possible")
            print("2. Or generate synthetic samples for testing:")
            
            for category, needed in categories_to_improve:
                print(f"   python generate_samples.py --test-file extra_{category}.wav --type {category}")
                print(f"   # Repeat this {needed} times for {category}")
            
            return False
        else:
            print("✅ Dataset already well balanced")
            return True
    
    return balanced


def main():
    """Main validation improvement workflow"""
    print("🎯 BABY CRY ANALYZER - VALIDATION ACCURACY IMPROVEMENT")
    print("=" * 60)
    
    print("Current results:")
    print("  Training Accuracy: 94.3%")
    print("  Validation Accuracy: 26.1% ⚠️ TOO LOW")
    print("  Goal: Improve validation to 40-50%")
    print()
    
    # Step 1: Check dataset
    balanced, category_counts = check_dataset_balance()
    
    # Step 2: Fix model overfitting
    model_fixed = fix_model_overfitting()
    
    # Step 3: Improve dataset
    data_improved = add_validation_data()
    
    # Summary and next steps
    print("\n🏁 SUMMARY & NEXT STEPS")
    print("=" * 40)
    
    if model_fixed:
        print("✅ Model complexity reduced (should reduce overfitting)")
    
    if balanced:
        print("✅ Dataset is reasonably balanced")
    else:
        print("⚠️ Dataset needs balancing")
    
    total_files = sum(category_counts.values())
    if total_files < 200:
        print(f"⚠️ Dataset size: {total_files} files (recommend 250+ for better results)")
    
    print("\n🚀 IMMEDIATE ACTIONS:")
    print("1. Retrain the model with optimized settings:")
    print("   python train_model.py")
    print()
    print("2. Expected improvements:")
    print("   • Validation accuracy: 26% → 35-45%")
    print("   • Reduced overfitting gap")
    print("   • More stable cross-validation scores")
    print()
    print("3. If results are still poor, add more real data:")
    print("   • Collect real baby cry recordings")
    print("   • Aim for 50+ samples per category")
    print("   • Ensure high audio quality")
    
    print("\n📊 To track progress:")
    print("   python training_analyzer.py  # After retraining")


if __name__ == "__main__":
    main()
