"""
Setup and startup script for Baby Cry Analyzer.
This script helps users get started with the application.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        return True


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\nChecking dependencies...")
    required_packages = [
        'flask', 'librosa', 'numpy', 'pandas', 'scikit-learn',
        'tensorflow', 'soundfile', 'scipy', 'matplotlib', 'seaborn',
        'flask_cors', 'joblib', 'werkzeug'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    return missing_packages


def install_dependencies():
    """Install missing dependencies."""
    print("\nInstalling dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
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
        print(f"✅ {directory}")


def check_dataset():
    """Check if dataset has audio files."""
    print("\nChecking dataset...")
    categories = ['hungry', 'pain', 'need_to_change', 'tired', 'attention']
    total_files = 0
    
    for category in categories:
        category_path = Path(f'dataset/{category}')
        wav_files = list(category_path.glob('*.wav'))
        count = len(wav_files)
        total_files += count
        status = "✅" if count > 0 else "⚠️"
        print(f"{status} {category}: {count} files")
    
    if total_files == 0:
        print("\n⚠️  No audio files found in dataset.")
        print("Please add .wav files to the category folders in the dataset directory.")
        return False
    elif total_files < 10:
        print(f"\n⚠️  Only {total_files} audio files found.")
        print("For better model performance, add more audio files (recommended: 20+ per category).")
        return True
    else:
        print(f"\n✅ Found {total_files} audio files total.")
        return True


def check_model():
    """Check if trained model exists."""
    print("\nChecking for trained model...")
    model_files = ['models/baby_cry_rf_model.pkl', 'models/baby_cry_svm_model.pkl']
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"✅ Found: {model_file}")
            return True
    
    print("⚠️  No trained model found.")
    print("Run 'python train_model.py' to train a model first.")
    return False


def show_next_steps():
    """Show next steps to the user."""
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    
    print("\n1. Add training data:")
    print("   - Place .wav files in dataset/<category>/ folders")
    print("   - Categories: hungry, pain, need_to_change, tired, attention")
    print("   - Recommended: 20+ files per category")
    
    print("\n2. Train the model:")
    print("   python train_model.py")
    
    print("\n3. Start the API server:")
    print("   python app.py")
    
    print("\n4. Test the API:")
    print("   python test_api.py --audio path/to/test_file.wav")
    
    print("\n5. Access the web interface:")
    print("   Open http://localhost:5000 in your browser")
    
    print("\nFor a quick demo (if you have audio files):")
    print("   python demo.py")


def main():
    """Main setup function."""
    print("🍼 Baby Cry Analyzer - Setup Script")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Installing dependencies...")
        if not install_dependencies():
            print("❌ Setup failed. Please install dependencies manually.")
            return
        print("\nRe-checking dependencies...")
        missing = check_dependencies()
        if missing:
            print(f"❌ Still missing: {', '.join(missing)}")
            return
    
    # Create directories
    create_directories()
    
    # Check dataset
    has_data = check_dataset()
    
    # Check model
    has_model = check_model()
    
    # Show status
    print("\n" + "="*60)
    print("SETUP COMPLETE")
    print("="*60)
    print(f"✅ Python: {sys.version_info.major}.{sys.version_info.minor}")
    print("✅ Dependencies: Installed")
    print("✅ Directories: Created")
    print(f"{'✅' if has_data else '⚠️ '} Dataset: {'Ready' if has_data else 'Needs audio files'}")
    print(f"{'✅' if has_model else '⚠️ '} Model: {'Ready' if has_model else 'Needs training'}")
    
    # Show next steps
    show_next_steps()
    
    if has_data and has_model:
        print("\n🎉 Everything is ready! You can start the API server now.")
    elif has_data:
        print("\n📊 Ready to train model with your data!")
    else:
        print("\n📁 Please add audio files to get started.")


if __name__ == "__main__":
    main()
