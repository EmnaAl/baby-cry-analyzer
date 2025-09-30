#!/usr/bin/env python3
"""
Project Structure Checker and Cleanup Tool
Identifies and removes unused files from the baby cry analyzer project
"""

import os
from pathlib import Path

def check_project_structure():
    """Check current project structure and identify files"""
    print("🔍 CURRENT PROJECT STRUCTURE")
    print("=" * 50)
    
    project_files = {}
    total_size = 0
    
    for root, dirs, files in os.walk("."):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if not file.startswith('.') and not file.endswith('.pyc'):
                file_path = Path(root) / file
                relative_path = file_path.relative_to('.')
                
                try:
                    file_size = file_path.stat().st_size
                    total_size += file_size
                except:
                    file_size = 0
                
                # Categorize files
                if file.endswith('.py'):
                    category = 'Python Scripts'
                elif file.endswith('.md'):
                    category = 'Documentation'
                elif file.endswith('.txt'):
                    category = 'Text Files'
                elif file.endswith('.wav'):
                    category = 'Audio Files'
                elif file.endswith(('.png', '.jpg', '.jpeg')):
                    category = 'Images/Plots'
                elif file.endswith(('.pkl', '.joblib')):
                    category = 'Model Files'
                elif file.endswith('.json'):
                    category = 'Config/Data Files'
                else:
                    category = 'Other Files'
                
                if category not in project_files:
                    project_files[category] = []
                project_files[category].append((str(relative_path), file_size))
    
    # Display structure
    for category, files in project_files.items():
        total_category_size = sum(size for _, size in files)
        print(f"\n📁 {category} ({len(files)} files, {total_category_size:,} bytes):")
        for file_path, size in sorted(files):
            print(f"  {file_path} ({size:,} bytes)")
    
    print(f"\n📊 TOTAL PROJECT SIZE: {total_size:,} bytes")
    return project_files

def identify_unused_files():
    """Identify files that are likely unused and can be removed"""
    print("\n🗑️  UNUSED FILES ANALYSIS")
    print("=" * 50)
    
    # Core required files (should NOT be removed)
    core_files = {
        'app.py',
        'model.py', 
        'audio_processor.py',
        'train_model.py',
        'config.py',
        'utils.py',
        'requirements.txt',
        'README.md',
        'QUICKSTART.md',
        'test_api.py',
        'generate_samples.py'
    }
    
    # Files that are definitely unused (development artifacts)
    unused_patterns = [
        'model_original_backup',
        'train_optimized',
        'train_4category',
        'simple_demo',
        'generate_dataset_inquiry',
        'integrate_premium_dataset',
        'dataset_downloader',
        'training_analyzer',
        'improve_validation',
        'check_dataset',
        'debug_features',
        'optimize_model',
        'conversion_summary',
        '_backup',
        'start.bat',
        'setup.py'
    ]
    
    potentially_unused = []
    
    # Check all Python files in current directory
    for file_path in Path('.').glob('*.py'):
        filename = file_path.name
        
        # Skip core files
        if filename in core_files:
            continue
            
        # Check if matches unused patterns
        if any(pattern in filename for pattern in unused_patterns):
            potentially_unused.append(file_path)
    
    # Check for other unused file types
    for pattern in ['*.bat', 'setup.py']:
        for file_path in Path('.').glob(pattern):
            if file_path not in potentially_unused:
                potentially_unused.append(file_path)
    
    print("Files identified for removal:")
    total_unused_size = 0
    for file_path in potentially_unused:
        if file_path.exists():
            file_size = file_path.stat().st_size
            total_unused_size += file_size
            print(f"  ✓ {file_path} ({file_size:,} bytes)")
        else:
            print(f"  ⚠ {file_path} (not found)")
    
    print(f"\nTotal size to be freed: {total_unused_size:,} bytes")
    return potentially_unused

def remove_unused_files(files_to_remove):
    """Remove unused files"""
    if not files_to_remove:
        print("✅ No unused files to remove!")
        return
    
    print(f"\n🗑️  REMOVING {len(files_to_remove)} UNUSED FILES")
    print("=" * 50)
    
    removed_count = 0
    failed_count = 0
    total_freed = 0
    
    for file_path in files_to_remove:
        try:
            if file_path.exists():
                file_size = file_path.stat().st_size
                file_path.unlink()
                print(f"✅ Removed: {file_path} ({file_size:,} bytes)")
                removed_count += 1
                total_freed += file_size
            else:
                print(f"⚠️  Not found: {file_path}")
        except Exception as e:
            print(f"❌ Failed to remove {file_path}: {e}")
            failed_count += 1
    
    print(f"\n📊 CLEANUP SUMMARY:")
    print(f"  ✅ Removed: {removed_count} files")
    print(f"  ❌ Failed: {failed_count} files")
    print(f"  💾 Space freed: {total_freed:,} bytes")

def show_final_structure():
    """Show the recommended final project structure"""
    print("\n🎯 RECOMMENDED FINAL PROJECT STRUCTURE")
    print("=" * 50)
    
    recommended_files = {
        'Core Application': [
            'app.py',
            'model.py', 
            'audio_processor.py',
            'train_model.py'
        ],
        'Utilities & Testing': [
            'config.py',
            'utils.py',
            'test_api.py',
            'generate_samples.py'
        ],
        'Documentation': [
            'README.md',
            'QUICKSTART.md',
            'requirements.txt'
        ]
    }
    
    recommended_dirs = [
        'dataset/hungry/',
        'dataset/need_to_change/',
        'dataset/pain/', 
        'dataset/tired/',
        'models/',
        'uploads/',
        'reports/',
        'logs/'
    ]
    
    print("\n📁 FILES:")
    for category, files in recommended_files.items():
        print(f"\n  {category}:")
        for file in files:
            status = "✅" if Path(file).exists() else "❌"
            print(f"    {status} {file}")
    
    print("\n📁 DIRECTORIES:")
    for dir_path in recommended_dirs:
        dir_obj = Path(dir_path)
        if dir_obj.exists():
            file_count = len(list(dir_obj.glob('*')))
            print(f"  ✅ {dir_path} ({file_count} files)")
        else:
            print(f"  ❌ {dir_path} (missing)")

def main():
    print("🧹 BABY CRY ANALYZER - PROJECT CLEANUP")
    print("=" * 60)
    
    # Check current structure
    project_files = check_project_structure()
    
    # Identify unused files
    unused_files = identify_unused_files()
    
    # Show final recommended structure
    show_final_structure()
    
    # Perform cleanup
    if unused_files:
        print(f"\n🤔 Found {len(unused_files)} unused development files.")
        print("These are backup and testing files that can be safely removed.")
        
        choice = input("\nRemove unused files? (y/N): ").lower().strip()
        if choice == 'y':
            remove_unused_files(unused_files)
            print("\n✨ Cleanup complete! Project is now cleaner.")
        else:
            print("\n📝 Files preserved. You can remove them manually later if needed.")
    else:
        print("\n✅ No unused files found. Project is already clean!")

if __name__ == "__main__":
    main()
