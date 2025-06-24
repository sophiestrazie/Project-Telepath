# tests/test_imports.py
"""
Test script to verify all imports work correctly.
"""

import sys
import traceback

def test_import(module_path, class_name=None):
    try:
        if class_name:
            exec(f"from {module_path} import {class_name}")
            print(f"✓ Successfully imported {class_name} from {module_path}")
        else:
            exec(f"import {module_path}")
            print(f"✓ Successfully imported {module_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to import {class_name or module_path} from {module_path}: {e}")
        return False

def main():
    print("Testing all project imports...")
    print("=" * 50)
    
    imports_to_test = [
        ("models.base_classifier", "BaseClassifier"),
        ("models.classical.svm", "SVMClassifier"),
        ("models.classical.random_forest", "RandomForestClassifier"),
        ("models.neural.mlp", "MLPClassifier"),
        ("data.loaders", "MultimodalDataLoader"),
        ("numpy", None),
        ("sklearn.svm", "SVC"),
        ("torch", None),
        ("torch.nn", None),
    ]
    
    success_count = 0
    total_count = len(imports_to_test)
    
    for module_path, class_name in imports_to_test:
        if test_import(module_path, class_name):
            success_count += 1
    
    print("=" * 50)
    print(f"Import test results: {success_count}/{total_count} successful")
    
    if success_count == total_count:
        print("🎉 All imports working correctly!")
        return 0
    else:
        print("❌ Some imports failed. Check your Python path and dependencies.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
