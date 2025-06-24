# PowerShell script to fix inconsistent import paths in fMRI prediction project
# This script standardizes all import statements to use absolute imports from project root

Write-Host "Starting import path standardization..." -ForegroundColor Green

# Define the project root directory (adjust as needed)
$ProjectRoot = Get-Location

# Define file patterns to process
$PythonFiles = Get-ChildItem -Path $ProjectRoot -Recurse -Include "*.py" -Exclude "__pycache__"

# Define import replacements
$ImportReplacements = @{
    "from ..base_classifier import BaseClassifier" = "from models.base_classifier import BaseClassifier"
    "from ...base_classifier import BaseClassifier" = "from models.base_classifier import BaseClassifier"
    "from base_classifier import BaseClassifier" = "from models.base_classifier import BaseClassifier"
    "from ..classical." = "from models.classical."
    "from ..neural." = "from models.neural."
    "from ...evaluation." = "from evaluation."
    "from ..evaluation." = "from evaluation."
    "from ..data." = "from data."
    "from ...data." = "from data."
}

Write-Host "Found $($PythonFiles.Count) Python files to process" -ForegroundColor Yellow

foreach ($File in $PythonFiles) {
    Write-Host "Processing: $($File.Name)" -ForegroundColor Cyan
    
    $Content = Get-Content -Path $File.FullName -Raw
    $OriginalContent = $Content
    $Modified = $false
    
    # Apply each replacement
    foreach ($OldImport in $ImportReplacements.Keys) {
        $NewImport = $ImportReplacements[$OldImport]
        
        if ($Content -match [regex]::Escape($OldImport)) {
            $Content = $Content -replace [regex]::Escape($OldImport), $NewImport
            Write-Host "  ✓ Replaced: $OldImport → $NewImport" -ForegroundColor Green
            $Modified = $true
        }
    }
    
    # Check for missing imports in neural network files
    if ($File.DirectoryName -like "*neural*") {
        $RequiredImports = @(
            "import torch",
            "import torch.nn as nn",
            "import torch.optim as optim",
            "from torch.utils.data import DataLoader, TensorDataset"
        )
        
        foreach ($RequiredImport in $RequiredImports) {
            if ($Content -notmatch [regex]::Escape($RequiredImport) -and 
                $Content -match "torch\.") {
                
                # Find the first import line and add the missing import
                $Lines = $Content -split "`n"
                $FirstImportIndex = -1
                
                for ($i = 0; $i -lt $Lines.Count; $i++) {
                    if ($Lines[$i] -match "^import |^from ") {
                        $FirstImportIndex = $i
                        break
                    }
                }
                
                if ($FirstImportIndex -ge 0) {
                    $Lines = $Lines[0..$FirstImportIndex] + $RequiredImport + $Lines[($FirstImportIndex+1)..($Lines.Count-1)]
                    $Content = $Lines -join "`n"
                    Write-Host "  ✓ Added missing import: $RequiredImport" -ForegroundColor Green
                    $Modified = $true
                }
            }
        }
    }
    
    # Save the file if it was modified
    if ($Modified) {
        Set-Content -Path $File.FullName -Value $Content -NoNewline
        Write-Host "  ✓ File updated successfully" -ForegroundColor Green
    } else {
        Write-Host "  • No changes needed" -ForegroundColor Gray
    }
}

Write-Host "`nCreating missing __init__.py files..." -ForegroundColor Yellow

# Create __init__.py files for subdirectories
$Directories = @(
    "models",
    "models\classical", 
    "models\neural",
    "data",
    "evaluation"
)

foreach ($Dir in $Directories) {
    $DirPath = Join-Path $ProjectRoot $Dir
    $InitFile = Join-Path $DirPath "__init__.py"
    
    if (Test-Path $DirPath) {
        if (-not (Test-Path $InitFile)) {
            # Create a basic __init__.py file
            $InitContent = "# $Dir module`n"
            Set-Content -Path $InitFile -Value $InitContent
            Write-Host "  ✓ Created: $InitFile" -ForegroundColor Green
        } else {
            Write-Host "  • Already exists: $InitFile" -ForegroundColor Gray
        }
    } else {
        Write-Host "  ⚠ Directory not found: $DirPath" -ForegroundColor Red
    }
}

Write-Host "`nValidating project structure..." -ForegroundColor Yellow

# Check for required files
$RequiredFiles = @(
    "models\base_classifier.py",
    "models\classical\svm.py",
    "models\classical\random_forest.py", 
    "models\neural\mlp.py",
    "models\neural\cnn.py",
    "data\loaders.py",
    "main.py"
)

$MissingFiles = @()
foreach ($File in $RequiredFiles) {
    $FilePath = Join-Path $ProjectRoot $File
    if (-not (Test-Path $FilePath)) {
        $MissingFiles += $File
    }
}

if ($MissingFiles.Count -gt 0) {
    Write-Host "`n⚠ Missing required files:" -ForegroundColor Red
    foreach ($File in $MissingFiles) {
        Write-Host "  • $File" -ForegroundColor Red
    }
} else {
    Write-Host "  ✓ All required files present" -ForegroundColor Green
}

Write-Host "`nChecking for notebook files that need conversion..." -ForegroundColor Yellow

$NotebookFiles = Get-ChildItem -Path $ProjectRoot -Recurse -Include "*.ipynb"
if ($NotebookFiles.Count -gt 0) {
    Write-Host "⚠ Found notebook files that should be converted to .py:" -ForegroundColor Red
    foreach ($Notebook in $NotebookFiles) {
        Write-Host "  • $($Notebook.Name)" -ForegroundColor Red
    }
    Write-Host "`nTo convert notebooks to Python files, use:" -ForegroundColor Yellow
    Write-Host "jupyter nbconvert --to script notebook_name.ipynb" -ForegroundColor Cyan
} else {
    Write-Host "  ✓ No notebook files found" -ForegroundColor Green
}

Write-Host "`n" + "="*60 -ForegroundColor Green
Write-Host "IMPORT STANDARDIZATION COMPLETE!" -ForegroundColor Green
Write-Host "="*60 -ForegroundColor Green

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Convert any .ipynb files to .py using jupyter nbconvert" -ForegroundColor Cyan
Write-Host "2. Test imports by running: python -c 'from models.classical.svm import SVMClassifier'" -ForegroundColor Cyan
Write-Host "3. Run the main experiment: python main.py --config config.json" -ForegroundColor Cyan

# Create a test script to verify imports
$TestScript = @"
#!/usr/bin/env python3
""Test script to verify all imports work correctly.""

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
"@

$TestScriptPath = Join-Path $ProjectRoot "test_imports.py"
Set-Content -Path $TestScriptPath -Value $TestScript
Write-Host "`nCreated test script: test_imports.py" -ForegroundColor Green
Write-Host "Run it with: python test_imports.py" -ForegroundColor Cyan

# Add this new section after all your existing code
Write-Host "`nRunning basic tests..." -ForegroundColor Yellow
pytest tests/test_config_validation.py -v

Write-Host "`nQuick verification command:" -ForegroundColor Cyan
Write-Host "python -c `"from models.base_classifier import BaseClassifier; print('✓ Base imports work')`""