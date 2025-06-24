# tests/test_config_validation.py

"""
This test file (test_config_validation.py) validates the structure and content of the experiment configuration YAML file.
It performs the following tests:

What this file tests:
1. File Existence: Verifies the config file exists at the expected location
2. YAML Syntax: Checks that the file contains valid YAML syntax
3. Required Sections: Validates presence of mandatory sections (different from folders):
   - 'experiments'
   - 'data'
   - 'preprocessing'
   - 'cross_validation'
4. Experiment Types: Ensures experiment types are from allowed values:
   - 'baseline'
   - 'neural_networks'
   - 'advanced'
5. Data Paths: Confirms required path configurations exist:
   - 'data_path'
   - 'preprocessed_path'

What this file does NOT test:
1. Content Validity: Does not validate the actual content/values within sections beyond basic structure
2. Path Accessibility: Does not verify that configured paths exist or are accessible
3. Parameter Values: Does not check if parameter values are within valid ranges
4. Logical Consistency: Does not validate relationships between different configuration parameters
5. Runtime Compatibility: Does not verify config compatibility with actual experiment code
6. Full Schema: Does not validate all possible configuration options, only critical ones
7. Environment Specifics: Does not check environment-specific configurations or overrides
"""
import pytest
import yaml
from pathlib import Path

# Use relative path from the test file
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "experiment_configs.yaml"

def test_config_files_exist():
    """Test that config files exist"""
    assert CONFIG_PATH.exists()

def test_config_syntax():
    """Test YAML syntax is valid"""
    try:
        with open(CONFIG_PATH, "r") as f:
            yaml.safe_load(f)
    except yaml.YAMLError as e:
        pytest.fail(f"YAML syntax error: {e}")

def test_required_sections():
    """Test that required sections exist"""
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    required_sections = ["experiments", "data", "preprocessing", "cross_validation"]
    for section in required_sections:
        assert section in config, f"Missing required section: {section}"

def test_experiment_types():
    """Test that experiment types are valid"""
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    valid_experiments = ["baseline", "neural_networks", "advanced"]
    for exp in config["experiments"]:
        assert exp in valid_experiments, f"Invalid experiment type: {exp}"

def test_data_paths():
    """Test that data paths are specified"""
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    assert "data_path" in config["data"], "Missing data_path"
    assert "preprocessed_path" in config["data"], "Missing preprocessed_path"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])