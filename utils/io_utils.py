# utils/io_utils.py
"""
I/O utilities for the multimodal fMRI stimulus prediction framework.

This module provides comprehensive file I/O operations, directory management,
model persistence, and result export functionality following SOLID principles.
"""

import json
import pickle
import joblib
import csv
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IOManager:
    """
    Centralized I/O management class following Single Responsibility Principle.
    
    Handles all file operations including saving/loading results, models,
    and generating reports.
    """
    
    def __init__(self, base_output_dir: str = "results"):
        """
        Initialize IOManager with base output directory.
        
        Args:
            base_output_dir: Base directory for all outputs
        """
        self.base_output_dir = Path(base_output_dir)
        self.setup_directories()
        
    def setup_directories(self) -> None:
        """Create necessary directory structure."""
        directories = [
            self.base_output_dir,
            self.base_output_dir / "models",
            self.base_output_dir / "figures",
            self.base_output_dir / "reports",
            self.base_output_dir / "exports",
            self.base_output_dir / "cache",
            "logs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")
    
    def save_results(self, 
                    results: Dict[str, Any], 
                    filename: str,
                    file_format: str = "json") -> Path:
        """
        Save experiment results in specified format.
        
        Args:
            results: Results dictionary to save
            filename: Output filename (without extension)
            file_format: Output format ('json', 'pickle', 'yaml')
            
        Returns:
            Path to saved file
        """
        # Clean results for serialization
        cleaned_results = self._clean_results_for_serialization(results)
        
        if file_format.lower() == "json":
            filepath = self.base_output_dir / f"{filename}.json"
            with open(filepath, 'w') as f:
                json.dump(cleaned_results, f, indent=2, default=str)
                
        elif file_format.lower() == "pickle":
            filepath = self.base_output_dir / f"{filename}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(results, f)  # Use original results for pickle
                
        elif file_format.lower() == "yaml":
            filepath = self.base_output_dir / f"{filename}.yaml"
            with open(filepath, 'w') as f:
                yaml.dump(cleaned_results, f, default_flow_style=False)
                
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        logger.info(f"Results saved to: {filepath}")
        return filepath
    
    def load_results(self, 
                    filepath: Union[str, Path],
                    file_format: Optional[str] = None) -> Dict[str, Any]:
        """
        Load experiment results from file.
        
        Args:
            filepath: Path to results file
            file_format: File format ('json', 'pickle', 'yaml'). Auto-detected if None
            
        Returns:
            Loaded results dictionary
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")
        
        # Auto-detect format from extension
        if file_format is None:
            file_format = filepath.suffix.lower().lstrip('.')
        
        if file_format == "json":
            with open(filepath, 'r') as f:
                results = json.load(f)
                
        elif file_format in ["pkl", "pickle"]:
            with open(filepath, 'rb') as f:
                results = pickle.load(f)
                
        elif file_format in ["yaml", "yml"]:
            with open(filepath, 'r') as f:
                results = yaml.safe_load(f)
                
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        logger.info(f"Results loaded from: {filepath}")
        return results
    
    def save_model(self, 
                  model: Any, 
                  model_name: str,
                  experiment_name: str = "default",
                  metadata: Optional[Dict[str, Any]] = None) -> Path:
        """
        Save trained model with metadata.
        
        Args:
            model: Trained model object
            model_name: Name of the model
            experiment_name: Name of the experiment
            metadata: Optional metadata dictionary
            
        Returns:
            Path to saved model directory
        """
        model_dir = self.base_output_dir / "models" / experiment_name / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model using joblib (recommended for sklearn models)
        model_path = model_dir / "model.pkl"
        joblib.dump(model, model_path)
        
        # Save metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'model_name': model_name,
            'experiment_name': experiment_name,
            'saved_timestamp': datetime.now().isoformat(),
            'model_type': str(type(model).__name__)
        })
        
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Model saved to: {model_dir}")
        return model_dir
    
    def load_model(self, 
                  model_path: Union[str, Path],
                  load_metadata: bool = True) -> Union[Any, tuple]:
        """
        Load trained model and optionally its metadata.
        
        Args:
            model_path: Path to model directory or model file
            load_metadata: Whether to load metadata
            
        Returns:
            Model object or (model, metadata) tuple if load_metadata=True
        """
        model_path = Path(model_path)
        
        # Handle both directory and file paths
        if model_path.is_dir():
            model_file = model_path / "model.pkl"
            metadata_file = model_path / "metadata.json"
        else:
            model_file = model_path
            metadata_file = model_path.parent / "metadata.json"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        # Load model
        model = joblib.load(model_file)
        
        if load_metadata:
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            logger.info(f"Model and metadata loaded from: {model_path}")
            return model, metadata
        
        logger.info(f"Model loaded from: {model_file}")
        return model
    
    def export_to_csv(self, 
                     data: Union[Dict[str, Any], pd.DataFrame], 
                     filename: str,
                     include_timestamp: bool = True) -> Path:
        """
        Export data to CSV format.
        
        Args:
            data: Data to export (dict or DataFrame)
            filename: Output filename (without extension)
            include_timestamp: Whether to include timestamp in filename
            
        Returns:
            Path to exported CSV file
        """
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{timestamp}"
        
        filepath = self.base_output_dir / "exports" / f"{filename}.csv"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, dict):
            # Convert dict to DataFrame for CSV export
            if 'summary' in data and isinstance(data['summary'], dict):
                # Handle experiment results structure
                df = self._results_dict_to_dataframe(data)
            else:
                # Simple dict to DataFrame conversion
                df = pd.DataFrame([data])
        else:
            df = data
        
        df.to_csv(filepath, index=False)
        logger.info(f"Data exported to CSV: {filepath}")
        return filepath
    
    def create_experiment_report(self, 
                               results: Dict[str, Any],
                               experiment_name: str = "experiment") -> Path:
        """
        Create comprehensive experiment report.
        
        Args:
            results: Experiment results dictionary
            experiment_name: Name of the experiment
            
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.base_output_dir / "reports" / f"{experiment_name}_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate report sections
        self._generate_summary_report(results, report_dir)
        self._generate_detailed_metrics_report(results, report_dir)
        self._generate_model_comparison_report(results, report_dir)
        
        # Create main report file
        main_report = self._create_main_report(results, experiment_name, report_dir)
        
        logger.info(f"Experiment report generated: {report_dir}")
        return report_dir
    
    def _clean_results_for_serialization(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Clean results dictionary for JSON serialization."""
        cleaned = {}
        
        for key, value in results.items():
            if isinstance(value, dict):
                # Handle nested dictionaries
                if key == 'classifiers':
                    # Special handling for classifier results
                    cleaned[key] = {}
                    for clf_name, clf_results in value.items():
                        cleaned[key][clf_name] = self._clean_classifier_results(clf_results)
                else:
                    cleaned[key] = self._clean_results_for_serialization(value)
            elif isinstance(value, np.ndarray):
                cleaned[key] = value.tolist()
            elif hasattr(value, '__dict__') and not callable(value):
                # Skip model objects but keep other serializable objects
                continue
            else:
                cleaned[key] = value
        
        return cleaned
    
    def _clean_classifier_results(self, clf_results: Dict[str, Any]) -> Dict[str, Any]:
        """Clean classifier results for serialization."""
        cleaned = {}
        
        for key, value in clf_results.items():
            if key == 'model':
                # Skip model objects
                continue
            elif isinstance(value, np.ndarray):
                cleaned[key] = value.tolist()
            elif isinstance(value, dict):
                cleaned[key] = self._clean_results_for_serialization(value)
            else:
                cleaned[key] = value
        
        return cleaned
    
    def _results_dict_to_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Convert results dictionary to DataFrame for CSV export."""
        rows = []
        
        if 'summary' in results and 'classifier_ranking' in results['summary']:
            for rank_info in results['summary']['classifier_ranking']:
                row = {
                    'Classifier': rank_info['classifier'],
                    'Mean_Accuracy': rank_info['mean_accuracy'],
                    'Std_Accuracy': rank_info['std_accuracy'],
                    'Runtime_Seconds': rank_info.get('runtime', 0)
                }
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _generate_summary_report(self, results: Dict[str, Any], report_dir: Path) -> None:
        """Generate summary report section."""
        summary_file = report_dir / "summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("EXPERIMENT SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            if 'summary' in results:
                summary = results['summary']
                f.write(f"Best Classifier: {summary.get('best_classifier', 'N/A')}\n")
                f.write(f"Best Accuracy: {summary.get('best_accuracy', 0):.4f}\n\n")
                
                f.write("CLASSIFIER RANKING:\n")
                f.write("-" * 30 + "\n")
                
                if 'classifier_ranking' in summary:
                    for i, clf_info in enumerate(summary['classifier_ranking']):
                        f.write(f"{i+1}. {clf_info['classifier']}: ")
                        f.write(f"{clf_info['mean_accuracy']:.4f} ± {clf_info['std_accuracy']:.4f}\n")
    
    def _generate_detailed_metrics_report(self, results: Dict[str, Any], report_dir: Path) -> None:
        """Generate detailed metrics report."""
        metrics_file = report_dir / "detailed_metrics.txt"
        
        with open(metrics_file, 'w') as f:
            f.write("DETAILED METRICS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            if 'classifiers' in results:
                for clf_name, clf_results in results['classifiers'].items():
                    f.write(f"\n{clf_name.upper()}\n")
                    f.write("-" * len(clf_name) + "\n")
                    
                    if 'classification_report' in clf_results:
                        f.write("Classification Report:\n")
                        report = clf_results['classification_report']
                        
                        # Write metrics for each class
                        for class_name, metrics in report.items():
                            if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                                f.write(f"  {class_name}: ")
                                f.write(f"precision={metrics.get('precision', 0):.3f}, ")
                                f.write(f"recall={metrics.get('recall', 0):.3f}, ")
                                f.write(f"f1-score={metrics.get('f1-score', 0):.3f}\n")
    
    def _generate_model_comparison_report(self, results: Dict[str, Any], report_dir: Path) -> None:
        """Generate model comparison report."""
        comparison_file = report_dir / "model_comparison.csv"
        
        # Export comparison data to CSV
        df = self._results_dict_to_dataframe(results)
        df.to_csv(comparison_file, index=False)
    
    def _create_main_report(self, 
                          results: Dict[str, Any], 
                          experiment_name: str, 
                          report_dir: Path) -> Path:
        """Create main HTML report."""
        main_report_file = report_dir / "main_report.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Experiment Report: {experiment_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .highlight {{ background-color: #ffffcc; }}
            </style>
        </head>
        <body>
            <h1>Experiment Report: {experiment_name}</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>Summary</h2>
            <p>This report contains the complete results of the multimodal fMRI stimulus prediction experiment.</p>
            
            <h2>Files Generated</h2>
            <ul>
                <li><a href="summary.txt">Summary Report</a></li>
                <li><a href="detailed_metrics.txt">Detailed Metrics</a></li>
                <li><a href="model_comparison.csv">Model Comparison (CSV)</a></li>
            </ul>
            
            <h2>Best Results</h2>
        """
        
        if 'summary' in results:
            summary = results['summary']
            html_content += f"""
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr class="highlight">
                    <td>Best Classifier</td>
                    <td>{summary.get('best_classifier', 'N/A')}</td>
                </tr>
                <tr class="highlight">
                    <td>Best Accuracy</td>
                    <td>{summary.get('best_accuracy', 0):.4f}</td>
                </tr>
            </table>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(main_report_file, 'w') as f:
            f.write(html_content)
        
        return main_report_file


# Standalone utility functions
def save_results(results: Dict[str, Any], 
                filepath: Union[str, Path],
                file_format: str = "json") -> None:
    """
    Standalone function to save results.
    
    Args:
        results: Results dictionary to save
        filepath: Output file path
        file_format: Output format ('json', 'pickle', 'yaml')
    """
    io_manager = IOManager()
    
    # Extract filename and directory from filepath
    filepath = Path(filepath)
    filename = filepath.stem
    
    # Update base directory if specified in filepath
    if filepath.parent != Path("."):
        io_manager.base_output_dir = filepath.parent
        io_manager.setup_directories()
    
    io_manager.save_results(results, filename, file_format)


def load_results(filepath: Union[str, Path],
                file_format: Optional[str] = None) -> Dict[str, Any]:
    """
    Standalone function to load results.
    
    Args:
        filepath: Path to results file
        file_format: File format (auto-detected if None)
        
    Returns:
        Loaded results dictionary
    """
    io_manager = IOManager()
    return io_manager.load_results(filepath, file_format)


def setup_directories(base_dir: Union[str, Path] = "results") -> None:
    """
    Standalone function to setup directory structure.
    
    Args:
        base_dir: Base directory for outputs
    """
    io_manager = IOManager(str(base_dir))


def save_model(model: Any, 
              model_name: str,
              experiment_name: str = "default",
              output_dir: str = "results",
              metadata: Optional[Dict[str, Any]] = None) -> Path:
    """
    Standalone function to save model.
    
    Args:
        model: Trained model object
        model_name: Name of the model
        experiment_name: Name of the experiment
        output_dir: Base output directory
        metadata: Optional metadata dictionary
        
    Returns:
        Path to saved model directory
    """
    io_manager = IOManager(output_dir)
    return io_manager.save_model(model, model_name, experiment_name, metadata)


def load_model(model_path: Union[str, Path],
              load_metadata: bool = True) -> Union[Any, tuple]:
    """
    Standalone function to load model.
    
    Args:
        model_path: Path to model directory or model file
        load_metadata: Whether to load metadata
        
    Returns:
        Model object or (model, metadata) tuple if load_metadata=True
    """
    io_manager = IOManager()
    return io_manager.load_model(model_path, load_metadata)


def export_to_csv(data: Union[Dict[str, Any], pd.DataFrame], 
                 filename: str,
                 output_dir: str = "results",
                 include_timestamp: bool = True) -> Path:
    """
    Standalone function to export data to CSV.
    
    Args:
        data: Data to export
        filename: Output filename
        output_dir: Output directory
        include_timestamp: Whether to include timestamp in filename
        
    Returns:
        Path to exported CSV file
    """
    io_manager = IOManager(output_dir)
    return io_manager.export_to_csv(data, filename, include_timestamp)


def create_experiment_report(results: Dict[str, Any],
                           experiment_name: str = "experiment",
                           output_dir: str = "results") -> Path:
    """
    Standalone function to create experiment report.
    
    Args:
        results: Experiment results dictionary
        experiment_name: Name of the experiment
        output_dir: Output directory
        
    Returns:
        Path to generated report directory
    """
    io_manager = IOManager(output_dir)
    return io_manager.create_experiment_report(results, experiment_name)


def backup_experiment(experiment_dir: Union[str, Path],
                     backup_dir: Union[str, Path] = "backups") -> Path:
    """
    Create backup of experiment directory.
    
    Args:
        experiment_dir: Directory to backup
        backup_dir: Backup destination directory
        
    Returns:
        Path to backup directory
    """
    experiment_dir = Path(experiment_dir)
    backup_dir = Path(backup_dir)
    
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{experiment_dir.name}_backup_{timestamp}"
    
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(experiment_dir, backup_path)
    
    logger.info(f"Experiment backed up to: {backup_path}")
    return backup_path