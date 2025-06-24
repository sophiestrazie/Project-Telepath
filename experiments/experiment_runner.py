# experiments/experiment_runner.py
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import time

from ..models import ClassifierFactory
from ..data.loaders import FMRIDataLoader
from ..utils.metrics import ModelEvaluator
from ..utils.io_utils import save_results, load_results

class ExperimentRunner:
    """Main experiment runner for classifier comparison"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_loader = FMRIDataLoader(self.config['data'])
        self.evaluator = ModelEvaluator()
        self.results = {}
        
    def run_experiment(self, experiment_name: str, subject_ids: List[str]) -> Dict[str, Any]:
        """Run a complete experiment"""
        print(f"Running experiment: {experiment_name}")
        
        experiment_config = self.config['experiments'][experiment_name]
        results = {
            'experiment_name': experiment_name,
            'classifiers': {},
            'summary': {}
        }
        
        # Load and prepare data
        all_X, all_y = self._load_all_subjects_data(subject_ids)
        
        # Run each classifier
        for classifier_config in experiment_config['classifiers']:
            classifier_type = classifier_config['type']
            print(f"  Running classifier: {classifier_type}")
            
            start_time = time.time()
            classifier_results = self._run_single_classifier(
                classifier_type, 
                classifier_config['config'], 
                all_X, 
                all_y
            )
            end_time = time.time()
            
            classifier_results['runtime'] = end_time - start_time
            results['classifiers'][classifier_type] = classifier_results
            
            print(f"    Completed in {end_time - start_time:.2f} seconds")
            print(f"    Mean CV Accuracy: {classifier_results['cv_scores'].mean():.4f} ± {classifier_results['cv_scores'].std():.4f}")
        
        # Generate summary
        results['summary'] = self._generate_summary(results['classifiers'])
        
        return results
    
    def _load_all_subjects_data(self, subject_ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Load and concatenate data from all subjects"""
        all_X = []
        all_y = []
        
        for subject_id in subject_ids:
            print(f"  Loading data for subject: {subject_id}")
            X, _ = self.data_loader.load_fmri_data(subject_id)
            y = self.data_loader.load_stimulus_labels(subject_id)
            
            all_X.append(X)
            all_y.append(y)
        
        return np.vstack(all_X), np.hstack(all_y)
    
    def _run_single_classifier(self, 
                             classifier_type: str, 
                             classifier_config: Dict[str, Any],
                             X: np.ndarray, 
                             y: np.ndarray) -> Dict[str, Any]:
        """Run a single classifier with cross-validation"""
        
        # Create classifier
        classifier = ClassifierFactory.create_classifier(classifier_type, classifier_config)
        
        # Cross-validation setup
        cv_config = self.config['cross_validation']
        cv = StratifiedKFold(
            n_splits=cv_config['n_folds'],
            shuffle=cv_config['shuffle'],
            random_state=cv_config['random_state']
        )
        
        # Perform cross-validation
        cv_scores = cross_val_score(classifier, X, y, cv=cv, scoring='accuracy')
        
        # Train final model on full dataset
        classifier.fit(X, y)
        
        # Make predictions for detailed evaluation
        y_pred = classifier.predict(X)
        y_pred_proba = classifier.predict_proba(X)
        
        # Get feature importance
        try:
            feature_importance = classifier.get_feature_importance()
        except Exception as e:
            print(f"    Warning: Could not get feature importance: {e}")
            feature_importance = None
        
        return {
            'classifier_type': classifier_type,
            'config': classifier_config,
            'cv_scores': cv_scores,
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std(),
            'classification_report': classification_report(y, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'feature_importance': feature_importance.tolist() if feature_importance is not None else None,
            'model': classifier  # Store trained model
        }
    
    def _generate_summary(self, classifier_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate experiment summary"""
        summary = {
            'best_classifier': None,
            'best_accuracy': 0.0,
            'classifier_ranking': []
        }
        
        # Rank classifiers by cross-validation score
        ranking = []
        for clf_type, results in classifier_results.items():
            ranking.append({
                'classifier': clf_type,
                'mean_accuracy': results['mean_cv_score'],
                'std_accuracy': results['std_cv_score'],
                'runtime': results['runtime']
            })
        
        ranking.sort(key=lambda x: x['mean_accuracy'], reverse=True)
        summary['classifier_ranking'] = ranking
        
        if ranking:
            summary['best_classifier'] = ranking[0]['classifier']
            summary['best_accuracy'] = ranking[0]['mean_accuracy']
        
        return summary
    
    def save_experiment_results(self, results: Dict[str, Any], output_path: str):
        """Save experiment results"""
        save_results(results, output_path)
        print(f"Results saved to: {output_path}")
    
    def run_hyperparameter_search(self, 
                                classifier_type: str, 
                                param_grid: Dict[str, List[Any]], 
                                X: np.ndarray, 
                                y: np.ndarray) -> Dict[str, Any]:
        """Run hyperparameter search for a specific classifier"""
        from sklearn.model_selection import GridSearchCV
        
        # Create base classifier
        base_classifier = ClassifierFactory.create_classifier(classifier_type, {})
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_classifier,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }