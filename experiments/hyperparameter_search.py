# experiments/hyperparameter_search.py
"""
Advanced hyperparameter optimization module for multimodal fMRI stimulus prediction.

This module provides comprehensive hyperparameter search capabilities including
grid search, random search, and Bayesian optimization following SOLID principles.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from abc import ABC, abstractmethod
import time
import warnings
from pathlib import Path
import logging

# Core ML libraries
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold,
    cross_val_score, validation_curve, learning_curve
)
from sklearn.metrics import accuracy_score, make_scorer
import joblib

# Statistical and optimization libraries
from scipy.stats import uniform, randint, loguniform
from scipy.optimize import minimize

# Ensure project imports work
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from models import ClassifierFactory
from utils.metrics import ModelEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperparameterSearcher(ABC):
    """
    Abstract base class for hyperparameter optimization strategies.
    
    Follows the Strategy pattern to enable different optimization approaches.
    """
    
    def __init__(self, 
                 classifier_type: str,
                 cv_folds: int = 5,
                 scoring: Union[str, Callable] = 'accuracy',
                 n_jobs: int = -1,
                 random_state: int = 42):
        """
        Initialize hyperparameter searcher.
        
        Args:
            classifier_type: Type of classifier to optimize
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for optimization
            n_jobs: Number of parallel jobs
            random_state: Random seed for reproducibility
        """
        self.classifier_type = classifier_type
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = None
        self.search_results_ = None
        
    @abstractmethod
    def search(self, 
              X: np.ndarray, 
              y: np.ndarray, 
              param_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method for hyperparameter search.
        
        Args:
            X: Training features
            y: Training labels
            param_space: Parameter search space
            
        Returns:
            Dictionary containing search results
        """
        pass
    
    def get_cv_strategy(self, y: np.ndarray) -> StratifiedKFold:
        """Get cross-validation strategy."""
        return StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
    
    def evaluate_params(self, 
                       X: np.ndarray, 
                       y: np.ndarray, 
                       params: Dict[str, Any]) -> float:
        """
        Evaluate a specific parameter configuration.
        
        Args:
            X: Training features
            y: Training labels
            params: Parameter configuration
            
        Returns:
            Cross-validation score
        """
        try:
            classifier = ClassifierFactory.create_classifier(self.classifier_type, params)
            cv_strategy = self.get_cv_strategy(y)
            scores = cross_val_score(
                classifier, X, y, 
                cv=cv_strategy, 
                scoring=self.scoring,
                n_jobs=1  # Use 1 to avoid nested parallelization
            )
            return np.mean(scores)
        except Exception as e:
            logger.warning(f"Failed to evaluate params {params}: {e}")
            return -np.inf


class GridSearchOptimizer(HyperparameterSearcher):
    """
    Grid search hyperparameter optimization.
    
    Exhaustively searches through all parameter combinations.
    """
    
    def search(self, 
              X: np.ndarray, 
              y: np.ndarray, 
              param_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Perform grid search optimization.
        
        Args:
            X: Training features
            y: Training labels
            param_space: Parameter grid as dict of parameter name -> list of values
            
        Returns:
            Dictionary containing search results
        """
        logger.info(f"Starting grid search for {self.classifier_type}")
        start_time = time.time()
        
        # Create base classifier
        base_classifier = ClassifierFactory.create_classifier(self.classifier_type, {})
        
        # Setup cross-validation
        cv_strategy = self.get_cv_strategy(y)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_classifier,
            param_grid=param_space,
            cv=cv_strategy,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=1,
            return_train_score=True
        )
        
        # Fit grid search
        grid_search.fit(X, y)
        
        # Store results
        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_
        self.search_results_ = grid_search.cv_results_
        
        end_time = time.time()
        search_time = end_time - start_time
        
        logger.info(f"Grid search completed in {search_time:.2f} seconds")
        logger.info(f"Best score: {self.best_score_:.4f}")
        logger.info(f"Best params: {self.best_params_}")
        
        return {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'cv_results': self.search_results_,
            'search_time': search_time,
            'n_combinations': len(grid_search.cv_results_['params']),
            'optimization_method': 'grid_search'
        }


class RandomSearchOptimizer(HyperparameterSearcher):
    """
    Random search hyperparameter optimization.
    
    Randomly samples from parameter distributions.
    """
    
    def __init__(self, 
                 classifier_type: str,
                 n_iter: int = 100,
                 **kwargs):
        """
        Initialize random search optimizer.
        
        Args:
            classifier_type: Type of classifier to optimize
            n_iter: Number of parameter settings to sample
            **kwargs: Additional arguments for parent class
        """
        super().__init__(classifier_type, **kwargs)
        self.n_iter = n_iter
    
    def search(self, 
              X: np.ndarray, 
              y: np.ndarray, 
              param_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform random search optimization.
        
        Args:
            X: Training features
            y: Training labels
            param_space: Parameter distributions as dict of parameter name -> distribution
            
        Returns:
            Dictionary containing search results
        """
        logger.info(f"Starting random search for {self.classifier_type}")
        start_time = time.time()
        
        # Create base classifier
        base_classifier = ClassifierFactory.create_classifier(self.classifier_type, {})
        
        # Setup cross-validation
        cv_strategy = self.get_cv_strategy(y)
        
        # Perform random search
        random_search = RandomizedSearchCV(
            estimator=base_classifier,
            param_distributions=param_space,
            n_iter=self.n_iter,
            cv=cv_strategy,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=1,
            random_state=self.random_state,
            return_train_score=True
        )
        
        # Fit random search
        random_search.fit(X, y)
        
        # Store results
        self.best_params_ = random_search.best_params_
        self.best_score_ = random_search.best_score_
        self.search_results_ = random_search.cv_results_
        
        end_time = time.time()
        search_time = end_time - start_time
        
        logger.info(f"Random search completed in {search_time:.2f} seconds")
        logger.info(f"Best score: {self.best_score_:.4f}")
        logger.info(f"Best params: {self.best_params_}")
        
        return {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'cv_results': self.search_results_,
            'search_time': search_time,
            'n_iterations': self.n_iter,
            'optimization_method': 'random_search'
        }


class BayesianOptimizer(HyperparameterSearcher):
    """
    Bayesian optimization for hyperparameter search.
    
    Uses Gaussian processes to model the objective function and guide search.
    """
    
    def __init__(self, 
                 classifier_type: str,
                 n_calls: int = 50,
                 n_initial_points: int = 10,
                 acquisition_function: str = 'EI',
                 **kwargs):
        """
        Initialize Bayesian optimizer.
        
        Args:
            classifier_type: Type of classifier to optimize
            n_calls: Number of evaluation calls
            n_initial_points: Number of random initial points
            acquisition_function: Acquisition function ('EI', 'PI', 'LCB')
            **kwargs: Additional arguments for parent class
        """
        super().__init__(classifier_type, **kwargs)
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.acquisition_function = acquisition_function
        
        # Try to import scikit-optimize
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            from skopt.acquisition import gaussian_ei, gaussian_pi, gaussian_lcb
            self.skopt_available = True
            self.gp_minimize = gp_minimize
            self.Real = Real
            self.Integer = Integer
            self.Categorical = Categorical
        except ImportError:
            self.skopt_available = False
            logger.warning("scikit-optimize not available. Falling back to random search.")
    
    def search(self, 
              X: np.ndarray, 
              y: np.ndarray, 
              param_space: Dict[str, Tuple]) -> Dict[str, Any]:
        """
        Perform Bayesian optimization.
        
        Args:
            X: Training features
            y: Training labels
            param_space: Parameter space as dict of parameter name -> (type, bounds/choices)
                        e.g., {'C': ('real', (0.1, 100)), 'kernel': ('categorical', ['rbf', 'linear'])}
            
        Returns:
            Dictionary containing search results
        """
        if not self.skopt_available:
            # Fallback to random search
            logger.warning("Using random search fallback")
            random_optimizer = RandomSearchOptimizer(
                self.classifier_type, 
                n_iter=self.n_calls,
                cv_folds=self.cv_folds,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                random_state=self.random_state
            )
            # Convert param_space for random search
            random_param_space = self._convert_param_space_for_random(param_space)
            return random_optimizer.search(X, y, random_param_space)
        
        logger.info(f"Starting Bayesian optimization for {self.classifier_type}")
        start_time = time.time()
        
        # Convert parameter space to skopt format
        dimensions, param_names = self._convert_param_space_to_skopt(param_space)
        
        # Define objective function
        def objective(params):
            param_dict = dict(zip(param_names, params))
            score = self.evaluate_params(X, y, param_dict)
            return -score  # Minimize negative score
        
        # Perform Bayesian optimization
        result = self.gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=self.n_calls,
            n_initial_points=self.n_initial_points,
            acquisition_function=self.acquisition_function.lower(),
            random_state=self.random_state,
            verbose=True
        )
        
        # Extract results
        best_params_list = result.x
        self.best_params_ = dict(zip(param_names, best_params_list))
        self.best_score_ = -result.fun  # Convert back to positive score
        
        end_time = time.time()
        search_time = end_time - start_time
        
        logger.info(f"Bayesian optimization completed in {search_time:.2f} seconds")
        logger.info(f"Best score: {self.best_score_:.4f}")
        logger.info(f"Best params: {self.best_params_}")
        
        return {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'optimization_result': result,
            'search_time': search_time,
            'n_calls': self.n_calls,
            'optimization_method': 'bayesian_optimization'
        }
    
    def _convert_param_space_to_skopt(self, param_space: Dict[str, Tuple]) -> Tuple[List, List]:
        """Convert parameter space to skopt format."""
        dimensions = []
        param_names = []
        
        for param_name, (param_type, bounds_or_choices) in param_space.items():
            param_names.append(param_name)
            
            if param_type == 'real':
                low, high = bounds_or_choices
                dimensions.append(self.Real(low, high, name=param_name))
            elif param_type == 'integer':
                low, high = bounds_or_choices
                dimensions.append(self.Integer(low, high, name=param_name))
            elif param_type == 'categorical':
                dimensions.append(self.Categorical(bounds_or_choices, name=param_name))
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
        
        return dimensions, param_names
    
    def _convert_param_space_for_random(self, param_space: Dict[str, Tuple]) -> Dict[str, Any]:
        """Convert parameter space for random search fallback."""
        random_param_space = {}
        
        for param_name, (param_type, bounds_or_choices) in param_space.items():
            if param_type == 'real':
                low, high = bounds_or_choices
                random_param_space[param_name] = uniform(low, high - low)
            elif param_type == 'integer':
                low, high = bounds_or_choices
                random_param_space[param_name] = randint(low, high + 1)
            elif param_type == 'categorical':
                random_param_space[param_name] = bounds_or_choices
            
        return random_param_space


class MultiObjectiveOptimizer(HyperparameterSearcher):
    """
    Multi-objective hyperparameter optimization.
    
    Optimizes multiple metrics simultaneously using Pareto efficiency.
    """
    
    def __init__(self, 
                 classifier_type: str,
                 objectives: List[str] = ['accuracy', 'f1_weighted'],
                 n_iter: int = 100,
                 **kwargs):
        """
        Initialize multi-objective optimizer.
        
        Args:
            classifier_type: Type of classifier to optimize
            objectives: List of metrics to optimize
            n_iter: Number of iterations
            **kwargs: Additional arguments for parent class
        """
        super().__init__(classifier_type, **kwargs)
        self.objectives = objectives
        self.n_iter = n_iter
        self.pareto_front_ = None
    
    def search(self, 
              X: np.ndarray, 
              y: np.ndarray, 
              param_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform multi-objective optimization.
        
        Args:
            X: Training features
            y: Training labels
            param_space: Parameter space
            
        Returns:
            Dictionary containing Pareto front and optimization results
        """
        logger.info(f"Starting multi-objective optimization for {self.classifier_type}")
        start_time = time.time()
        
        # Generate random parameter combinations
        param_combinations = self._generate_random_combinations(param_space, self.n_iter)
        
        # Evaluate all combinations
        results = []
        for i, params in enumerate(param_combinations):
            logger.info(f"Evaluating combination {i+1}/{len(param_combinations)}")
            
            objective_scores = []
            for objective in self.objectives:
                score = self._evaluate_single_objective(X, y, params, objective)
                objective_scores.append(score)
            
            results.append({
                'params': params,
                'objectives': dict(zip(self.objectives, objective_scores)),
                'scores': objective_scores
            })
        
        # Find Pareto front
        pareto_front = self._find_pareto_front(results)
        self.pareto_front_ = pareto_front
        
        # Select best compromise solution (closest to ideal point)
        best_solution = self._select_best_compromise(pareto_front)
        
        self.best_params_ = best_solution['params']
        self.best_score_ = np.mean(best_solution['scores'])
        
        end_time = time.time()
        search_time = end_time - start_time
        
        logger.info(f"Multi-objective optimization completed in {search_time:.2f} seconds")
        logger.info(f"Pareto front size: {len(pareto_front)}")
        logger.info(f"Best compromise params: {self.best_params_}")
        
        return {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'pareto_front': pareto_front,
            'all_results': results,
            'search_time': search_time,
            'optimization_method': 'multi_objective'
        }
    
    def _generate_random_combinations(self, param_space: Dict[str, Any], n_combinations: int) -> List[Dict]:
        """Generate random parameter combinations."""
        combinations = []
        
        for _ in range(n_combinations):
            combination = {}
            for param_name, param_values in param_space.items():
                if isinstance(param_values, list):
                    combination[param_name] = np.random.choice(param_values)
                elif hasattr(param_values, 'rvs'):  # scipy distribution
                    combination[param_name] = param_values.rvs()
                else:
                    combination[param_name] = np.random.choice(param_values)
            
            combinations.append(combination)
        
        return combinations
    
    def _evaluate_single_objective(self, X: np.ndarray, y: np.ndarray, params: Dict, objective: str) -> float:
        """Evaluate a single objective metric."""
        try:
            classifier = ClassifierFactory.create_classifier(self.classifier_type, params)
            cv_strategy = self.get_cv_strategy(y)
            scores = cross_val_score(classifier, X, y, cv=cv_strategy, scoring=objective, n_jobs=1)
            return np.mean(scores)
        except Exception as e:
            logger.warning(f"Failed to evaluate objective {objective} with params {params}: {e}")
            return 0.0
    
    def _find_pareto_front(self, results: List[Dict]) -> List[Dict]:
        """Find Pareto efficient solutions."""
        pareto_front = []
        
        for i, solution_a in enumerate(results):
            is_dominated = False
            
            for j, solution_b in enumerate(results):
                if i != j and self._dominates(solution_b['scores'], solution_a['scores']):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(solution_a)
        
        return pareto_front
    
    def _dominates(self, scores_a: List[float], scores_b: List[float]) -> bool:
        """Check if solution A dominates solution B."""
        return all(a >= b for a, b in zip(scores_a, scores_b)) and any(a > b for a, b in zip(scores_a, scores_b))
    
    def _select_best_compromise(self, pareto_front: List[Dict]) -> Dict:
        """Select best compromise solution from Pareto front."""
        if not pareto_front:
            raise ValueError("Empty Pareto front")
        
        # Calculate distances to ideal point (maximum in each objective)
        max_scores = [max(sol['scores'][i] for sol in pareto_front) for i in range(len(self.objectives))]
        
        best_distance = float('inf')
        best_solution = None
        
        for solution in pareto_front:
            # Calculate Euclidean distance to ideal point
            distance = np.sqrt(sum((max_scores[i] - solution['scores'][i])**2 for i in range(len(self.objectives))))
            
            if distance < best_distance:
                best_distance = distance
                best_solution = solution
        
        return best_solution


class HyperparameterSearchFactory:
    """
    Factory class for creating hyperparameter search instances.
    
    Follows the Factory pattern for easy extensibility.
    """
    
    _optimizers = {
        'grid_search': GridSearchOptimizer,
        'random_search': RandomSearchOptimizer,
        'bayesian': BayesianOptimizer,
        'multi_objective': MultiObjectiveOptimizer
    }
    
    @classmethod
    def create_optimizer(cls, 
                        optimizer_type: str,
                        classifier_type: str,
                        **kwargs) -> HyperparameterSearcher:
        """
        Create hyperparameter optimizer instance.
        
        Args:
            optimizer_type: Type of optimizer ('grid_search', 'random_search', 'bayesian', 'multi_objective')
            classifier_type: Type of classifier to optimize
            **kwargs: Additional arguments for optimizer
            
        Returns:
            HyperparameterSearcher instance
        """
        if optimizer_type not in cls._optimizers:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}. Available: {list(cls._optimizers.keys())}")
        
        optimizer_class = cls._optimizers[optimizer_type]
        return optimizer_class(classifier_type, **kwargs)
    
    @classmethod
    def get_available_optimizers(cls) -> List[str]:
        """Get list of available optimizer types."""
        return list(cls._optimizers.keys())


class ValidationCurveAnalyzer:
    """
    Analyzer for validation curves and learning curves.
    
    Provides insights into model performance vs hyperparameters.
    """
    
    def __init__(self, classifier_type: str, cv_folds: int = 5, scoring: str = 'accuracy'):
        """
        Initialize validation curve analyzer.
        
        Args:
            classifier_type: Type of classifier to analyze
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric
        """
        self.classifier_type = classifier_type
        self.cv_folds = cv_folds
        self.scoring = scoring
    
    def analyze_parameter(self, 
                         X: np.ndarray, 
                         y: np.ndarray,
                         param_name: str,
                         param_range: List[Any],
                         base_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze the effect of a single parameter on model performance.
        
        Args:
            X: Training features
            y: Training labels
            param_name: Name of parameter to analyze
            param_range: Range of parameter values to test
            base_params: Base parameter configuration
            
        Returns:
            Dictionary containing validation curve results
        """
        if base_params is None:
            base_params = {}
        
        logger.info(f"Analyzing parameter {param_name} for {self.classifier_type}")
        
        # Create base classifier
        classifier = ClassifierFactory.create_classifier(self.classifier_type, base_params)
        
        # Generate validation curve
        train_scores, validation_scores = validation_curve(
            classifier, X, y,
            param_name=param_name,
            param_range=param_range,
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42),
            scoring=self.scoring,
            n_jobs=-1
        )
        
        # Calculate statistics
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(validation_scores, axis=1)
        val_std = np.std(validation_scores, axis=1)
        
        return {
            'param_name': param_name,
            'param_range': param_range,
            'train_scores': train_scores.tolist(),
            'validation_scores': validation_scores.tolist(),
            'train_mean': train_mean.tolist(),
            'train_std': train_std.tolist(),
            'validation_mean': val_mean.tolist(),
            'validation_std': val_std.tolist(),
            'best_param_value': param_range[np.argmax(val_mean)],
            'best_score': np.max(val_mean)
        }
    
    def analyze_learning_curve(self, 
                              X: np.ndarray, 
                              y: np.ndarray,
                              params: Dict[str, Any] = None,
                              train_sizes: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze learning curves for the model.
        
        Args:
            X: Training features
            y: Training labels
            params: Model parameters
            train_sizes: Training set sizes to evaluate
            
        Returns:
            Dictionary containing learning curve results
        """
        if params is None:
            params = {}
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        logger.info(f"Analyzing learning curve for {self.classifier_type}")
        
        # Create classifier
        classifier = ClassifierFactory.create_classifier(self.classifier_type, params)
        
        # Generate learning curve
        train_sizes_abs, train_scores, validation_scores = learning_curve(
            classifier, X, y,
            train_sizes=train_sizes,
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42),
            scoring=self.scoring,
            n_jobs=-1
        )
        
        # Calculate statistics
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(validation_scores, axis=1)
        val_std = np.std(validation_scores, axis=1)
        
        return {
            'train_sizes': train_sizes_abs.tolist(),
            'train_scores': train_scores.tolist(),
            'validation_scores': validation_scores.tolist(),
            'train_mean': train_mean.tolist(),
            'train_std': train_std.tolist(),
            'validation_mean': val_mean.tolist(),
            'validation_std': val_std.tolist(),
            'final_train_score': train_mean[-1],
            'final_validation_score': val_mean[-1],
            'gap': train_mean[-1] - val_mean[-1]  # Overfitting indicator
        }


# Convenience functions for direct usage
def optimize_hyperparameters(classifier_type: str,
                            X: np.ndarray,
                            y: np.ndarray,
                            param_space: Dict[str, Any],
                            optimizer_type: str = 'grid_search',
                            **kwargs) -> Dict[str, Any]:
    """
    Convenience function for hyperparameter optimization.
    
    Args:
        classifier_type: Type of classifier to optimize
        X: Training features
        y: Training labels
        param_space: Parameter search space
        optimizer_type: Type of optimizer to use
        **kwargs: Additional arguments for optimizer
        
    Returns:
        Optimization results dictionary
    """
    optimizer = HyperparameterSearchFactory.create_optimizer(
        optimizer_type, classifier_type, **kwargs
    )
    return optimizer.search(X, y, param_space)


def analyze_parameter_sensitivity(classifier_type: str,
                                X: np.ndarray,
                                y: np.ndarray,
                                param_name: str,
                                param_range: List[Any],
                                base_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Convenience function for parameter sensitivity analysis.
    
    Args:
        classifier_type: Type of classifier to analyze
        X: Training features
        y: Training labels
        param_name: Name of parameter to analyze
        param_range: Range of parameter values
        base_params: Base parameter configuration
        
    Returns:
        Parameter analysis results
    """
    analyzer = ValidationCurveAnalyzer(classifier_type)
    return analyzer.analyze_parameter(X, y, param_name, param_range, base_params)


def get_default_param_spaces() -> Dict[str, Dict[str, Any]]:
    """
    Get default parameter spaces for common classifiers.
    
    Returns:
        Dictionary mapping classifier types to their parameter spaces
    """
    return {
        'svm': {
            'C': [0.1, 1.0, 10.0, 100.0],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        },
        'random_forest': {
            'n_estimators': [50, 100, 200, 500],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2']
        },
        'logistic_regression': {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'lbfgs', 'saga']
        }
    }


def get_bayesian_param_spaces() -> Dict[str, Dict[str, Tuple]]:
    """
    Get Bayesian optimization parameter spaces for common classifiers.
    
    Returns:
        Dictionary mapping classifier types to their Bayesian parameter spaces
    """
    return {
        'svm': {
            'C': ('real', (0.01, 100.0)),
            'gamma': ('real', (0.001, 1.0)),
            'kernel': ('categorical', ['rbf', 'linear'])
        },
        'random_forest': {
            'n_estimators': ('integer', (50, 500)),
            'max_depth': ('integer', (5, 50)),
            'min_samples_split': ('integer', (2, 20)),
            'max_features': ('categorical', ['sqrt', 'log2'])
        },
        'logistic_regression': {
            'C': ('real', (0.001, 100.0)),
            'penalty': ('categorical', ['l1', 'l2'])
        }
    }


# Example usage and testing
if __name__ == "__main__":
    # Example: Optimize SVM hyperparameters
    param_space = {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
    
    # Create optimizer
    optimizer = HyperparameterSearchFactory.create_optimizer(
        'grid_search', 'svm', cv_folds=5
    )
    
    print(f"Created {optimizer.__class__.__name__} for SVM")
    print(f"Parameter space: {param_space}")