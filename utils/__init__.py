
# utils/__init__.py
"""Utility functions and classes for the fMRI analysis pipeline."""

from .metrics import (
    ModelEvaluator,
    compute_classification_metrics,
    compute_confusion_matrix,
    plot_confusion_matrix,
    compute_roc_curves,
    compute_feature_importance_stats
)
#from .visualization import (
#    plot_results,
#    plot_classifier_comparison,
#    plot_feature_importance,
#    plot_brain_activation,
#    create_interactive_plots,
#    save_publication_figures
#)
#from .io_utils import (
#    save_results,
#    load_results,
#    setup_directories,
#    save_model,
#    load_model,
#    export_to_csv,
#    create_experiment_report
#)
#
__all__ = [
    # Metrics
    'ModelEvaluator',
    'compute_classification_metrics',
    'compute_confusion_matrix', 
    'plot_confusion_matrix',
    'compute_roc_curves',
    'compute_feature_importance_stats',
    
    # Visualization
    'plot_results',
    'plot_classifier_comparison',
    'plot_feature_importance',
    'plot_brain_activation',
    'create_interactive_plots',
    'save_publication_figures',
    
    # I/O utilities
    'save_results',
    'load_results',
    'setup_directories',
    'save_model',
    'load_model',
    'export_to_csv',
    'create_experiment_report'
]
