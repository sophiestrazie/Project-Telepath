# utils/visualization.py
"""
Comprehensive visualization utilities for multimodal fMRI stimulus prediction.

This module provides visualization tools for experiment results, classifier comparisons,
feature importance analysis, and brain activation patterns following SOLID principles.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings
import logging
from pathlib import Path

# Interactive plotting
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available. Interactive plots will be disabled.")

# Brain visualization
try:
    import nibabel as nib
    from nilearn import plotting, datasets
    from nilearn.image import load_img
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False
    logging.warning("Nilearn not available. Brain visualization will be disabled.")

# Ensure project imports work
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")


class ExperimentVisualizer:
    """
    Main visualizer class for fMRI experiment results.
    
    Provides comprehensive visualization capabilities for classifier comparison,
    performance analysis, and result interpretation.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size
            dpi: Resolution for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi
        self.color_palette = sns.color_palette("husl", 10)
        
    def plot_classifier_comparison(self, 
                                 results: Dict[str, Any],
                                 metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1'],
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive classifier comparison plot.
        
        Args:
            results: Experiment results dictionary
            metrics: Metrics to compare
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        if 'summary' not in results or 'classifier_ranking' not in results['summary']:
            raise ValueError("Results must contain 'summary' with 'classifier_ranking'")
        
        ranking = results['summary']['classifier_ranking']
        n_classifiers = len(ranking)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Accuracy ranking bar plot
        ax1 = fig.add_subplot(gs[0, 0])
        classifiers = [clf['classifier'] for clf in ranking]
        accuracies = [clf['mean_accuracy'] for clf in ranking]
        errors = [clf['std_accuracy'] for clf in ranking]
        
        bars = ax1.barh(classifiers, accuracies, xerr=errors, capsize=5,
                       color=self.color_palette[:n_classifiers])
        ax1.set_xlabel('Cross-Validation Accuracy')
        ax1.set_title('Classifier Performance Ranking')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, acc, err) in enumerate(zip(bars, accuracies, errors)):
            ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{acc:.3f}±{err:.3f}', va='center', fontsize=9)
        
        # 2. Runtime comparison
        ax2 = fig.add_subplot(gs[0, 1])
        runtimes = [clf.get('runtime', 0) for clf in ranking]
        
        bars2 = ax2.bar(range(len(classifiers)), runtimes, 
                       color=self.color_palette[:n_classifiers])
        ax2.set_xlabel('Classifier')
        ax2.set_ylabel('Runtime (seconds)')
        ax2.set_title('Training Runtime Comparison')
        ax2.set_xticks(range(len(classifiers)))
        ax2.set_xticklabels([clf[:8] for clf in classifiers], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, runtime in zip(bars2, runtimes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(runtimes),
                    f'{runtime:.1f}s', ha='center', va='bottom', fontsize=9)
        
        # 3. Detailed metrics comparison
        ax3 = fig.add_subplot(gs[1, :])
        
        # Get detailed metrics from classification reports
        metrics_data = []
        available_classifiers = []
        
        for clf_name in classifiers:
            if clf_name in results.get('classifiers', {}):
                clf_results = results['classifiers'][clf_name]
                if 'classification_report' in clf_results:
                    report = clf_results['classification_report']
                    if 'weighted avg' in report:
                        metrics_row = {
                            'Classifier': clf_name,
                            'Accuracy': accuracies[classifiers.index(clf_name)],
                            'Precision': report['weighted avg'].get('precision', 0),
                            'Recall': report['weighted avg'].get('recall', 0),
                            'F1-Score': report['weighted avg'].get('f1-score', 0)
                        }
                        metrics_data.append(metrics_row)
                        available_classifiers.append(clf_name)
        
        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data)
            
            # Create grouped bar chart
            x = np.arange(len(available_classifiers))
            width = 0.2
            
            for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1-Score']):
                if metric in df_metrics.columns:
                    values = df_metrics[metric].values
                    ax3.bar(x + i*width, values, width, label=metric, 
                           color=self.color_palette[i], alpha=0.8)
            
            ax3.set_xlabel('Classifier')
            ax3.set_ylabel('Score')
            ax3.set_title('Detailed Performance Metrics Comparison')
            ax3.set_xticks(x + width * 1.5)
            ax3.set_xticklabels([clf[:10] for clf in available_classifiers], rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1.1)
        
        plt.suptitle('Classifier Comparison Analysis', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Classifier comparison plot saved to {save_path}")
        
        return fig
    
    def plot_confusion_matrices(self,
                              results: Dict[str, Any],
                              classifiers: Optional[List[str]] = None,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrices for multiple classifiers.
        
        Args:
            results: Experiment results dictionary
            classifiers: List of classifier names to plot
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        if 'classifiers' not in results:
            raise ValueError("Results must contain 'classifiers' section")
        
        available_classifiers = list(results['classifiers'].keys())
        if classifiers is None:
            classifiers = available_classifiers[:4]  # Limit to 4 for display
        
        n_plots = len(classifiers)
        cols = min(2, n_plots)
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if n_plots == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, clf_name in enumerate(classifiers):
            if clf_name not in results['classifiers']:
                continue
                
            clf_results = results['classifiers'][clf_name]
            if 'confusion_matrix' not in clf_results:
                continue
            
            cm = np.array(clf_results['confusion_matrix'])
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot
            im = axes[i].imshow(cm_normalized, interpolation='nearest', cmap='Blues')
            axes[i].set_title(f'{clf_name}\nAccuracy: {clf_results.get("mean_cv_score", 0):.3f}')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            
            # Add text annotations
            thresh = cm_normalized.max() / 2.
            for row in range(cm.shape[0]):
                for col in range(cm.shape[1]):
                    axes[i].text(col, row, f'{cm[row, col]}\n({cm_normalized[row, col]:.2f})',
                               ha="center", va="center",
                               color="white" if cm_normalized[row, col] > thresh else "black",
                               fontsize=10)
            
            axes[i].set_ylabel('True Label')
            axes[i].set_xlabel('Predicted Label')
        
        # Hide empty subplots
        for i in range(len(classifiers), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Confusion Matrices Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Confusion matrices plot saved to {save_path}")
        
        return fig
    
    def plot_feature_importance(self,
                              results: Dict[str, Any],
                              classifier_name: str,
                              top_k: int = 20,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance for a specific classifier.
        
        Args:
            results: Experiment results dictionary
            classifier_name: Name of classifier to plot
            top_k: Number of top features to show
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        if classifier_name not in results.get('classifiers', {}):
            raise ValueError(f"Classifier {classifier_name} not found in results")
        
        clf_results = results['classifiers'][classifier_name]
        if 'feature_importance' not in clf_results or clf_results['feature_importance'] is None:
            raise ValueError(f"Feature importance not available for {classifier_name}")
        
        importance = np.array(clf_results['feature_importance'])
        
        # Get top features
        top_indices = np.argsort(importance)[-top_k:][::-1]
        top_importance = importance[top_indices]
        feature_names = [f'Feature_{i}' for i in top_indices]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Bar plot of top features
        bars = ax1.barh(range(len(top_importance)), top_importance, 
                       color=self.color_palette[0])
        ax1.set_yticks(range(len(top_importance)))
        ax1.set_yticklabels(feature_names)
        ax1.set_xlabel('Feature Importance')
        ax1.set_title(f'Top {top_k} Features - {classifier_name}')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, imp) in enumerate(zip(bars, top_importance)):
            ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{imp:.4f}', va='center', fontsize=9)
        
        # Histogram of all feature importances
        ax2.hist(importance, bins=50, color=self.color_palette[1], alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(importance), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(importance):.4f}')
        ax2.axvline(np.median(importance), color='orange', linestyle='--',
                   label=f'Median: {np.median(importance):.4f}')
        ax2.set_xlabel('Feature Importance')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Feature Importances')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Feature Importance Analysis - {classifier_name}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        return fig
    
    def plot_learning_curves(self,
                           learning_curve_results: Dict[str, Any],
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot learning curves for model performance analysis.
        
        Args:
            learning_curve_results: Results from learning curve analysis
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        train_sizes = learning_curve_results['train_sizes']
        train_mean = learning_curve_results['train_mean']
        train_std = learning_curve_results['train_std']
        val_mean = learning_curve_results['validation_mean']
        val_std = learning_curve_results['validation_std']
        
        # Learning curve
        axes[0].plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        axes[0].fill_between(train_sizes, 
                           np.array(train_mean) - np.array(train_std),
                           np.array(train_mean) + np.array(train_std),
                           alpha=0.2, color='blue')
        
        axes[0].plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        axes[0].fill_between(train_sizes,
                           np.array(val_mean) - np.array(val_std),
                           np.array(val_mean) + np.array(val_std),
                           alpha=0.2, color='red')
        
        axes[0].set_xlabel('Training Set Size')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Learning Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Performance gap analysis
        gap = np.array(train_mean) - np.array(val_mean)
        axes[1].plot(train_sizes, gap, 'o-', color='green', label='Training-Validation Gap')
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Training Set Size')
        axes[1].set_ylabel('Performance Gap')
        axes[1].set_title('Overfitting Analysis')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Learning curves plot saved to {save_path}")
        
        return fig
    
    def plot_hyperparameter_analysis(self,
                                   validation_results: Dict[str, Any],
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot hyperparameter validation curves.
        
        Args:
            validation_results: Results from validation curve analysis
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        param_name = validation_results['param_name']
        param_range = validation_results['param_range']
        train_mean = validation_results['train_mean']
        train_std = validation_results['train_std']
        val_mean = validation_results['validation_mean']
        val_std = validation_results['validation_std']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Convert param_range to string for plotting if needed
        x_values = range(len(param_range))
        x_labels = [str(p) for p in param_range]
        
        ax.plot(x_values, train_mean, 'o-', color='blue', label='Training Score')
        ax.fill_between(x_values,
                       np.array(train_mean) - np.array(train_std),
                       np.array(train_mean) + np.array(train_std),
                       alpha=0.2, color='blue')
        
        ax.plot(x_values, val_mean, 'o-', color='red', label='Validation Score')
        ax.fill_between(x_values,
                       np.array(val_mean) - np.array(val_std),
                       np.array(val_mean) + np.array(val_std),
                       alpha=0.2, color='red')
        
        # Mark best parameter
        best_idx = np.argmax(val_mean)
        ax.axvline(x=best_idx, color='green', linestyle='--', alpha=0.7,
                  label=f'Best: {param_range[best_idx]}')
        
        ax.set_xlabel(param_name)
        ax.set_ylabel('Score')
        ax.set_title(f'Validation Curve for {param_name}')
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_labels, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Hyperparameter analysis plot saved to {save_path}")
        
        return fig


class InteractiveVisualizer:
    """
    Interactive visualization using Plotly for web-based exploration.
    """
    
    def __init__(self):
        """Initialize interactive visualizer."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive visualization")
    
    def create_interactive_comparison(self, results: Dict[str, Any]) -> go.Figure:
        """
        Create interactive classifier comparison plot.
        
        Args:
            results: Experiment results dictionary
            
        Returns:
            Plotly figure object
        """
        ranking = results['summary']['classifier_ranking']
        
        # Prepare data
        classifiers = [clf['classifier'] for clf in ranking]
        accuracies = [clf['mean_accuracy'] for clf in ranking]
        std_errors = [clf['std_accuracy'] for clf in ranking]
        runtimes = [clf.get('runtime', 0) for clf in ranking]
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy Comparison', 'Runtime Analysis', 
                          'Accuracy vs Runtime', 'Error Bars'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Accuracy bar chart
        fig.add_trace(
            go.Bar(x=classifiers, y=accuracies, name='Accuracy',
                  error_y=dict(type='data', array=std_errors),
                  hovertemplate='<b>%{x}</b><br>Accuracy: %{y:.4f}<br>Std: %{error_y.array:.4f}<extra></extra>'),
            row=1, col=1
        )
        
        # Runtime bar chart
        fig.add_trace(
            go.Bar(x=classifiers, y=runtimes, name='Runtime',
                  hovertemplate='<b>%{x}</b><br>Runtime: %{y:.2f}s<extra></extra>'),
            row=1, col=2
        )
        
        # Scatter plot: Accuracy vs Runtime
        fig.add_trace(
            go.Scatter(x=runtimes, y=accuracies, mode='markers+text',
                      text=classifiers, textposition='top center',
                      name='Accuracy vs Runtime',
                      marker=dict(size=10, color=accuracies, colorscale='Viridis'),
                      hovertemplate='<b>%{text}</b><br>Runtime: %{x:.2f}s<br>Accuracy: %{y:.4f}<extra></extra>'),
            row=2, col=1
        )
        
        # Error bars detailed view
        fig.add_trace(
            go.Scatter(x=classifiers, y=accuracies,
                      error_y=dict(type='data', array=std_errors, visible=True),
                      mode='markers', name='Detailed Errors',
                      hovertemplate='<b>%{x}</b><br>Accuracy: %{y:.4f}±%{error_y.array:.4f}<extra></extra>'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Interactive Classifier Comparison Dashboard",
            showlegend=False,
            height=800
        )
        
        return fig
    
    def create_feature_importance_plot(self, 
                                     importance: np.ndarray,
                                     feature_names: Optional[List[str]] = None,
                                     top_k: int = 20) -> go.Figure:
        """
        Create interactive feature importance plot.
        
        Args:
            importance: Feature importance values
            feature_names: Names of features
            top_k: Number of top features to show
            
        Returns:
            Plotly figure object
        """
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importance))]
        
        # Get top features
        top_indices = np.argsort(importance)[-top_k:][::-1]
        top_importance = importance[top_indices]
        top_names = [feature_names[i] for i in top_indices]
        
        fig = go.Figure()
        
        # Add bar chart
        fig.add_trace(go.Bar(
            x=top_importance,
            y=top_names,
            orientation='h',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.6f}<extra></extra>',
            marker=dict(color=top_importance, colorscale='Viridis')
        ))
        
        fig.update_layout(
            title=f'Top {top_k} Feature Importances',
            xaxis_title='Importance',
            yaxis_title='Features',
            height=600,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig


class BrainVisualizer:
    """
    Brain visualization for neuroimaging data analysis.
    """
    
    def __init__(self):
        """Initialize brain visualizer."""
        if not NILEARN_AVAILABLE:
            raise ImportError("Nilearn is required for brain visualization")
    
    def plot_brain_activation(self,
                            activation_map: np.ndarray,
                            affine: np.ndarray,
                            threshold: float = 0.001,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot brain activation maps.
        
        Args:
            activation_map: 3D activation map
            affine: Affine transformation matrix
            threshold: Threshold for display
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        # Create NIfTI-like image
        from nilearn import image
        
        # Ensure 3D
        if activation_map.ndim == 4:
            activation_map = activation_map.mean(axis=-1)
        
        # Create temporary image
        img = image.new_img_like(
            reference_niimg=datasets.load_mni152_template(),
            data=activation_map,
            affine=affine
        )
        
        # Create figure
        fig = plt.figure(figsize=(15, 5))
        
        # Plot glass brain
        plotting.plot_glass_brain(
            img, threshold=threshold, colorbar=True,
            plot_abs=False, display_mode='lyrz',
            figure=fig
        )
        
        plt.suptitle('Brain Activation Pattern', fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Brain activation plot saved to {save_path}")
        
        return fig
    
    def plot_feature_brain_map(self,
                             feature_importance: np.ndarray,
                             brain_mask: np.ndarray,
                             affine: np.ndarray,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance on brain template.
        
        Args:
            feature_importance: 1D feature importance values
            brain_mask: 3D brain mask
            affine: Affine transformation matrix
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        # Map features back to brain space
        brain_importance = np.zeros(brain_mask.shape)
        brain_importance[brain_mask.astype(bool)] = feature_importance
        
        return self.plot_brain_activation(brain_importance, affine, save_path=save_path)


def plot_results(results: Dict[str, Any], 
                output_dir: Union[str, Path],
                create_interactive: bool = True) -> None:
    """
    Main function to generate all plots for experiment results.
    
    Args:
        results: Experiment results dictionary
        output_dir: Directory to save plots
        create_interactive: Whether to create interactive plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = ExperimentVisualizer()
    
    # Generate static plots
    try:
        # Classifier comparison
        fig1 = visualizer.plot_classifier_comparison(results)
        fig1.savefig(output_dir / 'classifier_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        # Confusion matrices
        fig2 = visualizer.plot_confusion_matrices(results)
        fig2.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        # Feature importance for best classifier
        if 'summary' in results and 'best_classifier' in results['summary']:
            best_clf = results['summary']['best_classifier']
            try:
                fig3 = visualizer.plot_feature_importance(results, best_clf)
                fig3.savefig(output_dir / f'feature_importance_{best_clf}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close(fig3)
            except Exception as e:
                logger.warning(f"Could not plot feature importance for {best_clf}: {e}")
        
        logger.info(f"Static plots saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error generating static plots: {e}")
    
    # Generate interactive plots
    if create_interactive and PLOTLY_AVAILABLE:
        try:
            interactive_viz = InteractiveVisualizer()
            
            # Interactive comparison
            fig_interactive = interactive_viz.create_interactive_comparison(results)
            fig_interactive.write_html(str(output_dir / 'interactive_comparison.html'))
            
            logger.info(f"Interactive plots saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error generating interactive plots: {e}")


def save_publication_figures(results: Dict[str, Any],
                           output_dir: Union[str, Path],
                           format: str = 'pdf') -> None:
    """
    Save publication-ready figures.
    
    Args:
        results: Experiment results dictionary
        output_dir: Directory to save figures
        format: Output format ('pdf', 'png', 'svg')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set publication style
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.direction': 'out',
        'ytick.direction': 'out'
    })
    
    visualizer = ExperimentVisualizer(figsize=(8, 6), dpi=300)
    
    # Generate publication figures
    try:
        # Main comparison figure
        fig = visualizer.plot_classifier_comparison(results)
        fig.savefig(output_dir / f'figure_1_comparison.{format}', 
                   bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        # Confusion matrix figure
        fig = visualizer.plot_confusion_matrices(results)
        fig.savefig(output_dir / f'figure_2_confusion.{format}', 
                   bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        logger.info(f"Publication figures saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error saving publication figures: {e}")
    
    # Reset style
    plt.rcdefaults()


# Example usage and testing
if __name__ == "__main__":
    # Create sample results for testing
    sample_results = {
        'summary': {
            'best_classifier': 'svm',
            'best_accuracy': 0.85,
            'classifier_ranking': [
                {'classifier': 'svm', 'mean_accuracy': 0.85, 'std_accuracy': 0.02, 'runtime': 120},
                {'classifier': 'random_forest', 'mean_accuracy': 0.82, 'std_accuracy': 0.03, 'runtime': 45},
                {'classifier': 'logistic_regression', 'mean_accuracy': 0.78, 'std_accuracy': 0.025, 'runtime': 20}
            ]
        },
        'classifiers': {
            'svm': {
                'confusion_matrix': [[50, 5], [3, 42]],
                'feature_importance': np.random.rand(100),
                'mean_cv_score': 0.85
            }
        }
    }
    
    # Test visualization
    visualizer = ExperimentVisualizer()
    fig = visualizer.plot_classifier_comparison(sample_results)
    plt.show()
    
    print("Visualization module test completed")