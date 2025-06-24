# main.py
#!/usr/bin/env python3
"""
Main script for running multimodal fMRI stimulus prediction experiments
"""

import argparse
from pathlib import Path
import yaml
import sys
import os
sys.path.append(os.path.abspath(r'C:\Users\PC\Desktop\AI4Good - Brain Project\AI4Good-MTL-Group-2\scripts\v2-strass-branch'))
from experiments.experiment_runner import ExperimentRunner
#from utils.visualization import plot_results
#from utils.io_utils import setup_directories

def main():
    parser = argparse.ArgumentParser(description='Run fMRI stimulus prediction experiments')
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config file')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name to run')
    parser.add_argument('--subjects', type=str, nargs='+', required=True, help='Subject IDs to include')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    
    args = parser.parse_args()
    
    # Setup directories
    output_dir = Path(args.output)
    #setup_directories(output_dir)
    
    # Run experiment
    runner = ExperimentRunner(args.config)
    results = runner.run_experiment(args.experiment, args.subjects)
    
    # Save results
    results_file = output_dir / f"{args.experiment}_results.json"
    runner.save_experiment_results(results, str(results_file))
    
    # Print summary
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Best Classifier: {results['summary']['best_classifier']}")
    print(f"Best Accuracy: {results['summary']['best_accuracy']:.4f}")
    print("\nRanking:")
    for i, clf in enumerate(results['summary']['classifier_ranking']):
        print(f"{i+1}. {clf['classifier']}: {clf['mean_accuracy']:.4f} ± {clf['std_accuracy']:.4f}")
    
    # Generate plots if requested
    #if args.plot:
        #plot_results(results, output_dir)
        #print(f"\nPlots saved to: {output_dir}")

if __name__ == "__main__":
    main()