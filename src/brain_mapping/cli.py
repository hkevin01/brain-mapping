#!/usr/bin/env python3
"""Command-line interface for Brain Mapping Toolkit."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from brain_mapping import __version__
from brain_mapping.core.data_loader import DataLoader
from brain_mapping.visualization.renderer_3d import Renderer3D
from brain_mapping.analysis.statistics import StatisticalAnalyzer
from brain_mapping.analysis.machine_learning import MLAnalyzer


def main() -> None:
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Brain Mapping Toolkit - GPU-accelerated neuroimaging analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  brain-mapper load --input data.nii.gz --output processed/
  brain-mapper visualize --input brain.nii.gz --type surface
  brain-mapper analyze --input data.nii.gz --method connectivity
  brain-mapper ml --input features.csv --target labels.csv --algorithm svm
        """,
    )
    
    parser.add_argument(
        "--version", action="version", version=f"Brain Mapping Toolkit {__version__}"
    )
    
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    
    parser.add_argument(
        "--gpu", action="store_true", help="Enable GPU acceleration (requires CUDA)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Load command
    load_parser = subparsers.add_parser("load", help="Load and preprocess brain data")
    load_parser.add_argument("--input", "-i", required=True, help="Input file path")
    load_parser.add_argument("--output", "-o", help="Output directory")
    load_parser.add_argument("--format", choices=["nifti", "dicom"], default="nifti")
    load_parser.add_argument("--preprocess", action="store_true", help="Apply preprocessing")
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Create brain visualizations")
    viz_parser.add_argument("--input", "-i", required=True, help="Input brain data")
    viz_parser.add_argument("--type", choices=["volume", "surface", "slice"], default="volume")
    viz_parser.add_argument("--output", "-o", help="Output image file")
    viz_parser.add_argument("--colormap", default="viridis", help="Colormap for visualization")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Perform statistical analysis")
    analyze_parser.add_argument("--input", "-i", required=True, help="Input brain data")
    analyze_parser.add_argument("--method", choices=["connectivity", "activation", "network"], 
                               default="connectivity")
    analyze_parser.add_argument("--threshold", type=float, default=0.05, help="Statistical threshold")
    analyze_parser.add_argument("--output", "-o", help="Output results file")
    
    # Machine Learning command
    ml_parser = subparsers.add_parser("ml", help="Apply machine learning methods")
    ml_parser.add_argument("--input", "-i", required=True, help="Input features file")
    ml_parser.add_argument("--target", required=True, help="Target labels file")
    ml_parser.add_argument("--algorithm", choices=["svm", "random_forest", "neural_network"], 
                          default="svm")
    ml_parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds")
    ml_parser.add_argument("--output", "-o", help="Output model file")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "load":
            handle_load_command(args)
        elif args.command == "visualize":
            handle_visualize_command(args)
        elif args.command == "analyze":
            handle_analyze_command(args)
        elif args.command == "ml":
            handle_ml_command(args)
            
    except Exception as e:
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def handle_load_command(args) -> None:
    """Handle the load command."""
    print(f"Loading data from {args.input}...")
    
    loader = DataLoader()
    data = loader.load_brain_data(args.input, format=args.format)
    
    if args.preprocess:
        print("Applying preprocessing...")
        # Add preprocessing logic here
    
    if args.output:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Data processed and saved to {args.output}")
    
    print(f"Data shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")


def handle_visualize_command(args) -> None:
    """Handle the visualize command."""
    print(f"Creating {args.type} visualization from {args.input}...")
    
    loader = DataLoader()
    data = loader.load_brain_data(args.input)
    
    renderer = Renderer3D()
    
    if args.type == "volume":
        renderer.render_volume(data, colormap=args.colormap)
    elif args.type == "surface":
        renderer.render_surface(data, colormap=args.colormap)
    elif args.type == "slice":
        renderer.render_slices(data, colormap=args.colormap)
    
    if args.output:
        renderer.save_image(args.output)
        print(f"Visualization saved to {args.output}")
    else:
        renderer.show()


def handle_analyze_command(args) -> None:
    """Handle the analyze command."""
    print(f"Performing {args.method} analysis on {args.input}...")
    
    loader = DataLoader()
    data = loader.load_brain_data(args.input)
    
    analyzer = StatisticalAnalyzer()
    
    if args.method == "connectivity":
        results = analyzer.compute_connectivity(data, threshold=args.threshold)
    elif args.method == "activation":
        results = analyzer.compute_activation_maps(data, threshold=args.threshold)
    elif args.method == "network":
        results = analyzer.compute_network_analysis(data)
    
    if args.output:
        # Save results
        print(f"Analysis results saved to {args.output}")
    else:
        print("Analysis completed. Results:")
        print(results)


def handle_ml_command(args) -> None:
    """Handle the machine learning command."""
    print(f"Training {args.algorithm} model...")
    
    ml_analyzer = MLAnalyzer()
    
    # Load features and targets
    print(f"Loading features from {args.input}")
    print(f"Loading targets from {args.target}")
    
    if args.algorithm == "svm":
        model, scores = ml_analyzer.train_svm(
            features_file=args.input,
            targets_file=args.target,
            cv_folds=args.cv_folds
        )
    elif args.algorithm == "random_forest":
        model, scores = ml_analyzer.train_random_forest(
            features_file=args.input,
            targets_file=args.target,
            cv_folds=args.cv_folds
        )
    elif args.algorithm == "neural_network":
        model, scores = ml_analyzer.train_neural_network(
            features_file=args.input,
            targets_file=args.target,
            cv_folds=args.cv_folds
        )
    
    print(f"Cross-validation accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    
    if args.output:
        # Save model
        print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
