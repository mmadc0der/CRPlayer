"""
Main entry point for screen page classification experiments.
Provides a unified interface for running the complete pipeline.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from data_loader import DatasetConfig, DatasetInspector
from dataset_inspector import DatasetAnalyzer
from experiment_runner import ExperimentRunner, ExperimentConfig
from auto_labeler import AutoLabelingPipeline
from distillation_pipeline import DistillationPipeline, DistillationConfig
from models import ModelFactory

console = Console()


def inspect_data(args):
    """Inspect available annotation data."""
    console.print("[bold blue]Inspecting available annotation data...[/bold blue]")
    
    config = DatasetConfig()
    analyzer = DatasetAnalyzer(config)
    
    # Inspect available data
    data_info = analyzer.inspector.inspect_available_data()
    
    if not data_info['datasets']:
        console.print("[red]No datasets found! Please ensure the annotation API is running and has data.[/red]")
        return
    
    # Display available datasets
    table = Table(title="Available Datasets")
    table.add_column("Project", style="cyan")
    table.add_column("Dataset", style="magenta")
    table.add_column("Type", style="green")
    table.add_column("Samples", style="yellow")
    table.add_column("Classes", style="blue")
    
    for dataset in data_info['datasets']:
        progress = dataset['progress']
        table.add_row(
            dataset['project_name'],
            dataset['dataset_name'],
            dataset['target_type'],
            str(progress.get('labeled', 0)),
            str(dataset['num_classes'])
        )
    
    console.print(table)
    
    # If dataset ID specified, analyze it
    if args.dataset_id:
        console.print(f"\n[bold green]Analyzing dataset {args.dataset_id}...[/bold green]")
        report = analyzer.generate_comprehensive_report(args.dataset_id)
        console.print(f"\n[green]Analysis complete! Check the output directory for detailed visualizations.[/green]")


def run_experiment(args):
    """Run classification experiments."""
    console.print("[bold blue]Running classification experiments...[/bold blue]")
    
    # Parse model types
    model_types = args.models.split(',') if args.models else ['resnet50', 'efficientnet_b0']
    
    # Create experiment configuration
    config = ExperimentConfig(
        dataset_id=args.dataset_id,
        model_types=model_types,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        experiment_name=args.experiment_name
    )
    
    # Run experiments
    runner = ExperimentRunner(config)
    results = runner.run_full_experiment()
    
    console.print(Panel(
        f"Experiments completed successfully!\n"
        f"Best model: {results['best_model']['model_type']}\n"
        f"Best F1 score: {results['best_model']['f1_score']:.4f}\n"
        f"Results saved to: {results['output_directory']}",
        title="Experiment Summary",
        border_style="green"
    ))


def auto_label(args):
    """Run auto-labeling on a corpus of screenshots."""
    console.print("[bold blue]Running auto-labeling...[/bold blue]")
    
    # Load image paths
    if args.image_list:
        with open(args.image_list, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
    else:
        # Get image paths from directory
        image_dir = Path(args.image_dir)
        image_paths = list(image_dir.glob("**/*.jpg")) + list(image_dir.glob("**/*.png"))
        image_paths = [str(p) for p in image_paths]
    
    if not image_paths:
        console.print("[red]No images found![/red]")
        return
    
    # Initialize auto-labeling pipeline
    config = DatasetConfig()
    pipeline = AutoLabelingPipeline(
        model_path=args.model_path,
        model_type=args.model_type,
        num_classes=args.num_classes,
        config=config
    )
    
    # Process corpus
    results = pipeline.process_corpus(
        image_paths=image_paths,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold,
        active_learning_samples=args.active_learning_samples
    )
    
    console.print(Panel(
        f"Auto-labeling completed!\n"
        f"High confidence: {results['high_confidence_count']}\n"
        f"Low confidence: {results['low_confidence_count']}\n"
        f"Active learning samples: {results['active_learning_samples']}\n"
        f"Results saved to: {results['output_directory']}",
        title="Auto-labeling Summary",
        border_style="green"
    ))


def run_distillation(args):
    """Run knowledge distillation experiments."""
    console.print("[bold blue]Running knowledge distillation...[/bold blue]")
    
    # Create distillation configuration
    config = DistillationConfig(
        teacher_model_path=args.teacher_model_path,
        teacher_model_type=args.teacher_model_type,
        student_model_type=args.student_model_type,
        num_classes=args.num_classes,
        temperature=args.temperature,
        alpha=args.alpha,
        use_attention_transfer=args.attention_transfer,
        use_feature_matching=args.feature_matching
    )
    
    # Initialize pipeline
    data_config = DatasetConfig()
    pipeline = DistillationPipeline(config, data_config)
    
    # Load data (you would need to implement this based on your data structure)
    # train_loader, val_loader, test_loader = load_data_for_distillation(args.dataset_id)
    
    # Train student model
    # results = pipeline.train_student(train_loader, val_loader, num_epochs=args.epochs)
    
    # Compare models
    # comparison = pipeline.compare_models(test_loader)
    
    console.print("[green]Distillation pipeline initialized![/green]")


def list_models(args):
    """List available model types."""
    console.print("[bold blue]Available model types:[/bold blue]")
    
    models = ModelFactory.get_available_models()
    
    table = Table(title="Available Models")
    table.add_column("Model Type", style="cyan")
    table.add_column("Description", style="magenta")
    
    model_descriptions = {
        'resnet50': 'ResNet-50 (25M parameters)',
        'resnet101': 'ResNet-101 (44M parameters)',
        'efficientnet_b0': 'EfficientNet-B0 (5M parameters)',
        'efficientnet_b3': 'EfficientNet-B3 (12M parameters)',
        'vit_base': 'Vision Transformer Base (86M parameters)',
        'vit_large': 'Vision Transformer Large (307M parameters)',
        'convnext_tiny': 'ConvNeXt Tiny (29M parameters)',
        'convnext_small': 'ConvNeXt Small (50M parameters)',
        'lightweight': 'Custom Lightweight CNN (1M parameters)'
    }
    
    for model in models:
        description = model_descriptions.get(model, "Custom model")
        table.add_row(model, description)
    
    console.print(table)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Screen Page Classification Pipeline")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Inspect data command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect available annotation data')
    inspect_parser.add_argument('--dataset-id', type=int, help='Specific dataset ID to analyze')
    
    # Run experiment command
    experiment_parser = subparsers.add_parser('experiment', help='Run classification experiments')
    experiment_parser.add_argument('--dataset-id', type=int, required=True, help='Dataset ID to use')
    experiment_parser.add_argument('--models', type=str, help='Comma-separated list of model types')
    experiment_parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    experiment_parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    experiment_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    experiment_parser.add_argument('--experiment-name', type=str, help='Experiment name')
    
    # Auto-label command
    autolabel_parser = subparsers.add_parser('autolabel', help='Auto-label screenshot corpus')
    autolabel_parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    autolabel_parser.add_argument('--model-type', type=str, required=True, help='Model type')
    autolabel_parser.add_argument('--num-classes', type=int, required=True, help='Number of classes')
    autolabel_parser.add_argument('--image-dir', type=str, help='Directory containing images')
    autolabel_parser.add_argument('--image-list', type=str, help='File containing list of image paths')
    autolabel_parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    autolabel_parser.add_argument('--confidence-threshold', type=float, default=0.9, help='Confidence threshold')
    autolabel_parser.add_argument('--active-learning-samples', type=int, default=100, help='Number of active learning samples')
    
    # Distillation command
    distillation_parser = subparsers.add_parser('distill', help='Run knowledge distillation')
    distillation_parser.add_argument('--teacher-model-path', type=str, required=True, help='Path to teacher model')
    distillation_parser.add_argument('--teacher-model-type', type=str, required=True, help='Teacher model type')
    distillation_parser.add_argument('--student-model-type', type=str, required=True, help='Student model type')
    distillation_parser.add_argument('--num-classes', type=int, required=True, help='Number of classes')
    distillation_parser.add_argument('--temperature', type=float, default=3.0, help='Distillation temperature')
    distillation_parser.add_argument('--alpha', type=float, default=0.7, help='Distillation alpha')
    distillation_parser.add_argument('--attention-transfer', action='store_true', help='Use attention transfer')
    distillation_parser.add_argument('--feature-matching', action='store_true', help='Use feature matching')
    distillation_parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    
    # List models command
    subparsers.add_parser('list-models', help='List available model types')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    try:
        if args.command == 'inspect':
            inspect_data(args)
        elif args.command == 'experiment':
            run_experiment(args)
        elif args.command == 'autolabel':
            auto_label(args)
        elif args.command == 'distill':
            run_distillation(args)
        elif args.command == 'list-models':
            list_models(args)
        else:
            console.print(f"[red]Unknown command: {args.command}[/red]")
            parser.print_help()
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()