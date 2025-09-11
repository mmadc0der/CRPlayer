"""
Experiment runner for screen page classification.
Orchestrates the complete pipeline from data loading to model training and evaluation.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from data_loader import DatasetInspector, DatasetConfig, create_data_loaders
from dataset_inspector import DatasetAnalyzer
from models import ModelFactory, get_model_info
from trainer import ClassificationTrainer, DistillationTrainer, get_device

console = Console()


class ExperimentConfig:
  """Configuration for running experiments."""

  def __init__(self,
               dataset_id: int,
               model_types: List[str] = None,
               num_epochs: int = 100,
               learning_rate: float = 1e-4,
               batch_size: int = 32,
               image_size: Tuple[int, int] = (224, 224),
               use_class_weights: bool = True,
               use_data_augmentation: bool = True,
               experiment_name: str = None,
               output_dir: str = "./experiments"):
    self.dataset_id = dataset_id
    self.model_types = model_types or ['resnet50', 'efficientnet_b0', 'vit_base']
    self.num_epochs = num_epochs
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.image_size = image_size
    self.use_class_weights = use_class_weights
    self.use_data_augmentation = use_data_augmentation
    self.experiment_name = experiment_name or f"experiment_{int(time.time())}"
    self.output_dir = output_dir


class ExperimentRunner:
  """Main class for running classification experiments."""

  def __init__(self, config: ExperimentConfig):
    self.config = config
    self.data_config = DatasetConfig(batch_size=config.batch_size, image_size=config.image_size)
    self.inspector = DatasetInspector(self.data_config)
    self.analyzer = DatasetAnalyzer(self.data_config)

    # Results storage
    self.results = {}
    self.best_models = {}

    console.print(f"[bold blue]Experiment Runner initialized[/bold blue]")
    console.print(f"[blue]Dataset ID: {config.dataset_id}[/blue]")
    console.print(f"[blue]Models to test: {', '.join(config.model_types)}[/blue]")

  def run_full_experiment(self) -> Dict[str, Any]:
    """Run the complete experiment pipeline."""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

      # Step 1: Data inspection and preparation
      task1 = progress.add_task("Inspecting and preparing data...", total=None)
      data_info = self._prepare_data()
      progress.update(task1, description="Data preparation complete")

      # Step 2: Run experiments for each model
      task2 = progress.add_task("Running model experiments...", total=len(self.config.model_types))

      for i, model_type in enumerate(self.config.model_types):
        progress.update(task2, description=f"Training {model_type}...")
        self._run_single_experiment(model_type, data_info)
        progress.update(task2, advance=1)

      progress.update(task2, description="All experiments complete")

      # Step 3: Compare results
      task3 = progress.add_task("Comparing results...", total=None)
      comparison_results = self._compare_results()
      progress.update(task3, description="Comparison complete")

    # Generate final report
    final_report = self._generate_final_report(comparison_results)

    console.print(
      Panel(
        f"Experiment completed successfully!\n"
        f"Best model: {final_report['best_model']['model_type']}\n"
        f"Best F1 score: {final_report['best_model']['f1_score']:.4f}\n"
        f"Results saved to: {final_report['output_directory']}",
        title="Experiment Summary",
        border_style="green"))

    return final_report

  def _prepare_data(self) -> Dict[str, Any]:
    """Prepare data for training."""

    # First, inspect the dataset
    console.print("[blue]Inspecting dataset...[/blue]")
    data_info = self.inspector.inspect_available_data()

    # Find the target dataset
    target_dataset = None
    for dataset in data_info['datasets']:
      if dataset['dataset_id'] == self.config.dataset_id:
        target_dataset = dataset
        break

    if not target_dataset:
      raise ValueError(f"Dataset {self.config.dataset_id} not found!")

    console.print(f"[green]Found dataset: {target_dataset['dataset_name']}[/green]")

    # Download and prepare the dataset
    console.print("[blue]Downloading dataset...[/blue]")
    download_result = self.inspector.download_dataset(self.config.dataset_id)

    # Load the labeled data
    labeled_data = download_result['dataframe']
    class_to_idx = download_result['metadata']['class_to_idx']
    idx_to_class = download_result['metadata']['idx_to_class']

    console.print(f"[green]Dataset loaded: {len(labeled_data)} samples, {len(class_to_idx)} classes[/green]")

    # Normalize labels to 0-based contiguous indices using metadata class IDs
    # Build mapping from API class_id -> 0..num_classes-1 (sorted by class_id)
    all_class_ids_sorted = sorted(idx_to_class.keys())
    id_to_idx = {cid: i for i, cid in enumerate(all_class_ids_sorted)}

    labeled_data_norm = labeled_data.copy()
    labeled_data_norm['class_id_norm'] = labeled_data_norm['class_id'].map(id_to_idx)

    # Prepare records with explicit normalized 'single_label_class_id' so Dataset uses it
    records_norm = []
    for row in labeled_data_norm.to_dict('records'):
      rec = dict(row)
      rec['single_label_class_id'] = int(row['class_id_norm']) if row['class_id_norm'] is not None else 0
      records_norm.append(rec)

    # Create data loaders
    console.print("[blue]Creating data loaders...[/blue]")
    train_loader, val_loader, test_loader = create_data_loaders(records_norm,
                                                                self.data_config.data_root,
                                                                class_to_idx,
                                                                self.data_config,
                                                                test_size=0.2,
                                                                val_size=0.1)

    # Compute class weights if needed
    class_weights = None
    if self.config.use_class_weights:
      class_weights = self._compute_class_weights(labeled_data_norm, len(class_to_idx))

    return {
      'train_loader': train_loader,
      'val_loader': val_loader,
      'test_loader': test_loader,
      'class_to_idx': class_to_idx,
      'idx_to_class': idx_to_class,
      'num_classes': len(class_to_idx),
      'class_weights': class_weights,
      'download_result': download_result
    }

  def _compute_class_weights(self, df, num_classes: int) -> torch.Tensor:
    """Compute class weights for handling imbalanced data."""
    # Use normalized 0-based class indices for weighting
    if 'class_id_norm' not in df.columns:
      raise ValueError("Expected 'class_id_norm' column in dataframe for class weighting")

    y = df['class_id_norm'].astype(int).values
    unique_classes = np.sort(df['class_id_norm'].astype(int).unique())

    console.print(
      f"[blue]Computing class weights: {unique_classes.size} unique classes in data (normalized), {num_classes} total classes expected[/blue]"
    )
    console.print(f"[blue]Normalized class indices present: {unique_classes.tolist()}[/blue]")

    # Compute class weights for the classes present in the data (expects numpy arrays)
    class_weights_present = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y)

    # Create a full weight tensor for all classes (as expected by the model)
    full_class_weights = torch.ones(num_classes)

    # Map the computed weights to the correct positions (already normalized indices)
    for idx, w in zip(unique_classes.tolist(), class_weights_present.tolist()):
      if 0 <= idx < num_classes:
        full_class_weights[idx] = float(w)

    console.print(f"[blue]Final class weights shape: {full_class_weights.shape}[/blue]")
    return full_class_weights

  def _run_single_experiment(self, model_type: str, data_info: Dict[str, Any]) -> Dict[str, Any]:
    """Run experiment for a single model type."""

    console.print(f"[bold green]Running experiment for {model_type}[/bold green]")

    # Create model
    model = ModelFactory.create_model(model_type, num_classes=data_info['num_classes'], pretrained=True)

    # Create trainer
    experiment_name = f"{self.config.experiment_name}_{model_type}"
    trainer = ClassificationTrainer(
      model=model,
      config=self.data_config,
      experiment_name=experiment_name,
      use_wandb=False,  # Set to True if you want to use Weights & Biases
      use_tensorboard=True)

    # Train the model
    training_results = trainer.train(train_loader=data_info['train_loader'],
                                     val_loader=data_info['val_loader'],
                                     num_epochs=self.config.num_epochs,
                                     learning_rate=self.config.learning_rate,
                                     class_weights=data_info['class_weights'])

    # Evaluate on test set
    device = get_device()
    # Ensure model is on the correct device before evaluation
    trainer.model.to(device)
    test_results = trainer.evaluate(data_info['test_loader'], device)

    # Store results
    experiment_result = {
      'model_type': model_type,
      'model_info': get_model_info(model),
      'training_results': training_results,
      'test_results': test_results,
      'experiment_name': experiment_name,
      'output_directory': str(trainer.output_dir)
    }

    self.results[model_type] = experiment_result

    # Track best model
    f1_score = test_results['test_metrics']['f1_score']
    if not self.best_models or f1_score > self.best_models['f1_score']:
      self.best_models = {'model_type': model_type, 'f1_score': f1_score, 'experiment_result': experiment_result}

    console.print(f"[green]{model_type} experiment completed! F1: {f1_score:.4f}[/green]")

    return experiment_result

  def _compare_results(self) -> Dict[str, Any]:
    """Compare results across all models."""

    comparison_table = Table(title="Model Comparison")
    comparison_table.add_column("Model", style="cyan")
    comparison_table.add_column("Parameters", style="blue")
    comparison_table.add_column("Size (MB)", style="green")
    comparison_table.add_column("Test F1", style="magenta")
    comparison_table.add_column("Test Acc", style="yellow")

    model_comparison = []

    for model_type, result in self.results.items():
      model_info = result['model_info']
      test_metrics = result['test_results']['test_metrics']

      comparison_table.add_row(model_type, f"{model_info['parameters']['total_parameters']:,}",
                               f"{model_info['model_size_mb']:.2f}", f"{test_metrics['f1_score']:.4f}",
                               f"{test_metrics['accuracy']:.4f}")

      model_comparison.append({
        'model_type': model_type,
        'parameters': model_info['parameters']['total_parameters'],
        'model_size_mb': model_info['model_size_mb'],
        'f1_score': test_metrics['f1_score'],
        'accuracy': test_metrics['accuracy'],
        'precision': test_metrics['precision'],
        'recall': test_metrics['recall']
      })

    console.print(comparison_table)

    # Sort by F1 score
    model_comparison.sort(key=lambda x: x['f1_score'], reverse=True)

    return {
      'model_comparison': model_comparison,
      'best_model': model_comparison[0] if model_comparison else None,
      'worst_model': model_comparison[-1] if model_comparison else None
    }

  def _generate_final_report(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate the final experiment report."""

    # Create output directory
    output_dir = Path(self.config.output_dir) / self.config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate report
    final_report = {
      'experiment_config': {
        'dataset_id': self.config.dataset_id,
        'model_types': self.config.model_types,
        'num_epochs': self.config.num_epochs,
        'learning_rate': self.config.learning_rate,
        'batch_size': self.config.batch_size,
        'image_size': self.config.image_size
      },
      'model_comparison': comparison_results['model_comparison'],
      'best_model': comparison_results['best_model'],
      'worst_model': comparison_results['worst_model'],
      'detailed_results': self.results,
      'output_directory': str(output_dir)
    }

    # Save report
    with open(output_dir / "final_report.json", 'w') as f:
      json.dump(final_report, f, indent=2, default=str)

    return final_report

  def run_distillation_experiment(self, teacher_model_type: str, student_model_type: str,
                                  data_info: Dict[str, Any]) -> Dict[str, Any]:
    """Run knowledge distillation experiment."""

    console.print(f"[bold blue]Running distillation: {teacher_model_type} -> {student_model_type}[/bold blue]")

    # Load the best teacher model
    if teacher_model_type not in self.results:
      raise ValueError(f"Teacher model {teacher_model_type} not found in results!")

    teacher_experiment = self.results[teacher_model_type]
    teacher_model = ModelFactory.create_model(teacher_model_type, num_classes=data_info['num_classes'], pretrained=True)

    # Load teacher weights (you would need to implement this based on your checkpoint format)
    # teacher_model.load_state_dict(torch.load(teacher_experiment['output_directory'] + '/best_model.pth'))

    # Create student model
    student_model = ModelFactory.create_model(student_model_type, num_classes=data_info['num_classes'], pretrained=True)

    # Create distillation trainer
    experiment_name = f"{self.config.experiment_name}_distillation_{teacher_model_type}_to_{student_model_type}"
    trainer = DistillationTrainer(model=student_model, config=self.data_config, experiment_name=experiment_name)

    # Train with distillation
    device = get_device()
    distillation_results = trainer.train_with_distillation(train_loader=data_info['train_loader'],
                                                           val_loader=data_info['val_loader'],
                                                           teacher_model=teacher_model,
                                                           num_epochs=self.config.num_epochs,
                                                           learning_rate=self.config.learning_rate,
                                                           device=device)

    # Evaluate student model
    trainer.model.to(device)  # Ensure model is on correct device
    test_results = trainer.evaluate(data_info['test_loader'], device)

    return {
      'teacher_model': teacher_model_type,
      'student_model': student_model_type,
      'distillation_results': distillation_results,
      'test_results': test_results,
      'experiment_name': experiment_name
    }


def main():
  """Main function for running experiments."""

  # Example configuration
  config = ExperimentConfig(
    dataset_id=1,  # Replace with actual dataset ID
    model_types=['resnet50', 'efficientnet_b0', 'lightweight'],
    num_epochs=50,
    learning_rate=1e-4,
    batch_size=32)

  # Create and run experiment
  runner = ExperimentRunner(config)
  results = runner.run_full_experiment()

  console.print("[green]Experiment completed successfully![/green]")


if __name__ == "__main__":
  main()
