"""
Training utilities for screen page classification experiments.
Includes training loops, evaluation metrics, and experiment management.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report)
import wandb
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

from models import BaseClassifier, DistillationLoss, get_model_info
from data_loader import DatasetConfig

console = Console()


def get_device() -> torch.device:
  """Get the appropriate device for training."""
  try:
    from device_utils import get_global_device_manager
    return get_global_device_manager().get_device()
  except ImportError:
    # Fallback to simple device detection
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MetricsTracker:
  """Track and compute various metrics during training."""

  def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
    self.num_classes = num_classes
    self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
    self.reset()

  def reset(self):
    """Reset all metrics."""
    self.predictions = []
    self.targets = []
    self.losses = []

  def update(self, predictions: torch.Tensor, targets: torch.Tensor, loss: float):
    """Update metrics with new batch."""
    self.predictions.extend(predictions.cpu().numpy())
    self.targets.extend(targets.cpu().numpy())
    self.losses.append(loss)

  def compute_metrics(self) -> Dict[str, Any]:
    """Compute comprehensive metrics."""
    if not self.predictions or not self.targets:
      return {}

    predictions = np.array(self.predictions)
    targets = np.array(self.targets)

    # Basic metrics
    accuracy = accuracy_score(targets, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(targets,
                                                                     predictions,
                                                                     average='weighted',
                                                                     zero_division=0)

    # Per-class metrics
    # Ensure per-class arrays are aligned to all classes [0..num_classes-1]
    all_labels = list(range(self.num_classes))
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
      targets, predictions, labels=all_labels, average=None, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(targets, predictions, labels=all_labels)

    # Average loss
    avg_loss = np.mean(self.losses)

    return {
      'accuracy': accuracy,
      'precision': precision,
      'recall': recall,
      'f1_score': f1,
      'avg_loss': avg_loss,
      'confusion_matrix': cm.tolist(),
      'per_class_metrics': {
        'precision': precision_per_class.tolist(),
        'recall': recall_per_class.tolist(),
        'f1_score': f1_per_class.tolist(),
        'support': support_per_class.tolist()
      },
      'class_names': self.class_names
    }


class EarlyStopping:
  """Early stopping utility to prevent overfitting."""

  def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
    self.patience = patience
    self.min_delta = min_delta
    self.restore_best_weights = restore_best_weights
    self.best_score = None
    self.counter = 0
    self.best_weights = None

  def __call__(self, val_score: float, model: nn.Module) -> bool:
    """Check if training should stop."""
    if self.best_score is None:
      self.best_score = val_score
      self.save_checkpoint(model)
    elif val_score < self.best_score + self.min_delta:
      self.counter += 1
      if self.counter >= self.patience:
        if self.restore_best_weights:
          model.load_state_dict(self.best_weights)
        return True
    else:
      self.best_score = val_score
      self.counter = 0
      self.save_checkpoint(model)

    return False

  def save_checkpoint(self, model: nn.Module):
    """Save the best model weights."""
    self.best_weights = model.state_dict().copy()


class ClassificationTrainer:
  """Main trainer class for classification experiments."""

  def __init__(self,
               model: BaseClassifier,
               config: DatasetConfig,
               experiment_name: str = "screen_classification",
               use_wandb: bool = False,
               use_tensorboard: bool = True):
    self.model = model
    self.config = config
    self.experiment_name = experiment_name
    self.use_wandb = use_wandb
    self.use_tensorboard = use_tensorboard

    # Setup directories (prefer experiments.output_dir from config.yaml if provided)
    base_experiments_dir = config.experiments_output_dir or (Path(config.output_dir) / "experiments")
    self.output_dir = Path(base_experiments_dir) / experiment_name
    self.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    self.setup_logging()

    # Initialize metrics
    self.num_classes = model.num_classes
    self.class_names = [f"Class_{i}" for i in range(self.num_classes)]

    # Training state
    self.current_epoch = 0
    self.best_val_f1 = 0.0
    self.training_history = []

    # Setup early stopping
    self.early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    console.print(f"[green]Trainer initialized for experiment: {experiment_name}[/green]")
    console.print(f"[blue]Output directory: {self.output_dir}[/blue]")

  def setup_logging(self):
    """Setup logging and experiment tracking."""
    if self.use_wandb:
      wandb.init(project="screen-classification", name=self.experiment_name, config=vars(self.config))

    if self.use_tensorboard:
      self.tb_writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))

  def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module,
                  device: torch.device) -> Dict[str, float]:
    """Train for one epoch."""
    self.model.train()
    metrics = MetricsTracker(self.num_classes, self.class_names)

    with Progress(TextColumn("[progress.description]{task.description}"),
                  BarColumn(),
                  TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                  TimeElapsedColumn(),
                  console=console) as progress:
      task = progress.add_task("Training", total=len(train_loader))

      for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = self.model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update metrics
        predictions = torch.argmax(output, dim=1)
        metrics.update(predictions, target, loss.item())

        progress.update(task, advance=1)

    return metrics.compute_metrics()

  def validate_epoch(self, val_loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    """Validate for one epoch."""
    self.model.eval()
    metrics = MetricsTracker(self.num_classes, self.class_names)

    with torch.no_grad():
      with Progress(TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeElapsedColumn(),
                    console=console) as progress:
        task = progress.add_task("Validating", total=len(val_loader))

        for data, target in val_loader:
          data, target = data.to(device), target.to(device)

          output = self.model(data)
          loss = criterion(output, target)

          # Update metrics
          predictions = torch.argmax(output, dim=1)
          metrics.update(predictions, target, loss.item())

          progress.update(task, advance=1)

    return metrics.compute_metrics()

  def train(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            num_epochs: int = 100,
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-4,
            class_weights: Optional[torch.Tensor] = None,
            scheduler: Optional[Any] = None) -> Dict[str, Any]:
    """Main training loop."""

    device = get_device()
    self.model.to(device)

    # Move class weights to device if provided
    if class_weights is not None:
      class_weights = class_weights.to(device)

    # Setup optimizer and criterion
    optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    console.print(f"[blue]Starting training on {device}[/blue]")
    console.print(f"[blue]Model: {self.model.__class__.__name__}[/blue]")
    console.print(f"[blue]Parameters: {sum(p.numel() for p in self.model.parameters()):,}[/blue]")

    # Training loop
    for epoch in range(num_epochs):
      self.current_epoch = epoch

      # Train
      train_metrics = self.train_epoch(train_loader, optimizer, criterion, device)

      # Validate
      val_metrics = self.validate_epoch(val_loader, criterion, device)

      # Update learning rate
      if scheduler:
        scheduler.step(val_metrics['f1_score'])

      # Log metrics
      self.log_metrics(epoch, train_metrics, val_metrics)

      # Save checkpoint
      self.save_checkpoint(epoch, val_metrics)

      # Early stopping check
      if self.early_stopping(val_metrics['f1_score'], self.model):
        console.print(f"[yellow]Early stopping at epoch {epoch}[/yellow]")
        break

      # Update best model
      if val_metrics['f1_score'] > self.best_val_f1:
        self.best_val_f1 = val_metrics['f1_score']
        self.save_best_model()

    # Final evaluation
    final_metrics = self.validate_epoch(val_loader, criterion, device)
    console.print(f"[green]Training completed! Best F1: {self.best_val_f1:.4f}[/green]")

    return {'best_val_f1': self.best_val_f1, 'final_metrics': final_metrics, 'training_history': self.training_history}

  def log_metrics(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
    """Log metrics to various backends."""

    # Console logging
    if epoch % 10 == 0 or epoch < 5:
      table = Table(title=f"Epoch {epoch}")
      table.add_column("Metric", style="cyan")
      table.add_column("Train", style="green")
      table.add_column("Val", style="magenta")

      table.add_row("Loss", f"{train_metrics['avg_loss']:.4f}", f"{val_metrics['avg_loss']:.4f}")
      table.add_row("Accuracy", f"{train_metrics['accuracy']:.4f}", f"{val_metrics['accuracy']:.4f}")
      table.add_row("F1 Score", f"{train_metrics['f1_score']:.4f}", f"{val_metrics['f1_score']:.4f}")

      console.print(table)

    # Store in history
    self.training_history.append({'epoch': epoch, 'train': train_metrics, 'val': val_metrics})

    # TensorBoard logging
    if self.use_tensorboard:
      self.tb_writer.add_scalar('Loss/Train', train_metrics['avg_loss'], epoch)
      self.tb_writer.add_scalar('Loss/Val', val_metrics['avg_loss'], epoch)
      self.tb_writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
      self.tb_writer.add_scalar('Accuracy/Val', val_metrics['accuracy'], epoch)
      self.tb_writer.add_scalar('F1/Train', train_metrics['f1_score'], epoch)
      self.tb_writer.add_scalar('F1/Val', val_metrics['f1_score'], epoch)

    # Weights & Biases logging
    if self.use_wandb:
      wandb.log({
        'epoch': epoch,
        'train_loss': train_metrics['avg_loss'],
        'val_loss': val_metrics['avg_loss'],
        'train_accuracy': train_metrics['accuracy'],
        'val_accuracy': val_metrics['accuracy'],
        'train_f1': train_metrics['f1_score'],
        'val_f1': val_metrics['f1_score']
      })

  def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
    """Save model checkpoint."""
    checkpoint = {
      'epoch': epoch,
      'model_state_dict': self.model.state_dict(),
      'optimizer_state_dict': None,  # Will be set if needed
      'metrics': metrics,
      'model_info': get_model_info(self.model)
    }

    torch.save(checkpoint, self.output_dir / f"checkpoint_epoch_{epoch}.pth")

  def save_best_model(self):
    """Save the best model."""
    best_model_path = self.output_dir / "best_model.pth"
    torch.save(
      {
        'model_state_dict': self.model.state_dict(),
        'model_info': get_model_info(self.model),
        'best_val_f1': self.best_val_f1
      }, best_model_path)

  def evaluate(self, test_loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    """Evaluate the model on test data."""
    self.model.eval()
    self.model.to(device)  # Ensure model is on the correct device
    metrics = MetricsTracker(self.num_classes, self.class_names)

    with torch.no_grad():
      for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = self.model(data)
        loss = nn.CrossEntropyLoss()(output, target)

        predictions = torch.argmax(output, dim=1)
        metrics.update(predictions, target, loss.item())

    test_metrics = metrics.compute_metrics()

    # Generate detailed report
    report = {
      'test_metrics': test_metrics,
      'confusion_matrix': test_metrics['confusion_matrix'],
      'per_class_metrics': test_metrics['per_class_metrics']
    }

    # Save evaluation results
    with open(self.output_dir / "evaluation_results.json", 'w') as f:
      json.dump(report, f, indent=2)

    # Display results
    self.display_evaluation_results(test_metrics)

    return report

  def display_evaluation_results(self, metrics: Dict[str, Any]):
    """Display evaluation results in a formatted table."""
    table = Table(title="Test Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Accuracy", f"{metrics['accuracy']:.4f}")
    table.add_row("Precision", f"{metrics['precision']:.4f}")
    table.add_row("Recall", f"{metrics['recall']:.4f}")
    table.add_row("F1 Score", f"{metrics['f1_score']:.4f}")
    table.add_row("Average Loss", f"{metrics['avg_loss']:.4f}")

    console.print(table)

    # Per-class metrics
    if 'per_class_metrics' in metrics:
      per_class_table = Table(title="Per-Class Metrics")
      per_class_table.add_column("Class", style="cyan")
      per_class_table.add_column("Precision", style="green")
      per_class_table.add_column("Recall", style="blue")
      per_class_table.add_column("F1", style="magenta")
      per_class_table.add_column("Support", style="yellow")

      # Robustly iterate over available classes
      num_classes = len(metrics.get('class_names', []))
      prec = metrics['per_class_metrics']['precision']
      rec = metrics['per_class_metrics']['recall']
      f1c = metrics['per_class_metrics']['f1_score']
      sup = metrics['per_class_metrics']['support']
      limit = min(num_classes, len(prec), len(rec), len(f1c), len(sup))

      for i in range(limit):
        class_name = metrics['class_names'][i]
        per_class_table.add_row(class_name, f"{prec[i]:.3f}", f"{rec[i]:.3f}", f"{f1c[i]:.3f}", f"{sup[i]}")

      console.print(per_class_table)


class DistillationTrainer(ClassificationTrainer):
  """Trainer for knowledge distillation experiments."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.distillation_loss = DistillationLoss(temperature=3.0, alpha=0.7)

  def train_with_distillation(self,
                              train_loader: DataLoader,
                              val_loader: DataLoader,
                              teacher_model: BaseClassifier,
                              num_epochs: int = 100,
                              learning_rate: float = 1e-4,
                              weight_decay: float = 1e-4,
                              device: torch.device = None) -> Dict[str, Any]:
    """Train student model using knowledge distillation."""

    if device is None:
      device = get_device()

    self.model.to(device)
    teacher_model.to(device)
    teacher_model.eval()  # Teacher should be in eval mode

    # Move distillation loss to device
    self.distillation_loss = self.distillation_loss.to(device)

    optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    console.print(f"[blue]Starting distillation training on {device}[/blue]")
    console.print(f"[blue]Student: {self.model.__class__.__name__}[/blue]")
    console.print(f"[blue]Teacher: {teacher_model.__class__.__name__}[/blue]")

    for epoch in range(num_epochs):
      self.current_epoch = epoch

      # Train with distillation
      train_metrics = self.train_epoch_distillation(train_loader, optimizer, teacher_model, device)

      # Validate
      val_metrics = self.validate_epoch(val_loader, nn.CrossEntropyLoss(), device)

      # Log metrics
      self.log_metrics(epoch, train_metrics, val_metrics)

      # Save checkpoint
      self.save_checkpoint(epoch, val_metrics)

      # Early stopping
      if self.early_stopping(val_metrics['f1_score'], self.model):
        console.print(f"[yellow]Early stopping at epoch {epoch}[/yellow]")
        break

    return {'best_val_f1': self.best_val_f1, 'training_history': self.training_history}

  def train_epoch_distillation(self, train_loader: DataLoader, optimizer: optim.Optimizer,
                               teacher_model: BaseClassifier, device: torch.device) -> Dict[str, float]:
    """Train one epoch with distillation."""
    self.model.train()
    teacher_model.eval()
    metrics = MetricsTracker(self.num_classes, self.class_names)

    with Progress(TextColumn("[progress.description]{task.description}"),
                  BarColumn(),
                  TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                  TimeElapsedColumn(),
                  console=console) as progress:
      task = progress.add_task("Distillation Training", total=len(train_loader))

      for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Forward pass
        student_output = self.model(data)

        with torch.no_grad():
          teacher_output = teacher_model(data)

        # Compute distillation loss
        loss = self.distillation_loss(student_output, teacher_output, target)

        loss.backward()
        optimizer.step()

        # Update metrics
        predictions = torch.argmax(student_output, dim=1)
        metrics.update(predictions, target, loss.item())

        progress.update(task, advance=1)

    return metrics.compute_metrics()


if __name__ == "__main__":
  # Example usage
  from models import ModelFactory
  from data_loader import create_data_loaders

  # Create a simple model for testing
  model = ModelFactory.create_model('resnet50', num_classes=10)

  # Create dummy data loaders (replace with real data)
  # train_loader, val_loader, test_loader = create_data_loaders(...)

  # Initialize trainer
  config = DatasetConfig()
  trainer = ClassificationTrainer(model, config, "test_experiment")

  print("Trainer initialized successfully!")
  print(f"Model info: {get_model_info(model)}")
