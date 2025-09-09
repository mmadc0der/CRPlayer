"""
Knowledge distillation pipeline for creating small, high-performance classifiers.
Implements various distillation strategies and techniques for model compression.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

from models import BaseClassifier, ModelFactory, DistillationLoss, get_model_info
from trainer import DistillationTrainer, ClassificationTrainer
from data_loader import DatasetConfig, create_data_loaders

console = Console()


class DistillationConfig:
    """Configuration for knowledge distillation experiments."""
    
    def __init__(
        self,
        teacher_model_path: str,
        teacher_model_type: str,
        student_model_type: str,
        num_classes: int,
        temperature: float = 3.0,
        alpha: float = 0.7,
        beta: float = 0.3,
        use_attention_transfer: bool = False,
        use_feature_matching: bool = False,
        use_relation_knowledge: bool = False
    ):
        self.teacher_model_path = teacher_model_path
        self.teacher_model_type = teacher_model_type
        self.student_model_type = student_model_type
        self.num_classes = num_classes
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.use_attention_transfer = use_attention_transfer
        self.use_feature_matching = use_feature_matching
        self.use_relation_knowledge = use_relation_knowledge


class AttentionTransferLoss(nn.Module):
    """Attention transfer loss for knowledge distillation."""
    
    def __init__(self, alpha: float = 0.3):
        super().__init__()
        self.alpha = alpha
    
    def forward(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute attention transfer loss."""
        
        if len(student_features) != len(teacher_features):
            raise ValueError("Student and teacher must have same number of feature maps")
        
        total_loss = 0.0
        for s_feat, t_feat in zip(student_features, teacher_features):
            # Compute attention maps
            s_attention = self._compute_attention_map(s_feat)
            t_attention = self._compute_attention_map(t_feat)
            
            # Compute MSE loss between attention maps
            loss = F.mse_loss(s_attention, t_attention)
            total_loss += loss
        
        return self.alpha * total_loss / len(student_features)
    
    def _compute_attention_map(self, features: torch.Tensor) -> torch.Tensor:
        """Compute attention map from feature tensor."""
        # Global average pooling to get attention weights
        attention = torch.mean(features, dim=1, keepdim=True)
        return attention


class FeatureMatchingLoss(nn.Module):
    """Feature matching loss for knowledge distillation."""
    
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
    
    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute feature matching loss."""
        
        # Ensure features have same dimensions
        if student_features.shape != teacher_features.shape:
            # Project student features to match teacher dimensions
            if student_features.shape[1] != teacher_features.shape[1]:
                projection = nn.Linear(student_features.shape[1], teacher_features.shape[1])
                student_features = projection(student_features)
        
        # Compute MSE loss between features
        loss = F.mse_loss(student_features, teacher_features)
        
        return self.alpha * loss


class RelationKnowledgeLoss(nn.Module):
    """Relation knowledge distillation loss."""
    
    def __init__(self, alpha: float = 0.3):
        super().__init__()
        self.alpha = alpha
    
    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute relation knowledge loss."""
        
        # Compute pairwise relations
        s_relations = self._compute_relations(student_features)
        t_relations = self._compute_relations(teacher_features)
        
        # Compute MSE loss between relations
        loss = F.mse_loss(s_relations, t_relations)
        
        return self.alpha * loss
    
    def _compute_relations(self, features: torch.Tensor) -> torch.Tensor:
        """Compute pairwise relations between samples."""
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute pairwise cosine similarity
        relations = torch.mm(features, features.t())
        
        return relations


class AdvancedDistillationLoss(nn.Module):
    """Advanced distillation loss combining multiple techniques."""
    
    def __init__(
        self,
        temperature: float = 3.0,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.1,
        use_attention_transfer: bool = False,
        use_feature_matching: bool = False,
        use_relation_knowledge: bool = False
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Initialize component losses
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        
        self.attention_loss = AttentionTransferLoss(beta) if use_attention_transfer else None
        self.feature_loss = FeatureMatchingLoss(gamma) if use_feature_matching else None
        self.relation_loss = RelationKnowledgeLoss(gamma) if use_relation_knowledge else None
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        student_features: Optional[List[torch.Tensor]] = None,
        teacher_features: Optional[List[torch.Tensor]] = None,
        student_embeddings: Optional[torch.Tensor] = None,
        teacher_embeddings: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute comprehensive distillation loss."""
        
        losses = {}
        
        # Soft distillation loss
        soft_loss = self.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)
        losses['soft_loss'] = soft_loss
        
        # Hard target loss
        hard_loss = self.ce_loss(student_logits, targets)
        losses['hard_loss'] = hard_loss
        
        # Total loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        losses['total_loss'] = total_loss
        
        # Additional losses
        if self.attention_loss and student_features and teacher_features:
            attention_loss = self.attention_loss(student_features, teacher_features)
            losses['attention_loss'] = attention_loss
            total_loss += attention_loss
        
        if self.feature_loss and student_embeddings is not None and teacher_embeddings is not None:
            feature_loss = self.feature_loss(student_embeddings, teacher_embeddings)
            losses['feature_loss'] = feature_loss
            total_loss += feature_loss
        
        if self.relation_loss and student_embeddings is not None and teacher_embeddings is not None:
            relation_loss = self.relation_loss(student_embeddings, teacher_embeddings)
            losses['relation_loss'] = relation_loss
            total_loss += relation_loss
        
        losses['final_loss'] = total_loss
        
        return losses


class DistillationPipeline:
    """Complete pipeline for knowledge distillation experiments."""
    
    def __init__(self, config: DistillationConfig, data_config: DatasetConfig):
        self.config = config
        self.data_config = data_config
        
        # Load teacher model
        self.teacher_model = self._load_teacher_model()
        
        # Create student model
        self.student_model = ModelFactory.create_model(
            config.student_model_type,
            num_classes=config.num_classes,
            pretrained=True
        )
        
        # Initialize advanced loss
        self.distillation_loss = AdvancedDistillationLoss(
            temperature=config.temperature,
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.beta,
            use_attention_transfer=config.use_attention_transfer,
            use_feature_matching=config.use_feature_matching,
            use_relation_knowledge=config.use_relation_knowledge
        )
        
        console.print(f"[green]Distillation pipeline initialized[/green]")
        console.print(f"[blue]Teacher: {config.teacher_model_type}[/blue]")
        console.print(f"[blue]Student: {config.student_model_type}[/blue]")
        console.print(f"[blue]Temperature: {config.temperature}[/blue]")
    
    def _load_teacher_model(self) -> BaseClassifier:
        """Load the pre-trained teacher model."""
        teacher_model = ModelFactory.create_model(
            self.config.teacher_model_type,
            num_classes=self.config.num_classes,
            pretrained=False
        )
        
        # Load checkpoint
        checkpoint = torch.load(self.config.teacher_model_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model_state_dict'])
        
        return teacher_model
    
    def train_student(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 100,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        device: torch.device = None
    ) -> Dict[str, Any]:
        """Train student model using knowledge distillation."""
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.teacher_model.to(device)
        self.student_model.to(device)
        self.teacher_model.eval()
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Training history
        training_history = []
        best_val_f1 = 0.0
        
        console.print(f"[blue]Starting distillation training on {device}[/blue]")
        
        for epoch in range(num_epochs):
            # Train epoch
            train_metrics = self._train_epoch_distillation(
                train_loader, optimizer, device
            )
            
            # Validate epoch
            val_metrics = self._validate_epoch_distillation(
                val_loader, device
            )
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Store history
            training_history.append({
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics
            })
            
            # Update best model
            if val_metrics['f1_score'] > best_val_f1:
                best_val_f1 = val_metrics['f1_score']
                self._save_best_student_model()
        
        console.print(f"[green]Distillation training completed! Best F1: {best_val_f1:.4f}[/green]")
        
        return {
            'best_val_f1': best_val_f1,
            'training_history': training_history
        }
    
    def _train_epoch_distillation(
        self,
        train_loader,
        optimizer: torch.optim.Optimizer,
        device: torch.device
    ) -> Dict[str, float]:
        """Train one epoch with advanced distillation."""
        self.student_model.train()
        self.teacher_model.eval()
        
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Distillation Training", total=len(train_loader))
                
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    student_output = self.student_model(data)
                    
                    with torch.no_grad():
                        teacher_output = self.teacher_model(data)
                    
                    # Compute distillation loss
                    losses = self.distillation_loss(
                        student_output,
                        teacher_output,
                        target
                    )
                    
                    # Backward pass
                    losses['final_loss'].backward()
                    optimizer.step()
                    
                    # Update metrics
                    total_loss += losses['final_loss'].item()
                    predictions.extend(torch.argmax(student_output, dim=1).cpu().numpy())
                    targets.extend(target.cpu().numpy())
                    
                    progress.update(task, advance=1)
        
        # Compute metrics
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted', zero_division=0
        )
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def _validate_epoch_distillation(
        self,
        val_loader,
        device: torch.device
    ) -> Dict[str, float]:
        """Validate one epoch."""
        self.student_model.eval()
        
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Validation", total=len(val_loader))
                
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    
                    # Forward pass
                    student_output = self.student_model(data)
                    teacher_output = self.teacher_model(data)
                    
                    # Compute loss
                    losses = self.distillation_loss(
                        student_output,
                        teacher_output,
                        target
                    )
                    
                    total_loss += losses['final_loss'].item()
                    predictions.extend(torch.argmax(student_output, dim=1).cpu().numpy())
                    targets.extend(target.cpu().numpy())
                    
                    progress.update(task, advance=1)
        
        # Compute metrics
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted', zero_division=0
        )
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def _log_metrics(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log training metrics."""
        if epoch % 10 == 0 or epoch < 5:
            table = Table(title=f"Distillation Epoch {epoch}")
            table.add_column("Metric", style="cyan")
            table.add_column("Train", style="green")
            table.add_column("Val", style="magenta")
            
            table.add_row("Loss", f"{train_metrics['loss']:.4f}", f"{val_metrics['loss']:.4f}")
            table.add_row("Accuracy", f"{train_metrics['accuracy']:.4f}", f"{val_metrics['accuracy']:.4f}")
            table.add_row("F1 Score", f"{train_metrics['f1_score']:.4f}", f"{val_metrics['f1_score']:.4f}")
            
            console.print(table)
    
    def _save_best_student_model(self):
        """Save the best student model."""
        # This would save to a specific path
        # Implementation depends on your file structure
        pass
    
    def compare_models(
        self,
        test_loader,
        device: torch.device = None
    ) -> Dict[str, Any]:
        """Compare teacher and student models."""
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.teacher_model.to(device)
        self.student_model.to(device)
        self.teacher_model.eval()
        self.student_model.eval()
        
        teacher_predictions = []
        student_predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                
                teacher_output = self.teacher_model(data)
                student_output = self.student_model(data)
                
                teacher_predictions.extend(torch.argmax(teacher_output, dim=1).cpu().numpy())
                student_predictions.extend(torch.argmax(student_output, dim=1).cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        # Compute metrics
        teacher_accuracy = accuracy_score(targets, teacher_predictions)
        student_accuracy = accuracy_score(targets, student_predictions)
        
        teacher_precision, teacher_recall, teacher_f1, _ = precision_recall_fscore_support(
            targets, teacher_predictions, average='weighted', zero_division=0
        )
        
        student_precision, student_recall, student_f1, _ = precision_recall_fscore_support(
            targets, student_predictions, average='weighted', zero_division=0
        )
        
        # Model size comparison
        teacher_info = get_model_info(self.teacher_model)
        student_info = get_model_info(self.student_model)
        
        comparison = {
            'teacher': {
                'accuracy': teacher_accuracy,
                'precision': teacher_precision,
                'recall': teacher_recall,
                'f1_score': teacher_f1,
                'parameters': teacher_info['parameters']['total_parameters'],
                'model_size_mb': teacher_info['model_size_mb']
            },
            'student': {
                'accuracy': student_accuracy,
                'precision': student_precision,
                'recall': student_recall,
                'f1_score': student_f1,
                'parameters': student_info['parameters']['total_parameters'],
                'model_size_mb': student_info['model_size_mb']
            },
            'compression_ratio': teacher_info['parameters']['total_parameters'] / student_info['parameters']['total_parameters'],
            'size_reduction': (teacher_info['model_size_mb'] - student_info['model_size_mb']) / teacher_info['model_size_mb'],
            'performance_retention': student_f1 / teacher_f1 if teacher_f1 > 0 else 0
        }
        
        # Display comparison
        self._display_comparison(comparison)
        
        return comparison
    
    def _display_comparison(self, comparison: Dict[str, Any]):
        """Display model comparison in a formatted table."""
        
        table = Table(title="Teacher vs Student Model Comparison")
        table.add_column("Metric", style="cyan")
        table.add_column("Teacher", style="green")
        table.add_column("Student", style="magenta")
        table.add_column("Difference", style="yellow")
        
        # Accuracy
        acc_diff = comparison['student']['accuracy'] - comparison['teacher']['accuracy']
        table.add_row(
            "Accuracy",
            f"{comparison['teacher']['accuracy']:.4f}",
            f"{comparison['student']['accuracy']:.4f}",
            f"{acc_diff:+.4f}"
        )
        
        # F1 Score
        f1_diff = comparison['student']['f1_score'] - comparison['teacher']['f1_score']
        table.add_row(
            "F1 Score",
            f"{comparison['teacher']['f1_score']:.4f}",
            f"{comparison['student']['f1_score']:.4f}",
            f"{f1_diff:+.4f}"
        )
        
        # Parameters
        param_diff = comparison['student']['parameters'] - comparison['teacher']['parameters']
        table.add_row(
            "Parameters",
            f"{comparison['teacher']['parameters']:,}",
            f"{comparison['student']['parameters']:,}",
            f"{param_diff:+,}"
        )
        
        # Model Size
        size_diff = comparison['student']['model_size_mb'] - comparison['teacher']['model_size_mb']
        table.add_row(
            "Size (MB)",
            f"{comparison['teacher']['model_size_mb']:.2f}",
            f"{comparison['student']['model_size_mb']:.2f}",
            f"{size_diff:+.2f}"
        )
        
        console.print(table)
        
        # Summary metrics
        summary_table = Table(title="Compression Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="magenta")
        
        summary_table.add_row("Compression Ratio", f"{comparison['compression_ratio']:.2f}x")
        summary_table.add_row("Size Reduction", f"{comparison['size_reduction']:.2%}")
        summary_table.add_row("Performance Retention", f"{comparison['performance_retention']:.2%}")
        
        console.print(summary_table)


def main():
    """Example usage of the distillation pipeline."""
    
    # Example configuration
    config = DistillationConfig(
        teacher_model_path="/path/to/teacher_model.pth",
        teacher_model_type="resnet50",
        student_model_type="lightweight",
        num_classes=10,
        temperature=3.0,
        alpha=0.7,
        use_attention_transfer=True,
        use_feature_matching=True
    )
    
    data_config = DatasetConfig()
    
    # Initialize pipeline
    pipeline = DistillationPipeline(config, data_config)
    
    # Example training (you would need actual data loaders)
    # results = pipeline.train_student(train_loader, val_loader)
    
    console.print("[green]Distillation pipeline initialized![/green]")


if __name__ == "__main__":
    main()