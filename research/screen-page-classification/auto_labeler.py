"""
Auto-labeling system for large screenshot corpus using trained classifiers.
Implements confidence-based filtering and active learning strategies.
"""

import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
import pandas as pd
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

from models import BaseClassifier, ModelFactory
from data_loader import DatasetConfig

console = Console()


class ScreenshotDataset(Dataset):
    """Dataset for unlabeled screenshots."""
    
    def __init__(
        self,
        image_paths: List[str],
        transform: Optional[transforms.Compose] = None
    ):
        self.image_paths = image_paths
        self.transform = transform or self._get_default_transform()
    
    def _get_default_transform(self) -> transforms.Compose:
        """Get default image transformations."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, image_path


class ConfidenceFilter:
    """Filter predictions based on confidence thresholds."""
    
    def __init__(self, confidence_threshold: float = 0.9):
        self.confidence_threshold = confidence_threshold
    
    def filter_predictions(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        image_paths: List[str]
    ) -> Dict[str, Any]:
        """Filter predictions based on confidence."""
        
        high_confidence_mask = confidences >= self.confidence_threshold
        low_confidence_mask = confidences < self.confidence_threshold
        
        high_confidence_data = {
            'predictions': predictions[high_confidence_mask],
            'confidences': confidences[high_confidence_mask],
            'image_paths': [image_paths[i] for i in range(len(image_paths)) if high_confidence_mask[i]]
        }
        
        low_confidence_data = {
            'predictions': predictions[low_confidence_mask],
            'confidences': confidences[low_confidence_mask],
            'image_paths': [image_paths[i] for i in range(len(image_paths)) if low_confidence_mask[i]]
        }
        
        return {
            'high_confidence': high_confidence_data,
            'low_confidence': low_confidence_data,
            'total_samples': len(predictions),
            'high_confidence_count': np.sum(high_confidence_mask),
            'low_confidence_count': np.sum(low_confidence_mask),
            'confidence_ratio': np.sum(high_confidence_mask) / len(predictions)
        }


class ActiveLearningSelector:
    """Select samples for active learning based on uncertainty and diversity."""
    
    def __init__(self, diversity_weight: float = 0.5):
        self.diversity_weight = diversity_weight
    
    def select_samples(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        features: np.ndarray,
        image_paths: List[str],
        num_samples: int = 100
    ) -> List[str]:
        """Select samples for active learning."""
        
        # Calculate uncertainty (1 - confidence)
        uncertainties = 1 - confidences
        
        # Calculate diversity using pairwise distances
        if len(features) > 1:
            distances = pairwise_distances(features, metric='cosine')
            diversity_scores = np.mean(distances, axis=1)
        else:
            diversity_scores = np.ones(len(features))
        
        # Combine uncertainty and diversity
        combined_scores = (1 - self.diversity_weight) * uncertainties + self.diversity_weight * diversity_scores
        
        # Select top samples
        top_indices = np.argsort(combined_scores)[-num_samples:]
        
        return [image_paths[i] for i in top_indices]


class AutoLabeler:
    """Main auto-labeling system."""
    
    def __init__(
        self,
        model: BaseClassifier,
        config: DatasetConfig,
        confidence_threshold: float = 0.9,
        batch_size: int = 32
    ):
        self.model = model
        self.config = config
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        
        self.confidence_filter = ConfidenceFilter(confidence_threshold)
        self.active_learning_selector = ActiveLearningSelector()
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        console.print(f"[blue]AutoLabeler initialized on {self.device}[/blue]")
        console.print(f"[blue]Confidence threshold: {confidence_threshold}[/blue]")
    
    def predict_batch(
        self,
        image_paths: List[str],
        return_features: bool = False
    ) -> Dict[str, Any]:
        """Predict labels for a batch of images."""
        
        # Create dataset and dataloader
        dataset = ScreenshotDataset(image_paths)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        predictions = []
        confidences = []
        features = []
        
        with torch.no_grad():
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Predicting", total=len(dataloader))
                
                for batch_images, batch_paths in dataloader:
                    batch_images = batch_images.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(batch_images)
                    
                    # Get predictions and confidences
                    batch_predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                    batch_confidences = torch.max(F.softmax(outputs, dim=1), dim=1)[0].cpu().numpy()
                    
                    predictions.extend(batch_predictions)
                    confidences.extend(batch_confidences)
                    
                    # Extract features if needed
                    if return_features:
                        # This would need to be implemented based on your model architecture
                        # For now, we'll use the logits as features
                        batch_features = outputs.cpu().numpy()
                        features.extend(batch_features)
                    
                    progress.update(task, advance=1)
        
        return {
            'predictions': np.array(predictions),
            'confidences': np.array(confidences),
            'features': np.array(features) if return_features else None,
            'image_paths': image_paths
        }
    
    def auto_label_corpus(
        self,
        image_paths: List[str],
        output_file: Optional[str] = None,
        return_features: bool = False
    ) -> Dict[str, Any]:
        """Auto-label a large corpus of screenshots."""
        
        console.print(f"[blue]Auto-labeling {len(image_paths)} screenshots...[/blue]")
        
        # Predict on all images
        results = self.predict_batch(image_paths, return_features=return_features)
        
        # Filter by confidence
        filtered_results = self.confidence_filter.filter_predictions(
            results['predictions'],
            results['confidences'],
            results['image_paths']
        )
        
        # Generate summary
        summary = self._generate_summary(filtered_results)
        
        # Save results
        if output_file:
            self._save_results(filtered_results, output_file)
        
        # Display results
        self._display_results(summary)
        
        return {
            'filtered_results': filtered_results,
            'summary': summary,
            'raw_results': results
        }
    
    def _generate_summary(self, filtered_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics."""
        
        high_conf = filtered_results['high_confidence']
        low_conf = filtered_results['low_confidence']
        
        # Class distribution for high confidence predictions
        if len(high_conf['predictions']) > 0:
            class_counts = np.bincount(high_conf['predictions'])
            class_distribution = {f"class_{i}": int(count) for i, count in enumerate(class_counts)}
        else:
            class_distribution = {}
        
        # Confidence statistics
        if len(high_conf['confidences']) > 0:
            confidence_stats = {
                'mean': float(np.mean(high_conf['confidences'])),
                'std': float(np.std(high_conf['confidences'])),
                'min': float(np.min(high_conf['confidences'])),
                'max': float(np.max(high_conf['confidences']))
            }
        else:
            confidence_stats = {}
        
        return {
            'total_samples': filtered_results['total_samples'],
            'high_confidence_count': filtered_results['high_confidence_count'],
            'low_confidence_count': filtered_results['low_confidence_count'],
            'confidence_ratio': filtered_results['confidence_ratio'],
            'class_distribution': class_distribution,
            'confidence_stats': confidence_stats
        }
    
    def _save_results(self, results: Dict[str, Any], output_file: str):
        """Save results to file."""
        
        # Convert to serializable format
        save_data = {
            'high_confidence': {
                'predictions': results['high_confidence']['predictions'].tolist(),
                'confidences': results['high_confidence']['confidences'].tolist(),
                'image_paths': results['high_confidence']['image_paths']
            },
            'low_confidence': {
                'predictions': results['low_confidence']['predictions'].tolist(),
                'confidences': results['low_confidence']['confidences'].tolist(),
                'image_paths': results['low_confidence']['image_paths']
            },
            'metadata': {
                'total_samples': results['total_samples'],
                'high_confidence_count': results['high_confidence_count'],
                'low_confidence_count': results['low_confidence_count'],
                'confidence_ratio': results['confidence_ratio']
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        console.print(f"[green]Results saved to {output_file}[/green]")
    
    def _display_results(self, summary: Dict[str, Any]):
        """Display results in a formatted table."""
        
        # Main statistics table
        table = Table(title="Auto-labeling Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Samples", str(summary['total_samples']))
        table.add_row("High Confidence", str(summary['high_confidence_count']))
        table.add_row("Low Confidence", str(summary['low_confidence_count']))
        table.add_row("Confidence Ratio", f"{summary['confidence_ratio']:.2%}")
        
        console.print(table)
        
        # Class distribution table
        if summary['class_distribution']:
            class_table = Table(title="High Confidence Class Distribution")
            class_table.add_column("Class", style="cyan")
            class_table.add_column("Count", style="magenta")
            
            for class_name, count in summary['class_distribution'].items():
                class_table.add_row(class_name, str(count))
            
            console.print(class_table)
        
        # Confidence statistics
        if summary['confidence_stats']:
            conf_table = Table(title="Confidence Statistics")
            conf_table.add_column("Statistic", style="cyan")
            conf_table.add_column("Value", style="magenta")
            
            for stat, value in summary['confidence_stats'].items():
                conf_table.add_row(stat.title(), f"{value:.4f}")
            
            console.print(conf_table)
    
    def select_for_active_learning(
        self,
        image_paths: List[str],
        num_samples: int = 100
    ) -> List[str]:
        """Select samples for active learning."""
        
        console.print(f"[blue]Selecting {num_samples} samples for active learning...[/blue]")
        
        # Get predictions and features
        results = self.predict_batch(image_paths, return_features=True)
        
        # Select samples
        selected_paths = self.active_learning_selector.select_samples(
            results['predictions'],
            results['confidences'],
            results['features'],
            results['image_paths'],
            num_samples
        )
        
        console.print(f"[green]Selected {len(selected_paths)} samples for active learning[/green]")
        
        return selected_paths
    
    def filter_by_class(
        self,
        image_paths: List[str],
        target_classes: List[int],
        min_confidence: float = 0.8
    ) -> Dict[str, List[str]]:
        """Filter images by predicted class."""
        
        console.print(f"[blue]Filtering images by classes: {target_classes}[/blue]")
        
        # Get predictions
        results = self.predict_batch(image_paths)
        
        # Filter by class and confidence
        filtered_by_class = {}
        
        for class_id in target_classes:
            class_mask = (results['predictions'] == class_id) & (results['confidences'] >= min_confidence)
            filtered_by_class[f'class_{class_id}'] = [
                results['image_paths'][i] for i in range(len(results['image_paths'])) if class_mask[i]
            ]
        
        # Display results
        table = Table(title="Filtered Images by Class")
        table.add_column("Class", style="cyan")
        table.add_column("Count", style="magenta")
        
        for class_name, paths in filtered_by_class.items():
            table.add_row(class_name, str(len(paths)))
        
        console.print(table)
        
        return filtered_by_class


class AutoLabelingPipeline:
    """Complete pipeline for auto-labeling large screenshot corpus."""
    
    def __init__(
        self,
        model_path: str,
        model_type: str,
        num_classes: int,
        config: DatasetConfig
    ):
        self.config = config
        self.num_classes = num_classes
        
        # Load model
        self.model = ModelFactory.create_model(model_type, num_classes)
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Initialize auto-labeler
        self.auto_labeler = AutoLabeler(self.model, config)
        
        console.print(f"[green]Auto-labeling pipeline initialized[/green]")
        console.print(f"[blue]Model: {model_type}[/blue]")
        console.print(f"[blue]Classes: {num_classes}[/blue]")
    
    def process_corpus(
        self,
        image_paths: List[str],
        output_dir: str,
        confidence_threshold: float = 0.9,
        active_learning_samples: int = 100
    ) -> Dict[str, Any]:
        """Process a large corpus of screenshots."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        console.print(f"[bold blue]Processing corpus of {len(image_paths)} images[/bold blue]")
        
        # Step 1: Auto-label all images
        console.print("[blue]Step 1: Auto-labeling all images...[/blue]")
        auto_label_results = self.auto_labeler.auto_label_corpus(
            image_paths,
            output_file=str(output_path / "auto_label_results.json")
        )
        
        # Step 2: Select samples for active learning
        console.print("[blue]Step 2: Selecting samples for active learning...[/blue]")
        active_learning_samples = self.auto_labeler.select_for_active_learning(
            image_paths,
            active_learning_samples
        )
        
        # Save active learning samples
        with open(output_path / "active_learning_samples.json", 'w') as f:
            json.dump(active_learning_samples, f, indent=2)
        
        # Step 3: Filter by class for easy review
        console.print("[blue]Step 3: Filtering by class...[/blue]")
        class_filtered = self.auto_labeler.filter_by_class(
            image_paths,
            list(range(self.num_classes))
        )
        
        # Save class-filtered results
        with open(output_path / "class_filtered.json", 'w') as f:
            json.dump(class_filtered, f, indent=2)
        
        # Generate summary report
        summary_report = {
            'total_images': len(image_paths),
            'high_confidence_count': auto_label_results['summary']['high_confidence_count'],
            'low_confidence_count': auto_label_results['summary']['low_confidence_count'],
            'confidence_ratio': auto_label_results['summary']['confidence_ratio'],
            'active_learning_samples': len(active_learning_samples),
            'class_distribution': auto_label_results['summary']['class_distribution'],
            'output_directory': str(output_path)
        }
        
        with open(output_path / "summary_report.json", 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        console.print(Panel(
            f"Corpus processing completed!\n"
            f"High confidence: {summary_report['high_confidence_count']}\n"
            f"Low confidence: {summary_report['low_confidence_count']}\n"
            f"Active learning samples: {summary_report['active_learning_samples']}\n"
            f"Results saved to: {output_path}",
            title="Processing Summary",
            border_style="green"
        ))
        
        return summary_report


def main():
    """Example usage of the auto-labeling system."""
    
    # Example configuration
    config = DatasetConfig()
    
    # Example image paths (replace with actual paths)
    image_paths = [
        "/path/to/image1.jpg",
        "/path/to/image2.jpg",
        # ... more paths
    ]
    
    # Initialize auto-labeler
    model = ModelFactory.create_model('resnet50', num_classes=10)
    auto_labeler = AutoLabeler(model, config)
    
    # Auto-label corpus
    results = auto_labeler.auto_label_corpus(image_paths)
    
    console.print("[green]Auto-labeling completed![/green]")


if __name__ == "__main__":
    main()