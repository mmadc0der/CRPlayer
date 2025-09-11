"""
Example usage of the screen page classification infrastructure.
Demonstrates the complete pipeline from data inspection to model deployment.
"""

import os
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from data_loader import DatasetConfig, DatasetInspector
from dataset_inspector import DatasetAnalyzer
from experiment_runner import ExperimentRunner, ExperimentConfig
from auto_labeler import AutoLabelingPipeline
from distillation_pipeline import DistillationPipeline, DistillationConfig
from models import ModelFactory

console = Console()


def example_data_inspection():
    """Example: Inspect available data and analyze dataset."""
    console.print(Panel("Example 1: Data Inspection", style="bold blue"))
    
    # Initialize configuration
    config = DatasetConfig()
    analyzer = DatasetAnalyzer(config)
    
    # Inspect available data
    console.print("Inspecting available annotation data...")
    data_info = analyzer.inspector.inspect_available_data()
    
    if not data_info['datasets']:
        console.print("No datasets found. Please ensure the annotation API is running.")
        return
    
    # Display available datasets
    console.print(f"Found {len(data_info['datasets'])} datasets across {data_info['total_projects']} projects")
    
    # Analyze the first dataset
    if data_info['datasets']:
        dataset_id = data_info['datasets'][0]['dataset_id']
        console.print(f"Analyzing dataset {dataset_id}...")
        
        # Generate comprehensive report
        report = analyzer.generate_comprehensive_report(dataset_id)
        console.print(f"Analysis complete! Check {report['output_directory']} for visualizations.")


def example_experiment_training():
    """Example: Run classification experiments."""
    console.print(Panel("Example 2: Classification Experiments", style="bold blue"))
    
    # Configuration
    config = ExperimentConfig(
        dataset_id=1,  # Replace with actual dataset ID
        model_types=['resnet50', 'efficientnet_b0', 'lightweight'],
        num_epochs=10,  # Reduced for example
        learning_rate=1e-4,
        batch_size=16,  # Reduced for example
        experiment_name="example_experiment"
    )
    
    # Run experiments
    runner = ExperimentRunner(config)
    
    try:
        results = runner.run_full_experiment()
        console.print(f"Best model: {results['best_model']['model_type']}")
        console.print(f"Best F1 score: {results['best_model']['f1_score']:.4f}")
    except Exception as e:
        console.print(f"Experiment failed (expected if no data): {e}")


def example_auto_labeling():
    """Example: Auto-label screenshot corpus."""
    console.print(Panel("Example 3: Auto-labeling", style="bold blue"))
    
    # Create a dummy model for demonstration
    model = ModelFactory.create_model('resnet50', num_classes=10)
    
    # Save dummy model
    model_path = "/tmp/dummy_model.pth"
    torch.save({'model_state_dict': model.state_dict()}, model_path)
    
    # Initialize auto-labeling pipeline
    config = DatasetConfig()
    pipeline = AutoLabelingPipeline(
        model_path=model_path,
        model_type="resnet50",
        num_classes=10,
        config=config
    )
    
    # Example image paths (replace with actual paths)
    image_paths = [
        "/path/to/screenshot1.jpg",
        "/path/to/screenshot2.jpg",
        "/path/to/screenshot3.jpg"
    ]
    
    console.print("Auto-labeling pipeline initialized.")
    console.print("To use with real data, provide actual image paths and trained model.")


def example_distillation():
    """Example: Knowledge distillation."""
    console.print(Panel("Example 4: Knowledge Distillation", style="bold blue"))
    
    # Create dummy models
    teacher_model = ModelFactory.create_model('resnet50', num_classes=10)
    student_model = ModelFactory.create_model('lightweight', num_classes=10)
    
    # Save dummy teacher model
    teacher_path = "/tmp/dummy_teacher.pth"
    torch.save({'model_state_dict': teacher_model.state_dict()}, teacher_path)
    
    # Configure distillation
    config = DistillationConfig(
        teacher_model_path=teacher_path,
        teacher_model_type="resnet50",
        student_model_type="lightweight",
        num_classes=10,
        temperature=3.0,
        alpha=0.7,
        use_attention_transfer=True,
        use_feature_matching=True
    )
    
    # Initialize pipeline
    data_config = DatasetConfig()
    pipeline = DistillationPipeline(config, data_config)
    
    console.print("Distillation pipeline initialized.")
    console.print("To use with real data, provide actual data loaders and trained teacher model.")


def example_model_comparison():
    """Example: Compare different model architectures."""
    console.print(Panel("Example 5: Model Comparison", style="bold blue"))
    
    num_classes = 10
    models_to_compare = ['resnet50', 'efficientnet_b0', 'lightweight']
    
    console.print("Comparing model architectures:")
    
    for model_type in models_to_compare:
        model = ModelFactory.create_model(model_type, num_classes)
        info = get_model_info(model)
        
        console.print(f"\n{model_type.upper()}:")
        console.print(f"  Parameters: {info['parameters']['total_parameters']:,}")
        console.print(f"  Size: {info['model_size_mb']:.2f} MB")
        console.print(f"  Embedding size: {info['embedding_size']}")


def example_custom_model():
    """Example: Create and use custom model architecture."""
    console.print(Panel("Example 6: Custom Model", style="bold blue"))
    
    from models import BaseClassifier
    import torch.nn as nn
    
    class CustomScreenClassifier(BaseClassifier):
        """Custom classifier for screen page classification."""
        
        def __init__(self, num_classes: int, dropout_rate: float = 0.3):
            super().__init__(num_classes, dropout_rate)
            
            # Custom architecture optimized for screenshots
            self.features = nn.Sequential(
                # Initial convolution
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
                
                # Block 1
                self._make_layer(64, 64, 2),
                self._make_layer(64, 128, 2, stride=2),
                self._make_layer(128, 256, 2, stride=2),
                self._make_layer(256, 512, 2, stride=2),
                
                # Global average pooling
                nn.AdaptiveAvgPool2d((1, 1))
            )
            
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(256, num_classes)
            )
        
        def _make_layer(self, in_channels, out_channels, blocks, stride=1):
            """Create a layer with multiple blocks."""
            layers = []
            
            # First block with potential stride
            layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
            # Remaining blocks
            for _ in range(1, blocks):
                layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
            
            return nn.Sequential(*layers)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.features(x)
            return self.classifier(features)
        
        def get_embedding_size(self) -> int:
            return 256
    
    # Create and test custom model
    custom_model = CustomScreenClassifier(num_classes=10)
    info = get_model_info(custom_model)
    
    console.print("Custom Screen Classifier:")
    console.print(f"  Parameters: {info['parameters']['total_parameters']:,}")
    console.print(f"  Size: {info['model_size_mb']:.2f} MB")
    console.print(f"  Embedding size: {info['embedding_size']}")
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = custom_model(x)
        console.print(f"  Output shape: {output.shape}")


def main():
    """Run all examples."""
    console.print(Panel("Screen Page Classification - Example Usage", style="bold green"))
    
    try:
        # Example 1: Data Inspection
        example_data_inspection()
        
        # Example 2: Experiment Training
        example_experiment_training()
        
        # Example 3: Auto-labeling
        example_auto_labeling()
        
        # Example 4: Distillation
        example_distillation()
        
        # Example 5: Model Comparison
        example_model_comparison()
        
        # Example 6: Custom Model
        example_custom_model()
        
        console.print(Panel(
            "All examples completed successfully!\n"
            "Check the individual example functions for detailed usage patterns.",
            style="bold green"
        ))
        
    except Exception as e:
        console.print(f"[red]Error running examples: {e}[/red]")
        console.print("This is expected if the annotation API is not running or no data is available.")


if __name__ == "__main__":
    # Import torch here to avoid issues if not installed
    import torch
    from models import get_model_info
    
    main()