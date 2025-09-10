# Screen Page Classification Research Infrastructure

A comprehensive PyTorch-based infrastructure for screen page classification experiments, designed to handle the complete pipeline from data inspection to model deployment.

## Overview

This infrastructure supports the complete workflow for screen page classification:

1. **Data Inspection**: Analyze labeled datasets, class balance, and data quality
2. **Model Training**: Fine-tune heavy classifiers on labeled data
3. **Auto-labeling**: Use trained models to label larger screenshot corpus
4. **Knowledge Distillation**: Create small, high-performance classifiers for production

## Features

### 🔍 Data Inspection & Analysis

- **Dataset Inspector**: Comprehensive analysis of class balance, data quality, and statistics
- **Visualization**: Automatic generation of charts and reports
- **API Integration**: Direct integration with annotation API for data download

### 🚀 Model Training

- **Multiple Architectures**: ResNet, EfficientNet, Vision Transformer, ConvNeXt, and custom lightweight models
- **Advanced Training**: Support for class weights, data augmentation, and early stopping
- **Experiment Tracking**: Integration with TensorBoard and Weights & Biases
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, and confusion matrices

### 🏷️ Auto-labeling System

- **Confidence-based Filtering**: Filter predictions based on confidence thresholds
- **Active Learning**: Select samples for manual review using uncertainty and diversity
- **Batch Processing**: Efficient processing of large image corpus
- **Class-based Filtering**: Organize results by predicted class

### 🎓 Knowledge Distillation

- **Advanced Distillation**: Multiple distillation techniques including attention transfer and feature matching
- **Model Compression**: Create lightweight models for production deployment
- **Performance Comparison**: Compare teacher and student model performance

## Installation

1. **Install Dependencies**:

```bash
pip install -r requirements.txt
```

2. **Setup Annotation API** (if using labeled data):

```bash
cd /workspace/tools/annotation
python app.py
```

## Quick Start

### 1. Inspect Available Data

```bash
python main.py inspect
```

This will show all available datasets and their statistics.

### 2. Analyze Specific Dataset

```bash
python main.py inspect --dataset-id 1
```

This will generate a comprehensive analysis report with visualizations.

### 3. Run Classification Experiments

```bash
python main.py experiment --dataset-id 1 --models resnet50,efficientnet_b0,lightweight --epochs 50
```

This will train multiple models and compare their performance.

### 4. Auto-label Screenshot Corpus

```bash
python main.py autolabel \
    --model-path /path/to/trained_model.pth \
    --model-type resnet50 \
    --num-classes 10 \
    --image-dir /path/to/screenshots \
    --output-dir /path/to/output \
    --confidence-threshold 0.9
```

### 5. Run Knowledge Distillation

```bash
python main.py distill \
    --teacher-model-path /path/to/teacher_model.pth \
    --teacher-model-type resnet50 \
    --student-model-type lightweight \
    --num-classes 10 \
    --temperature 3.0 \
    --alpha 0.7
```

## Architecture

### Core Components

- **`data_loader.py`**: Data loading utilities and PyTorch datasets
- **`dataset_inspector.py`**: Dataset analysis and visualization
- **`models.py`**: Model definitions and factory
- **`trainer.py`**: Training loops and evaluation
- **`experiment_runner.py`**: Experiment orchestration
- **`auto_labeler.py`**: Auto-labeling system
- **`distillation_pipeline.py`**: Knowledge distillation
- **`main.py`**: Command-line interface

### Model Architectures

| Model           | Parameters | Description                        |
| --------------- | ---------- | ---------------------------------- |
| ResNet-50       | 25M        | Standard ResNet architecture       |
| ResNet-101      | 44M        | Deeper ResNet variant              |
| EfficientNet-B0 | 5M         | EfficientNet with compound scaling |
| EfficientNet-B3 | 12M        | Larger EfficientNet variant        |
| ViT-Base        | 86M        | Vision Transformer base model      |
| ViT-Large       | 307M       | Vision Transformer large model     |
| ConvNeXt-Tiny   | 29M        | Modern ConvNet architecture        |
| ConvNeXt-Small  | 50M        | Larger ConvNeXt variant            |
| Lightweight     | 1M         | Custom lightweight CNN             |

## Usage Examples

### Complete Experiment Pipeline

```python
from experiment_runner import ExperimentRunner, ExperimentConfig

# Configure experiment
config = ExperimentConfig(
    dataset_id=1,
    model_types=['resnet50', 'efficientnet_b0', 'lightweight'],
    num_epochs=100,
    learning_rate=1e-4,
    batch_size=32
)

# Run experiments
runner = ExperimentRunner(config)
results = runner.run_full_experiment()

print(f"Best model: {results['best_model']['model_type']}")
print(f"Best F1 score: {results['best_model']['f1_score']:.4f}")
```

### Auto-labeling Large Corpus

```python
from auto_labeler import AutoLabelingPipeline
from data_loader import DatasetConfig

# Initialize pipeline
config = DatasetConfig()
pipeline = AutoLabelingPipeline(
    model_path="/path/to/model.pth",
    model_type="resnet50",
    num_classes=10,
    config=config
)

# Process corpus
image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg", ...]
results = pipeline.process_corpus(
    image_paths=image_paths,
    output_dir="/path/to/output",
    confidence_threshold=0.9
)
```

### Knowledge Distillation

```python
from distillation_pipeline import DistillationPipeline, DistillationConfig
from data_loader import DatasetConfig

# Configure distillation
config = DistillationConfig(
    teacher_model_path="/path/to/teacher.pth",
    teacher_model_type="resnet50",
    student_model_type="lightweight",
    num_classes=10,
    temperature=3.0,
    alpha=0.7,
    use_attention_transfer=True
)

# Initialize pipeline
data_config = DatasetConfig()
pipeline = DistillationPipeline(config, data_config)

# Train student model
results = pipeline.train_student(train_loader, val_loader)
```

## Configuration

### Dataset Configuration

```python
from data_loader import DatasetConfig

config = DatasetConfig(
    annotation_api_url="http://localhost:5000",
    data_root="/workspace/data",
    output_dir="/workspace/research/screen-page-classification/data",
    image_size=(224, 224),
    batch_size=32,
    num_workers=4,
    test_size=0.2,
    val_size=0.1,
    random_state=42
)
```

### Experiment Configuration

```python
from experiment_runner import ExperimentConfig

config = ExperimentConfig(
    dataset_id=1,
    model_types=['resnet50', 'efficientnet_b0'],
    num_epochs=100,
    learning_rate=1e-4,
    batch_size=32,
    image_size=(224, 224),
    use_class_weights=True,
    use_data_augmentation=True,
    experiment_name="my_experiment"
)
```

## Output Structure

```
experiments/
├── my_experiment/
│   ├── resnet50/
│   │   ├── best_model.pth
│   │   ├── checkpoint_epoch_*.pth
│   │   ├── tensorboard/
│   │   └── evaluation_results.json
│   ├── efficientnet_b0/
│   │   └── ...
│   └── final_report.json
```

## Advanced Features

### Custom Model Architecture

```python
from models import BaseClassifier

class CustomClassifier(BaseClassifier):
    def __init__(self, num_classes: int, dropout_rate: float = 0.5):
        super().__init__(num_classes, dropout_rate)
        # Define your architecture here
        self.features = nn.Sequential(...)
        self.classifier = nn.Sequential(...)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(features)

    def get_embedding_size(self) -> int:
        return 512
```

### Custom Loss Functions

```python
from trainer import ClassificationTrainer

class CustomTrainer(ClassificationTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_loss = YourCustomLoss()

    def train_epoch(self, train_loader, optimizer, criterion, device):
        # Override with custom training logic
        pass
```

## Troubleshooting

### Common Issues

1. **No datasets found**: Ensure the annotation API is running and has data
2. **CUDA out of memory**: Reduce batch size or use CPU
3. **Model loading errors**: Check model path and architecture compatibility

### Debug Mode

```bash
python main.py experiment --dataset-id 1 --epochs 5  # Quick test
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- Hugging Face for the transformers library
- TIMM for the model implementations
- Rich for the beautiful console output
