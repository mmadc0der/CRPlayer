# Clean Knowledge Distillation Pipeline

A streamlined knowledge distillation pipeline that uses a pre-trained ResNet50 teacher model to train a lightweight student model. This implementation focuses on simplicity and ease of use while maintaining powerful distillation capabilities.

## Features

- ✅ **Uses existing trained teacher model** (`best_teacher_model.pth`)
- ✅ **Balanced sampling** for handling class imbalance
- ✅ **Advanced data augmentation** (random masking, rotation, color jitter)
- ✅ **Clean, minimal dependencies** (no heavy formatting libraries)
- ✅ **Comprehensive loss tracking** (soft + hard targets)
- ✅ **Real-time training visualization**
- ✅ **Production-ready student model**

## Quick Start

### 1. Prerequisites

Make sure you have:

- A trained teacher model saved as `best_teacher_model.pth`
- A dataset CSV file with image paths and class labels
- Python environment with required packages

### 2. Install Dependencies

```bash
pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn pillow tqdm
```

### 3. Basic Usage

```python
from clean_distillation_pipeline import CleanDistillationPipeline, load_dataset_from_csv

# Load your dataset
df, class_names, class_id_col = load_dataset_from_csv("path/to/your/dataset.csv")

# Initialize pipeline
pipeline = CleanDistillationPipeline(
    teacher_model_path="best_teacher_model.pth",
    num_classes=len(class_names)
)

# Create data loaders
train_loader, val_loader, test_loader, _, _, _ = pipeline.create_data_loaders(
    df, data_root="./data", batch_size=32
)

# Train student model
history, best_val_f1 = pipeline.train_student(train_loader, val_loader, num_epochs=30)

# Evaluate models
results = pipeline.evaluate_models(test_loader)
print(f"Compression ratio: {results['compression_ratio']:.1f}x")
print(f"Performance retention: {results['performance_retention']:.2%}")

# Save student model
pipeline.save_student_model("best_student_model.pth")
```

### 4. Jupyter Notebook Demo

Run the `clean_distillation_demo.ipynb` notebook for a complete walkthrough with visualizations.

## Key Components

### SimpleDistillationLoss

Combines soft targets (teacher predictions) and hard targets (ground truth) with configurable weights:

- **Soft Loss**: KL divergence between teacher and student predictions
- **Hard Loss**: Cross-entropy with ground truth labels
- **Temperature**: Controls the softness of teacher predictions

### LightweightStudent

A compact CNN architecture designed for production deployment:

- 4 convolutional blocks with batch normalization
- Global average pooling
- 2-layer classifier head
- ~1M parameters (vs ~25M for ResNet50)

### BalancedSampler

Handles class imbalance by ensuring equal representation of all classes during training:

- Oversamples minority classes
- Randomly samples majority classes
- Maintains balanced batches

### ImageDataset

Robust dataset loader with advanced augmentation:

- **Training**: Random masking, rotation, color jitter, rescaling
- **Validation/Test**: Standard normalization only
- **Fallback**: Random tensors for missing images

## Loss Function Details

The distillation loss combines multiple components:

```python
total_loss = α * soft_loss + (1-α) * hard_loss

where:
- soft_loss = KL_div(student_logits/T, teacher_logits/T) * T²
- hard_loss = CrossEntropy(student_logits, ground_truth)
- α = 0.7 (weight for soft targets)
- T = 3.0 (temperature for softening)
```

## Configuration Options

### Distillation Parameters

- `temperature`: Softening temperature for teacher predictions (default: 3.0)
- `alpha`: Weight for soft vs hard targets (default: 0.7)
- `learning_rate`: Student model learning rate (default: 1e-3)
- `weight_decay`: L2 regularization (default: 1e-4)

### Data Parameters

- `batch_size`: Training batch size (default: 32)
- `test_size`: Fraction for test set (default: 0.2)
- `val_size`: Fraction for validation set (default: 0.1)
- `data_root`: Root directory for images (default: "./data")

## Expected Results

Typical performance on screen classification tasks:

- **Compression Ratio**: 20-30x parameter reduction
- **Performance Retention**: 85-95% of teacher performance
- **Model Size**: ~1MB vs ~25MB for teacher
- **Inference Speed**: 3-5x faster than teacher

## File Structure

```
research/screen-page-classification/
├── clean_distillation_pipeline.py    # Main pipeline implementation
├── clean_distillation_demo.ipynb     # Jupyter notebook demo
├── CLEAN_DISTILLATION_README.md      # This file
├── best_teacher_model.pth            # Your trained teacher model
└── data/
    ├── annotations.csv               # Your dataset CSV
    └── images/                       # Your image files
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Missing images**: Pipeline creates random tensors as fallback
3. **Class imbalance**: BalancedSampler handles this automatically
4. **Poor performance**: Try adjusting temperature or alpha values

### Performance Tips

1. **Use GPU**: Significantly faster training
2. **Preload images**: Set `preload=True` in ImageDataset
3. **Adjust batch size**: Larger batches for better GPU utilization
4. **Monitor loss components**: Ensure both soft and hard losses are decreasing

## Advanced Usage

### Custom Student Architecture

```python
class CustomStudent(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Your custom architecture
        pass

    def forward(self, x):
        # Your forward pass
        pass

# Use in pipeline
pipeline.student_model = CustomStudent(num_classes)
```

### Custom Loss Function

```python
class CustomDistillationLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.7):
        super().__init__()
        # Your custom loss implementation
        pass

    def forward(self, student_logits, teacher_logits, targets):
        # Your loss computation
        pass

# Use in pipeline
pipeline.distillation_loss = CustomDistillationLoss()
```

## Contributing

This is a clean, focused implementation. For additional features:

1. Fork the repository
2. Add your enhancements
3. Maintain the clean, minimal dependency approach
4. Test with your datasets
5. Submit a pull request

## License

This implementation is part of the CRPlayer project and follows the same licensing terms.
