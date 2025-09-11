# Screen Page Classification Research Pipeline Overview

## 🎯 Pipeline Architecture

This research infrastructure provides a comprehensive PyTorch-based pipeline for screen page classification experiments, designed to handle the complete workflow from data inspection to model deployment.

### Core Components

```
research/screen-page-classification/
├── 📊 Data Management
│   ├── data_loader.py          # Dataset loading and API integration
│   ├── dataset_inspector.py    # Data analysis and visualization
│   └── auto_labeler.py         # Automated labeling system
├── 🧠 Model Architectures
│   ├── models.py               # Model definitions and factory
│   └── distillation_pipeline.py # Knowledge distillation
├── 🚀 Training & Evaluation
│   ├── trainer.py              # Training loops and metrics
│   ├── experiment_runner.py    # Experiment orchestration
│   └── main.py                 # CLI interface
└── 📈 Visualization
    └── notebooks/              # Jupyter notebooks for analysis
```

## 🔬 Supported Model Architectures

| Model Type | Parameters | Use Case | Device Requirements |
|------------|------------|----------|-------------------|
| **ResNet-50** | 25M | Heavy classifier, high accuracy | GPU recommended |
| **ResNet-101** | 44M | Deeper variant, better performance | GPU required |
| **EfficientNet-B0** | 5M | Balanced performance/size | GPU recommended |
| **EfficientNet-B3** | 12M | Higher accuracy variant | GPU recommended |
| **ViT-Base** | 86M | Vision Transformer, SOTA | GPU required |
| **ViT-Large** | 307M | Maximum performance | High-end GPU required |
| **ConvNeXt-Tiny** | 29M | Modern ConvNet | GPU recommended |
| **ConvNeXt-Small** | 50M | Larger ConvNeXt | GPU recommended |
| **Lightweight** | 1M | Production deployment | CPU/Edge device |

## 🎯 Most Promising Experiment Scenario

### **Knowledge Distillation Pipeline** ⭐

**Why this scenario is most promising:**

1. **Production Ready**: Creates deployable models for real-world applications
2. **Performance vs Size Trade-off**: Balances accuracy with inference speed
3. **Scalability**: Enables deployment on edge devices and mobile platforms
4. **Cost Effective**: Reduces computational requirements for inference
5. **Proven Results**: Knowledge distillation typically achieves 90-95% of teacher performance with 10-100x smaller models

### Experiment Flow:
```
Heavy Teacher Model (ResNet-50/EfficientNet) 
    ↓ Knowledge Transfer
Lightweight Student Model (Custom CNN)
    ↓ Optimization
Production-Ready Classifier
```

## 🔧 Device Management & Type Safety

### Device Requirements by Model Type

#### **GPU Models** (Recommended for training)
- ResNet-50/101: 4-8GB VRAM
- EfficientNet-B0/B3: 2-6GB VRAM  
- ViT-Base: 8-12GB VRAM
- ViT-Large: 16-24GB VRAM
- ConvNeXt variants: 4-8GB VRAM

#### **CPU/Edge Models** (Production deployment)
- Lightweight CNN: <1GB RAM
- Quantized models: <500MB RAM

### Type Safety Considerations

```python
# Device detection and management
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model placement with proper typing
model = model.to(device, dtype=torch.float32)

# Data loading with device consistency
data = data.to(device, dtype=torch.float32)
```

## 📊 Pipeline Workflow

### 1. **Data Inspection & Preparation**
- Connect to annotation API
- Analyze class balance and data quality
- Generate comprehensive data reports
- Download and prepare datasets

### 2. **Model Training & Comparison**
- Train multiple architectures
- Compare performance metrics
- Track training progress with TensorBoard
- Implement early stopping and checkpointing

### 3. **Knowledge Distillation**
- Select best teacher model
- Train lightweight student model
- Compare teacher vs student performance
- Optimize for production deployment

### 4. **Auto-labeling & Active Learning**
- Use trained models for large-scale labeling
- Implement confidence-based filtering
- Select samples for manual review
- Continuously improve model performance

## 🎨 Visualization Strategy

### Notebook 1: Pipeline Overview & Data Analysis
- Data distribution analysis
- Class balance visualization
- Data quality assessment
- Pipeline architecture diagrams

### Notebook 2: Model Comparison & Performance
- Training curves and metrics
- Model performance comparison
- Confusion matrices and ROC curves
- Resource utilization analysis

### Notebook 3: Knowledge Distillation & Production
- Teacher vs student performance
- Model compression analysis
- Inference speed benchmarks
- Production deployment strategies

## 🚀 Getting Started

### Quick Start Commands

```bash
# Inspect available data
python main.py inspect

# Run full experiment pipeline
python main.py experiment --dataset-id 1 --models resnet50,efficientnet_b0,lightweight

# Run knowledge distillation
python main.py distill --teacher-model-path ./experiments/best_model.pth --student-model-type lightweight

# Auto-label large corpus
python main.py autolabel --model-path ./experiments/best_model.pth --image-dir ./screenshots --output-dir ./labeled_data
```

### Configuration

The pipeline supports YAML configuration for easy customization:

```yaml
data:
  annotation_api_url: "http://localhost:5000"
  data_root: "/workspace/data"
  batch_size: 32
  image_size: [224, 224]

experiments:
  output_dir: "/workspace/research/experiments"
  use_class_weights: true
  use_data_augmentation: true
```

## 📈 Expected Outcomes

### Performance Targets
- **Teacher Model**: 85-95% accuracy on screen classification
- **Student Model**: 80-90% accuracy (90-95% of teacher performance)
- **Inference Speed**: 10-100x faster than teacher model
- **Model Size**: 1-10MB vs 25-300MB for teacher models

### Production Benefits
- Real-time classification on mobile devices
- Reduced server costs for inference
- Lower latency for user interactions
- Scalable deployment across edge devices

## 🔍 Research Focus Areas

1. **Model Compression**: Advanced pruning and quantization techniques
2. **Architecture Search**: Neural architecture search for optimal student models
3. **Multi-modal Learning**: Incorporating text and layout information
4. **Active Learning**: Intelligent sample selection for annotation
5. **Domain Adaptation**: Transfer learning across different app types

This pipeline provides a solid foundation for screen page classification research with a clear path from experimentation to production deployment.