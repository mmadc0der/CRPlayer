# Screen Page Classification Research Notebooks

This directory contains comprehensive Jupyter notebooks for analyzing and visualizing the screen page classification research pipeline.

## 📚 Notebook Overview

### 1. Pipeline Overview & Data Analysis (`01_Pipeline_Overview_and_Data_Analysis.py`)
**Purpose**: Comprehensive analysis of the classification pipeline and data characteristics.

**Key Features**:
- Pipeline architecture visualization
- Data distribution and class balance analysis
- Device management and model requirements analysis
- Mock dataset creation for demonstration
- Data preparation for training experiments

**Outputs**:
- `data_analysis.png`: Data distribution visualizations
- `model_requirements.png`: Model resource requirements
- `prepared_data.pkl`: Preprocessed data for training

### 2. Model Comparison & Performance Visualization (`02_Model_Comparison_and_Performance_Visualization.py`)
**Purpose**: Train and compare multiple model architectures with detailed performance analysis.

**Key Features**:
- Multi-model training and comparison
- Performance metrics visualization
- Confusion matrix and ROC curve analysis
- Best model selection and evaluation
- Training progress tracking

**Outputs**:
- `model_comparison.png`: Model performance comparison charts
- `best_model_analysis.png`: Detailed analysis of best model
- `best_model_*.pth`: Saved best model checkpoints

### 3. Knowledge Distillation & Production Optimization (`03_Knowledge_Distillation_and_Production_Optimization.py`)
**Purpose**: Implement knowledge distillation and optimize models for production deployment.

**Key Features**:
- Knowledge distillation from teacher to student models
- Multiple distillation configuration experiments
- Production optimization analysis
- Inference performance benchmarking
- Deployment recommendations

**Outputs**:
- `distillation_analysis.png`: Distillation results visualization
- `deployment_recommendations.md`: Production deployment guide

## 🚀 Quick Start

### Prerequisites
```bash
# Install required packages
pip install -r requirements.txt

# Ensure you're in the research directory
cd /workspace/research/screen-page-classification
```

### Running the Notebooks

1. **Start with Data Analysis**:
```bash
python notebooks/01_Pipeline_Overview_and_Data_Analysis.py
```

2. **Run Model Comparison**:
```bash
python notebooks/02_Model_Comparison_and_Performance_Visualization.py
```

3. **Execute Knowledge Distillation**:
```bash
python notebooks/03_Knowledge_Distillation_and_Production_Optimization.py
```

## 🔧 Device Management & Type Safety

### CUDA/GPU Support
All notebooks include proper device management:
- Automatic CUDA detection and fallback to CPU
- Proper model and data placement on devices
- Memory usage monitoring and optimization
- Type safety with PyTorch tensors

### Device Requirements by Model Type

| Model Type | Min VRAM | Recommended VRAM | CPU Compatible |
|------------|----------|------------------|----------------|
| ResNet-50 | 2 GB | 4 GB | ✅ |
| ResNet-101 | 4 GB | 8 GB | ✅ |
| EfficientNet-B0 | 1 GB | 2 GB | ✅ |
| EfficientNet-B3 | 2 GB | 4 GB | ✅ |
| ViT-Base | 4 GB | 8 GB | ✅ |
| ViT-Large | 8 GB | 16 GB | ❌ |
| ConvNeXt-Tiny | 2 GB | 4 GB | ✅ |
| ConvNeXt-Small | 4 GB | 8 GB | ✅ |
| Lightweight | 0.5 GB | 1 GB | ✅ |

### Memory Optimization
- Automatic memory cleanup after model operations
- Batch size adjustment based on available memory
- Gradient checkpointing for large models
- Mixed precision training support

## 📊 Expected Results

### Performance Targets
- **Teacher Model**: 85-95% accuracy on screen classification
- **Student Model**: 80-90% accuracy (90-95% of teacher performance)
- **Inference Speed**: 10-100x faster than teacher model
- **Model Size**: 1-10MB vs 25-300MB for teacher models

### Key Metrics Tracked
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **Inference Time**: Time per prediction in milliseconds
- **Model Size**: Memory footprint in MB
- **Training Time**: Time to train each model
- **Memory Usage**: Peak memory consumption

## 🎯 Most Promising Experiment Scenario

### Knowledge Distillation Pipeline ⭐

**Why this scenario is most promising**:
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

## 📈 Visualization Features

### Data Analysis Visualizations
- Class distribution bar charts and pie charts
- Class balance analysis with imbalance ratios
- Session distribution histograms
- Data quality metrics

### Model Comparison Visualizations
- Accuracy and F1 score comparisons
- Precision vs Recall scatter plots
- Training time and model size comparisons
- Performance vs Efficiency trade-off analysis

### Distillation Analysis Visualizations
- Teacher vs Student performance comparison
- Knowledge transfer efficiency metrics
- Model compression ratios
- Training time analysis

### Production Optimization Visualizations
- Inference performance benchmarking
- Memory usage analysis
- Quantization impact assessment
- Deployment scenario recommendations

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in data loaders
   - Use gradient checkpointing
   - Switch to CPU training for large models

2. **Model Loading Errors**:
   - Check model path and architecture compatibility
   - Ensure proper device placement
   - Verify model state dict keys

3. **Data Loading Issues**:
   - Ensure annotation API is running
   - Check data paths and permissions
   - Verify data format compatibility

### Debug Mode
```bash
# Run with reduced epochs for quick testing
python notebooks/02_Model_Comparison_and_Performance_Visualization.py --epochs 5
```

## 📁 File Structure

```
notebooks/
├── README.md                                    # This file
├── 01_Pipeline_Overview_and_Data_Analysis.py   # Data analysis notebook
├── 02_Model_Comparison_and_Performance_Visualization.py  # Model comparison
├── 03_Knowledge_Distillation_and_Production_Optimization.py  # Distillation
├── prepared_data.pkl                           # Preprocessed data
├── data_analysis.png                          # Data visualization
├── model_requirements.png                     # Model requirements chart
├── model_comparison.png                       # Model comparison charts
├── best_model_analysis.png                    # Best model analysis
├── distillation_analysis.png                  # Distillation results
└── deployment_recommendations.md              # Production recommendations
```

## 🤝 Contributing

1. Follow the existing code structure and naming conventions
2. Add proper error handling and logging
3. Include comprehensive docstrings
4. Test on both CPU and GPU environments
5. Update this README when adding new features

## 📄 License

This project is licensed under the MIT License - see the main project LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Hugging Face for the transformers library
- TIMM for the model implementations
- Rich for the beautiful console output