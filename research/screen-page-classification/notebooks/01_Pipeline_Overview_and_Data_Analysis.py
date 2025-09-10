#!/usr/bin/env python3
"""
Screen Page Classification Pipeline - Overview & Data Analysis

This notebook provides a comprehensive overview of the screen page classification 
research pipeline and performs detailed data analysis.

Objectives:
- Understand the complete classification pipeline architecture
- Analyze available datasets and their characteristics
- Visualize data distribution and class balance
- Assess data quality and identify potential issues
- Prepare data for model training experiments
"""

# Import required libraries
import sys
import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Add the parent directory to the path to import our modules
sys.path.append(str(Path.cwd().parent))

from data_loader import DatasetConfig, DatasetInspector, create_data_loaders
from dataset_inspector import DatasetAnalyzer
from models import ModelFactory, get_model_info
from trainer import get_device

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def main():
    print("✅ Libraries imported successfully!")
    print(f"🔧 PyTorch version: {torch.__version__}")
    print(f"🎯 Device available: {get_device()}")
    print(f"📊 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🚀 CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"💾 CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create mock dataset for demonstration
    mock_data, class_names = create_mock_dataset()
    df = pd.DataFrame(mock_data)
    
    print(f"\n📊 Mock dataset created with {len(mock_data)} samples")
    print(f"🏷️ Classes: {', '.join(class_names)}")
    print(f"\n📈 Class distribution:")
    print(df['class_name'].value_counts())
    
    # Analyze data distribution
    analyze_data_distribution(df, class_names)
    
    # Analyze model requirements
    analyze_model_requirements()
    
    # Demonstrate device management
    demonstrate_device_management()
    
    # Prepare training data
    train_data, val_data, test_data, class_to_idx, idx_to_class = prepare_training_data(
        mock_data, class_names
    )
    
    print("\n💾 Analysis complete! Check the generated plots and data.")

def create_mock_dataset():
    """Create a mock dataset for demonstration purposes."""
    np.random.seed(42)
    
    # Define screen page types
    screen_types = [
        'Home Screen', 'Settings', 'Profile', 'Search', 'Chat',
        'Gallery', 'Video Player', 'Shopping Cart', 'Login', 'Dashboard'
    ]
    
    # Create mock data with realistic distribution
    n_samples = 1000
    
    # Simulate class imbalance (common in real datasets)
    class_probs = np.array([0.25, 0.15, 0.12, 0.10, 0.08, 0.08, 0.07, 0.06, 0.05, 0.04])
    
    data = []
    for i in range(n_samples):
        class_id = np.random.choice(len(screen_types), p=class_probs)
        
        # Simulate session and frame data
        session_id = f"session_{np.random.randint(1, 100)}"
        frame_id = f"frame_{np.random.randint(1, 1000)}"
        
        # Simulate image path (mock)
        frame_path_rel = f"screenshots/{session_id}/{frame_id}.jpg"
        
        data.append({
            'session_id': session_id,
            'frame_id': frame_id,
            'frame_path_rel': frame_path_rel,
            'single_label_class_id': class_id,
            'class_name': screen_types[class_id]
        })
    
    return data, screen_types

def analyze_data_distribution(df, class_names):
    """Create comprehensive data analysis visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Class distribution bar plot
    class_counts = df['class_name'].value_counts()
    axes[0, 0].bar(range(len(class_counts)), class_counts.values, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Class Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Screen Page Type')
    axes[0, 0].set_ylabel('Number of Samples')
    axes[0, 0].set_xticks(range(len(class_counts)))
    axes[0, 0].set_xticklabels(class_counts.index, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, v in enumerate(class_counts.values):
        axes[0, 0].text(i, v + 5, str(v), ha='center', va='bottom')
    
    # 2. Class distribution pie chart
    axes[0, 1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    # 3. Class balance analysis
    class_balances = class_counts / class_counts.sum()
    imbalance_ratio = class_counts.max() / class_counts.min()
    
    axes[1, 0].bar(range(len(class_balances)), class_balances.values, color='lightcoral', alpha=0.7)
    axes[1, 0].set_title(f'Class Balance (Imbalance Ratio: {imbalance_ratio:.2f})', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Screen Page Type')
    axes[1, 0].set_ylabel('Proportion of Dataset')
    axes[1, 0].set_xticks(range(len(class_balances)))
    axes[1, 0].set_xticklabels(class_balances.index, rotation=45, ha='right')
    
    # Add horizontal line for balanced distribution
    balanced_ratio = 1.0 / len(class_balances)
    axes[1, 0].axhline(y=balanced_ratio, color='red', linestyle='--', alpha=0.7, label='Balanced')
    axes[1, 0].legend()
    
    # 4. Session distribution
    session_counts = df['session_id'].value_counts()
    axes[1, 1].hist(session_counts.values, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Samples per Session Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Number of Samples per Session')
    axes[1, 1].set_ylabel('Number of Sessions')
    
    plt.tight_layout()
    plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print data quality metrics
    print("\n📊 Data Quality Metrics:")
    print(f"Total samples: {len(df)}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Number of sessions: {df['session_id'].nunique()}")
    print(f"Average samples per session: {len(df) / df['session_id'].nunique():.1f}")
    print(f"Class imbalance ratio: {df['class_name'].value_counts().max() / df['class_name'].value_counts().min():.2f}")
    print(f"Most common class: {df['class_name'].value_counts().index[0]} ({df['class_name'].value_counts().iloc[0]} samples)")
    print(f"Least common class: {df['class_name'].value_counts().index[-1]} ({df['class_name'].value_counts().iloc[-1]} samples)")

def analyze_model_requirements():
    """Analyze device requirements for different model types."""
    model_info = {
        'Model Type': ['ResNet-50', 'ResNet-101', 'EfficientNet-B0', 'EfficientNet-B3', 
                      'ViT-Base', 'ViT-Large', 'ConvNeXt-Tiny', 'ConvNeXt-Small', 'Lightweight'],
        'Parameters (M)': [25, 44, 5, 12, 86, 307, 29, 50, 1],
        'Model Size (MB)': [100, 176, 20, 48, 344, 1228, 116, 200, 4],
        'Min VRAM (GB)': [2, 4, 1, 2, 4, 8, 2, 4, 0.5],
        'Recommended VRAM (GB)': [4, 8, 2, 4, 8, 16, 4, 8, 1],
        'Inference Time (ms)': [15, 25, 8, 12, 30, 60, 20, 35, 3],
        'Training Time (min/epoch)': [5, 8, 3, 4, 10, 20, 6, 9, 1]
    }
    
    df_models = pd.DataFrame(model_info)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Model size comparison
    axes[0, 0].bar(df_models['Model Type'], df_models['Model Size (MB)'], color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Model Size (MB)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. VRAM requirements
    x_pos = np.arange(len(df_models))
    width = 0.35
    axes[0, 1].bar(x_pos - width/2, df_models['Min VRAM (GB)'], width, label='Minimum', alpha=0.7)
    axes[0, 1].bar(x_pos + width/2, df_models['Recommended VRAM (GB)'], width, label='Recommended', alpha=0.7)
    axes[0, 1].set_title('VRAM Requirements', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('VRAM (GB)')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(df_models['Model Type'], rotation=45)
    axes[0, 1].legend()
    
    # 3. Inference time comparison
    axes[1, 0].bar(df_models['Model Type'], df_models['Inference Time (ms)'], color='lightcoral', alpha=0.7)
    axes[1, 0].set_title('Inference Time Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Inference Time (ms)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Training time comparison
    axes[1, 1].bar(df_models['Model Type'], df_models['Training Time (min/epoch)'], color='lightgreen', alpha=0.7)
    axes[1, 1].set_title('Training Time per Epoch', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Training Time (min/epoch)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_requirements.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Display model requirements table
    print("\n🔧 Model Requirements Summary:")
    print(df_models.to_string(index=False))

def demonstrate_device_management():
    """Demonstrate proper device management and type safety for PyTorch models."""
    print("\n🔧 Device Management Demonstration")
    print("=" * 50)
    
    # Get available device
    device = get_device()
    print(f"Available device: {device}")
    
    # Check CUDA availability and properties
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
    else:
        print("CUDA not available, using CPU")
    
    # Demonstrate model creation and device placement
    print("\n📱 Model Creation and Device Placement:")
    
    # Test different model types
    test_models = ['resnet50', 'efficientnet_b0', 'lightweight']
    class_names = ['Home Screen', 'Settings', 'Profile', 'Search', 'Chat',
                   'Gallery', 'Video Player', 'Shopping Cart', 'Login', 'Dashboard']
    
    for model_type in test_models:
        try:
            print(f"\nTesting {model_type}...")
            
            # Create model
            model = ModelFactory.create_model(model_type, num_classes=len(class_names))
            
            # Get model info
            info = get_model_info(model)
            
            # Move to device
            model = model.to(device)
            
            # Test forward pass
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            
            with torch.no_grad():
                output = model(dummy_input)
            
            print(f"  ✅ Model created successfully")
            print(f"  📊 Parameters: {info['parameters']['total_parameters']:,}")
            print(f"  💾 Model size: {info['model_size_mb']:.2f} MB")
            print(f"  🎯 Output shape: {output.shape}")
            print(f"  🔧 Device: {next(model.parameters()).device}")
            
        except Exception as e:
            print(f"  ❌ Error with {model_type}: {e}")
    
    # Memory usage analysis
    if torch.cuda.is_available():
        print(f"\n💾 GPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
        # Clear cache
        torch.cuda.empty_cache()
        print(f"  After clearing cache: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

def prepare_training_data(data, class_names, test_size=0.2, val_size=0.1):
    """Prepare data with proper train/validation/test splits."""
    
    # Create class mapping
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(data)
    
    # Split data
    train_data, test_data = train_test_split(
        data, 
        test_size=test_size, 
        random_state=42, 
        stratify=[item['single_label_class_id'] for item in data]
    )
    
    train_data, val_data = train_test_split(
        train_data, 
        test_size=val_size / (1 - test_size), 
        random_state=42, 
        stratify=[item['single_label_class_id'] for item in train_data]
    )
    
    # Print split information
    print(f"\n📊 Data Split Summary:")
    print(f"  Total samples: {len(data)}")
    print(f"  Training: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")
    print(f"  Validation: {len(val_data)} ({len(val_data)/len(data)*100:.1f}%)")
    print(f"  Test: {len(test_data)} ({len(test_data)/len(data)*100:.1f}%)")
    
    # Check class distribution in each split
    def analyze_split(split_data, split_name):
        split_df = pd.DataFrame(split_data)
        class_dist = split_df['class_name'].value_counts()
        print(f"\n  {split_name} class distribution:")
        for class_name in class_names:
            count = class_dist.get(class_name, 0)
            percentage = count / len(split_data) * 100
            print(f"    {class_name}: {count} ({percentage:.1f}%)")
    
    analyze_split(train_data, "Training")
    analyze_split(val_data, "Validation")
    analyze_split(test_data, "Test")
    
    return train_data, val_data, test_data, class_to_idx, idx_to_class

if __name__ == "__main__":
    main()