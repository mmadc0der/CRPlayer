#!/usr/bin/env python3
"""
Screen Page Classification Pipeline - Model Comparison & Performance Visualization

This notebook focuses on model training, comparison, and performance analysis.

Objectives:
- Train multiple model architectures on the dataset
- Compare model performance across different metrics
- Visualize training progress and convergence
- Analyze model behavior and identify best performers
- Prepare for knowledge distillation experiments
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, auc
import time
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

# Add the parent directory to the path to import our modules
sys.path.append(str(Path.cwd().parent))

from data_loader import DatasetConfig, create_data_loaders
from models import ModelFactory, get_model_info
from trainer import ClassificationTrainer, get_device, MetricsTracker

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

console = Console()

def main():
    print("🚀 Model Comparison and Performance Visualization")
    print("=" * 60)
    
    # Load prepared data
    try:
        import pickle
        with open('prepared_data.pkl', 'rb') as f:
            data_info = pickle.load(f)
        
        train_data = data_info['train_data']
        val_data = data_info['val_data']
        test_data = data_info['test_data']
        class_names = data_info['class_names']
        class_to_idx = data_info['class_to_idx']
        idx_to_class = data_info['idx_to_class']
        
        print(f"✅ Loaded prepared data: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        print(f"🏷️ Classes: {', '.join(class_names)}")
        
    except FileNotFoundError:
        print("❌ Prepared data not found. Please run notebook 1 first.")
        return
    
    # Initialize data configuration
    config = DatasetConfig()
    device = get_device()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, config.data_root, class_to_idx, config, test_size=0.0, val_size=0.0
    )
    
    # Create validation and test loaders separately
    val_loader, _ = create_data_loaders(
        val_data, config.data_root, class_to_idx, config, test_size=0.0, val_size=0.0
    )
    _, test_loader = create_data_loaders(
        test_data, config.data_root, class_to_idx, config, test_size=0.0, val_size=0.0
    )
    
    print(f"📊 Data loaders created successfully")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Define models to compare
    model_types = ['resnet50', 'efficientnet_b0', 'lightweight']
    
    # Train and compare models
    results = train_and_compare_models(model_types, train_loader, val_loader, test_loader, 
                                     class_names, device, config)
    
    # Visualize results
    visualize_model_comparison(results, class_names)
    
    # Analyze best model performance
    analyze_best_model_performance(results, test_loader, class_names, device)
    
    print("\n✅ Model comparison complete! Check the generated visualizations.")

def train_and_compare_models(model_types, train_loader, val_loader, test_loader, 
                            class_names, device, config):
    """Train multiple models and compare their performance."""
    
    results = {}
    
    for model_type in model_types:
        console.print(f"\n[bold blue]Training {model_type}...[/bold blue]")
        
        try:
            # Create model
            model = ModelFactory.create_model(model_type, num_classes=len(class_names))
            model = model.to(device)
            
            # Get model info
            model_info = get_model_info(model)
            
            # Create trainer
            trainer = ClassificationTrainer(
                model=model,
                config=config,
                experiment_name=f"comparison_{model_type}",
                use_wandb=False,
                use_tensorboard=False
            )
            
            # Train model
            start_time = time.time()
            training_results = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=10,  # Reduced for demo
                learning_rate=1e-4,
                class_weights=None
            )
            training_time = time.time() - start_time
            
            # Evaluate on test set
            test_results = trainer.evaluate(test_loader, device)
            
            # Store results
            results[model_type] = {
                'model_info': model_info,
                'training_results': training_results,
                'test_results': test_results,
                'training_time': training_time,
                'trainer': trainer
            }
            
            console.print(f"[green]✅ {model_type} completed! F1: {test_results['test_metrics']['f1_score']:.4f}[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Error training {model_type}: {e}[/red]")
            continue
    
    return results

def visualize_model_comparison(results, class_names):
    """Create comprehensive visualizations comparing model performance."""
    
    if not results:
        print("❌ No results to visualize")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract metrics for comparison
    model_names = list(results.keys())
    accuracies = [results[m]['test_results']['test_metrics']['accuracy'] for m in model_names]
    f1_scores = [results[m]['test_results']['test_metrics']['f1_score'] for m in model_names]
    precisions = [results[m]['test_results']['test_metrics']['precision'] for m in model_names]
    recalls = [results[m]['test_results']['test_metrics']['recall'] for m in model_names]
    training_times = [results[m]['training_time'] for m in model_names]
    model_sizes = [results[m]['model_info']['model_size_mb'] for m in model_names]
    
    # 1. Accuracy comparison
    bars1 = axes[0, 0].bar(model_names, accuracies, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    for bar, acc in zip(bars1, accuracies):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{acc:.3f}', ha='center', va='bottom')
    
    # 2. F1 Score comparison
    bars2 = axes[0, 1].bar(model_names, f1_scores, color='lightcoral', alpha=0.7)
    axes[0, 1].set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    for bar, f1 in zip(bars2, f1_scores):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{f1:.3f}', ha='center', va='bottom')
    
    # 3. Precision vs Recall
    axes[0, 2].scatter(precisions, recalls, s=100, alpha=0.7, c=range(len(model_names)), cmap='viridis')
    for i, model in enumerate(model_names):
        axes[0, 2].annotate(model, (precisions[i], recalls[i]), xytext=(5, 5), textcoords='offset points')
    axes[0, 2].set_xlabel('Precision')
    axes[0, 2].set_ylabel('Recall')
    axes[0, 2].set_title('Precision vs Recall', fontsize=14, fontweight='bold')
    
    # 4. Training time comparison
    bars4 = axes[1, 0].bar(model_names, training_times, color='lightgreen', alpha=0.7)
    axes[1, 0].set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Training Time (seconds)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    for bar, time in zip(bars4, training_times):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                       f'{time:.1f}s', ha='center', va='bottom')
    
    # 5. Model size comparison
    bars5 = axes[1, 1].bar(model_names, model_sizes, color='gold', alpha=0.7)
    axes[1, 1].set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Model Size (MB)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    for bar, size in zip(bars5, model_sizes):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       f'{size:.1f}MB', ha='center', va='bottom')
    
    # 6. Performance vs Efficiency scatter plot
    axes[1, 2].scatter(model_sizes, f1_scores, s=100, alpha=0.7, c=range(len(model_names)), cmap='viridis')
    for i, model in enumerate(model_names):
        axes[1, 2].annotate(model, (model_sizes[i], f1_scores[i]), xytext=(5, 5), textcoords='offset points')
    axes[1, 2].set_xlabel('Model Size (MB)')
    axes[1, 2].set_ylabel('F1 Score')
    axes[1, 2].set_title('Performance vs Efficiency', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create detailed comparison table
    create_comparison_table(results)

def create_comparison_table(results):
    """Create a detailed comparison table of model performance."""
    
    table = Table(title="Model Performance Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("Parameters", style="blue")
    table.add_column("Size (MB)", style="green")
    table.add_column("Accuracy", style="magenta")
    table.add_column("F1 Score", style="yellow")
    table.add_column("Training Time (s)", style="red")
    
    for model_name, result in results.items():
        model_info = result['model_info']
        test_metrics = result['test_results']['test_metrics']
        
        table.add_row(
            model_name,
            f"{model_info['parameters']['total_parameters']:,}",
            f"{model_info['model_size_mb']:.2f}",
            f"{test_metrics['accuracy']:.4f}",
            f"{test_metrics['f1_score']:.4f}",
            f"{result['training_time']:.1f}"
        )
    
    console.print(table)

def analyze_best_model_performance(results, test_loader, class_names, device):
    """Analyze the performance of the best model in detail."""
    
    if not results:
        print("❌ No results to analyze")
        return
    
    # Find best model based on F1 score
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_results']['test_metrics']['f1_score'])
    best_result = results[best_model_name]
    
    console.print(f"\n[bold green]Best Model: {best_model_name}[/bold green]")
    console.print(f"F1 Score: {best_result['test_results']['test_metrics']['f1_score']:.4f}")
    
    # Get the best model
    best_model = best_result['trainer'].model
    best_model.eval()
    
    # Get predictions and true labels
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = best_model(data)
            probabilities = torch.softmax(output, dim=1)
            predictions = torch.argmax(output, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    
    # Create detailed performance analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(all_targets, all_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0, 0].set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # 2. Per-class F1 scores
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_predictions, average=None, zero_division=0
    )
    
    axes[0, 1].bar(range(len(class_names)), f1, color='lightcoral', alpha=0.7)
    axes[0, 1].set_title('Per-Class F1 Scores', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Class')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_xticks(range(len(class_names)))
    axes[0, 1].set_xticklabels(class_names, rotation=45, ha='right')
    
    # Add value labels
    for i, v in enumerate(f1):
        axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 3. ROC Curves for each class
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    
    # Binarize the output
    y_bin = label_binarize(all_targets, classes=range(len(class_names)))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], all_probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    for i, color in zip(range(len(class_names)), colors):
        axes[1, 0].plot(fpr[i], tpr[i], color=color, lw=2,
                       label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    axes[1, 0].plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    axes[1, 0].set_xlim([0.0, 1.0])
    axes[1, 0].set_ylim([0.0, 1.05])
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('ROC Curves', fontsize=14, fontweight='bold')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4. Prediction confidence distribution
    max_probs = np.max(all_probabilities, axis=1)
    axes[1, 1].hist(max_probs, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Maximum Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='50% threshold')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('best_model_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed classification report
    print(f"\n📊 Detailed Classification Report for {best_model_name}:")
    print(classification_report(all_targets, all_predictions, target_names=class_names))
    
    # Save best model for distillation
    best_model_path = f"best_model_{best_model_name}.pth"
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'model_info': best_result['model_info'],
        'test_metrics': best_result['test_results']['test_metrics'],
        'class_names': class_names
    }, best_model_path)
    
    print(f"\n💾 Best model saved to: {best_model_path}")
    print("   This model will be used as the teacher in knowledge distillation experiments.")

if __name__ == "__main__":
    main()