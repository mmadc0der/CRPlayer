"""
Model definitions for screen page classification experiments.
Includes various architectures for different use cases: heavy classifiers, lightweight models, and distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from transformers import AutoModel, AutoConfig
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class BaseClassifier(nn.Module):
  """Base class for all classification models."""

  def __init__(self, num_classes: int, dropout_rate: float = 0.5):
    super().__init__()
    self.num_classes = num_classes
    self.dropout_rate = dropout_rate

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError

  def get_embedding_size(self) -> int:
    """Get the size of the feature embedding."""
    raise NotImplementedError


class ResNetClassifier(BaseClassifier):
  """ResNet-based classifier with configurable architecture."""

  def __init__(self,
               num_classes: int,
               model_name: str = "resnet50",
               pretrained: bool = True,
               dropout_rate: float = 0.5,
               freeze_backbone: bool = False):
    super().__init__(num_classes, dropout_rate)

    # Load pretrained model
    if model_name.startswith("resnet"):
      self.backbone = getattr(models, model_name)(pretrained=pretrained)
      # Remove the original classifier and avgpool
      self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove avgpool and fc
      # Get the feature size from the last conv layer
      self.embedding_size = self.backbone[-1].num_features
    else:
      raise ValueError(f"Unsupported ResNet model: {model_name}")

    # Freeze backbone if requested
    if freeze_backbone:
      for param in self.backbone.parameters():
        param.requires_grad = False

    # Custom classifier head
    self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout_rate), nn.Linear(self.embedding_size, 512),
                                    nn.ReLU(inplace=True), nn.Dropout(dropout_rate), nn.Linear(512, num_classes))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    features = self.backbone(x)
    # Apply global average pooling before classifier
    features = F.adaptive_avg_pool2d(features, (1, 1))
    return self.classifier(features)

  def get_embedding_size(self) -> int:
    return 512


class EfficientNetClassifier(BaseClassifier):
  """EfficientNet-based classifier using timm library."""

  def __init__(self,
               num_classes: int,
               model_name: str = "efficientnet_b0",
               pretrained: bool = True,
               dropout_rate: float = 0.5,
               freeze_backbone: bool = False):
    super().__init__(num_classes, dropout_rate)

    # Load pretrained model
    self.backbone = timm.create_model(
      model_name,
      pretrained=pretrained,
      num_classes=0,  # Remove classifier
      global_pool='avg')
    self.embedding_size = self.backbone.num_features

    # Freeze backbone if requested
    if freeze_backbone:
      for param in self.backbone.parameters():
        param.requires_grad = False

    # Custom classifier head
    self.classifier = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(self.embedding_size, 512),
                                    nn.ReLU(inplace=True), nn.Dropout(dropout_rate), nn.Linear(512, num_classes))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    features = self.backbone(x)
    return self.classifier(features)

  def get_embedding_size(self) -> int:
    return 512


class VisionTransformerClassifier(BaseClassifier):
  """Vision Transformer-based classifier."""

  def __init__(self,
               num_classes: int,
               model_name: str = "vit_base_patch16_224",
               pretrained: bool = True,
               dropout_rate: float = 0.5,
               freeze_backbone: bool = False):
    super().__init__(num_classes, dropout_rate)

    # Load pretrained model
    self.backbone = timm.create_model(
      model_name,
      pretrained=pretrained,
      num_classes=0,  # Remove classifier
      global_pool='')
    self.embedding_size = self.backbone.num_features

    # Freeze backbone if requested
    if freeze_backbone:
      for param in self.backbone.parameters():
        param.requires_grad = False

    # Custom classifier head
    self.classifier = nn.Sequential(nn.LayerNorm(self.embedding_size), nn.Dropout(dropout_rate),
                                    nn.Linear(self.embedding_size, 512), nn.ReLU(inplace=True),
                                    nn.Dropout(dropout_rate), nn.Linear(512, num_classes))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    features = self.backbone(x)
    # Global average pooling for ViT
    features = features.mean(dim=1)
    return self.classifier(features)

  def get_embedding_size(self) -> int:
    return 512


class ConvNeXtClassifier(BaseClassifier):
  """ConvNeXt-based classifier."""

  def __init__(self,
               num_classes: int,
               model_name: str = "convnext_tiny",
               pretrained: bool = True,
               dropout_rate: float = 0.5,
               freeze_backbone: bool = False):
    super().__init__(num_classes, dropout_rate)

    # Load pretrained model
    self.backbone = timm.create_model(
      model_name,
      pretrained=pretrained,
      num_classes=0,  # Remove classifier
      global_pool='avg')
    self.embedding_size = self.backbone.num_features

    # Freeze backbone if requested
    if freeze_backbone:
      for param in self.backbone.parameters():
        param.requires_grad = False

    # Custom classifier head
    self.classifier = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(self.embedding_size, 512),
                                    nn.ReLU(inplace=True), nn.Dropout(dropout_rate), nn.Linear(512, num_classes))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    features = self.backbone(x)
    return self.classifier(features)

  def get_embedding_size(self) -> int:
    return 512


class LightweightClassifier(BaseClassifier):
  """Lightweight classifier for production deployment."""

  def __init__(self, num_classes: int, input_size: int = 224, dropout_rate: float = 0.3):
    super().__init__(num_classes, dropout_rate)

    # Lightweight CNN architecture
    self.features = nn.Sequential(
      # Block 1
      nn.Conv2d(3, 32, 3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.Conv2d(32, 32, 3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, 2),

      # Block 2
      nn.Conv2d(32, 64, 3, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 64, 3, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, 2),

      # Block 3
      nn.Conv2d(64, 128, 3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 128, 3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, 2),

      # Block 4
      nn.Conv2d(128, 256, 3, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, 3, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.AdaptiveAvgPool2d((1, 1)))

    # Calculate the size after feature extraction
    self.embedding_size = 256

    # Classifier head
    self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout_rate), nn.Linear(self.embedding_size, 128),
                                    nn.ReLU(inplace=True), nn.Dropout(dropout_rate), nn.Linear(128, num_classes))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    features = self.features(x)
    return self.classifier(features)

  def get_embedding_size(self) -> int:
    return 128


class DistillationLoss(nn.Module):
  """Knowledge distillation loss combining hard and soft targets."""

  def __init__(self, temperature: float = 3.0, alpha: float = 0.7):
    super().__init__()
    self.temperature = temperature
    self.alpha = alpha
    self.kl_div = nn.KLDivLoss(reduction='batchmean')
    self.ce_loss = nn.CrossEntropyLoss()

  def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Soft distillation loss
    soft_loss = self.kl_div(F.log_softmax(student_logits / self.temperature, dim=1),
                            F.softmax(teacher_logits / self.temperature, dim=1)) * (self.temperature**2)

    # Hard target loss
    hard_loss = self.ce_loss(student_logits, targets)

    # Combine losses
    total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

    return total_loss


class ModelFactory:
  """Factory class for creating different model architectures."""

  @staticmethod
  def create_model(model_type: str, num_classes: int, **kwargs) -> BaseClassifier:
    """Create a model instance based on the specified type."""

    model_configs = {
      'resnet50': {
        'class': ResNetClassifier,
        'model_name': 'resnet50'
      },
      'resnet101': {
        'class': ResNetClassifier,
        'model_name': 'resnet101'
      },
      'efficientnet_b0': {
        'class': EfficientNetClassifier,
        'model_name': 'efficientnet_b0'
      },
      'efficientnet_b3': {
        'class': EfficientNetClassifier,
        'model_name': 'efficientnet_b3'
      },
      'vit_base': {
        'class': VisionTransformerClassifier,
        'model_name': 'vit_base_patch16_224'
      },
      'vit_large': {
        'class': VisionTransformerClassifier,
        'model_name': 'vit_large_patch16_224'
      },
      'convnext_tiny': {
        'class': ConvNeXtClassifier,
        'model_name': 'convnext_tiny'
      },
      'convnext_small': {
        'class': ConvNeXtClassifier,
        'model_name': 'convnext_small'
      },
      'lightweight': {
        'class': LightweightClassifier
      }
    }

    if model_type not in model_configs:
      raise ValueError(f"Unsupported model type: {model_type}. Available: {list(model_configs.keys())}")

    config = model_configs[model_type]
    model_class = config['class']

    # Remove model_name from kwargs if it exists
    model_kwargs = {k: v for k, v in kwargs.items() if k != 'model_name'}

    if 'model_name' in config:
      return model_class(num_classes=num_classes, model_name=config['model_name'], **model_kwargs)
    else:
      return model_class(num_classes=num_classes, **model_kwargs)

  @staticmethod
  def get_available_models() -> List[str]:
    """Get list of available model types."""
    return [
      'resnet50', 'resnet101', 'efficientnet_b0', 'efficientnet_b3', 'vit_base', 'vit_large', 'convnext_tiny',
      'convnext_small', 'lightweight'
    ]


def count_parameters(model: nn.Module) -> Dict[str, int]:
  """Count the number of parameters in a model."""
  total_params = sum(p.numel() for p in model.parameters())
  trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

  return {
    'total_parameters': total_params,
    'trainable_parameters': trainable_params,
    'frozen_parameters': total_params - trainable_params
  }


def get_model_info(model: BaseClassifier) -> Dict[str, Any]:
  """Get comprehensive information about a model."""
  param_counts = count_parameters(model)

  return {
    'model_type': model.__class__.__name__,
    'num_classes': model.num_classes,
    'embedding_size': model.get_embedding_size(),
    'dropout_rate': model.dropout_rate,
    'parameters': param_counts,
    'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
  }


if __name__ == "__main__":
  # Example usage
  num_classes = 10

  # Test different model architectures
  models_to_test = ['resnet50', 'efficientnet_b0', 'vit_base', 'lightweight']

  for model_type in models_to_test:
    print(f"\n--- {model_type.upper()} ---")
    model = ModelFactory.create_model(model_type, num_classes)
    info = get_model_info(model)

    print(f"Model: {info['model_type']}")
    print(f"Parameters: {info['parameters']['total_parameters']:,}")
    print(f"Trainable: {info['parameters']['trainable_parameters']:,}")
    print(f"Size: {info['model_size_mb']:.2f} MB")

    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
      output = model(x)
      print(f"Output shape: {output.shape}")
