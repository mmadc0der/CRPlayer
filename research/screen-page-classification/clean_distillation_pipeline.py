"""
Clean Knowledge Distillation Pipeline
A streamlined distillation pipeline that uses a pre-trained teacher model to train a lightweight student model.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import v2
from torchvision.io import decode_image

from models import ModelFactory

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class SimpleDistillationLoss(nn.Module):
  """Simple knowledge distillation loss combining hard and soft targets."""

  def __init__(self, temperature: float = 3.0, alpha: float = 0.7):
    super().__init__()
    self.temperature = temperature
    self.alpha = alpha
    self.kl_div = nn.KLDivLoss(reduction='batchmean')
    self.ce_loss = nn.CrossEntropyLoss()

  def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
              targets: torch.Tensor) -> Dict[str, torch.Tensor]:
    # Soft distillation loss (teacher knowledge)
    soft_loss = self.kl_div(F.log_softmax(student_logits / self.temperature, dim=1),
                            F.softmax(teacher_logits / self.temperature, dim=1)) * (self.temperature**2)

    # Hard target loss (ground truth)
    hard_loss = self.ce_loss(student_logits, targets)

    # Combined loss
    total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

    return {'total_loss': total_loss, 'soft_loss': soft_loss, 'hard_loss': hard_loss}


class LightweightStudent(nn.Module):
  """Lightweight CNN student model for distillation."""

  def __init__(self, num_classes: int, dropout_rate: float = 0.3):
    super().__init__()

    self.features = nn.Sequential(
      # Block 1
      nn.Conv2d(3, 32, 3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.Conv2d(32, 32, 3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(3, 3),

      # Block 2
      nn.Conv2d(32, 32, 3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.Conv2d(32, 96, 3, padding=1),
      nn.BatchNorm2d(96),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(3, 3),

      # Block 3
      nn.Conv2d(96, 384, 3, padding=1),
      nn.BatchNorm2d(384),
      nn.ReLU(inplace=True),
      nn.AdaptiveAvgPool2d((1, 1)))

    self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout_rate), nn.Linear(384, 192), nn.ReLU(inplace=True),
                                    nn.Dropout(dropout_rate), nn.Linear(192, num_classes))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    features = self.features(x)
    return self.classifier(features)


class BalancedSampler:
  """Balanced sampler for handling class imbalance."""

  def __init__(self, dataset, samples_per_class=None):
    self.dataset = dataset
    self.labels = [dataset[i][1] for i in range(len(dataset))]
    self.class_counts = {}

    # Count samples per class
    for label in self.labels:
      self.class_counts[label] = self.class_counts.get(label, 0) + 1

    # Determine samples per class
    if samples_per_class is None:
      self.samples_per_class = max(self.class_counts.values())
    else:
      self.samples_per_class = samples_per_class

    # Create balanced indices
    self.balanced_indices = []
    for class_id in range(len(self.class_counts)):
      class_indices = [i for i, label in enumerate(self.labels) if label == class_id]
      if not len(class_indices): continue

      # Oversample if class has fewer samples than target
      if len(class_indices) < self.samples_per_class:
        repeat_factor = self.samples_per_class // len(class_indices)
        remainder = self.samples_per_class % len(class_indices)

        balanced_class_indices = class_indices * repeat_factor
        balanced_class_indices.extend(class_indices[:remainder])
      else:
        # Randomly sample if class has more samples than target
        balanced_class_indices = np.random.choice(class_indices, size=self.samples_per_class, replace=False).tolist()

      self.balanced_indices.extend(balanced_class_indices)

    # Shuffle the balanced indices
    np.random.shuffle(self.balanced_indices)

  def __iter__(self):
    return iter(self.balanced_indices)

  def __len__(self):
    return len(self.balanced_indices)


class RandomMask:
    """
    Random block mask for tensor images.

    - Input: torch.Tensor with shape (C, H, W), dtype float, values in [0, 1].
    - Output: torch.Tensor same shape/dtype. By default a non-inplace transform
              (does not modify the original tensor) unless inplace=True.

    Raises:
        TypeError: if input is not a floating-point tensor.
        ValueError: if input does not have 3 dims or plausible channel count.
    """
    def __init__(self, mask_prob=0.5, min_mask_ratio=0.1, max_mask_ratio=0.5, inplace=True):
        self.mask_prob = float(mask_prob)
        self.min_mask_ratio = float(min_mask_ratio)
        self.max_mask_ratio = float(max_mask_ratio)
        self.inplace = bool(inplace)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # Validate dtype
        if not torch.is_floating_point(image):
            raise TypeError("RandomMask expects a floating-point tensor (values in [0,1])")

        # Validate shape
        if image.ndim != 3:
            raise ValueError(f"RandomMask expects a 3D tensor (C,H,W), got shape {tuple(image.shape)}.")
        c, h, w = image.shape
        if c not in (1, 3, 4) and c > 4:
            # still allow non-standard channels, but warn via ValueError for typical mistakes
            pass

        # Quick exit if not applying mask
        if torch.rand(1).item() >= self.mask_prob:
            return image if self.inplace else image.clone()

        # Work on a copy unless inplace
        out = image if self.inplace else image.clone()

        # Sample mask size (at least 1 px)
        mask_ratio = torch.empty(1).uniform_(self.min_mask_ratio, self.max_mask_ratio).item()
        mask_h = max(1, int(h * mask_ratio))
        mask_w = max(1, int(w * mask_ratio))

        # Sample mask position (clamped)
        if h - mask_h > 0:
            mask_y = torch.randint(0, h - mask_h + 1, (1,)).item()
        else:
            mask_y = 0
            mask_h = h

        if w - mask_w > 0:
            mask_x = torch.randint(0, w - mask_w + 1, (1,)).item()
        else:
            mask_x = 0
            mask_w = w

        device = out.device

        # Choose color vs noise (float in [0,1])
        if torch.rand(1).item() < 0.5:
            # solid color per channel
            mask_color = torch.rand((c, 1, 1), dtype=image.dtype, device=device)
            out[:, mask_y:mask_y+mask_h, mask_x:mask_x+mask_w] = mask_color
        else:
            # per-pixel noise
            noise = torch.rand((c, mask_h, mask_w), dtype=image.dtype, device=device)
            out[:, mask_y:mask_y+mask_h, mask_x:mask_x+mask_w] = noise

        return out

class AdvancedAugmentation:
    """Combined augmentation strategy for handling class imbalance."""
    def __init__(self, is_training=True):
        self.is_training = is_training
        
        # Base transforms
        self.base_transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((236, 236)),  # Slightly larger for cropping
            v2.CenterCrop((224, 224)),
            v2.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
        ])
        
        # Training augmentations
        if is_training:
            self.augment_transforms = v2.Compose([
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize((256, 256)),
                v2.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Rescaling
                v2.RandomRotation(5),  # ±5 degrees
                v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Color modification
                RandomMask(mask_prob=0.6, min_mask_ratio=0.2, max_mask_ratio=0.4),  # Random masking
                RandomMask(mask_prob=0.6, min_mask_ratio=0.2, max_mask_ratio=0.4),
                v2.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
            ])
        else:
            self.augment_transforms = self.base_transforms
    
    def __call__(self, image):
        return self.augment_transforms(image)


class ImageDataset(Dataset):
  """Dataset for loading images with augmentation."""

  def __init__(self, df, data_root, class_id_col, is_training=True):
    self.df = df.reset_index(drop=True)
    self.data_root = Path(data_root)
    self.class_id_col = class_id_col

    # Create augmentation transforms
    self.transform = AdvancedAugmentation(is_training)

    self.images = []
    self.labels = []
    print("Preloading images...")
    for idx in tqdm(range(len(self.df)), desc="Loading images"):
      row = self.df.iloc[idx]

      # Try to load image from different possible path columns
      image_path = None
      if 'frame_path_rel' in row and pd.notna(row['frame_path_rel']):
        image_path = self.data_root / row['frame_path_rel'].strip('./')
      elif 'image_path' in row and pd.notna(row['image_path']):
        image_path = self.data_root / row['image_path'].strip('./')
      elif 'frame_id' in row and pd.notna(row['frame_id']):
        session_id = row.get('session_id', '')
        if session_id:
          image_path = self.data_root / "raw" / session_id / row['frame_id']

      # Load image or create random tensor if not found
      try:
        if image_path and image_path.exists():
          image = v2.functional.to_dtype(decode_image(image_path, "RGB"), torch.float16, scale=True)

          self.images.append(image)
          self.labels.append(torch.scalar_tensor(row[self.class_id_col]).to(torch.int64))
        else:
          print(f"Warning: image with path '{image_path}' not found -> skipping")
      except Exception as e:
        print(f"Warning: image {idx} cant be loaded: {e} -> skipping")
        continue
    print(f"Preloaded {len(self.images)} images")

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    return self.transform(self.images[idx]), self.labels[idx]

    

class CleanDistillationPipeline:
  """Clean distillation pipeline without heavy dependencies."""

  def __init__(self, teacher_model_path: str, num_classes: int, device: str = None):
    self.teacher_model_path = teacher_model_path
    self.num_classes = num_classes
    self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Load teacher model
    self.teacher_model = self._load_teacher_model()
    self.teacher_model.to(self.device)
    self.teacher_model.eval()

    # Create student model
    self.student_model = LightweightStudent(num_classes)
    self.student_model.to(self.device)

    # Initialize distillation loss
    self.distillation_loss = SimpleDistillationLoss(temperature=3.0, alpha=0.7)

    print(f"Distillation pipeline initialized on {self.device}")
    print(f"Teacher model loaded from: {teacher_model_path}")
    print(f"Student model created with {sum(p.numel() for p in self.student_model.parameters()):,} parameters")

  def _load_teacher_model(self):
    """Load the pre-trained teacher model."""
    teacher_model = ModelFactory.create_model("resnet50",
                                              num_classes=self.num_classes,
                                              pretrained=False)

    # Load checkpoint
    checkpoint = torch.load(self.teacher_model_path, map_location='cpu')
    teacher_model.load_state_dict(checkpoint['model_state_dict'])

    return teacher_model

  def create_data_loaders(self,
                          df,
                          data_root="./data",
                          batch_size=32,
                          test_size=0.2,
                          val_size=0.1,
                          class_id_col='class_id'):
    """Create data loaders with balanced sampling."""

    # Split data
    train_df, temp_df = train_test_split(df, test_size=test_size + val_size, random_state=42, stratify=df[class_id_col])
    val_df, test_df = train_test_split(temp_df,
                                       test_size=test_size / (test_size + val_size),
                                       random_state=42,
                                       stratify=temp_df[class_id_col])

    # Create datasets
    train_dataset = ImageDataset(train_df, data_root, class_id_col, is_training=True)
    val_dataset = ImageDataset(val_df, data_root, class_id_col, is_training=False)
    test_dataset = ImageDataset(test_df, data_root, class_id_col, is_training=False)

    # Create balanced sampler for training
    balanced_sampler = BalancedSampler(train_dataset)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=balanced_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_df, val_df, test_df

  def train_student(self, train_loader, val_loader, num_epochs=50, learning_rate=1e-3, weight_decay=1e-4):
    """Train student model using knowledge distillation."""

    # Setup optimizer
    optimizer = optim.AdamW(self.student_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training history
    history = {
      'train_loss': [],
      'train_acc': [],
      'val_loss': [],
      'val_acc': [],
      'val_f1': [],
      'soft_loss': [],
      'hard_loss': []
    }

    best_val_f1 = 0.0
    best_model_state = None

    print(f"Starting distillation training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
      # Training phase
      train_metrics = self._train_epoch(train_loader, optimizer)

      # Validation phase
      val_metrics = self._validate_epoch(val_loader)

      # Store history
      history['train_loss'].append(train_metrics['loss'])
      history['train_acc'].append(train_metrics['accuracy'])
      history['val_loss'].append(val_metrics['loss'])
      history['val_acc'].append(val_metrics['accuracy'])
      history['val_f1'].append(val_metrics['f1_score'])
      history['soft_loss'].append(train_metrics['soft_loss'])
      history['hard_loss'].append(train_metrics['hard_loss'])

      # Update best model
      if val_metrics['f1_score'] > best_val_f1:
        best_val_f1 = val_metrics['f1_score']
        best_model_state = self.student_model.state_dict().copy()

      # Print progress
      if epoch % 5 == 0 or epoch < 5:
        print(f"Epoch {epoch:2d}: Train Loss: {train_metrics['loss']:.4f}, "
              f"Train Acc: {train_metrics['accuracy']:.4f}, "
              f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.4f}, "
              f"Val F1: {val_metrics['f1_score']:.4f}")

    # Load best model
    if best_model_state is not None:
      self.student_model.load_state_dict(best_model_state)

    print(f"Training completed! Best validation F1: {best_val_f1:.4f}")

    return history, best_val_f1

  def _train_epoch(self, train_loader, optimizer):
    """Train one epoch."""
    self.student_model.train()
    self.teacher_model.eval()

    total_loss = 0.0
    total_soft_loss = 0.0
    total_hard_loss = 0.0
    predictions = []
    targets = []

    for data, target in tqdm(train_loader, desc="Training", leave=False):
      data, target = data.to(self.device), target.to(self.device)

      optimizer.zero_grad()

      # Forward pass
      student_output = self.student_model(data)

      # Teacher forward (no gradients)
      with torch.no_grad():
        teacher_output = self.teacher_model(data)

      # Compute distillation loss
      losses = self.distillation_loss(student_output, teacher_output, target)

      # Backward pass
      losses['total_loss'].backward()
      optimizer.step()

      # Update metrics
      total_loss += losses['total_loss'].item()
      total_soft_loss += losses['soft_loss'].item()
      total_hard_loss += losses['hard_loss'].item()
      predictions.extend(torch.argmax(student_output, dim=1).cpu().numpy())
      targets.extend(target.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(targets, predictions)

    return {
      'loss': total_loss / len(train_loader),
      'soft_loss': total_soft_loss / len(train_loader),
      'hard_loss': total_hard_loss / len(train_loader),
      'accuracy': accuracy
    }

  def _validate_epoch(self, val_loader):
    """Validate one epoch."""
    self.student_model.eval()

    total_loss = 0.0
    predictions = []
    targets = []

    with torch.no_grad():
      for data, target in tqdm(val_loader, desc="Validation", leave=False):
        data, target = data.to(self.device), target.to(self.device)

        # Forward pass
        student_output = self.student_model(data)
        teacher_output = self.teacher_model(data)

        # Compute loss
        losses = self.distillation_loss(student_output, teacher_output, target)

        total_loss += losses['total_loss'].item()
        predictions.extend(torch.argmax(student_output, dim=1).cpu().numpy())
        targets.extend(target.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average='weighted')

    return {'loss': total_loss / len(val_loader), 'accuracy': accuracy, 'f1_score': f1}

  def evaluate_models(self, test_loader):
    """Evaluate both teacher and student models."""
    self.teacher_model.eval()
    self.student_model.eval()

    teacher_predictions = []
    student_predictions = []
    targets = []

    with torch.no_grad():
      for data, target in tqdm(test_loader, desc="Evaluation"):
        data, target = data.to(self.device), target.to(self.device)

        teacher_output = self.teacher_model(data)
        student_output = self.student_model(data)

        teacher_predictions.extend(torch.argmax(teacher_output, dim=1).cpu().numpy())
        student_predictions.extend(torch.argmax(student_output, dim=1).cpu().numpy())
        targets.extend(target.cpu().numpy())

    # Compute metrics
    teacher_acc = accuracy_score(targets, teacher_predictions)
    student_acc = accuracy_score(targets, student_predictions)

    teacher_f1 = f1_score(targets, teacher_predictions, average='weighted')
    student_f1 = f1_score(targets, student_predictions, average='weighted')

    # Model size comparison
    teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
    student_params = sum(p.numel() for p in self.student_model.parameters())

    return {
      'teacher': {
        'accuracy': teacher_acc,
        'f1_score': teacher_f1,
        'parameters': teacher_params
      },
      'student': {
        'accuracy': student_acc,
        'f1_score': student_f1,
        'parameters': student_params
      },
      'compression_ratio': teacher_params / student_params,
      'performance_retention': student_f1 / teacher_f1 if teacher_f1 > 0 else 0
    }

  def plot_training_history(self, history):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Distillation Training Progress', fontsize=16, fontweight='bold')

    epochs = range(len(history['train_loss']))

    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # F1 score curve
    axes[1, 0].plot(epochs, history['val_f1'], 'g-', label='Validation F1 Score', linewidth=2)
    axes[1, 0].set_title('F1 Score Progress')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Loss components
    axes[1, 1].plot(epochs, history['soft_loss'], 'purple', label='Soft Loss (Teacher)', linewidth=2)
    axes[1, 1].plot(epochs, history['hard_loss'], 'orange', label='Hard Loss (Ground Truth)', linewidth=2)
    axes[1, 1].set_title('Loss Components')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

  def save_student_model(self, path: str):
    """Save the trained student model."""
    torch.save({'model_state_dict': self.student_model.state_dict(), 'num_classes': self.num_classes}, path)
    print(f"Student model saved to: {path}")


def load_dataset_from_csv(csv_path: str, data_root: str = "./data"):
  """Load dataset from CSV file."""
  df = pd.read_csv(csv_path)

  # Determine class column
  if 'class_name' in df.columns:
    class_names = df['class_name'].unique().tolist()
    class_id_col = 'class_id'
  elif 'single_label_class_id' in df.columns:
    unique_class_ids = sorted(df['single_label_class_id'].unique())
    class_names = [f"class_{cid}" for cid in unique_class_ids]
    class_id_col = 'single_label_class_id'
    # Add class_name column for consistency
    df['class_name'] = df[class_id_col].map({cid: f"class_{cid}" for cid in unique_class_ids})
  else:
    raise ValueError("No class_id or single_label_class_id column found")

  # Create mapping from original class IDs to 0-based indices
  unique_class_ids = sorted(df[class_id_col].unique())
  class_id_to_index = {class_id: idx for idx, class_id in enumerate(unique_class_ids)}
  
  # Map class IDs to 0-based indices
  df[f'{class_id_col}_original'] = df[class_id_col].copy()  # Keep original for reference
  df[class_id_col] = df[class_id_col].map(class_id_to_index)

  while True:
    dist = df[class_id_col].value_counts().sort_index()
    dist = dist.where(dist < 7).dropna()
    if dist.empty: break
    df = pd.concat([df, df.where(df[class_id_col].isin(dist.index)).dropna()])
  
  print(f"Dataset loaded: {len(df)} samples, {len(class_names)} classes")
  print(f"Class names: {class_names}")

  return df, class_names, class_id_col

def main():
  """Example usage of the clean distillation pipeline."""

  # Configuration
  teacher_model_path = "best_teacher_model.pth"
  csv_path = "./data/annotations.csv"  # Adjust path as needed
  data_root = "./data"
  num_classes = 10  # Adjust based on your dataset

  # Load dataset
  print("Loading dataset...")
  df, class_names, class_id_col = load_dataset_from_csv(csv_path, data_root)
  num_classes = len(class_names)

  # Initialize distillation pipeline
  print("Initializing distillation pipeline...")
  pipeline = CleanDistillationPipeline(teacher_model_path, num_classes)

  # Create data loaders
  print("Creating data loaders...")
  train_loader, val_loader, test_loader, train_df, val_df, test_df = pipeline.create_data_loaders(
    df, data_root, batch_size=32, class_id_col=class_id_col)

  print(f"Training samples: {len(train_df)}")
  print(f"Validation samples: {len(val_df)}")
  print(f"Test samples: {len(test_df)}")

  # Train student model
  print("Starting distillation training...")
  history, best_val_f1 = pipeline.train_student(train_loader, val_loader, num_epochs=30)

  # Plot training history
  pipeline.plot_training_history(history)

  # Evaluate models
  print("Evaluating models...")
  results = pipeline.evaluate_models(test_loader)

  print("\n=== Model Comparison ===")
  print(
    f"Teacher - Accuracy: {results['teacher']['accuracy']:.4f}, F1: {results['teacher']['f1_score']:.4f}, Params: {results['teacher']['parameters']:,}"
  )
  print(
    f"Student - Accuracy: {results['student']['accuracy']:.4f}, F1: {results['student']['f1_score']:.4f}, Params: {results['student']['parameters']:,}"
  )
  print(f"Compression Ratio: {results['compression_ratio']:.1f}x")
  print(f"Performance Retention: {results['performance_retention']:.2%}")

  # Save student model
  pipeline.save_student_model("best_student_model.pth")

  print("Distillation pipeline completed successfully!")


if __name__ == "__main__":
  main()
