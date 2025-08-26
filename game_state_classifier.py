#!/usr/bin/env python3
"""
Game State Classification Model
Tiny CNN model to classify game states: menu, loading, battle, final
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class GameStateDataset(Dataset):
    """Dataset for game state classification."""
    
    def __init__(self, annotation_file: str, transform=None):
        """
        Initialize dataset from annotation file.
        
        Args:
            annotation_file: Path to annotation JSON file
            transform: Image transformations
        """
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.session_id = self.annotations['session_id']
        self.frames = [f for f in self.annotations['frames'] if f['game_state'] is not None]
        self.transform = transform
        
        # State to index mapping
        self.state_to_idx = {
            'menu': 0,
            'loading': 1, 
            'battle': 2,
            'final': 3
        }
        self.idx_to_state = {v: k for k, v in self.state_to_idx.items()}
        
        print(f"Loaded {len(self.frames)} annotated frames from session {self.session_id}")
        
        # Print class distribution
        state_counts = {}
        for frame in self.frames:
            state = frame['game_state']
            state_counts[state] = state_counts.get(state, 0) + 1
        
        print("Class distribution:")
        for state, count in state_counts.items():
            print(f"  {state}: {count} frames ({count/len(self.frames)*100:.1f}%)")
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame_info = self.frames[idx]
        
        # Load image
        session_dir = Path("game_data/sessions") / self.session_id
        img_path = session_dir / frame_info['filename']
        
        # Read image with OpenCV (BGR format)
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL format for transforms
        img = transforms.ToPILImage()(img)
        
        if self.transform:
            img = self.transform(img)
        
        # Get label
        state = frame_info['game_state']
        label = self.state_to_idx[state]
        
        return img, label, frame_info['frame_id']


class TinyGameStateClassifier(nn.Module):
    """Tiny CNN for game state classification - optimized for speed."""
    
    def __init__(self, num_classes=4, input_size=(224, 224)):
        """
        Initialize tiny classifier.
        
        Args:
            num_classes: Number of game states (4)
            input_size: Input image size
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Tiny feature extractor - very lightweight
        self.features = nn.Sequential(
            # First block - reduce spatial size quickly
            nn.Conv2d(3, 16, kernel_size=7, stride=4, padding=3),  # 224->56
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # 56->28
            
            # Second block
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),  # 28->14
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # 14->7
            
            # Third block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 7->7
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def predict_state(self, x):
        """Predict game state with confidence."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1)
            confidence = torch.max(probs, dim=1)[0]
        
        return pred_idx, confidence, probs


class GameStateTrainer:
    """Trainer for game state classification model."""
    
    def __init__(self, model, device='cuda'):
        """
        Initialize trainer.
        
        Args:
            model: Classification model
            device: Training device
        """
        self.model = model.to(device)
        self.device = device
        
        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target, _) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}: Loss {loss.item():.4f}, '
                      f'Acc {100.*correct/total:.2f}%')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate model."""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target, _ in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = val_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, all_preds, all_targets
    
    def train(self, train_loader, val_loader, epochs=20):
        """Train the model."""
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, val_preds, val_targets = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_game_state_model.pth')
                print(f'New best model saved! Val Acc: {val_acc:.2f}%')
        
        print(f'\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%')
        return best_val_acc
    
    def plot_training_history(self):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, self.val_accuracies, 'g-', label='Val Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()


def create_data_transforms():
    """Create data transforms for training and validation."""
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),  # Light augmentation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def train_game_state_classifier(annotation_file: str, 
                               val_split: float = 0.2,
                               batch_size: int = 16,
                               epochs: int = 20):
    """
    Train game state classifier from annotation file.
    
    Args:
        annotation_file: Path to annotation JSON file
        val_split: Validation split ratio
        batch_size: Training batch size
        epochs: Number of training epochs
    """
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create transforms
    train_transform, val_transform = create_data_transforms()
    
    # Load full dataset
    full_dataset = GameStateDataset(annotation_file, transform=train_transform)
    
    # Split dataset
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Update validation dataset transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model = TinyGameStateClassifier(num_classes=4)
    
    # Create trainer
    trainer = GameStateTrainer(model, device)
    
    # Train model
    best_acc = trainer.train(train_loader, val_loader, epochs)
    
    # Plot training history
    trainer.plot_training_history()
    
    return model, trainer, best_acc


def test_real_time_classification():
    """Test real-time classification with live stream."""
    from android_stream_gpu import GPUAndroidStreamer
    
    # Load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TinyGameStateClassifier(num_classes=4)
    model.load_state_dict(torch.load('best_game_state_model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    # State names
    state_names = ['menu', 'loading', 'battle', 'final']
    
    # Transform for inference
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    def classify_frame(tensor, pts, timestamp):
        """Classify game state in real-time."""
        try:
            # Convert tensor to numpy and prepare for transform
            frame_np = tensor.cpu().numpy()
            frame_np = np.transpose(frame_np, (1, 2, 0))  # CHW to HWC
            frame_np = (frame_np * 255).astype(np.uint8)  # [0,1] to [0,255]
            
            # Apply transform
            input_tensor = transform(frame_np).unsqueeze(0).to(device)
            
            # Classify
            pred_idx, confidence, probs = model.predict_state(input_tensor)
            
            # Get results
            pred_state = state_names[pred_idx.item()]
            conf_score = confidence.item()
            
            # Print results every 30 frames
            if pts % 30 == 0:
                print(f"Frame {pts}: {pred_state} ({conf_score:.3f} confidence)")
                
                # Show all probabilities
                prob_str = " | ".join([f"{state}: {prob:.3f}" 
                                     for state, prob in zip(state_names, probs[0])])
                print(f"  Probabilities: {prob_str}")
                
        except Exception as e:
            print(f"Classification error: {e}")
    
    # Start streaming with classification
    streamer = GPUAndroidStreamer(max_fps=30, max_size=800)
    
    print("Starting real-time game state classification...")
    print("States: menu, loading, battle, final")
    print("Press Ctrl+C to stop")
    
    try:
        streamer.start_streaming(frame_callback=classify_frame)
        
        # Keep running
        import time
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping classification...")
        streamer.stop_streaming()
        print("Done!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python game_state_classifier.py train <annotation_file>")
        print("  python game_state_classifier.py test")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "train":
        if len(sys.argv) < 3:
            print("Please provide annotation file path")
            sys.exit(1)
        
        annotation_file = sys.argv[2]
        print(f"Training game state classifier from: {annotation_file}")
        
        model, trainer, best_acc = train_game_state_classifier(
            annotation_file=annotation_file,
            epochs=25,
            batch_size=16
        )
        
        print(f"Training completed with best accuracy: {best_acc:.2f}%")
        
    elif command == "test":
        print("Testing real-time classification...")
        test_real_time_classification()
        
    else:
        print("Unknown command. Use 'train' or 'test'")
