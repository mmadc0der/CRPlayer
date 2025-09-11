#!/usr/bin/env python3
"""
Game State Classifier Consumer Implementation
Separate module for real-time game state classification.
"""

import torch
import numpy as np
from typing import Optional, List
import time

from core.stream_pipeline import StreamConsumer, FrameData


class ClassifierConsumer(StreamConsumer):
  """Consumer that classifies game states in real-time."""

  def __init__(self,
               consumer_id: str,
               stream_buffer,
               model_path: Optional[str] = None,
               classification_interval: int = 30,
               confidence_threshold: float = 0.5):
    """
        Initialize classifier consumer.
        
        Args:
            consumer_id: Unique consumer identifier
            stream_buffer: Shared stream buffer
            model_path: Path to trained classification model
            classification_interval: Classify every Nth frame
            confidence_threshold: Minimum confidence for predictions
        """
    super().__init__(consumer_id, stream_buffer)

    self.model = None
    self.transform = None
    self.classification_interval = classification_interval
    self.confidence_threshold = confidence_threshold
    self.state_names = ['menu', 'loading', 'battle', 'final']

    # Classification history
    self.predictions = []
    self.last_prediction = None
    self.last_confidence = 0.0

    if model_path:
      self._load_model(model_path)

  def _load_model(self, model_path: str):
    """Load classification model."""
    try:
      from analysis.game_state_classifier import TinyGameStateClassifier
      import torchvision.transforms as transforms

      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

      self.model = TinyGameStateClassifier(num_classes=4)
      self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
      self.model.to(device)
      self.model.eval()

      self.transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])

      print(f"[CLASSIFIER] Model loaded: {model_path}")
      print(f"[CLASSIFIER] Device: {device}")
      print(f"[CLASSIFIER] Classification interval: every {self.classification_interval} frames")

    except Exception as e:
      print(f"[ERROR] Failed to load classifier model: {e}")

  def process_frame(self, frame_data: FrameData) -> bool:
    """Classify game state."""
    if self.model is None:
      return True

    try:
      # Only classify at specified intervals
      if frame_data.frame_id % self.classification_interval != 0:
        return True

      # Prepare frame for classification
      frame_np = frame_data.tensor.cpu().numpy()
      frame_np = np.transpose(frame_np, (1, 2, 0))  # CHW to HWC
      frame_np = (frame_np * 255).astype(np.uint8)

      # Apply transform
      input_tensor = self.transform(frame_np).unsqueeze(0).to(frame_data.tensor.device)

      # Classify
      with torch.no_grad():
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1)
        confidence = torch.max(probs, dim=1)[0]

      # Get results
      pred_state = self.state_names[pred_idx.item()]
      conf_score = confidence.item()

      # Store prediction
      prediction_data = {
        'frame_id': frame_data.frame_id,
        'timestamp': frame_data.timestamp,
        'state': pred_state,
        'confidence': conf_score,
        'all_probs': probs.cpu().numpy().flatten().tolist()
      }

      self.predictions.append(prediction_data)

      # Update current state if confidence is high enough
      if conf_score >= self.confidence_threshold:
        if pred_state != self.last_prediction:
          print(f"[CLASSIFIER] State change: {self.last_prediction} -> {pred_state} "
                f"(confidence: {conf_score:.3f})")

        self.last_prediction = pred_state
        self.last_confidence = conf_score

      # Keep only recent predictions (last 100)
      if len(self.predictions) > 100:
        self.predictions = self.predictions[-100:]

      return True

    except Exception as e:
      print(f"[ERROR] Classifier error: {e}")
      return True

  def get_current_state(self) -> tuple[Optional[str], float]:
    """Get current game state and confidence."""
    return self.last_prediction, self.last_confidence

  def get_state_history(self, n: int = 10) -> List[dict]:
    """Get recent state predictions."""
    return self.predictions[-n:] if self.predictions else []

  def get_state_distribution(self, window_size: int = 50) -> dict:
    """Get distribution of states in recent window."""
    if not self.predictions:
      return {}

    recent = self.predictions[-window_size:]
    distribution = {}

    for pred in recent:
      state = pred['state']
      if pred['confidence'] >= self.confidence_threshold:
        distribution[state] = distribution.get(state, 0) + 1

    # Normalize to percentages
    total = sum(distribution.values())
    if total > 0:
      distribution = {k: (v / total) * 100 for k, v in distribution.items()}

    return distribution
