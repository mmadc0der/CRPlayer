from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import logging


class AutoLabelUnavailable(Exception):
  """Raised when autolabel dependencies or model are unavailable."""


class AutoLabelService:
  """
  Wrapper around the research screen-page-classification pipeline to run
  inference for a single image.

  Assumptions:
  - Classification only (SingleLabelClassification) 
  - Model path: ./data/models/<target_type>/model.pth or ./data/models/model.pth
  - Checkpoint must contain model_info with architecture details
  """

  def __init__(self):
    self._log = logging.getLogger("annotation.service.AutoLabelService")
    self._loaded: Dict[str, Any] = {}

  def _project_root(self) -> Path:
    """Get project root, preferring Docker mount point."""
    # In Docker, we're at /app with data at /app/data
    if Path("/app/data").exists():
      return Path("/app")

    # Local development: look upwards for data directory
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
      candidate = p.parent if p.is_file() else p
      if (candidate / "data").exists():
        return candidate

    # Fallback to working directory
    return Path.cwd()

  def _resolve_model_path(self, target_type_name: str) -> Path:
    """Find model checkpoint, raising error if not found."""
    root = self._project_root()

    # Try target-type specific path first
    candidate = (root / "data" / "models" / str(target_type_name) / "model.pth").resolve()
    if candidate.exists():
      return candidate

    # Fallback to general model path
    fallback = (root / "data" / "models" / "model.pth").resolve()
    if fallback.exists():
      return fallback

    # No model found - this is an error
    raise AutoLabelUnavailable(f"Model not found. Looked for:\n"
                               f"  - {candidate}\n"
                               f"  - {fallback}\n"
                               "Please ensure a trained model is available.")

  def _lazy_load(self, target_type_name: str, num_classes: int):
    """Load model from checkpoint with proper architecture detection."""
    # Reuse cached model keyed by target_type_name
    key = f"{target_type_name}:{num_classes}"
    if key in self._loaded:
      return self._loaded[key]

    try:
      import torch  # noqa: F401
    except Exception as e:
      raise AutoLabelUnavailable(f"torch not available: {e}")

    # Get model path - will raise if not found
    model_path = self._resolve_model_path(target_type_name)

    try:
      import torch

      # Load checkpoint to detect architecture
      ckpt = torch.load(str(model_path), map_location="cpu")

      # Extract model info from checkpoint
      if not isinstance(ckpt, dict):
        raise AutoLabelUnavailable(f"Invalid checkpoint format at {model_path}")

      model_info = ckpt.get("model_info")
      if not model_info:
        raise AutoLabelUnavailable(f"Checkpoint at {model_path} missing 'model_info'. "
                                   "This checkpoint may not be from the research training pipeline.")

      model_class_name = str(model_info.get("model_type", "")).strip()
      checkpoint_num_classes = model_info.get("num_classes", num_classes)

      # Log if class count differs
      if checkpoint_num_classes != num_classes:
        self._log.warning(f"Model trained with {checkpoint_num_classes} classes, "
                          f"but dataset has {num_classes} classes")

      # Map model class to factory key
      model_type_map = {
        "ResNetClassifier": "resnet50",
        "EfficientNetClassifier": "efficientnet_b0",
        "VisionTransformerClassifier": "vit_base",
        "ConvNeXtClassifier": "convnext_tiny",
        "LightweightClassifier": "lightweight",
      }

      model_type = model_type_map.get(model_class_name)
      if not model_type:
        raise AutoLabelUnavailable(f"Unknown model type '{model_class_name}' in checkpoint. "
                                   f"Expected one of: {list(model_type_map.keys())}")

      self._log.info(f"Loading {model_class_name} as {model_type} from {model_path}")

      # Import research ModelFactory
      try:
        import importlib.util
        import sys

        # Only look in research directory within project (not outside data/)
        spec_root = self._project_root() / "research" / "screen-page-classification"
        models_path = spec_root / "models.py"

        if not models_path.exists():
          raise ImportError(f"Research models.py not found at {models_path}")

        # Import as separate module to avoid collision with tools/annotation/models
        spec = importlib.util.spec_from_file_location("spc_models", str(models_path))
        if spec and spec.loader:
          spc_models = importlib.util.module_from_spec(spec)
          sys.modules["spc_models"] = spc_models
          spec.loader.exec_module(spc_models)  # type: ignore[attr-defined]
          ModelFactory = getattr(spc_models, "ModelFactory", None)

          if not ModelFactory:
            raise ImportError("ModelFactory not found in research models")
      except Exception as e:
        raise AutoLabelUnavailable(f"Cannot import research models: {e}\n"
                                   "Ensure research/screen-page-classification/models.py is available.")

      # Create model with detected architecture
      self._log.info(f"Creating {model_type} model with {checkpoint_num_classes} classes")
      model = ModelFactory.create_model(
        model_type,
        num_classes=checkpoint_num_classes,  # Use checkpoint's class count
        pretrained=False  # We'll load weights from checkpoint
      )

      # Load state dict
      state_dict = ckpt.get("model_state_dict")
      if state_dict is None:
        raise AutoLabelUnavailable(f"No 'model_state_dict' in checkpoint at {model_path}")

      # Load with strict=True to catch architecture mismatches
      try:
        model.load_state_dict(state_dict, strict=True)
      except RuntimeError as e:
        # Try with strict=False and log warning
        self._log.warning(f"Loading with strict=False due to: {e}")
        model.load_state_dict(state_dict, strict=False)

      model.eval()

      # Move to appropriate device
      device = "cuda" if torch.cuda.is_available() else "cpu"
      model.to(device)
      self._log.info(f"Model loaded on {device}")

      self._loaded[key] = {"model": model, "device": device}
      return self._loaded[key]

    except AutoLabelUnavailable:
      raise
    except Exception as e:
      raise AutoLabelUnavailable(f"Failed to load model: {e}")

  def predict_single(self,
                     image_path: Path,
                     target_type_name: str,
                     class_names_in_order: List[str],
                     confidence_threshold: float = 0.9) -> Dict[str, Any]:
    """Run classification for a single image and map to dataset class.

    Returns: { index, category_name, confidence, meets_threshold }
    """
    try:
      import torch
      import torchvision.transforms as T
      from PIL import Image
      import torch.nn.functional as F
    except Exception as e:
      raise AutoLabelUnavailable(f"Required dependencies missing: {e}")

    num_classes = max(1, len(class_names_in_order))
    bundle = self._lazy_load(target_type_name, num_classes)
    model = bundle["model"]
    device = bundle["device"]

    # Preprocess image
    img = Image.open(str(image_path)).convert("RGB")
    tfm = T.Compose([
      T.Resize((224, 224)),
      T.ToTensor(),
      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = tfm(img).unsqueeze(0)
    x = x.to(device)

    # Run inference
    with torch.inference_mode():
      logits = model(x)
      probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
      idx = int(probs.argmax())
      conf = float(probs[idx])

    # Map to class name
    idx = max(0, min(idx, len(class_names_in_order) - 1))
    name = class_names_in_order[idx] if class_names_in_order else str(idx)

    return {
      "index": idx,
      "category_name": name,
      "confidence": conf,
      "meets_threshold": bool(conf >= float(confidence_threshold)),
    }
