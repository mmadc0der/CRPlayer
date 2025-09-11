from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import logging


class AutoLabelUnavailable(Exception):
  """Raised when autolabel dependencies or model are unavailable."""


class AutoLabelService:
  """
  Lightweight wrapper around the research screen-page-classification pipeline to run
  inference for a single image. Designed to fail gracefully when optional
  dependencies are missing.

  Assumptions:
  - Classification only (SingleLabelClassification) mapping index -> dataset class order
  - Model path: ./data/models/<target_type>/model.pth relative to project root
  """

  def __init__(self):
    self._log = logging.getLogger("annotation.service.AutoLabelService")
    self._loaded: Dict[str, Any] = {}

  def _project_root(self) -> Path:
    # Heuristic: look upwards for a marker directory (data/) from this file
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
      candidate = p.parent if p.is_file() else p
      if (candidate / "data").exists():
        return candidate
    # Fallback to working directory
    return Path.cwd()

  def _resolve_model_path(self, target_type_name: str) -> Path:
    root = self._project_root()
    return (root / "data" / "models" / str(target_type_name) / "model.pth").resolve()

  def _lazy_load(self, target_type_name: str, num_classes: int):
    # Reuse cached model keyed by target_type_name
    key = f"{target_type_name}:{num_classes}"
    if key in self._loaded:
      return self._loaded[key]

    try:
      import torch  # noqa: F401
    except Exception as e:
      raise AutoLabelUnavailable(f"torch not available: {e}")

    # Import research factory lazily
    try:
      # Allow both installed package or local path
      import importlib
      spec_root = self._project_root() / "research" / "screen-page-classification"
      if spec_root.exists():
        import sys
        sys.path.insert(0, str(spec_root))
      models_mod = importlib.import_module("models")
      from models import ModelFactory  # type: ignore
    except Exception as e:
      raise AutoLabelUnavailable(f"autolabel pipeline not importable: {e}")

    model_path = self._resolve_model_path(target_type_name)
    if not model_path.exists():
      raise AutoLabelUnavailable(f"model file not found: {model_path}")

    try:
      import torch
      model = ModelFactory.create_model("lightweight", num_classes=num_classes)
      ckpt = torch.load(str(model_path), map_location="cpu")
      # Support both pure state_dict and {'model_state_dict': ...}
      state = ckpt.get("model_state_dict") if isinstance(ckpt, dict) else None
      if state is None and isinstance(ckpt, dict):
        # try common keys
        for k in ("state_dict", "model", "weights"):
          if k in ckpt and isinstance(ckpt[k], dict):
            state = ckpt[k]
            break
      if state is None:
        state = ckpt
      model.load_state_dict(state, strict=False)
      model.eval()
      device = ("cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu")
      model.to(device)
      self._loaded[key] = {"model": model, "device": device}
      return self._loaded[key]
    except Exception as e:
      raise AutoLabelUnavailable(f"failed to load model: {e}")

  def predict_single(self,
                     image_path: Path,
                     target_type_name: str,
                     class_names_in_order: List[str],
                     confidence_threshold: float = 0.9) -> Dict[str, Any]:
    """Run classification for a single image and map to dataset class id by order.

    Returns: { index, category_name, confidence }
    """
    try:
      import torch
      import torchvision.transforms as T
      from PIL import Image
      import torch.nn.functional as F
    except Exception as e:
      raise AutoLabelUnavailable(f"autolabel dependencies missing: {e}")

    num_classes = max(1, len(class_names_in_order))
    bundle = self._lazy_load(target_type_name, num_classes)
    model = bundle["model"]
    device = bundle["device"]

    # Preprocess
    img = Image.open(str(image_path)).convert("RGB")
    tfm = T.Compose([
      T.Resize((224, 224)),
      T.ToTensor(),
      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = tfm(img).unsqueeze(0)
    x = x.to(device)

    with torch.inference_mode():
      logits = model(x)
      probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
      idx = int(probs.argmax())
      conf = float(probs[idx])

    # Clamp index into range
    idx = max(0, min(idx, len(class_names_in_order) - 1))
    name = class_names_in_order[idx] if class_names_in_order else str(idx)
    return {
      "index": idx,
      "category_name": name,
      "confidence": conf,
      "meets_threshold": bool(conf >= float(confidence_threshold)),
    }

