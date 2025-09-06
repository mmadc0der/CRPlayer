"""
DEPRECATED: Legacy dataset models used for filesystem-style export manifests.

The system now uses DB-backed repositories and services. These classes are
kept only for backward compatibility with any export tooling that still
consumes JSON manifests. Prefer querying the DB and generating exports
directly from query results.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
import warnings


@dataclass
class DatasetSample:
  """DEPRECATED: Single sample structure for legacy export manifests."""

  frame_path: str
  label: Any
  session_id: str
  frame_id: str
  confidence: float = 1.0
  metadata: Dict[str, Any] = field(default_factory=dict)

  def to_dict(self) -> Dict[str, Any]:
    return {
      "frame_path": self.frame_path,
      "label": self.label,
      "session_id": self.session_id,
      "frame_id": self.frame_id,
      "confidence": self.confidence,
      "metadata": self.metadata,
    }


@dataclass
class DatasetManifest:
  """DEPRECATED: Dataset manifest for legacy filesystem-style exports.

    Instantiating this class will emit a deprecation warning. Prefer using
    DB-backed queries and producing export artifacts directly.
    """

  dataset_id: str
  annotation_type: str
  project_name: str
  created_at: datetime
  source_sessions: List[str] = field(default_factory=list)
  samples: List[DatasetSample] = field(default_factory=list)
  splits: Dict[str, float] = field(default_factory=lambda: {"train": 0.8, "val": 0.2})
  categories: List[str] = field(default_factory=list)
  targets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
  metadata: Dict[str, Any] = field(default_factory=dict)

  def add_sample(self, sample: DatasetSample):
    """DEPRECATED: Add a sample to the legacy manifest."""
    self.samples.append(sample)
    if sample.session_id not in self.source_sessions:
      self.source_sessions.append(sample.session_id)

  def get_split_samples(self, split_name: str) -> List[DatasetSample]:
    """DEPRECATED: Get samples for a split from legacy manifest."""
    if split_name not in self.splits:
      return []

    split_ratio = self.splits[split_name]
    total_samples = len(self.samples)

    if split_name == "train":
      end_idx = int(total_samples * split_ratio)
      return self.samples[:end_idx]
    elif split_name == "val":
      start_idx = int(total_samples * self.splits.get("train", 0.8))
      return self.samples[start_idx:]
    else:
      return []

  def get_statistics(self) -> Dict[str, Any]:
    """DEPRECATED: Compute stats from legacy manifest."""
    stats = {
      "total_samples": len(self.samples),
      "source_sessions": len(self.source_sessions),
      "annotation_type": self.annotation_type,
      "splits": {},
    }

    for split_name in self.splits:
      split_samples = self.get_split_samples(split_name)
      stats["splits"][split_name] = len(split_samples)

    # Category distribution for classification
    if self.annotation_type == "classification":
      label_counts = {}
      for sample in self.samples:
        label = str(sample.label)
        label_counts[label] = label_counts.get(label, 0) + 1
      stats["label_distribution"] = label_counts

    return stats

  def to_dict(self) -> Dict[str, Any]:
    """DEPRECATED: Serialize legacy manifest to dict."""
    return {
      "dataset_id": self.dataset_id,
      "annotation_type": self.annotation_type,
      "project_name": self.project_name,
      "created_at": self.created_at.isoformat(),
      "source_sessions": self.source_sessions,
      "samples": [sample.to_dict() for sample in self.samples],
      "splits": self.splits,
      "categories": self.categories,
      "targets": self.targets,
      "metadata": self.metadata,
      "statistics": self.get_statistics(),
    }

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "DatasetManifest":
    """DEPRECATED: Deserialize legacy manifest from dict."""
    manifest = cls(
      dataset_id=data["dataset_id"],
      annotation_type=data["annotation_type"],
      project_name=data["project_name"],
      created_at=datetime.fromisoformat(data["created_at"]),
      source_sessions=data.get("source_sessions", []),
      splits=data.get("splits", {
        "train": 0.8,
        "val": 0.2
      }),
      categories=data.get("categories", []),
      targets=data.get("targets", {}),
      metadata=data.get("metadata", {}),
    )

    # Load samples
    samples_data = data.get("samples", [])
    for sample_data in samples_data:
      sample = DatasetSample(
        frame_path=sample_data["frame_path"],
        label=sample_data["label"],
        session_id=sample_data["session_id"],
        frame_id=sample_data["frame_id"],
        confidence=sample_data.get("confidence", 1.0),
        metadata=sample_data.get("metadata", {}),
      )
      manifest.samples.append(sample)

    warnings.warn(
      "Loading legacy DatasetManifest. Prefer DB-backed export flows.",
      DeprecationWarning,
      stacklevel=2,
    )
    return manifest
