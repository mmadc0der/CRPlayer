from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from pathlib import Path
import json

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


@dataclass
class Record:
    image_path: str
    label: int


class JsonlImageDataset(Dataset):
    def __init__(self, dataset_dir: str, split_file: str, num_classes: int, image_size: int) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.samples: List[Record] = []
        with open(self.dataset_dir / split_file, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.samples.append(Record(image_path=obj["image"], label=int(obj["label"])))
        self.num_classes = num_classes
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rec = self.samples[idx]
        img = Image.open(self.dataset_dir / rec.image_path).convert("RGB")
        return self.transform(img), rec.label


def make_loader(dataset_dir: str, split_file: str, num_classes: int, image_size: int, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    ds = JsonlImageDataset(dataset_dir=dataset_dir, split_file=split_file, num_classes=num_classes, image_size=image_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

