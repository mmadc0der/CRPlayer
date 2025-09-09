from __future__ import annotations

import argparse
import yaml
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train import LitClassifier
from utils.data import make_loader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt_path", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    loader: DataLoader = make_loader(
        dataset_dir=cfg["dataset_dir"],
        split_file=cfg["split"],
        num_classes=int(cfg["num_classes"]),
        image_size=int(cfg["image_size"]),
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg.get("num_workers", 4)),
        shuffle=False,
    )

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    hparams = ckpt["hyper_parameters"]
    model = LitClassifier(
        model_name=hparams["model_name"],
        num_classes=hparams["num_classes"],
        lr=hparams["lr"],
        weight_decay=hparams["weight_decay"],
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            pred = logits.argmax(dim=1)
            total += y.size(0)
            correct += (pred == y).sum().item()
            loss_sum += loss.item() * y.size(0)
    acc = correct / max(1, total)
    loss_avg = loss_sum / max(1, total)
    print({"acc": acc, "loss": loss_avg, "total": total})


if __name__ == "__main__":
    main()

