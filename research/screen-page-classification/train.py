from __future__ import annotations

import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from utils.data import make_loader


class LitClassifier(pl.LightningModule):
    def __init__(self, model_name: str, num_classes: int, lr: float, weight_decay: float) -> None:
        super().__init__()
        self.save_hyperparameters()
        if model_name == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.head = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model_name={model_name}")

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    output_dir = Path(cfg.get("output_dir", "runs/exp"))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader: DataLoader = make_loader(
        dataset_dir=cfg["dataset_dir"],
        split_file=cfg["train_split"],
        num_classes=int(cfg["num_classes"]),
        image_size=int(cfg["image_size"]),
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg.get("num_workers", 4)),
        shuffle=True,
    )
    val_loader: DataLoader = make_loader(
        dataset_dir=cfg["dataset_dir"],
        split_file=cfg["val_split"],
        num_classes=int(cfg["num_classes"]),
        image_size=int(cfg["image_size"]),
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg.get("num_workers", 4)),
        shuffle=False,
    )

    model = LitClassifier(
        model_name=cfg.get("model_name", "resnet18"),
        num_classes=int(cfg["num_classes"]),
        lr=float(cfg.get("learning_rate", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-2)),
    )

    logger = TensorBoardLogger(save_dir=str(output_dir), name="tb")
    ckpt = ModelCheckpoint(dirpath=str(output_dir / "checkpoints"), monitor="val/acc", mode="max", save_top_k=1)
    lrmon = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=int(cfg.get("max_epochs", 10)),
        logger=logger,
        callbacks=[ckpt, lrmon],
        precision=int(cfg.get("precision", 32)),
        default_root_dir=str(output_dir),
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()

