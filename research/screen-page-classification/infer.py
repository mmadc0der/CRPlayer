from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T

from train import LitClassifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

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

    image_size = 224
    tfm = T.Compose(
        [
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = Image.open(args.image).convert("RGB")
    x = tfm(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred = int(probs.argmax().item())
    print({"pred": pred, "probs": probs.tolist()})


if __name__ == "__main__":
    main()

