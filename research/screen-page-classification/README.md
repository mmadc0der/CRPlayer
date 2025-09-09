Screen Page Classification Research Pipeline

Overview

This directory contains a research-ready pipeline for training and evaluating a screen page classification model using the annotation tool in `tools/annotation`.

Components

- prepare_dataset.py: Export labeled data from the annotation DB into a training-ready dataset directory.
- train.py: Train a classifier using PyTorch Lightning.
- evaluate.py: Evaluate a trained checkpoint on a validation/test split.
- infer.py: Run inference on images or frames by session/idx via the annotation services.
- configs/: Example YAML configs for dataset paths, hyperparameters, and output paths.
- utils/: Shared utilities.

Quickstart

1) Ensure the annotation API is running and the SQLite DB is populated (see tools/annotation/README.md). Set ANNOTATION_DB env var if needed.

2) Prepare dataset

```bash
python prepare_dataset.py --output_dir /tmp/screen_page_ds --dataset_id 1 --split 0.8 0.1 0.1
```

3) Train

```bash
python train.py --config configs/train_example.yaml
```

4) Evaluate

```bash
python evaluate.py --config configs/eval_example.yaml --ckpt_path runs/exp1/checkpoints/best.ckpt
```

5) Inference

```bash
python infer.py --ckpt_path runs/exp1/checkpoints/best.ckpt --image /path/to/image.png
```

Notes

- The dataset exporter uses the annotation DB via services in `tools/annotation`. It requires the Python path to include the repo root or uses relative imports.
- See configs for adjustable parameters.

