Project: AD vs NC Classifier (ConvNeXt)

Overview
- Keeps existing `main.py` unchanged for debugging.
- Adds a modular, assignment-style structure under `code/adni` with a CLI in `code/cli.py`.

Structure
- `code/adni/models.py` — Model blocks and `CustomConvNeXt`.
- `code/adni/data.py` — `ADNIDataset`, data prep, and DataLoaders.
- `code/adni/train_loop.py` — Train/eval loops and plotting.
- `code/adni/eval_loop.py` — Checkpoint evaluation utilities.
- `code/cli.py` — CLI with `train` and `eval` subcommands.

Quick Start
1) Train
   python code/cli.py train --data-dir AD_NC --epochs 50 --batch-size 32 --lr 1e-4 --output-dir .

2) Evaluate a checkpoint
   python code/cli.py eval --data-dir AD_NC --checkpoint best_model.pth --batch-size 32

Notes
- Defaults assume dataset at `AD_NC/{train,test}/{NC,AD}`.
- Artifacts `best_model.pth`, `training_history.png`, and `confusion_matrix.png` save to `--output-dir`.
- If your assignment specifies file names, function signatures, or CLI flags, please share it so we can align exactly.

