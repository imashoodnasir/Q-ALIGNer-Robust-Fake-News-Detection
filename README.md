# Q-ALIGNer (Minimal Working Implementation)

This repo provides a **working, minimal** implementation of the paper's core ideas:
- Classical encoders for **text** (DistilBERT) and **image** (ResNet18)
- Projection to a **shared latent space**
- A **quantum entanglement** layer (PennyLane) for fusion
- Multi-term objective with **Cross-Entropy + InfoNCE + Swap + Bures (proxy) + optional robustness**
- Evaluation with **accuracy** and **ECE**

> If you don't have the datasets ready, the loaders automatically fall back to a **small synthetic dataset** so you can run and verify end-to-end training.

## File Structure
```
qaligner/
  __init__.py
  config.py                # hyperparameters & paths
  datasets.py              # dataset + dataloaders (CSV expected; synthetic fallback)
  losses.py                # losses including InfoNCE, swap, Bures (proxy), ECE
  adversarial.py           # simple FGSM for images
  models/
    encoders.py            # TextEncoder (HF) + ImageEncoder (ResNet18)
    quantum_layer.py       # PennyLane circuit (angle encoding + entanglement)
    qaligner.py            # Full model wiring + heads
main_train.py              # training script
requirements.txt
```

## Install
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

> If you cannot download pretrained weights (air-gapped), the code still runs with **randomly initialized** encoders.

## Data Format
Provide CSV files for train/val/test with columns: `text,image_path,label`.
Set paths in `qaligner/config.py`. Example:
```
data/train.csv
data/val.csv
data/test.csv
data/images/...
```
If files are **missing**, a tiny synthetic dataset is auto-generated in `data/images/` so the code runs.

## Run
```bash
python main_train.py
```
- Best checkpoint saved to `checkpoints/qaligner_best.pt`
- Prints validation metrics each epoch and final test metrics.

## Notes
- Quantum layer keeps **num_qubits** small (default 6) for speed.
- `Config.use_quantum=False` turns Q-ALIGNer into a classical multimodal baseline (still trains).
- Robustness training uses **image FGSM** as a simple example (set `lambda_robust` and `adv_eps` in config).

## Reproducibility Tips
- Set `device="cpu"` in `config.py` if you don't have a GPU.
- Adjust `batch_size`, `epochs` for your hardware.
- When using real datasets, consider increasing `proj_dim`, `epochs`, and enabling PGD steps.
