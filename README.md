# Q-ALIGNer: Quantum Entanglement-Driven Multimodal Fake News Detection

This repository provides a minimal working implementation of the **Q-ALIGNer** framework, 
a quantum-inspired multimodal architecture for robust fake news detection. 
It integrates **classical feature extraction**, **quantum state encoding**, 
**learnable entanglement fusion**, **contrastive alignment**, and **robustness-aware training**.

---

## 📂 File Structure
```
qaligner/
  ├── config.py              # Configuration (paths, hyperparameters)
  ├── datasets.py            # Dataset loader (CSV + image/text preprocessing)
  ├── losses.py              # CE, InfoNCE, Swap, Bures (proxy), calibration
  ├── adversarial.py         # FGSM adversarial image attack
  ├── train.py               # Training and evaluation loops
  ├── models/
  │     ├── encoders.py      # TextEncoder (BERT) + ImageEncoder (ResNet18)
  │     ├── quantum_layer.py # PennyLane quantum entanglement layer
  │     └── qaligner.py      # Full Q-ALIGNer architecture
main_train.py                 # Training script
requirements.txt              # Dependencies
README.md                     # Documentation
```

---

## ⚙️ Installation
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📊 Data Format
The model expects CSV files with the following structure:
```
text,image_path,label
"Some news headline","img1.jpg",1
"Another article","img2.jpg",0
```
- `text`: textual claim or article excerpt
- `image_path`: relative path to the associated image
- `label`: 0 (real) or 1 (fake)

Images should be placed in the folder defined by `Config.image_root` in `config.py`.

⚠️ If no dataset is provided, the code automatically generates a **synthetic toy dataset** so the pipeline can be tested.

---

## 🚀 Running Training
```bash
python main_train.py
```
- Trains Q-ALIGNer on the dataset (or synthetic fallback)
- Saves best checkpoint to `checkpoints/qaligner_best.pt`
- Prints validation metrics per epoch and test metrics at the end

---

## 🧩 Features
- **Text Encoder**: DistilBERT (transformer-based)
- **Image Encoder**: ResNet18 (lightweight CNN)
- **Quantum Fusion**: PennyLane variational quantum circuit with entangling gates
- **Composite Loss**: CE + InfoNCE + Swap + Bures (proxy) + Robustness
- **Adversarial Training**: Simple FGSM perturbations for images
- **Uncertainty Calibration**: Expected Calibration Error (ECE)

---

## 🔧 Tips
- Set `use_quantum=False` in `config.py` to disable quantum layer and use a classical baseline.
- Use GPU (`device="cuda"`) if available, else CPU fallback is automatic.
- Adjust `batch_size`, `epochs`, and `proj_dim` for real datasets vs. synthetic toy data.
- Extend adversarial module to include **PGD** or **text perturbations** for stronger robustness testing.

---

## 📜 License
This project is released under the MIT License.
