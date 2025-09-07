import os, random, numpy as np, torch
from qaligner.config import Config
from qaligner.datasets import build_loaders
from qaligner.models.qaligner import QAligner
from qaligner.train import train_one_epoch, evaluate
from torch.optim import AdamW

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def main():
    cfg = Config()
    set_seed(cfg.seed)
    if not torch.cuda.is_available() and cfg.device == "cuda":
        cfg.device = "cpu"
    print("Using device:", cfg.device)
    train_loader, val_loader, test_loader = build_loaders(cfg)
    model = QAligner(num_classes=cfg.num_classes,
                     text_model=cfg.text_model_name,
                     image_model=cfg.image_model_name,
                     proj_dim=cfg.proj_dim,
                     use_quantum=cfg.use_quantum,
                     num_qubits=cfg.num_qubits,
                     q_depth=cfg.q_depth,
                     device=cfg.device).to(cfg.device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_acc = -1.0
    for epoch in range(1, cfg.epochs+1):
        print(f"Epoch {epoch}/{cfg.epochs}")
        tr_loss = train_one_epoch(model, train_loader, cfg, optimizer)
        val_metrics = evaluate(model, val_loader, cfg)
        print(f"train_loss={tr_loss:.4f} | val_loss={val_metrics['loss']:.4f} "
              f"| val_acc={val_metrics['acc']:.4f} | val_ece={val_metrics['ece']:.4f}")
        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/qaligner_best.pt')
            print('Saved best checkpoint -> checkpoints/qaligner_best.pt')

    test_metrics = evaluate(model, test_loader, cfg)
    print(f"TEST: loss={test_metrics['loss']:.4f} | acc={test_metrics['acc']:.4f} | ece={test_metrics['ece']:.4f}")

if __name__ == "__main__":
    main()
