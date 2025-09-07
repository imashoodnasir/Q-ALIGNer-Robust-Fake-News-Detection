from dataclasses import dataclass

@dataclass
class Config:
    # data
    train_csv: str = "data/train.csv"   # columns: text,image_path,label
    val_csv: str   = "data/val.csv"
    test_csv: str  = "data/test.csv"
    image_root: str = "data/images"
    num_classes: int = 2
    # model
    text_model_name: str = "distilbert-base-uncased"
    image_model_name: str = "resnet18"
    proj_dim: int = 64
    use_quantum: bool = True
    num_qubits: int = 6           # keep small for speed
    q_depth: int = 2
    # training
    batch_size: int = 8
    epochs: int = 3
    lr: float = 2e-4
    weight_decay: float = 1e-4
    device: str = "cuda"
    # losses
    lambda_infonce: float = 0.1
    lambda_swap: float = 0.05
    lambda_bures: float = 0.05
    lambda_robust: float = 0.2
    temperature: float = 0.07
    # adversarial
    adv_eps: float = 1e-2
    adv_pgd_steps: int = 0  # 0 -> disable PGD by default
    # misc
    seed: int = 42
