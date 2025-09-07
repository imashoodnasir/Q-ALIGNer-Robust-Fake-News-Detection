import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

class NewsMultimodalDataset(Dataset):
    def __init__(self, csv_path, image_root, tokenizer_name, max_len=160, training=True, synthetic_if_missing=True):
        self.df = pd.read_csv(csv_path) if os.path.exists(csv_path) else None
        self.image_root = image_root
        self.training = training
        self.synthetic = False
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=False)
        if self.df is None or any([col not in (self.df.columns if self.df is not None else []) for col in ["text","image_path","label"]]):
            # Build a tiny synthetic dataset if CSV not present
            self.synthetic = True
            self.df = pd.DataFrame({
                "text": [
                    "Breaking: Cats discovered to read quantum physics.",
                    "Official sources confirm event is fake news.",
                    "Scientists reveal images manipulated in article.",
                    "Local news: community garden festival is real."
                ],
                "image_path": ["synthetic1.jpg","synthetic2.jpg","synthetic3.jpg","synthetic4.jpg"],
                "label": [1,0,1,0]
            })
            os.makedirs(image_root, exist_ok=True)
            for ip in self.df["image_path"].unique():
                img = Image.new("RGB", (256,256), color=(200,200,200))
                img.save(os.path.join(image_root, ip))

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["text"])
        img_path = os.path.join(self.image_root, str(row["image_path"]))
        label = int(row["label"])

        enc = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        for k in enc: enc[k] = enc[k].squeeze(0)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224,224), color=(128,128,128))
        img = self.transform(img)

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "image": img,
            "label": torch.tensor(label, dtype=torch.long)
        }

def build_loaders(cfg):
    train_ds = NewsMultimodalDataset(cfg.train_csv, cfg.image_root, cfg.text_model_name, training=True)
    val_ds   = NewsMultimodalDataset(cfg.val_csv, cfg.image_root, cfg.text_model_name, training=False)
    test_ds  = NewsMultimodalDataset(cfg.test_csv, cfg.image_root, cfg.text_model_name, training=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader
