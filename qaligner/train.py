import torch, math
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from .losses import cross_entropy_loss, infonce_loss, swap_loss, bures_loss, ece_score
from .adversarial import get_adversarial_images

def train_one_epoch(model, loader, cfg, optimizer):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        for k in list(batch.keys()):
            batch[k] = batch[k].to(cfg.device)
        out = model(batch)
        ce = cross_entropy_loss(out["logits"], batch["label"])
        nce = infonce_loss(out["t"], out["v"], cfg.temperature)
        swp = swap_loss(out["t"], out["v"])
        bur = bures_loss(out["t"], out["v"])

        loss = ce + cfg.lambda_infonce * nce + cfg.lambda_swap * swp + cfg.lambda_bures * bur

        # adversarial robustness (images only for simplicity)
        if cfg.lambda_robust > 0 and cfg.adv_eps > 0:
            adv_images = get_adversarial_images(model, batch, eps=cfg.adv_eps).to(cfg.device)
            adv_batch = dict(batch)
            adv_batch["image"] = adv_images
            adv_out = model(adv_batch)
            adv_ce = cross_entropy_loss(adv_out["logits"], batch["label"])
            loss = loss + cfg.lambda_robust * adv_ce

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, cfg):
    model.eval()
    import numpy as np
    total_loss, n = 0.0, 0
    correct = 0
    all_probs = []
    all_labels = []
    for batch in tqdm(loader, desc="eval", leave=False):
        for k in list(batch.keys()):
            batch[k] = batch[k].to(cfg.device)
        out = model(batch)
        probs = F.softmax(out["logits"], dim=-1)
        preds = probs.argmax(dim=-1)
        ce = cross_entropy_loss(out["logits"], batch["label"])
        total_loss += ce.item() * batch["label"].size(0)
        correct += (preds == batch["label"]).sum().item()
        n += batch["label"].size(0)
        all_probs.append(probs.detach().cpu())
        all_labels.append(batch["label"].detach().cpu())

    if n == 0:
        return {"loss": float("nan"), "acc": float("nan"), "ece": float("nan")}
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    ece = ece_score(all_probs, all_labels).item()
    return {"loss": total_loss / n, "acc": correct / n, "ece": ece}
