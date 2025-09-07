import torch
import torch.nn.functional as F

def cross_entropy_loss(logits, labels):
    return F.cross_entropy(logits, labels)

def infonce_loss(t, v, temperature=0.07):
    # NT-Xent over batch: similarities between matched t/v vs mismatches
    t = F.normalize(t, dim=-1)
    v = F.normalize(v, dim=-1)
    logits = (t @ v.T) / temperature
    labels = torch.arange(t.size(0), device=t.device)
    loss_t = F.cross_entropy(logits, labels)
    loss_v = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_t + loss_v)

def swap_loss(t, v):
    # Encourage similarity (1 - cosine)
    sim = F.cosine_similarity(t, v, dim=-1)
    return (1.0 - sim).mean()

def bures_loss(t, v, eps=1e-6):
    # Proxy: use Euclidean distance between normalized vectors as surrogate
    t = F.normalize(t, dim=-1)
    v = F.normalize(v, dim=-1)
    return ((t - v).pow(2).sum(dim=-1) + eps).mean()

def ece_score(probs, labels, n_bins=15):
    # Expected Calibration Error (scalar)
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)
    bins = torch.linspace(0, 1, n_bins+1, device=probs.device)
    ece = torch.zeros((), device=probs.device)
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if mask.any():
            ece += (mask.float().mean()) * (accuracies[mask].float().mean() - confidences[mask].mean()).abs()
    return ece
