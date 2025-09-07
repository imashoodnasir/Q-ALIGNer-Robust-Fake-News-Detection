import torch

@torch.no_grad()
def fgsm_attack(images, eps, grad):
    # images in [0,1] approximately; clamp after perturb
    perturbed = images + eps * grad.sign()
    return torch.clamp(perturbed, 0.0, 1.0)

def get_adversarial_images(model, batch, eps=1e-2):
    # compute gradient wrt input image
    images = batch["image"].detach().clone().requires_grad_(True)
    adv_batch = {k: v for k, v in batch.items()}
    adv_batch["image"] = images
    out = model(adv_batch)
    logits = out["logits"]
    loss = torch.nn.functional.cross_entropy(logits, batch["label"])
    loss.backward()
    grad = images.grad.detach()
    adv_images = fgsm_attack(images.detach(), eps, grad)
    return adv_images
