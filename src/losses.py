import torch
import torch.nn.functional as F

def clip_contrastive_loss(z_i, z_t, temperature: float = 0.07):
    z_i = F.normalize(z_i, dim=-1)
    z_t = F.normalize(z_t, dim=-1)
    logits_per_image = (z_i @ z_t.t()) / temperature
    logits_per_text  = logits_per_image.t()
    labels = torch.arange(z_i.size(0), device=z_i.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text,  labels)
    return 0.5 * (loss_i + loss_t)
