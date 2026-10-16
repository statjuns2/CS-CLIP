"""Training losses.

Both losses follow the Structure-CLIP implementation (Huang et al., AAAI 2024,
github.com/zjukg/Structure-CLIP, model/utils.py: WinoLoss and MarginLoss).
All features are expected to be L2-normalized, so similarities are cosine similarities.
"""
import torch
import torch.nn.functional as F


def contrastive_margin_loss(image_feats, text_feats, margin=0.2):
    """Bidirectional in-batch margin ranking loss (Structure-CLIP's WinoLoss, is_hard=False).

    image -> text: the positive caption must beat the hardest other caption in the batch.
    text -> image: the positive image must beat the mean similarity over all images in the batch.
    """
    sim = image_feats @ text_feats.T
    pos = torch.diag(sim)
    eye = torch.eye(sim.shape[0], device=sim.device)
    hardest_caption = torch.max(sim - 10 * eye, dim=1).values
    mean_image = torch.mean(sim.T, dim=1)
    loss = F.relu(hardest_caption + margin - pos) + F.relu(mean_image + margin - pos)
    return loss.mean()


def hard_negative_hinge_loss(image_feats, pos_text_feats, neg_text_feats, margin=0.1):
    """max(0, margin - sim(v, c) + sim(v, c^-)), averaged over the batch."""
    pos = torch.diag(image_feats @ pos_text_feats.T)
    neg = torch.diag(image_feats @ neg_text_feats.T)
    return F.relu(margin - pos + neg).mean()
