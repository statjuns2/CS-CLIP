"""Fine-tune CLIP ViT-B/32 on CS-CLIP hard negatives.

Default hyper-parameters are the ones used for the released checkpoint.
"""
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from csclip import load_model
from csclip.data import Collate, HardNegativeDataset
from csclip.losses import contrastive_margin_loss, hard_negative_hinge_loss


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_file", required=True, help="JSON Lines file of hard negatives")
    p.add_argument("--image_root", required=True, help="COCO root containing train2014/")
    p.add_argument("--output_dir", default="checkpoints/cs-clip")
    p.add_argument("--init", default="openai", help="initial weights passed to csclip.load_model")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-6, help="constant learning rate (no schedule)")
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--contrastive_margin", type=float, default=0.2)
    p.add_argument("--hinge_margin", type=float, default=0.1)
    p.add_argument("--hinge_weight", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=120)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_loss(model, images, captions, negatives, args):
    image_feats = F.normalize(model.encode_image(images), dim=-1)
    text_feats = F.normalize(model.encode_text(captions), dim=-1)
    neg_feats = F.normalize(model.encode_text(negatives), dim=-1)
    contrastive = contrastive_margin_loss(image_feats, text_feats, args.contrastive_margin)
    hinge = hard_negative_hinge_loss(image_feats, text_feats, neg_feats, args.hinge_margin)
    return contrastive + args.hinge_weight * hinge, contrastive, hinge


def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    model, _, tokenizer = load_model(args.init, device=args.device)
    dataset = HardNegativeDataset(args.train_file, args.image_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        collate_fn=Collate(tokenizer), pin_memory=True)
    # logit_scale is not used by the losses, so it receives no gradient and stays at 100.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"{len(dataset):,} training instances, {len(loader):,} steps per epoch")

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        for step, (images, captions, negatives) in enumerate(loader):
            images = images.to(args.device, non_blocking=True)
            captions = captions.to(args.device, non_blocking=True)
            negatives = negatives.to(args.device, non_blocking=True)

            loss, contrastive, hinge = compute_loss(model, images, captions, negatives, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} loss {loss.item():.4f} "
                      f"(contrastive {contrastive.item():.4f}, hinge {hinge.item():.4f}) "
                      f"{time.time() - start:.0f}s", flush=True)

        path = os.path.join(args.output_dir, f"epoch_{epoch}.safetensors")
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
