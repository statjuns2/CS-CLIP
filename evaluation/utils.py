"""Helpers shared by the evaluation scripts."""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import csclip


def get_parser(description):
    """Argument parser with the options every evaluation script uses."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", default=csclip.DEFAULT_MODEL_ID,
                        help="Hugging Face repo id, local model directory, checkpoint file, "
                             "or 'openai' for the original CLIP ViT-B/32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_json", default=None, help="Optional path to save the metrics as JSON")
    return parser


def load_model(args):
    model, preprocess, tokenizer = csclip.load_model(args.model, args.device)
    return model, preprocess, tokenizer


def load_image(path, box=None):
    """Open an image as RGB. `box` = (x, y, w, h) crops the image before preprocessing."""
    image = Image.open(path).convert("RGB")
    if box is not None:
        x, y, w, h = box
        image = image.crop((x, y, x + w, y + h))
    return image


class PairDataset(Dataset):
    """One image with a positive and a negative caption per sample.

    Each sample is a dict with keys "image" (full path), "positive", "negative"
    and optionally "box" = (x, y, w, h).
    """

    def __init__(self, samples, preprocess):
        self.samples = samples
        self.preprocess = preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        image = self.preprocess(load_image(s["image"], s.get("box")))
        return image, s["positive"], s["negative"]


@torch.no_grad()
def score_pairs(model, preprocess, tokenizer, samples, args, desc="scoring"):
    """Cosine similarity of each image with its positive and its negative caption.

    Returns two float32 arrays (pos_sims, neg_sims). A sample counts as correct
    when pos_sim > neg_sim (a tie counts as wrong).
    """
    loader = DataLoader(PairDataset(samples, preprocess), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)
    pos_sims, neg_sims = [], []
    for images, positives, negatives in tqdm(loader, desc=desc):
        image_feats = csclip.encode_images(model, images.to(args.device))
        pos_feats = csclip.encode_texts(model, tokenizer(list(positives)).to(args.device))
        neg_feats = csclip.encode_texts(model, tokenizer(list(negatives)).to(args.device))
        pos_sims.append((image_feats * pos_feats).sum(dim=-1).float().cpu())
        neg_sims.append((image_feats * neg_feats).sum(dim=-1).float().cpu())
    return torch.cat(pos_sims).numpy(), torch.cat(neg_sims).numpy()


def macro_accuracy(correct, groups, exclude=()):
    """Mean over groups of the per-group accuracy, skipping groups listed in `exclude`."""
    correct = np.asarray(correct, dtype=float)
    groups = np.asarray(groups)
    accs = [correct[groups == g].mean() for g in np.unique(groups) if g not in exclude]
    return float(np.mean(accs))


def save_json(metrics, path):
    if path is None:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {path}")
