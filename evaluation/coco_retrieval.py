"""Zero-shot image-text retrieval on the COCO 5K Karpathy test split.

Data: coco_karpathy_test.json (5,000 images, 25,010 captions; the BLIP/ARO copy at
https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json)
and the COCO 2014 images. Each entry is {"image": "val2014/COCO_val2014_....jpg", "caption": [...]}.

  * Image-to-text (I2T) R@K: fraction of the 5,000 images for which at least one of
    their captions is among the top-K of all 25,010 captions.
  * Text-to-image (T2I) R@K: fraction of the 25,010 captions whose image is among
    the top-K of the 5,000 images.

    python -m evaluation.coco_retrieval --annotation data/coco/coco_karpathy_test.json --image_dir data/coco
"""

import json
import os

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import csclip
from evaluation.utils import get_parser, load_image, load_model, save_json

KS = (1, 5, 10)


class ImageDataset(Dataset):
    def __init__(self, paths, preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.preprocess(load_image(self.paths[index]))


def load_annotations(annotation_file, image_dir):
    with open(annotation_file) as f:
        data = json.load(f)
    image_paths = [os.path.join(image_dir, item["image"]) for item in data]
    captions, caption_to_image = [], []
    for image_index, item in enumerate(data):
        for caption in item["caption"]:
            captions.append(caption)
            caption_to_image.append(image_index)
    return image_paths, captions, torch.tensor(caption_to_image)


@torch.no_grad()
def encode_all(model, preprocess, tokenizer, image_paths, captions, args):
    loader = DataLoader(ImageDataset(image_paths, preprocess), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)
    image_feats = [csclip.encode_images(model, images.to(args.device)).float().cpu()
                   for images in tqdm(loader, desc="images")]
    text_feats = []
    for start in tqdm(range(0, len(captions), args.batch_size), desc="captions"):
        tokens = tokenizer(captions[start:start + args.batch_size]).to(args.device)
        text_feats.append(csclip.encode_texts(model, tokens).float().cpu())
    return torch.cat(image_feats), torch.cat(text_feats)


def recall_at_k(image_feats, text_feats, caption_to_image):
    sims = image_feats.cpu() @ text_feats.cpu().T  # (num_images, num_captions)
    num_images = sims.shape[0]
    max_k = max(KS)

    # The test split contains identical captions for different images, which get exactly
    # tied scores. Their order decides a few image-to-text hits, so we rank with
    # torch.sort on CPU exactly as in the paper's code (topk, a stable sort or a GPU sort
    # resolve ties differently and can move I2T R@1/R@5 by a few hundredths of a point).
    top_captions = sims.sort(dim=1, descending=True).indices[:, :max_k]
    top_images = sims.T.sort(dim=1, descending=True).indices[:, :max_k]

    # Image-to-text: does any caption of the image appear in the top K?
    i2t_hits = caption_to_image[top_captions] == torch.arange(num_images)[:, None]
    # Text-to-image: does the caption's image appear in the top K?
    t2i_hits = top_images == caption_to_image[:, None]

    metrics = {}
    for k in KS:
        metrics[f"i2t_r{k}"] = i2t_hits[:, :k].any(dim=1).double().mean().item()
    for k in KS:
        metrics[f"t2i_r{k}"] = t2i_hits[:, :k].any(dim=1).double().mean().item()
    metrics["num_images"] = num_images
    metrics["num_captions"] = len(caption_to_image)
    return metrics


def evaluate(model, preprocess, tokenizer, args):
    image_paths, captions, caption_to_image = load_annotations(args.annotation, args.image_dir)
    image_feats, text_feats = encode_all(model, preprocess, tokenizer, image_paths, captions, args)
    return recall_at_k(image_feats, text_feats, caption_to_image)


def main():
    parser = get_parser("COCO 5K (Karpathy test) retrieval")
    parser.add_argument("--annotation", required=True, help="Path to coco_karpathy_test.json")
    parser.add_argument("--image_dir", required=True,
                        help="COCO root directory that contains val2014/ (paths in the annotation are relative to it)")
    args = parser.parse_args()

    model, preprocess, tokenizer = load_model(args)
    metrics = evaluate(model, preprocess, tokenizer, args)

    print(f"\nCOCO 5K retrieval ({metrics['num_images']} images, {metrics['num_captions']} captions), recall %")
    print("  I2T  " + "  ".join(f"R@{k} {100 * metrics[f'i2t_r{k}']:6.2f}" for k in KS))
    print("  T2I  " + "  ".join(f"R@{k} {100 * metrics[f't2i_r{k}']:6.2f}" for k in KS))
    save_json(metrics, args.output_json)


if __name__ == "__main__":
    main()
