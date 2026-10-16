import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, InterpolationMode, Normalize, RandomResizedCrop, ToTensor

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def train_transform(size=224):
    return Compose([
        RandomResizedCrop(size, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize(CLIP_MEAN, CLIP_STD),
    ])


class HardNegativeDataset(Dataset):
    """Image-caption pairs with one hard negative caption each.

    Reads a JSON Lines file with records
      {"image_id": int, "image_path": "train2014/COCO_train2014_....jpg",
       "caption": str, "negative_caption": str, "type": "REPLACE-OBJ"}
    where image_path is relative to image_root (the COCO root directory).
    """

    def __init__(self, path, image_root, transform=None):
        with open(path) as f:
            self.items = [json.loads(line) for line in f if line.strip()]
        self.image_root = image_root
        self.transform = transform or train_transform()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(os.path.join(self.image_root, item["image_path"])).convert("RGB")
        return self.transform(image), item["caption"], item["negative_caption"]


class Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        images, captions, negatives = zip(*batch)
        return torch.stack(images), self.tokenizer(list(captions)), self.tokenizer(list(negatives))
