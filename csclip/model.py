import os

import open_clip
import torch

ARCH = "ViT-B-32-quickgelu"
DEFAULT_MODEL_ID = "juns/CS-CLIP"


def load_model(model_id=DEFAULT_MODEL_ID, device="cpu"):
    """Load CS-CLIP (or vanilla CLIP) as an open_clip model.

    model_id can be
      - a Hugging Face Hub repo id (default: the released CS-CLIP checkpoint),
      - a local directory containing open_clip_config.json + open_clip_model.safetensors,
      - a checkpoint file (.safetensors / .pt state_dict) saved by train.py,
      - "openai" for the original OpenAI CLIP ViT-B/32.

    Returns (model, preprocess, tokenizer). The model is in fp32 and eval mode.
    """
    if model_id == "openai":
        model, _, preprocess = open_clip.create_model_and_transforms(ARCH, pretrained="openai", precision="fp32")
    elif os.path.isdir(model_id):
        model, _, preprocess = open_clip.create_model_and_transforms(f"local-dir:{model_id}")
    elif os.path.isfile(model_id):
        model, _, preprocess = open_clip.create_model_and_transforms(ARCH, pretrained=model_id, precision="fp32")
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(f"hf-hub:{model_id}")
    tokenizer = open_clip.get_tokenizer(ARCH)
    return model.float().to(device).eval(), preprocess, tokenizer


@torch.no_grad()
def encode_images(model, images):
    feats = model.encode_image(images)
    return feats / feats.norm(dim=-1, keepdim=True)


@torch.no_grad()
def encode_texts(model, tokens):
    feats = model.encode_text(tokens)
    return feats / feats.norm(dim=-1, keepdim=True)
