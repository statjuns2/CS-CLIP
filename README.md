# CS-CLIP: Compositional Scene Graph-guided CLIP for Robust Compositional Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-2609.08242-b31b1b.svg)](https://arxiv.org/abs/2609.08242)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-juns%2FCS--CLIP-yellow)](https://huggingface.co/juns/CS-CLIP)
[![Data](https://img.shields.io/badge/%F0%9F%A4%97%20Data-juns%2FCS--CLIP--data-yellow)](https://huggingface.co/datasets/juns/CS-CLIP-data)

Official code, checkpoint and data for **CS-CLIP** (Findings of EMNLP 2026).

> SeongJun Jeong, Minjoon Jung, Woo Suk Choi, Youwon Jang, Byoung-Tak Zhang. Seoul National University.

CS-CLIP fine-tunes CLIP on hard negatives that target one compositional element at a time.
Each COCO caption is parsed into a scene graph; an object, attribute or relation span is masked,
a BERT masked language model proposes fluent replacements, and a RoBERTa NLI model keeps the one
that most contradicts the original caption. Seven negative types are generated
(Replace / Swap / Add × Object / Attribute / Relation).

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Results](#results)
- [Data](#data)
- [Evaluation](#evaluation)
- [Training](#training)
- [Hard-negative generation](#hard-negative-generation)
- [Citation](#citation)

## Installation

```bash
git clone https://github.com/<github-id>/CS-CLIP && cd CS-CLIP
pip install -r requirements.txt          # model, training, evaluation
pip install -r negatives/requirements.txt  # only for hard-negative generation
```

## Quick start

The checkpoint is a standard CLIP ViT-B/32 and loads with open_clip, Hugging Face Transformers, or this repository.

```python
import torch, open_clip
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:juns/CS-CLIP")
tokenizer = open_clip.get_tokenizer("hf-hub:juns/CS-CLIP")
model.eval()

image = preprocess(Image.open("example.jpg")).unsqueeze(0)
text = tokenizer(["a man riding a brown horse", "a brown man riding a horse"])
with torch.no_grad():
    image_feats = model.encode_image(image, normalize=True)
    text_feats = model.encode_text(text, normalize=True)
print(image_feats @ text_feats.T)
```

```python
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("juns/CS-CLIP")
processor = CLIPProcessor.from_pretrained("juns/CS-CLIP")
```

```python
import csclip

model, preprocess, tokenizer = csclip.load_model("juns/CS-CLIP", device="cuda")
```

## Results

Results of the released checkpoint (seed 120), reproduced with the scripts in `evaluation/`:

| SugarCrepe Rel / Obj / Att | Swap Obj / Att | Add Obj / Att | Avg | Std | VG-Rel | VG-Att | COCO I2T / T2I R@1 | GQA |
|---|---|---|---|---|---|---|---|---|
| 83.71 / 94.31 / 87.94 | 78.05 / 81.98 | 90.83 / 89.02 | 88.59 | 5.18 | 85.96 | 77.36 | 53.20 / 42.17 | 71.27 |

Table 1 of the paper reports the mean and standard deviation over three training seeds (SugarCrepe Avg 88.6 ± 0.2); this checkpoint is one of them.

## Data

Everything we created is on the Hugging Face Hub at [`juns/CS-CLIP-data`](https://huggingface.co/datasets/juns/CS-CLIP-data):

| File | Content |
|---|---|
| `train/cs_clip_hard_negatives.jsonl` | 1,941,340 caption / hard-negative pairs on 78,614 COCO train2014 images: the exact training set of the released checkpoint |
| `eval/gqa_two_choice.json` | GQA two-choice benchmark (10,000 instances from the balanced validation split) |
| `eval/controlled_{object,attribute,relation}.json` | Controlled test cases built from VG-Relation and VG-Attribute (23,937 / 28,748 / 13,595 pairs) |

```bash
hf download juns/CS-CLIP-data --repo-type dataset --local-dir data/csclip
```

Each training record looks like

```json
{"image_id": 318556, "image_path": "train2014/COCO_train2014_000000318556.jpg",
 "caption": "A very clean and well decorated empty bathroom",
 "negative_caption": "A very clean and well decorated empty bed", "type": "REPLACE-OBJ"}
```

where `type` is one of `REPLACE-OBJ`, `REPLACE-ATT`, `REPLACE-REL`, `SWAP-OBJ`, `SWAP-ATT`, `ADD-OBJ`, `ADD-ATT`.
Images are not redistributed; download them from the original sources below.

## Evaluation

### Data preparation

```bash
mkdir -p data && cd data
# SugarCrepe: the 7,512-pair version used in the paper (a later upstream commit removed one swap-obj pair)
git clone https://github.com/RAIVNLab/sugar-crepe && git -C sugar-crepe checkout 343ccb9be752dcdb003ff92b46f035dff1ebf263
# COCO images (val2014 for retrieval, val2017 for SugarCrepe) and the Karpathy test split
mkdir -p coco && cd coco
wget http://images.cocodataset.org/zips/val2014.zip http://images.cocodataset.org/zips/val2017.zip
unzip -q val2014.zip && unzip -q val2017.zip
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json
cd ..
# ARO: VG-Relation / VG-Attribute annotations and images (pip install gdown)
mkdir -p aro && cd aro
gdown 1kX2iCHEv0CADL8dSO1nMdW-V0NqIAiP3 -O visual_genome_relation.json
gdown 13tWvOrNOLHxl3Rm9cR3geAdHx2qR3-Tw -O visual_genome_attribution.json
gdown 1qaPlrwhGNMrR3a11iopZUT_GPP_LrgP9 -O vgr_vga_images.zip && unzip -q vgr_vga_images.zip   # -> images/
cd ..
# GQA images
mkdir -p gqa && cd gqa && wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip && unzip -q images.zip && cd ..
# Our benchmark files: GQA two-choice and the controlled test cases
hf download juns/CS-CLIP-data --repo-type dataset --include "eval/*" --local-dir csclip
cd ..
```

### Running

Every script takes `--model` (default `juns/CS-CLIP`; a local directory, a checkpoint file from `train.py`, or `openai` for the original CLIP ViT-B/32), `--device`, `--batch_size` and an optional `--output_json`.

```bash
python -m evaluation.sugarcrepe     --data_dir data/sugar-crepe/data --image_dir data/coco/val2017
python -m evaluation.aro            --data_dir data/aro --image_dir data/aro/images
python -m evaluation.coco_retrieval --annotation data/coco/coco_karpathy_test.json --image_dir data/coco
python -m evaluation.gqa            --data data/csclip/eval/gqa_two_choice.json --image_dir data/gqa/images
python -m evaluation.controlled     --data_dir data/csclip/eval --image_dir data/aro/images

# or all at once (paths can be overridden with environment variables)
bash scripts/eval_all.sh
MODEL=openai OUT_DIR=outputs/eval_clip bash scripts/eval_all.sh
```

Protocol notes:

- SugarCrepe Avg is the accuracy over all 7,512 pairs; Std is the standard deviation of the seven subset accuracies.
- VG-Relation and VG-Attribute use full images (`--crop` uses ARO's boxes instead) and macro accuracy: over relations after ARO's list of excluded relations, and over attribute pairs.
- The controlled test cases report accuracy and Cohen's d between positive and negative similarity scores.
- A prediction counts as correct only if the positive scores strictly higher than the negative.

## Training

Download COCO train2014 images and the training file, then train on one GPU:

```bash
mkdir -p data/coco && cd data/coco && wget http://images.cocodataset.org/zips/train2014.zip && unzip -q train2014.zip && cd ../..
hf download juns/CS-CLIP-data --repo-type dataset --include "train/*" --local-dir data/csclip
bash scripts/train.sh
MODEL=checkpoints/cs-clip/epoch_2.safetensors bash scripts/eval_all.sh
```

In our runs, one epoch took about 3.5–4 hours on a single A100 80GB (fp32, batch size 128). Re-trained weights may differ slightly from the released checkpoint.

## Hard-negative generation

The released training file is exactly what CS-CLIP was trained on, so you do not need to regenerate it.
The scripts in `negatives/` show how it was made. They read only `annotations/captions_train2014.json`, not images.

```bash
pip install -r negatives/requirements.txt
mkdir -p data/coco && cd data/coco && wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip && unzip -q annotations_trainval2014.zip && cd ../..

# 1. Relation vocabulary (optional; shipped as negatives/relation_vocab_coco_train2014.json)
python negatives/build_relation_vocab.py --captions_json data/coco/annotations/captions_train2014.json \
    --output negatives/relation_vocab_coco_train2014.json --workers 8

# 2. Generate 9 shards (one GPU each) and merge them
COCO_ROOT=data/coco bash scripts/generate_negatives.sh

#    or a single shard
CUDA_VISIBLE_DEVICES=0 python negatives/generate.py --coco_root data/coco \
    --relation_vocab negatives/relation_vocab_coco_train2014.json \
    --output_dir outputs/negatives --num_shards 9 --shard_idx 0 --workers 4

# 3. Merge shards in order, drop negatives equal to the caption, keep the first negative per (caption, type)
python negatives/merge_dedup.py --inputs outputs/negatives/negatives_contradiction_shard0{0..8}of09.jsonl \
    --output outputs/negatives/csclip_negatives_train2014.jsonl
```

- **Models.** `bert-base-uncased` proposes candidates and `roberta-large-mnli` ranks them; both are downloaded from the Hugging Face Hub. Scene graphs come from SceneGraphParser with spaCy `en_core_web_sm` 3.8.0 (other model versions change the parses noticeably).
- **Cost.** One 400-caption batch takes about 3–4 minutes on an A100. That is about 6–8 hours per shard and 50–75 GPU-hours in total, with about 17 GB of GPU memory per shard.
- **Output.** The 9 shard files hold about 5.1M negatives (one per target), and about 2.04M remain after merging.
- **Released file.** The released file also excludes 4,167 of the 82,783 train2014 images that were missing from an auxiliary file during development, so it is a subset of what the scripts produce. Pass the image ids of the released file with `--keep_image_ids` to `merge_dedup.py` to apply the same restriction. On the original generator output this reproduces the released file record for record.
- **Ablations (Table 4).** SG+MLM: `PREFER=mlm EXTRA_ARGS=--use_wordnet`. SG+Random: `PREFER=random EXTRA_ARGS=--random_from_full_vocab`.

## Citation

```bibtex
@inproceedings{jeong2026csclip,
  title     = {{CS-CLIP}: Compositional Scene Graph-guided {CLIP} for Robust Compositional Reasoning},
  author    = {Jeong, SeongJun and Jung, Minjoon and Choi, Woo Suk and Jang, Youwon and Zhang, Byoung-Tak},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026}
}
```

## Acknowledgments

This code builds on [OpenAI CLIP](https://github.com/openai/CLIP), [OpenCLIP](https://github.com/mlfoundations/open_clip), [Structure-CLIP](https://github.com/zjukg/Structure-CLIP) (training losses), [SceneGraphParser](https://github.com/vacancy/SceneGraphParser), and the [ARO](https://github.com/mertyg/vision-language-models-are-bows) and [SugarCrepe](https://github.com/RAIVNLab/sugar-crepe) benchmarks.

## License

The code is released under the [MIT License](LICENSE). The model weights are fine-tuned from OpenAI CLIP (MIT). The data files are derived from COCO captions, Visual Genome and GQA annotations and are released under CC BY 4.0; images remain subject to their original terms.
