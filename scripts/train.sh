#!/usr/bin/env bash
# Train CS-CLIP (CLIP ViT-B/32, 2 epochs, one GPU) with the settings of the released checkpoint.
# Run from the repository root:
#   TRAIN_FILE=data/csclip/train/cs_clip_hard_negatives.jsonl COCO_DIR=data/coco bash scripts/train.sh
set -euo pipefail

TRAIN_FILE=${TRAIN_FILE:-data/csclip/train/cs_clip_hard_negatives.jsonl}
COCO_DIR=${COCO_DIR:-data/coco}            # must contain train2014/
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/cs-clip}
SEED=${SEED:-120}

python train.py \
    --train_file "$TRAIN_FILE" \
    --image_root "$COCO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs 2 \
    --batch_size 128 \
    --lr 2e-6 \
    --weight_decay 0.1 \
    --contrastive_margin 0.2 \
    --hinge_margin 0.1 \
    --hinge_weight 5 \
    --seed "$SEED"

# The final checkpoint is $OUTPUT_DIR/epoch_2.safetensors; evaluate it with
#   MODEL=$OUTPUT_DIR/epoch_2.safetensors bash scripts/eval_all.sh
