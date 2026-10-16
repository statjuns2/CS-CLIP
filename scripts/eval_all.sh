#!/usr/bin/env bash
# Run every CS-CLIP evaluation. Run from the repository root:
#   bash scripts/eval_all.sh
# Override any path below with an environment variable, e.g.
#   MODEL=openai OUT_DIR=outputs/eval_clip bash scripts/eval_all.sh
set -euo pipefail

MODEL=${MODEL:-juns/CS-CLIP}                     # HF repo id, local dir, checkpoint file, or "openai"
DEVICE=${DEVICE:-cuda}
BATCH_SIZE=${BATCH_SIZE:-128}
OUT_DIR=${OUT_DIR:-outputs/eval}

DATA=${DATA:-data}
CSCLIP_EVAL_DIR=${CSCLIP_EVAL_DIR:-$DATA/csclip/eval}           # gqa_two_choice.json, controlled_*.json (HF dataset)
SUGARCREPE_DIR=${SUGARCREPE_DIR:-$DATA/sugar-crepe/data}        # replace_rel.json, ..., add_att.json
COCO_DIR=${COCO_DIR:-$DATA/coco}                                # val2014/, val2017/
COCO_ANNOTATION=${COCO_ANNOTATION:-$DATA/coco/coco_karpathy_test.json}
ARO_DIR=${ARO_DIR:-$DATA/aro}                                   # visual_genome_{relation,attribution}.json
ARO_IMAGE_DIR=${ARO_IMAGE_DIR:-$DATA/aro/images}
GQA_IMAGE_DIR=${GQA_IMAGE_DIR:-$DATA/gqa/images}

COMMON=(--model "$MODEL" --device "$DEVICE" --batch_size "$BATCH_SIZE")
mkdir -p "$OUT_DIR"

python -m evaluation.sugarcrepe "${COMMON[@]}" --data_dir "$SUGARCREPE_DIR" \
    --image_dir "$COCO_DIR/val2017" --output_json "$OUT_DIR/sugarcrepe.json"

python -m evaluation.aro "${COMMON[@]}" --data_dir "$ARO_DIR" \
    --image_dir "$ARO_IMAGE_DIR" --output_json "$OUT_DIR/aro.json"

python -m evaluation.coco_retrieval "${COMMON[@]}" --annotation "$COCO_ANNOTATION" \
    --image_dir "$COCO_DIR" --output_json "$OUT_DIR/coco_retrieval.json"

python -m evaluation.gqa "${COMMON[@]}" --data "$CSCLIP_EVAL_DIR/gqa_two_choice.json" \
    --image_dir "$GQA_IMAGE_DIR" --output_json "$OUT_DIR/gqa.json"

python -m evaluation.controlled "${COMMON[@]}" --data_dir "$CSCLIP_EVAL_DIR" \
    --image_dir "$ARO_IMAGE_DIR" --output_json "$OUT_DIR/controlled.json"
