#!/usr/bin/env bash
# Generate the CS-CLIP hard negatives for COCO train2014 and merge them.
#
#   COCO_ROOT=/path/to/coco bash scripts/generate_negatives.sh
#
# COCO_ROOT must contain annotations/captions_train2014.json (images are not read).
# One shard per GPU, all shards in parallel; then shards are merged and deduplicated.
# Environment variables (defaults in brackets):
#   OUT [outputs/negatives]      output directory
#   PREFER [contradiction]       contradiction (CS-CLIP) | mlm (SG+MLM) | random (SG+Random)
#   GPUS [0,1,2,3,4,5,6,7,8]     GPU ids, one per shard (reused round-robin if fewer than NUM_SHARDS)
#   NUM_SHARDS [9]               keep 9 to reproduce the paper data
#   WORKERS [4]                  CPU processes per shard for scene-graph parsing
#   KEEP_IMAGE_IDS []            optional image-id list passed to merge_dedup.py
#   EXTRA_ARGS []                extra generate.py flags, e.g. "--use_wordnet" for SG+MLM
#                                or "--random_from_full_vocab" for SG+Random (Table 4)
set -euo pipefail
cd "$(dirname "$0")/.."

COCO_ROOT=${COCO_ROOT:?set COCO_ROOT to the COCO root directory}
OUT=${OUT:-outputs/negatives}
PREFER=${PREFER:-contradiction}
GPUS=${GPUS:-0,1,2,3,4,5,6,7,8}
NUM_SHARDS=${NUM_SHARDS:-9}
WORKERS=${WORKERS:-4}
KEEP_IMAGE_IDS=${KEEP_IMAGE_IDS:-}
EXTRA_ARGS=${EXTRA_ARGS:-}
VOCAB=negatives/relation_vocab_coco_train2014.json
export TOKENIZERS_PARALLELISM=false

IFS=, read -ra GPU_LIST <<< "$GPUS"
mkdir -p "$OUT/logs"

pids=()
shards=()
for ((i = 0; i < NUM_SHARDS; i++)); do
  gpu=${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}
  CUDA_VISIBLE_DEVICES=$gpu python negatives/generate.py \
      --coco_root "$COCO_ROOT" --relation_vocab "$VOCAB" --output_dir "$OUT" \
      --num_shards "$NUM_SHARDS" --shard_idx "$i" --nli_prefer "$PREFER" --workers "$WORKERS" $EXTRA_ARGS \
      > "$OUT/logs/${PREFER}_shard${i}.log" 2>&1 &
  pids+=($!)
  shards+=("$OUT/$(printf 'negatives_%s_shard%02dof%02d.jsonl' "$PREFER" "$i" "$NUM_SHARDS")")
done
for p in "${pids[@]}"; do
  wait "$p"
done

merge_args=(--inputs "${shards[@]}" --output "$OUT/csclip_negatives_${PREFER}_train2014.jsonl")
if [[ -n "$KEEP_IMAGE_IDS" ]]; then
  merge_args+=(--keep_image_ids "$KEEP_IMAGE_IDS")
fi
python negatives/merge_dedup.py "${merge_args[@]}"
