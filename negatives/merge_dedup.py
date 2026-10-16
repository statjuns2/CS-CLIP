#!/usr/bin/env python3
"""Merge the shard outputs of generate.py into the final hard-negative file.

Steps, in this order:
  1. read the shard files in the order given (shard 0, 1, ..., 8);
  2. drop records whose negative equals the caption (e.g. a SWAP-OBJ that did
     not change anything);
  3. optionally keep only images listed in --keep_image_ids;
  4. keep the first record per (caption text, type). The key is the caption
     text, not the image: a caption string shared by several images keeps only
     its first occurrence.

Usage:
    python negatives/merge_dedup.py \
        --inputs outputs/negatives/negatives_contradiction_shard*of09.jsonl \
        --output outputs/negatives/csclip_negatives_train2014.jsonl
"""

import argparse
import json
from collections import Counter
from typing import Iterable, Iterator, Optional, Set


def read_jsonl(paths: Iterable[str]) -> Iterator[dict]:
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def load_image_ids(path: str) -> Set[int]:
    """One integer image id per line, or a JSON list of ids."""
    with open(path, "r") as f:
        text = f.read().strip()
    if text.startswith("["):
        return {int(x) for x in json.loads(text)}
    return {int(x) for x in text.split()}


def merge_dedup(records: Iterable[dict], keep_image_ids: Optional[Set[int]] = None) -> Iterator[dict]:
    seen = set()
    for r in records:
        if r["negative_caption"] == r["caption"]:
            continue
        if keep_image_ids is not None and int(r["image_id"]) not in keep_image_ids:
            continue
        key = (r["caption"], r["type"])
        if key in seen:
            continue
        seen.add(key)
        yield r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, help="shard JSONL files, in shard order")
    ap.add_argument("--output", required=True)
    ap.add_argument("--keep_image_ids", default=None, help="optional file of image ids to keep")
    args = ap.parse_args()

    keep = load_image_ids(args.keep_image_ids) if args.keep_image_ids else None
    counts = Counter()
    with open(args.output, "w", encoding="utf-8") as fout:
        for r in merge_dedup(read_jsonl(args.inputs), keep):
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            counts[r["type"]] += 1
    print(f"wrote {sum(counts.values())} records to {args.output}")
    for op, n in sorted(counts.items()):
        print(f"  {op}: {n}")


if __name__ == "__main__":
    main()
