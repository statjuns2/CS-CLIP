#!/usr/bin/env python3
"""Build the training-set variants used in the paper's ablations and analyses.

All variants are derived from the released training file
(train/cs_clip_hard_negatives.jsonl) and are trained with the unchanged train.py.

  Table 4, leave-one-out      --without object | attribute | relation
  Table 5, CS-CLIP-noSG       --random_position
  Figure 5, sample efficiency --num_samples N

--without drops every negative type that targets the element:
    object     REPLACE-OBJ, SWAP-OBJ, ADD-OBJ
    attribute  REPLACE-ATT, SWAP-ATT, ADD-ATT
    relation   REPLACE-REL

--random_position keeps the word that the pipeline introduced but puts it at a
random position: the word of the negative that differs from the caption replaces
a randomly chosen word of the caption. Records without such a word (pure swaps)
keep their negative. The procedure and seed are the ones used for the paper; the
original run also drew random numbers for an auxiliary field that is not part of
the released file, so the output is not identical to the file behind Table 5.

--num_samples draws N records without replacement with random.Random(seed).
With the default seed (42) this gives exactly the subsets used for Figure 5.

Options are applied in the order: --without, --random_position, --num_samples.

Examples:
    python negatives/make_variant.py --input data/csclip/train/cs_clip_hard_negatives.jsonl \
        --output data/csclip/train/without_relation.jsonl --without relation
    python negatives/make_variant.py --input data/csclip/train/cs_clip_hard_negatives.jsonl \
        --output data/csclip/train/no_sg.jsonl --random_position
    python negatives/make_variant.py --input data/csclip/train/cs_clip_hard_negatives.jsonl \
        --output data/csclip/train/subset_10k.jsonl --num_samples 10000
"""

import argparse
import json
import os
import random
from collections import Counter
from typing import List, Optional

ELEMENT_TYPES = {
    "object": {"REPLACE-OBJ", "SWAP-OBJ", "ADD-OBJ"},
    "attribute": {"REPLACE-ATT", "SWAP-ATT", "ADD-ATT"},
    "relation": {"REPLACE-REL"},
}


def read_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def differing_word(caption: str, negative: str, rng: random.Random) -> Optional[str]:
    """A word of ``negative`` that is not in ``caption`` (None if there is none).

    Same number of words: one of the positions where the two differ.
    Otherwise: one of the words of the negative that do not occur in the caption.
    """
    cap_words, neg_words = caption.split(), negative.split()
    if not neg_words:
        return None
    if len(cap_words) == len(neg_words):
        diff = [n for c, n in zip(cap_words, neg_words) if c != n]
    else:
        in_caption = set(cap_words)
        diff = [w for w in neg_words if w not in in_caption]
    return rng.choice(diff) if diff else None


def random_position_negative(caption: str, negative: str, rng: random.Random) -> str:
    """Replace a random word of the caption with the word the negative introduced."""
    caption = caption.strip()
    word = differing_word(caption, negative, rng)
    if word is None:
        return negative
    words = caption.split()
    if not words:
        return negative
    words[rng.randint(0, len(words) - 1)] = word
    return " ".join(words)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="cs_clip_hard_negatives.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--without", choices=sorted(ELEMENT_TYPES), default=None,
                    help="drop the negatives that target this element (Table 4, leave-one-out)")
    ap.add_argument("--random_position", action="store_true",
                    help="place the introduced word at a random position (Table 5, CS-CLIP-noSG)")
    ap.add_argument("--num_samples", type=int, default=None,
                    help="sample this many records without replacement (Figure 5)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.without is None and not args.random_position and args.num_samples is None:
        ap.error("give at least one of --without, --random_position, --num_samples")

    records = read_jsonl(args.input)
    print(f"read {len(records):,} records from {args.input}")

    if args.without is not None:
        drop = ELEMENT_TYPES[args.without]
        records = [r for r in records if r["type"] not in drop]
        print(f"without {args.without} negatives ({', '.join(sorted(drop))}): {len(records):,} records")

    if args.random_position:
        rng = random.Random(args.seed)
        unchanged = 0
        for r in records:
            new = random_position_negative(r["caption"], r["negative_caption"], rng)
            unchanged += new == r["negative_caption"]
            r["negative_caption"] = new
        print(f"random position: {unchanged:,} negatives left unchanged")

    if args.num_samples is not None and len(records) >= args.num_samples:
        records = random.Random(args.seed).sample(records, args.num_samples)
        print(f"sampled {len(records):,} records (seed {args.seed})")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fout:
        for r in records:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records):,} records to {args.output}")
    for op, n in sorted(Counter(r["type"] for r in records).items()):
        print(f"  {op}: {n:,}")


if __name__ == "__main__":
    main()
