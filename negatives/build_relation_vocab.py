#!/usr/bin/env python3
"""Step 0: build the relation vocabulary used by REPLACE-REL.

REPLACE-REL only keeps BERT candidates that already occur as a single-word
relation in some COCO train2014 caption scene graph. This script parses every
unique caption of ``captions_train2014.json`` with SceneGraphParser and writes
that set of relation words (lower-cased) to a small JSON file:

    {"num_captions": 400962, "relations": ["above", "across", ...]}

Only set membership is used downstream, so the result does not depend on the
caption order or on the number of worker processes. The result for COCO
train2014 (3,123 words) is shipped as negatives/relation_vocab_coco_train2014.json,
so running this script is optional.

Usage
-----
    # from the COCO captions (400,962 unique captions; about 1 CPU-hour, 8 min with 8 workers)
    python negatives/build_relation_vocab.py \
        --captions_json $COCO_ROOT/annotations/captions_train2014.json \
        --output negatives/relation_vocab_coco_train2014.json --workers 8

    # from an existing {caption: sng_parser.parse(caption)} dump
    python negatives/build_relation_vocab.py \
        --from_graph_json coco_train2014_caption_sng_graph.json \
        --output negatives/relation_vocab_coco_train2014.json
"""

import argparse
import json
import os
from multiprocessing import Pool
from typing import Iterable, List, Set

from tqdm import tqdm


def relation_words(graph: dict) -> List[str]:
    """Single-word relations of one scene graph, lower-cased."""
    words = []
    for rel in graph.get("relations", []):
        word = rel.get("relation")
        if word and len(word.split()) == 1:
            words.append(word.lower())
    return words


def vocab_from_graphs(graphs: Iterable[dict]) -> Set[str]:
    vocab = set()
    for g in graphs:
        vocab.update(relation_words(g))
    return vocab


def _parse_relations(caption: str) -> List[str]:
    import sng_parser  # imported lazily so that each worker loads spaCy once
    return relation_words(sng_parser.parse(caption))


def load_unique_captions(captions_json: str) -> List[str]:
    with open(captions_json, "r") as f:
        anns = json.load(f)["annotations"]
    return list(dict.fromkeys(a["caption"] for a in anns))


def vocab_from_captions(captions: List[str], workers: int) -> Set[str]:
    vocab = set()
    if workers > 1:
        with Pool(workers) as pool:
            for words in tqdm(pool.imap(_parse_relations, captions, chunksize=256),
                              total=len(captions), desc="Parsing captions"):
                vocab.update(words)
    else:
        for cap in tqdm(captions, desc="Parsing captions"):
            vocab.update(_parse_relations(cap))
    return vocab


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--captions_json", help="COCO captions_train2014.json")
    src.add_argument("--from_graph_json", help="existing JSON dict {caption: sng_parser graph}")
    ap.add_argument("--output", required=True, help="output vocabulary JSON")
    ap.add_argument("--workers", type=int, default=1, help="parser processes (captions_json only)")
    args = ap.parse_args()

    if args.captions_json:
        captions = load_unique_captions(args.captions_json)
        vocab = vocab_from_captions(captions, args.workers)
        num_captions = len(captions)
    else:
        with open(args.from_graph_json, "r") as f:
            graphs = json.load(f)
        vocab = vocab_from_graphs(graphs.values())
        num_captions = len(graphs)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"num_captions": num_captions, "relations": sorted(vocab)}, f, indent=0, ensure_ascii=False)
        f.write("\n")
    print(f"{len(vocab)} relation words from {num_captions} captions -> {args.output}")


if __name__ == "__main__":
    main()
