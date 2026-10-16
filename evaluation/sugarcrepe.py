"""SugarCrepe (image-to-text, 7 subsets).

Data: the 7 json files from https://github.com/RAIVNLab/sugar-crepe (data/*.json,
format {id: {"filename", "caption", "negative_caption"}}) and the COCO val2017 images.
The paper uses the 7,512-pair release (commit 343ccb9); later commits removed one
swap_obj pair (7,511 pairs), which changes swap-obj / Avg slightly.

A pair is correct when sim(image, caption) > sim(image, negative_caption).
Avg is the accuracy over all pairs; Std is the standard deviation (ddof=0) of the
7 subset accuracies.

    python -m evaluation.sugarcrepe --data_dir data/sugar-crepe/data --image_dir data/coco/val2017
"""

import json
import os

import numpy as np

from evaluation.utils import get_parser, load_model, save_json, score_pairs

SUBSETS = ["replace_rel", "replace_obj", "replace_att", "swap_obj", "swap_att", "add_obj", "add_att"]
PAPER_NUM_PAIRS = 7512


def load_samples(data_dir, image_dir):
    samples = []
    for subset in SUBSETS:
        with open(os.path.join(data_dir, f"{subset}.json")) as f:
            data = json.load(f)
        for pair_id, item in data.items():
            samples.append({
                "subset": subset,
                "id": pair_id,
                "image": os.path.join(image_dir, item["filename"]),
                "positive": item["caption"],
                "negative": item["negative_caption"],
            })
    return samples


def evaluate(model, preprocess, tokenizer, args):
    """Returns (metrics, samples, correct)."""
    samples = load_samples(args.data_dir, args.image_dir)
    if len(samples) != PAPER_NUM_PAIRS:
        print(f"Note: found {len(samples)} pairs; the paper uses the {PAPER_NUM_PAIRS}-pair release.")
    pos, neg = score_pairs(model, preprocess, tokenizer, samples, args, desc="SugarCrepe")
    correct = pos > neg
    subsets = np.array([s["subset"] for s in samples])

    metrics = {s: float(correct[subsets == s].mean()) for s in SUBSETS}
    metrics["avg"] = float(correct.mean())
    metrics["std"] = float(np.std([metrics[s] for s in SUBSETS]))
    metrics["num_pairs"] = len(samples)
    return metrics, samples, correct


def main():
    parser = get_parser(__doc__.splitlines()[0])
    parser.add_argument("--data_dir", required=True, help="Directory with the 7 SugarCrepe json files")
    parser.add_argument("--image_dir", required=True, help="COCO val2017 image directory")
    args = parser.parse_args()

    model, preprocess, tokenizer = load_model(args)
    metrics, _, _ = evaluate(model, preprocess, tokenizer, args)

    print(f"\nSugarCrepe ({metrics['num_pairs']} pairs), accuracy %")
    for name in SUBSETS + ["avg", "std"]:
        print(f"  {name:<12} {100 * metrics[name]:6.2f}")
    save_json(metrics, args.output_json)


if __name__ == "__main__":
    main()
