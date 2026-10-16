"""Controlled test cases (paper Section 3 / Appendix C).

Each sample is a Visual Genome image (from the ARO image folder) with a true caption
and a negative that changes exactly one object, attribute or relation. For each set
we report
  * accuracy: fraction of samples with sim(image, positive) > sim(image, negative),
  * Cohen's d = (mean(pos) - mean(neg)) / s_pooled, where pos / neg are the
    image-text cosine similarities and s_pooled = sqrt((sd_pos^2 + sd_neg^2) / 2)
    (sample standard deviations).
Images are used in full (no crop).

Data: controlled_{object,attribute,relation}.json from the CS-CLIP Hugging Face dataset.

    python -m evaluation.controlled --data_dir data/csclip/eval --image_dir data/aro/images
"""

import json
import os

import numpy as np

from evaluation.utils import get_parser, load_model, save_json, score_pairs

SETS = ["object", "attribute", "relation"]


def load_samples(data_file, image_dir):
    with open(data_file) as f:
        data = json.load(f)
    return [{"image": os.path.join(image_dir, item["image"]),
             "positive": item["positive"],
             "negative": item["negative"]} for item in data]


def cohens_d(pos, neg):
    pos, neg = np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)
    s_pooled = np.sqrt((pos.std(ddof=1) ** 2 + neg.std(ddof=1) ** 2) / 2)
    return float((pos.mean() - neg.mean()) / s_pooled)


def evaluate(model, preprocess, tokenizer, args, name):
    """Returns (metrics, pos_sims, neg_sims) for one controlled set."""
    samples = load_samples(os.path.join(args.data_dir, f"controlled_{name}.json"), args.image_dir)
    pos, neg = score_pairs(model, preprocess, tokenizer, samples, args, desc=f"controlled-{name}")
    metrics = {"accuracy": float((pos > neg).mean()), "cohens_d": cohens_d(pos, neg), "num_samples": len(samples)}
    return metrics, pos, neg


def main():
    parser = get_parser("Controlled test cases (object / attribute / relation replacement)")
    parser.add_argument("--data_dir", required=True, help="Directory with controlled_{object,attribute,relation}.json")
    parser.add_argument("--image_dir", required=True, help="ARO Visual Genome image directory")
    parser.add_argument("--sets", nargs="+", default=SETS, choices=SETS)
    args = parser.parse_args()

    model, preprocess, tokenizer = load_model(args)
    results = {}
    for name in args.sets:
        results[name], _, _ = evaluate(model, preprocess, tokenizer, args, name)

    print("\nControlled test cases")
    for name, m in results.items():
        print(f"  {name:<10} accuracy {100 * m['accuracy']:6.2f}%   Cohen's d {m['cohens_d']:.3f}   n={m['num_samples']}")
    save_json(results, args.output_json)


if __name__ == "__main__":
    main()
