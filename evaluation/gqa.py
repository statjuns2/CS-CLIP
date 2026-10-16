"""GQA two-choice benchmark (built for the CS-CLIP paper).

Each of the 10,000 samples pairs a GQA image with two texts that share the question:
"<question> <correct full answer>" (positive) and "<question> <wrong full answer>"
(negative). A sample is correct when sim(image, positive) > sim(image, negative).

Data: gqa_two_choice.json from the CS-CLIP Hugging Face dataset and the GQA images
(https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip).

    python -m evaluation.gqa --data data/csclip/eval/gqa_two_choice.json --image_dir data/gqa/images
"""

import json
import os

from evaluation.utils import get_parser, load_model, save_json, score_pairs


def load_samples(data_file, image_dir):
    with open(data_file) as f:
        data = json.load(f)
    return [{"image": os.path.join(image_dir, item["image"]),
             "positive": item["positive"],
             "negative": item["negative"]} for item in data]


def evaluate(model, preprocess, tokenizer, args):
    """Returns (metrics, samples, correct)."""
    samples = load_samples(args.data, args.image_dir)
    pos, neg = score_pairs(model, preprocess, tokenizer, samples, args, desc="GQA")
    correct = pos > neg
    return {"accuracy": float(correct.mean()), "num_samples": len(samples)}, samples, correct


def main():
    parser = get_parser("GQA two-choice")
    parser.add_argument("--data", required=True, help="Path to gqa_two_choice.json")
    parser.add_argument("--image_dir", required=True, help="GQA image directory")
    args = parser.parse_args()

    model, preprocess, tokenizer = load_model(args)
    metrics, _, _ = evaluate(model, preprocess, tokenizer, args)
    print(f"\nGQA two-choice (n={metrics['num_samples']}): accuracy {100 * metrics['accuracy']:.2f}%")
    save_json(metrics, args.output_json)


if __name__ == "__main__":
    main()
