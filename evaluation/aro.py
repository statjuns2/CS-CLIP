"""VG-Relation and VG-Attribution from ARO (Yuksekgonul et al., 2023).

Data: visual_genome_relation.json, visual_genome_attribution.json and the image
folder from https://github.com/mertyg/vision-language-models-are-bows.

Protocol used in the paper:
  * the model sees the full image (no bounding-box crop; use --crop for ARO's crop),
  * a sample is correct when sim(image, true_caption) > sim(image, false_caption),
  * VG-Relation: macro accuracy over relations, excluding ARO's list of
    symmetric / ambiguous relations (DROP_RELATIONS below),
  * VG-Attribution: macro accuracy over all attribute pairs (ARO's notebook keeps
    only pairs with at least 25 samples; we do not apply that filter).

    python -m evaluation.aro --data_dir data/aro --image_dir data/aro/images
"""

import json
import os

from evaluation.utils import get_parser, load_model, macro_accuracy, save_json, score_pairs

# Relations excluded from the VG-Relation macro accuracy (same list as ARO's
# "Replicate ARO" notebook).
DROP_RELATIONS = [
    "adjusting", "attached to", "between", "bigger than", "biting", "boarding", "brushing", "chewing",
    "cleaning", "climbing", "close to", "coming from", "coming out of", "contain", "crossing", "dragging",
    "draped over", "drinking", "drinking from", "driving", "driving down", "driving on", "eating from",
    "eating in", "enclosing", "exiting", "facing", "filled with", "floating in", "floating on", "flying",
    "flying above", "flying in", "flying over", "flying through", "full of", "going down", "going into",
    "going through", "grazing in", "growing in", "growing on", "guiding", "hanging from", "hanging in",
    "hanging off", "hanging over", "higher than", "holding onto", "hugging", "in between", "jumping off",
    "jumping on", "jumping over", "kept in", "larger than", "leading", "leaning over", "leaving", "licking",
    "longer than", "looking in", "looking into", "looking out", "looking over", "looking through",
    "lying next to", "lying on top of", "making", "mixed with", "mounted on", "moving", "on the back of",
    "on the edge of", "on the front of", "on the other side of", "opening", "painted on", "parked at",
    "parked beside", "parked by", "parked in", "parked in front of", "parked near", "parked next to",
    "perched on", "petting", "piled on", "playing", "playing in", "playing on", "playing with", "pouring",
    "reaching for", "reading", "reflected on", "riding on", "running in", "running on", "running through",
    "seen through", "sitting behind", "sitting beside", "sitting by", "sitting in front of", "sitting near",
    "sitting next to", "sitting under", "skiing down", "skiing on", "sleeping in", "sleeping on",
    "smiling at", "sniffing", "splashing", "sprinkled on", "stacked on", "standing against",
    "standing around", "standing behind", "standing beside", "standing in front of", "standing near",
    "standing next to", "staring at", "stuck in", "surrounding", "swimming in", "swinging", "talking to",
    "topped with", "touching", "traveling down", "traveling on", "tying", "typing on", "underneath",
    "wading in", "waiting for", "walking across", "walking by", "walking down", "walking next to",
    "walking through", "working in", "working on", "worn on", "wrapped around", "wrapped in",
    "by", "of", "near", "next to", "with", "beside", "on the side of", "around",
]

ANNOTATIONS = {
    "relation": "visual_genome_relation.json",
    "attribution": "visual_genome_attribution.json",
}


def load_samples(annotation_file, image_dir, crop):
    with open(annotation_file) as f:
        data = json.load(f)
    samples = []
    for item in data:
        sample = {
            "image": os.path.join(image_dir, item["image_path"]),
            "positive": item["true_caption"],
            "negative": item["false_caption"],
        }
        if crop:
            sample["box"] = (item["bbox_x"], item["bbox_y"], item["bbox_w"], item["bbox_h"])
        if "relation_name" in item:
            sample["group"] = item["relation_name"]
        else:
            sample["group"] = f"{item['attributes'][0]}_{item['attributes'][1]}"
        samples.append(sample)
    return samples


def evaluate(model, preprocess, tokenizer, args, subset):
    """Returns (metrics, samples, correct) for subset 'relation' or 'attribution'."""
    samples = load_samples(os.path.join(args.data_dir, ANNOTATIONS[subset]), args.image_dir, args.crop)
    pos, neg = score_pairs(model, preprocess, tokenizer, samples, args, desc=f"VG-{subset}")
    correct = pos > neg
    groups = [s["group"] for s in samples]
    exclude = DROP_RELATIONS if subset == "relation" else ()
    metrics = {
        "macro_accuracy": macro_accuracy(correct, groups, exclude),
        "micro_accuracy": float(correct.mean()),
        "num_samples": len(samples),
    }
    return metrics, samples, correct


def main():
    parser = get_parser("VG-Relation / VG-Attribution (ARO)")
    parser.add_argument("--data_dir", required=True,
                        help="Directory with visual_genome_relation.json and visual_genome_attribution.json")
    parser.add_argument("--image_dir", required=True, help="ARO Visual Genome image directory")
    parser.add_argument("--subsets", nargs="+", default=["relation", "attribution"], choices=list(ANNOTATIONS))
    parser.add_argument("--crop", action="store_true",
                        help="Crop each image to the annotated bounding box as in the ARO code. "
                             "Off by default: the paper reports full-image results.")
    args = parser.parse_args()

    model, preprocess, tokenizer = load_model(args)
    results = {}
    for subset in args.subsets:
        results[f"vg_{subset}"], _, _ = evaluate(model, preprocess, tokenizer, args, subset)

    print(f"\nARO ({'cropped' if args.crop else 'full'} images), accuracy %")
    for name, m in results.items():
        print(f"  {name:<16} macro {100 * m['macro_accuracy']:6.2f}   "
              f"(micro {100 * m['micro_accuracy']:6.2f}, n={m['num_samples']})")
    save_json(results, args.output_json)


if __name__ == "__main__":
    main()
