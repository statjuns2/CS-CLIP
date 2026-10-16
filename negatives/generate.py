#!/usr/bin/env python3
"""Generate CS-CLIP scene-graph-guided hard negatives for COCO train2014 captions.

Each caption is parsed into a scene graph with SceneGraphParser (spaCy). Seven
operations then edit the caption:

  REPLACE-OBJ  replace an object word                     BERT MLM candidates
  REPLACE-ATT  replace an adjective                       BERT MLM candidates
  REPLACE-REL  replace a one-word relation                BERT MLM candidates that are COCO relation words
  SWAP-OBJ     swap the subject and object of a relation  rule-based
  SWAP-ATT     swap the adjectives of two objects         rule-based
  ADD-OBJ      insert "and a <noun>" after an object      BERT MLM candidates (noun heuristic)
  ADD-ATT      insert an adjective before an object       BERT MLM candidates (adjective heuristic)

For the MLM-based operations the candidate that is kept is chosen by
``--nli_prefer``:
  contradiction  highest RoBERTa-large-MNLI P(contradiction | caption, negative)  (CS-CLIP, default)
  mlm            highest BERT MLM probability
  random         no MLM: tokens sampled uniformly from the BERT vocabulary (same word-class
                 filters), then one picked at random

A negative is kept only if its own scene graph has at least one triple. Several
negatives of the same type can be produced per caption (one per target); the
released training set keeps the first one per (caption, type), see merge_dedup.py.

The defaults (--mlm_topk 16, WordNet off, --batch_size 400, --seed 42, 9 shards)
are the settings used to generate the CS-CLIP training data. Table 4 ablations:
  SG+MLM     --nli_prefer mlm --use_wordnet
  SG+Random  --nli_prefer random --random_from_full_vocab

Example (shard 0 of 9, one GPU):
    CUDA_VISIBLE_DEVICES=0 python negatives/generate.py \
        --coco_root /path/to/coco \
        --relation_vocab negatives/relation_vocab_coco_train2014.json \
        --output_dir outputs/negatives --num_shards 9 --shard_idx 0
"""

import argparse
import itertools
import json
import os
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import sng_parser
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoTokenizer

MLM_MODEL = "bert-base-uncased"
NLI_MODEL = "roberta-large-mnli"  # label order: [contradiction, neutral, entailment]

# Output order of a caption's negatives (and the label kept when two operations
# produce the same sentence: the earlier operation in this list wins).
OP_ORDER = ["REPLACE-REL", "SWAP-OBJ", "SWAP-ATT", "REPLACE-OBJ", "REPLACE-ATT", "ADD-OBJ", "ADD-ATT"]

Candidate = Tuple[str, float]  # (word, MLM probability)


# ---------------------------------------------------------------------------
# Scene graphs
# ---------------------------------------------------------------------------
def scene_graph_triples(g: dict) -> List[List[str]]:
    """(subject, relation, object) triples plus (object, "is", adjective) triples."""
    heads = [e["head"] for e in g["entities"]]
    triples = [[heads[r["subject"]], r["relation"], heads[r["object"]]] for r in g.get("relations", [])]
    for e in g["entities"]:
        for m in e.get("modifiers", []):
            if m.get("dep") == "amod":
                triples.append([e["head"], "is", m["lemma_span"]])
    return triples


def has_triples(sentence: str) -> bool:
    """True if the parsed sentence has at least one triple (parser errors count as no triple)."""
    try:
        return bool(scene_graph_triples(sng_parser.parse(sentence)))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Rule-based operations
# ---------------------------------------------------------------------------
def swap_attributes(caption: str, g: dict) -> List[str]:
    """SWAP-ATT: swap the first adjectives of every pair of objects (plain substring replace)."""
    adjs = []
    for e in g["entities"]:
        for m in e.get("modifiers", []):
            if m.get("dep") == "amod":
                adjs.append(m["lemma_span"])
                break
    outs = []
    for a1, a2 in itertools.combinations(adjs, 2):
        s = caption.replace(a1, "__TEMP__", 1).replace(a2, a1, 1).replace("__TEMP__", a2, 1)
        outs.append(s)
    return outs


def swap_subject_object(caption: str, g: dict) -> List[str]:
    """SWAP-OBJ: swap subject and object phrases of each relation.

    Works on the lower-cased caption and upper-cases the first letter. When a
    phrase is not found the (re-cased) caption is still emitted; such outputs are
    dropped later if they equal the caption (merge_dedup.py).
    """
    ents = g["entities"]
    outs = []
    for rel in g.get("relations", []):
        s = caption.lower()
        subj = ents[rel["subject"]]["lemma_span"].lower()
        obj = ents[rel["object"]]["lemma_span"].lower()
        if subj in s and obj in s:
            s = s.replace(subj, "__TEMP__", 1).replace(obj, subj, 1).replace("__TEMP__", obj, 1)
        if s:
            outs.append(s[0].upper() + s[1:])
    return outs


# ---------------------------------------------------------------------------
# Word-class heuristics used to filter MLM candidates
# ---------------------------------------------------------------------------
_COLOR_ADJ = {
    "red", "orange", "yellow", "green", "blue", "purple", "violet", "indigo", "pink",
    "brown", "black", "white", "gray", "grey", "beige", "gold", "silver", "cyan", "magenta",
    "navy", "teal", "maroon", "turquoise", "lavender", "olive"}
_SIZE_ADJ = {
    "big", "small", "large", "tiny", "huge", "massive", "little", "short", "tall", "long", "wide",
    "narrow", "thick", "thin", "slim", "giant", "mini", "micro", "macro"}
_QUALITY_ADJ = {
    "good", "bad", "better", "best", "worse", "worst", "fine", "poor", "great", "excellent", "awful",
    "perfect", "nice", "terrible", "decent", "superb", "awesome"}
_COMMON_NOUNS = {
    "time", "year", "people", "way", "day", "man", "woman", "child", "world", "school", "state",
    "family", "student", "group", "country", "problem", "hand", "part", "place", "case", "week",
    "company", "system", "program", "question", "work", "government", "number", "night", "point"}
_NOUN_SUFFIXES = ("tion", "sion", "ment", "ness", "ity", "ship", "hood", "age", "ery", "ary", "ism",
                  "ist", "ance", "ence", "ure", "dom", "al")
_ADJ_SUFFIXES = ("ous", "ive", "able", "ible", "al", "ful", "less", "ic", "ish", "ary", "ant", "ent", "y")
_STRIP_PUNCT = re.compile(r"^[^\w\-]+|[^\w\-]+$")


def is_clean_token(tok: str) -> bool:
    """Whole-word WordPiece token made of at least two letters."""
    if tok.startswith("##"):
        return False
    core = tok.lstrip("#")
    return core.isalpha() and len(core) > 1


def normalize_token(tok: str) -> str:
    if not tok or tok.lower() in {"[mask]", "<mask>"} or tok.startswith("##"):
        return ""
    while tok.startswith(("Ġ", "▁")):
        tok = tok[1:]
    tok = _STRIP_PUNCT.sub("", tok).strip().lower()
    if not tok or tok.isdigit() or tok == "_":
        return ""
    return tok


class WordClassFilter:
    """Noun / adjective / relation tests for single MLM tokens.

    Nouns and adjectives are recognised by word lists and suffix rules. The CS-CLIP
    data was generated with these rules only (WordNet off); ``use_wordnet=True``
    additionally accepts any word that has a WordNet synset of the right POS.
    """

    def __init__(self, relation_vocab: set, use_wordnet: bool = False):
        self.relations = relation_vocab
        self._wn = None
        self._wn_cache: Dict[Tuple[str, str], bool] = {}
        if use_wordnet:
            from nltk.corpus import wordnet
            wordnet.synsets("dog", pos="n")  # raises LookupError if the corpus is missing
            self._wn = wordnet

    def _wn_has_pos(self, word: str, pos: str) -> bool:
        if self._wn is None or not word:
            return False
        key = (word, pos)
        if key not in self._wn_cache:
            try:
                self._wn_cache[key] = bool(self._wn.synsets(word, pos=pos))
            except Exception:
                self._wn_cache[key] = False
        return self._wn_cache[key]

    def is_noun(self, tok: str) -> bool:
        t = normalize_token(tok)
        if not t:
            return False
        if self._wn_has_pos(t, "n") or t in _COMMON_NOUNS:
            return True
        if len(t) >= 4 and t.endswith(_NOUN_SUFFIXES):
            if t.endswith("al"):
                return not self._wn_has_pos(t, "a")
            return True
        if tok[0].isupper() and tok.lower() != tok:
            return len(tok) > 1
        return False

    def is_attribute(self, tok: str) -> bool:
        t = normalize_token(tok)
        if not t:
            return False
        if self._wn_has_pos(t, "a") or self._wn_has_pos(t, "s"):
            return True
        if t in _COLOR_ADJ or t in _SIZE_ADJ or t in _QUALITY_ADJ:
            return True
        if len(t) >= 4 and t.endswith(_ADJ_SUFFIXES):
            return not (t.endswith("al") and self._wn_has_pos(t, "n"))
        if len(t) >= 5 and t.endswith(("er", "est")):
            return t not in {"other", "interest", "forest"}
        return False

    def is_relation(self, tok: str) -> bool:
        t = normalize_token(tok)
        return bool(t) and t in self.relations

    def get(self, kind: Optional[str]) -> Callable[[str], bool]:
        return {None: lambda tok: True, "noun": self.is_noun,
                "attribute": self.is_attribute, "relation": self.is_relation}[kind]


def load_relation_vocab(path: str) -> set:
    with open(path, "r") as f:
        return set(json.load(f)["relations"])


# ---------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------
def _a_or_an(word: str) -> str:
    return "an" if re.match(r"^[aeiouAEIOU]", word) else "a"


def _first_word_match(text: str, word: str):
    """First case-insensitive whole-word match of ``word`` in ``text``."""
    return re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class NegativeGenerator:
    def __init__(self, device: str, relation_vocab: set, mlm_topk: int = 16,
                 nli_prefer: str = "contradiction", use_wordnet: bool = False,
                 random_full_vocab: bool = False, seed: int = 42):
        self.device = device
        self.topk = mlm_topk
        self.prefer = nli_prefer
        self.random_full_vocab = random_full_vocab
        self.words = WordClassFilter(relation_vocab, use_wordnet)
        # A dedicated RNG: library imports cannot shift it (e.g. importing spaCy draws
        # from the global ``random`` state in some environments).
        self.rng = random.Random(seed)
        torch.manual_seed(seed)

        on_gpu = device.startswith("cuda")
        dtype = torch.float16 if on_gpu else torch.float32
        self.tok = AutoTokenizer.from_pretrained(MLM_MODEL)
        self.mlm = AutoModelForMaskedLM.from_pretrained(MLM_MODEL, torch_dtype=dtype).eval().to(device)
        self.mask = self.tok.mask_token
        self.mask_id = self.tok.mask_token_id
        self.nli_tok = self.nli = None
        if nli_prefer == "contradiction":
            self.nli_tok = AutoTokenizer.from_pretrained(NLI_MODEL)
            self.nli = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL, torch_dtype=dtype).eval().to(device)
        self._vocab_cache: Dict[Optional[str], List[str]] = {}

    # ----------------------------------------------------------------- MLM
    def _vocab_tokens(self, kind: Optional[str]) -> List[str]:
        """Random-mode candidate pool, in token-id order.

        Default: clean BERT tokens of the word class ``kind``. With
        ``random_full_vocab`` the whole vocabulary is used without any filtering
        (sub-words, [unused*] and special tokens included), as for the paper's
        SG+Random data.
        """
        if self.random_full_vocab:
            kind = "full"
        if kind not in self._vocab_cache:
            vocab = self.tok.get_vocab()
            toks = sorted(vocab, key=vocab.get)
            if kind != "full":
                keep = self.words.get(kind)
                toks = [t for t in toks if is_clean_token(t) and keep(t)]
            self._vocab_cache[kind] = toks
        return self._vocab_cache[kind]

    def mlm_candidates(self, masked: List[str], kind: Optional[str] = None) -> List[List[Candidate]]:
        """Up to ``topk`` (token, probability) candidates for the first [MASK] of each sentence.

        mlm / contradiction: the 128 most likely tokens (probabilities renormalised
        over those 128) are filtered (whole words of word class ``kind``) and the
        first ``topk`` survivors kept.
        random: ``topk`` tokens sampled uniformly from the pool of ``_vocab_tokens``.
        """
        if not masked:
            return []
        enc = self.tok(masked, return_tensors="pt", padding=True, truncation=True)
        if self.prefer == "random":
            pool = self._vocab_tokens(kind)
            if not pool:
                return [[] for _ in masked]
            out = []
            for ids in enc["input_ids"]:
                if not (ids == self.mask_id).any():
                    out.append([])
                    continue
                k = min(self.topk, len(pool))
                out.append([(t, 1.0 / k) for t in self.rng.sample(pool, k)])
            return out

        enc = {k: v.to(self.device) for k, v in enc.items()}
        # The MLM runs under fp16 autocast (LayerNorm/softmax in fp32), the NLI model
        # in plain fp16; both are kept as in the original data generation.
        ctx = torch.autocast("cuda", dtype=torch.float16) if self.device.startswith("cuda") else torch.no_grad()
        with torch.inference_mode(), ctx:
            logits = self.mlm(**enc).logits
        keep = self.words.get(kind)
        n_top = max(self.topk * 5, 128)
        out = []
        for b, ids in enumerate(enc["input_ids"]):
            pos = (ids == self.mask_id).nonzero(as_tuple=False)
            if pos.numel() == 0:
                out.append([])
                continue
            vals, idx = torch.topk(logits[b, int(pos[0])], k=n_top)
            probs = F.softmax(vals, dim=-1).tolist()
            toks = self.tok.convert_ids_to_tokens(idx.tolist())
            kept = [(t, p) for t, p in zip(toks, probs) if is_clean_token(t) and keep(t)]
            out.append(kept[:self.topk])
        return out

    def mlm_words(self, masked: str, kind: Optional[str] = None, exclude: Optional[str] = None) -> List[Candidate]:
        """Candidates for one masked sentence as plain words, minus ``exclude`` (case-insensitive)."""
        words = []
        for tok, p in self.mlm_candidates([masked], kind)[0]:
            w = self.tok.convert_tokens_to_string([tok]).strip()
            if exclude and w.lower() == exclude.lower():
                continue
            words.append((w, p))
        return words

    # ----------------------------------------------------------------- ranking
    @torch.no_grad()
    def contradiction_probs(self, premises: List[str], hypotheses: List[str]) -> List[float]:
        enc = self.nli_tok(premises, hypotheses, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        probs = self.nli(**enc).logits.softmax(-1).cpu().tolist()
        return [float(p[0]) for p in probs]

    def selection_scores(self, premises: Sequence[str], hypotheses: List[str], mlm_probs: List[float]) -> List[float]:
        """Score used to pick one candidate (premises, hypotheses and mlm_probs are aligned)."""
        if not hypotheses:
            return []
        if self.prefer == "contradiction":
            return self.contradiction_probs(list(premises), hypotheses)
        if self.prefer == "mlm":
            return list(mlm_probs)
        return [self.rng.random() for _ in hypotheses]

    @staticmethod
    def best(hypotheses: List[str], scores: List[float]) -> str:
        """Highest-scoring hypothesis; ties go to the earlier (more MLM-likely) candidate."""
        return hypotheses[max(range(len(scores)), key=scores.__getitem__)]

    # ----------------------------------------------------------------- REPLACE-REL
    def replace_relations(self, captions: List[str], graphs: List[dict]) -> List[List[str]]:
        """REPLACE-REL for a whole batch: one MLM call and one NLI call for all relations."""
        masked, metas = [], []
        for ci, (cap, g) in enumerate(zip(captions, graphs)):
            for rel in g.get("relations", []):
                word = rel.get("relation")
                if not word or len(word.split()) != 1:
                    continue
                m = _first_word_match(cap, word)
                if not m:
                    continue
                masked.append(cap[:m.start()] + self.mask + cap[m.end():])
                metas.append((ci, m.start(), m.end(), word))

        outs: List[List[str]] = [[] for _ in captions]
        if not masked:
            return outs
        cand_lists = self.mlm_candidates(masked, kind="relation")

        groups, premises, hyps, mlm_probs = [], [], [], []
        for (ci, start, end, word), cands in zip(metas, cand_lists):
            cap = captions[ci]
            n0 = len(hyps)
            for tok, p in cands:
                if tok.lower() == word.lower():
                    continue
                premises.append(cap)
                hyps.append(cap[:start] + tok + cap[end:])
                mlm_probs.append(p)
            groups.append((ci, n0, len(hyps)))
        scores = self.selection_scores(premises, hyps, mlm_probs)
        for ci, a, b in groups:
            if b > a:
                outs[ci].append(self.best(hyps[a:b], scores[a:b]))
        return outs

    # ----------------------------------------------------------------- REPLACE-OBJ / REPLACE-ATT
    def _replace_words(self, caption: str, targets: List[str]) -> List[str]:
        """Mask the first whole-word occurrence of each target and keep the best replacement.

        No word-class filter is applied to the candidates (any clean token).
        """
        outs = []
        for target in targets:
            m = _first_word_match(caption, target)
            if not m:
                continue
            masked = caption[:m.start()] + self.mask + caption[m.end():]
            cands = self.mlm_words(masked, kind=None, exclude=target)
            if not cands:
                continue
            hyps = [masked.replace(self.mask, w, 1) for w, _ in cands]
            scores = self.selection_scores([caption] * len(hyps), hyps, [p for _, p in cands])
            outs.append(self.best(hyps, scores))
        return outs

    def replace_object(self, caption: str, g: dict) -> List[str]:
        """REPLACE-OBJ: every single-word object head, in scene-graph order."""
        heads = [e.get("head") or e.get("lemma_span") for e in g.get("entities", [])]
        return self._replace_words(caption, [h for h in heads if h and len(h.split()) == 1])

    def replace_attribute(self, caption: str, g: dict) -> List[str]:
        """REPLACE-ATT: every single-word adjective (lemma), in scene-graph order."""
        adjs = []
        for e in g.get("entities", []):
            for m in e.get("modifiers", []):
                if m.get("dep") == "amod":
                    span = m.get("lemma_span") or m.get("span")
                    if span and len(span.split()) == 1:
                        adjs.append(span)
        return self._replace_words(caption, adjs)

    # ----------------------------------------------------------------- ADD-OBJ
    def add_object(self, caption: str, g: dict) -> List[str]:
        """ADD-OBJ: for each object, insert " and a <noun>" right after it.

        Candidates come from the caption-free template "<object> and [MASK]" and must
        not already occur in the caption. The insertion point is the first
        case-insensitive substring match of the object (no word boundary).
        """
        heads = [e.get("head") or e.get("lemma_span") for e in g.get("entities", [])]
        cap_words = set(re.findall(r"[A-Za-z]+", caption.lower()))
        outs = []
        for target in [h for h in heads if h]:
            cands = self.mlm_words(f"{target} and {self.mask}", kind="noun")
            if not cands:
                continue
            cands = [(w, p) for w, p in cands if w.lower() not in cap_words]
            if not cands:
                continue
            m = re.search(re.escape(target), caption, flags=re.IGNORECASE)
            if not m:
                continue
            hyps = [caption[:m.end()] + f" and {_a_or_an(w)} {w}" + caption[m.end():] for w, _ in cands]
            scores = self.selection_scores([caption] * len(hyps), hyps, [p for _, p in cands])
            outs.append(self.best(hyps, scores))
        return outs

    # ----------------------------------------------------------------- ADD-ATT
    def add_attribute(self, caption: str, g: dict) -> List[str]:
        """ADD-ATT: for each object, add one adjective.

        If the object already has adjectives, one is drawn at random (``base``), the
        template "<base> and [MASK] <object>" is filled and "<base> " becomes
        "<base> and <new> " (only when <base> directly precedes the object word).
        Otherwise "[MASK] <object>" is filled and "<new> " is inserted before the
        object word, without "and".
        """
        cap_words = set(re.findall(r"[A-Za-z]+", caption.lower()))
        outs = []
        for e in g.get("entities", []):
            head = e.get("head") or e.get("lemma_span")
            if not head:
                continue
            adjs = [m.get("lemma_span") for m in e.get("modifiers", [])
                    if m.get("dep") == "amod" and m.get("lemma_span")]
            adjs = [a for a in adjs if len(a.split()) == 1]

            if adjs:
                base = self.rng.choice(adjs)
                cands = self.mlm_words(f"{base} and {self.mask} {head}", kind="attribute")
                cands = [(w, p) for w, p in cands if w.lower() not in cap_words]
                if not cands:
                    continue
                pat = re.compile(rf"\b{re.escape(base)}\b\s+(?=\b{re.escape(head)}\b)", flags=re.IGNORECASE)
                if not pat.search(caption):
                    continue
                hyps = [pat.sub(f"{base} and {w} ", caption, count=1) for w, _ in cands]
            else:
                m = _first_word_match(caption, head)
                if not m:
                    continue
                cands = self.mlm_words(f"{self.mask} {head}", kind="attribute")
                cands = [(w, p) for w, p in cands if w.lower() not in cap_words]
                if not cands:
                    continue
                hyps = [caption[:m.start()] + f"{w} " + caption[m.start():] for w, _ in cands]

            scores = self.selection_scores([caption] * len(hyps), hyps, [p for _, p in cands])
            outs.append(self.best(hyps, scores))
        return outs

    # ----------------------------------------------------------------- batch
    def generate_batch(self, captions: List[str], graphs: List[dict]) -> List[List[Tuple[str, str]]]:
        """All (negative, type) pairs per caption, in OP_ORDER, first type wins on duplicates.

        The call order (REPLACE-REL for the whole batch, then the per-caption
        operations) fixes the RNG stream and the MLM/NLI batch shapes.
        """
        per_op = {"REPLACE-REL": self.replace_relations(captions, graphs),
                  "SWAP-OBJ": [swap_subject_object(c, g) for c, g in zip(captions, graphs)],
                  "SWAP-ATT": [swap_attributes(c, g) for c, g in zip(captions, graphs)]}
        for op in ("REPLACE-OBJ", "REPLACE-ATT", "ADD-OBJ", "ADD-ATT"):
            per_op[op] = [[] for _ in captions]
        for i, (cap, g) in enumerate(zip(captions, graphs)):
            per_op["REPLACE-OBJ"][i] = self.replace_object(cap, g)
            per_op["REPLACE-ATT"][i] = self.replace_attribute(cap, g)
            per_op["ADD-OBJ"][i] = self.add_object(cap, g)
            per_op["ADD-ATT"][i] = self.add_attribute(cap, g)

        results = []
        for i in range(len(captions)):
            seen: Dict[str, str] = {}
            for op in OP_ORDER:
                for s in per_op[op][i]:
                    seen.setdefault(s, op)
            results.append(list(seen.items()))
        return results


# ---------------------------------------------------------------------------
# Data / sharding
# ---------------------------------------------------------------------------
def load_coco_captions(captions_json: str) -> List[dict]:
    with open(captions_json, "r") as f:
        return json.load(f)["annotations"]


def shard_range(n: int, num_shards: int, shard_idx: int) -> Tuple[int, int]:
    """Contiguous split; the last shard takes the remainder."""
    size = n // num_shards
    start = size * shard_idx
    end = size * (shard_idx + 1) if shard_idx < num_shards - 1 else n
    return start, end


def image_path(image_id: int) -> str:
    return f"train2014/COCO_train2014_{image_id:012d}.jpg"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco_root", default=None, help="COCO root containing annotations/captions_train2014.json")
    ap.add_argument("--captions_json", default=None, help="overrides <coco_root>/annotations/captions_train2014.json")
    ap.add_argument("--relation_vocab", required=True, help="output of build_relation_vocab.py")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_shards", type=int, default=9)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=400,
                    help="captions per step; also fixes the REPLACE-REL MLM/NLI batch (keep 400 to reproduce)")
    ap.add_argument("--nli_prefer", default="contradiction", choices=["contradiction", "mlm", "random"])
    ap.add_argument("--mlm_topk", type=int, default=16, help="MLM candidates per mask (16 for CS-CLIP)")
    ap.add_argument("--use_wordnet", action="store_true",
                    help="also accept nouns/adjectives found in WordNet (off for the CS-CLIP data; "
                         "on for the SG+MLM data of Table 4)")
    ap.add_argument("--random_from_full_vocab", action="store_true",
                    help="--nli_prefer random only: sample from the whole BERT vocabulary without any "
                         "filtering (how the SG+Random data of Table 4 was generated)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=1, help="processes for scene-graph parsing")
    ap.add_argument("--max_batches", type=int, default=None, help="debug: stop after this many batches")
    args = ap.parse_args()

    if args.captions_json is None:
        if args.coco_root is None:
            ap.error("give --coco_root or --captions_json")
        args.captions_json = os.path.join(args.coco_root, "annotations", "captions_train2014.json")
    if not 0 <= args.shard_idx < args.num_shards:
        ap.error("need 0 <= shard_idx < num_shards")

    gen = NegativeGenerator(args.device, load_relation_vocab(args.relation_vocab), mlm_topk=args.mlm_topk,
                            nli_prefer=args.nli_prefer, use_wordnet=args.use_wordnet,
                            random_full_vocab=args.random_from_full_vocab, seed=args.seed)

    anns = load_coco_captions(args.captions_json)
    start, end = shard_range(len(anns), args.num_shards, args.shard_idx)
    anns = anns[start:end]
    num_batches = (len(anns) + args.batch_size - 1) // args.batch_size
    if args.max_batches is not None:
        num_batches = min(num_batches, args.max_batches)
    print(f"{end - start} captions in shard {args.shard_idx}/{args.num_shards} "
          f"(annotations [{start}, {end})), {num_batches} batches")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"negatives_{args.nli_prefer}_shard{args.shard_idx:02d}"
                                             f"of{args.num_shards:02d}.jsonl")
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    pmap = (lambda f, xs: list(pool.map(f, xs, chunksize=16))) if pool else (lambda f, xs: [f(x) for x in xs])

    n_written, t0 = 0, time.time()
    with open(out_path + ".partial", "w", encoding="utf-8") as fout:
        for b in tqdm(range(num_batches), desc=f"shard {args.shard_idx}"):
            batch = anns[b * args.batch_size:(b + 1) * args.batch_size]
            captions = [a["caption"] for a in batch]
            graphs = pmap(sng_parser.parse, captions)
            negatives = gen.generate_batch(captions, graphs)

            unique = list(dict.fromkeys(s for negs in negatives for s, _ in negs))
            valid = dict(zip(unique, pmap(has_triples, unique)))
            for a, negs in zip(batch, negatives):
                for neg, op in negs:
                    if not valid[neg]:
                        continue
                    rec = {"image_id": int(a["image_id"]), "image_path": image_path(a["image_id"]),
                           "caption": a["caption"], "negative_caption": neg, "type": op}
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_written += 1
            fout.flush()
    if pool is not None:
        pool.shutdown()
    os.replace(out_path + ".partial", out_path)
    dt = time.time() - t0
    print(f"wrote {n_written} negatives to {out_path} ({dt:.0f}s, {dt / max(num_batches, 1):.1f}s per batch)")


if __name__ == "__main__":
    main()
