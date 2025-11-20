# src/dataloaders.py
import torch
import random

from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List, Tuple, Dict
from collections import Counter
from datasets import load_dataset
import spacy


@dataclass
class Example:
    token_ids: List[int]
    label: int
    distance: int
    ent1_start: int
    ent1_text: str
    ent2_start: int
    ent2_text: str


class FOFETextDataset(Dataset):
    """PyTorch Dataset for FOFE text classification."""
    
    def __init__(self, examples: List[Example], max_len: int, pad_id: int = 0):
        self.examples = examples
        self.max_len = max_len
        self.pad_id = pad_id

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        ids = ex.token_ids
        
        # Truncate if too long (keep recent tokens at end)
        if len(ids) > self.max_len:
            ids = ids[-self.max_len:]
            
        # Pad on left so recent tokens are at end (FOFE forgets old)
        pad_len = self.max_len - len(ids)
        padded = [self.pad_id] * pad_len + ids
        
        x = torch.tensor(padded, dtype=torch.long)
        y = torch.tensor(ex.label, dtype=torch.long)
        return x, y

def build_vocab(tokens_list: List[List[str]], min_freq: int = 10) -> Dict[str, int]:
    """Build vocabulary from list of tokenized documents."""
    cnt = Counter()
    for toks in tokens_list:
        cnt.update(toks)
    
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for tok, f in cnt.items():
        if f >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab

def tokens_to_ids(tokens: List[str], vocab: Dict[str, int], force_keep: List[str] = None) -> List[int]:
    unk = vocab["<UNK>"]
    force_keep = set(force_keep or [])

    ids = []
    for t in tokens:
        if t in force_keep and t not in vocab:
            # inject entity into vocab with new ID
            vocab[t] = len(vocab)
        ids.append(vocab.get(t, unk))
    return ids

def extract_ner_examples_from_doc(
    text: str,
    nlp,
    vocab: Dict[str, int],
    max_seq_len: int = 4096,
    target_types=("PERSON", "ORG", "GPE"),
    min_distance: int = 32,
    max_distance: int = 2048,
    num_pairs_per_doc: int = 30) -> List[Example]:
    """
    Extract NER-based examples from a single document.
    
    Process:
    1. Run spaCy NER on document
    2. Find entity pairs of same type at controlled distances
    3. Label = 1 if surface forms match, 0 otherwise
    4. Build sequence from doc start to second entity mention
    """
    doc = nlp(text)
    tokens = [t.text for t in doc]
    
    if len(tokens) < min_distance + 4:
        return []

    # Extract entity mentions: (start_idx, end_idx, text, label)
    ents = [
        (ent.start, ent.end, ent.text, ent.label_)
        for ent in doc.ents
        if ent.label_ in target_types
    ]
    
    if len(ents) < 2:
        return []

    examples: List[Example] = []
    seen = set()
    entity_tokens = {ent.text.lower().strip() for ent in doc.ents}
    token_ids_full = tokens_to_ids(tokens, vocab, force_keep=entity_tokens)

    # Index entities by type
    by_type: Dict[str, List[int]] = {}
    for idx, (_, _, _, label) in enumerate(ents):
        by_type.setdefault(label, []).append(idx)

    rng = random.Random(0)

    # Generate up to num_pairs_per_doc pairs
    for _ in range(num_pairs_per_doc):
        # Choose entity type with at least 2 mentions
        types_ok = [t for t, idxs in by_type.items() if len(idxs) >= 2]
        if not types_ok:
            break
        ent_type = rng.choice(types_ok)
        idxs = by_type[ent_type]

        # Pick two different mentions
        i1, i2 = rng.sample(idxs, 2)
        start1, end1, text1, _ = ents[i1]
        start2, end2, text2, _ = ents[i2]

        # Enforce order: first then second
        if start2 <= start1:
            start1, end1, text1, start2, end2, text2 = (
                start2, end2, text2, start1, end1, text1
            )

        distance = start2 - start1
        if distance < min_distance or distance > max_distance:
            continue

        # Build sequence from start to end of second mention
        end_pos = min(end2, len(token_ids_full))
        seq_ids = token_ids_full[:end_pos]
        
        if len(seq_ids) < 4:
            continue
        if len(seq_ids) > max_seq_len:
            seq_ids = seq_ids[-max_seq_len:]

        label = 1 if text1 == text2 else 0
        
        key = tuple(seq_ids)
        if key in seen:
            continue
        seen.add(key)
        examples.append(Example(
            token_ids=seq_ids,
            label=label,
            distance=distance,
            ent1_start=start1,
            ent1_text=text1,
            ent2_start=start2,
            ent2_text=text2
        ))

    return examples

def build_corpus_examples(
    split: str = "train",
    max_docs: int = 200,
    max_seq_len: int = 4096,
    min_distance: int = 32,
    max_distance: int = 2048,) -> Tuple[List[Example], Dict[str, int]]:
    """
    Load WikiText-103 and build NER-based examples.
    
    Returns:
        examples: List of Example objects
        vocab: Token vocabulary mapping
    """
    print("Loading WikiText-103:", split)
    ds = load_dataset("wikitext", "wikitext-103-v1", split=split)

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])
    nlp.add_pipe("sentencizer")

    # Collect texts
    texts = []
    for i, row in enumerate(ds):
        if i >= max_docs:
            break
        txt = row["text"]
        if txt and not txt.isspace():
            texts.append(txt)

    # First pass: build vocab
    print("Building vocab...")
    all_tokens = []
    for txt in texts:
        doc = nlp.make_doc(txt)
        all_tokens.append([t.text for t in doc])
    vocab = build_vocab(all_tokens, min_freq=2)
    print("Vocab size:", len(vocab))

    # Second pass: extract examples
    print("Extracting examples...")
    examples: List[Example] = []
    for idx, txt in enumerate(texts):
        exs = extract_ner_examples_from_doc(
            txt, nlp, vocab,
            max_seq_len=max_seq_len,
            min_distance=min_distance,
            max_distance=max_distance,
        )
        examples.extend(exs)
        if (idx + 1) % 20 == 0:
            print(f"Processed {idx+1}/{len(texts)} docs, total examples: {len(examples)}")

    print("Total examples:", len(examples))
    return examples, vocab

def bucket_examples_by_distance(
    examples: List[Example],
    buckets: List[Tuple[int, int]],) -> Dict[Tuple[int, int], List[Example]]:
    """
    Group examples into distance buckets.
    
    Args:
        examples: List of examples
        buckets: List of (min_distance, max_distance) inclusive ranges
        
    Returns:
        Dictionary mapping bucket ranges to lists of examples
    """
    out: Dict[Tuple[int, int], List[Example]] = {b: [] for b in buckets}
    for ex in examples:
        for b in buckets:
            lo, hi = b
            if lo <= ex.distance <= hi:
                out[b].append(ex)
                break
    return out

import random

def visualize_examples(examples: List[Example], vocab: Dict[str, int], tokens_per_row: int = 20):
    if not examples:
        print("No examples to visualize.")
        return

    rev_vocab = {v: k for k, v in vocab.items()}
    ex = random.choice(examples)

    tokens = [rev_vocab.get(t, "<UNK>") for t in ex.token_ids if t != vocab["<PAD>"]]

    # ANSI red highlight
    def red(text):
        return f"\033[91m{text}\033[0m"

    highlighted = tokens.copy()

    spans = [
        (ex.ent1_start, len(ex.ent1_text.split())),
        (ex.ent2_start, len(ex.ent2_text.split()))
    ]

    # Apply highlight in reverse order
    for start, length in sorted(spans, reverse=True):
        end = start + length
        if start < len(highlighted):
            highlighted[start:end] = [red(" ".join(highlighted[start:end]))]

    print("\n" + "=" * 100)
    print(f"Label: {ex.label} | Distance: {ex.distance}")
    print("-" * 100)

    # Print wrapped by token count
    for i in range(0, len(highlighted), tokens_per_row):
        print(" ".join(highlighted[i:i + tokens_per_row]))

    print("=" * 100)

