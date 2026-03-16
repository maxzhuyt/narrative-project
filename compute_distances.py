#!/usr/bin/env python3
"""
Compute cosine embedding distances for all stories (literary + baseline).

Reads _endings.json files, encodes each generated ending and the actual
continuation (sentences[pos:]) with Qwen3-Embedding-0.6B, computes mean
cosine distance per position, and saves results to:

  narrative_project/distances/all_distances.json

Format:
  {
    "<sid>": {
      "name": "...",
      "group": "literary" | "baseline",
      "token_pcts": [float, ...],     # % tokens revealed at each position
      "emb_distances": [float, ...]   # mean cosine dist per position
    },
    ...
  }

The notebook only needs to load this file — no GPU required for plotting.
"""

import os
import re
import json
import gc
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# ── Paths ────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "/project/jevans/maxzhuyt/models/Qwen3-Embedding-0.6B"

LITERARY_RESULTS = "/project/jevans/maxzhuyt/narrative_project/results_scaled"
LITERARY_CORPUS  = "/project/jevans/maxzhuyt/narrative_project/NEWCORPUS_CLEANED"
LITERARY_IDS     = ["00066", "00135", "00056", "00015", "00144", "00166"]

BASELINE_RESULTS = "/project/jevans/maxzhuyt/narrative_project/BASELINE_CORPUS/results"
BASELINE_CORPUS  = "/project/jevans/maxzhuyt/narrative_project/BASELINE_CORPUS"

OUTPUT_DIR  = "/project/jevans/maxzhuyt/narrative_project/distances"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "all_distances.json")


# ── Sentence splitting (same regex as run_endings_scaled.py) ─────────────────

def split_sentences(text):
    boundaries = [0]
    for m in re.finditer(
        r'[.!?]["\u201d\u2019)]*\s+(?=[A-Z\u201c"(\[])', text
    ):
        boundaries.append(m.end())
    parts = []
    for i in range(len(boundaries)):
        start = boundaries[i]
        end   = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            parts.append(chunk)
    return parts


# ── Embedding helpers ─────────────────────────────────────────────────────────

def load_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {EMBEDDING_MODEL} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        EMBEDDING_MODEL, trust_remote_code=True, dtype=torch.float16
    ).to(device).eval()
    return tokenizer, model, device


def encode(texts, tokenizer, model, device, batch_size=256):
    """Return L2-normalised embeddings, shape (N, dim)."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        # mean pool over non-padding tokens
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        emb  = torch.nn.functional.normalize(emb, dim=-1)
        all_embs.append(emb.cpu().float().numpy())
    return np.vstack(all_embs)


def cos_dist(embs, ref):
    """Mean cosine distance of embs (N, dim) from ref (1, dim)."""
    sims = embs @ ref.T          # (N, 1)
    return float(np.mean(1.0 - sims[:, 0]))


# ── Token percentage helper ──────────────────────────────────────────────────

def token_pcts(sentences, positions, tokenizer):
    """Map sentence positions → % tokens revealed."""
    sent_lens = [len(tokenizer.encode(s, add_special_tokens=False))
                 for s in sentences]
    cum   = np.cumsum([0] + sent_lens)
    total = cum[-1]
    return {pos: round(float(cum[pos]) / total * 100, 1) for pos in positions}


# ── Per-story processing ──────────────────────────────────────────────────────

def process_story(sid, endings_path, corpus_path, group,
                  tokenizer, model, device):
    with open(endings_path) as f:
        d = json.load(f)
    with open(corpus_path) as f:
        raw = f.read()

    name      = next(l.strip() for l in raw.splitlines() if l.strip())
    sentences = split_sentences(raw)
    n_sents   = len(sentences)

    pos_list = [p["position"] for p in d["positions"]]
    tok_map  = token_pcts(sentences, pos_list, tokenizer)

    pcts_out  = []
    dists_out = []

    n = len(d["positions"])
    for idx, p in enumerate(d["positions"]):
        pos          = p["position"]
        continuation = " ".join(sentences[pos:])
        endings      = p["endings"]

        cont_emb = encode([continuation], tokenizer, model, device)   # (1, dim)
        end_embs = encode(endings,       tokenizer, model, device)    # (100, dim)

        pcts_out.append(tok_map[pos])
        dists_out.append(cos_dist(end_embs, cont_emb))

        if (idx + 1) % 20 == 0 or idx == n - 1:
            print(f"  [{sid}] {idx+1}/{n} positions  "
                  f"(last dist={dists_out[-1]:.4f})", flush=True)

    return {
        "name":          name,
        "group":         group,
        "token_pcts":    pcts_out,
        "emb_distances": dists_out,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load existing results to allow resume
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            all_results = json.load(f)
        print(f"Resuming: {len(all_results)} stories already done")
    else:
        all_results = {}

    tokenizer, model, device = load_embedding_model()

    # ── Literary stories ─────────────────────────────────────────────────────
    for sid in LITERARY_IDS:
        if sid in all_results:
            print(f"[{sid}] already done, skipping")
            continue
        endings_path = os.path.join(LITERARY_RESULTS, f"{sid}_endings.json")
        corpus_path  = os.path.join(LITERARY_CORPUS,  f"{sid}.txt")
        if not os.path.exists(endings_path):
            print(f"[{sid}] endings file missing, skipping")
            continue
        print(f"\n{'='*60}\n[{sid}] literary")
        all_results[sid] = process_story(
            sid, endings_path, corpus_path, "literary",
            tokenizer, model, device
        )
        with open(OUTPUT_FILE, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  → saved checkpoint ({len(all_results)} total)")
        gc.collect()
        torch.cuda.empty_cache()

    # ── Baseline stories ─────────────────────────────────────────────────────
    baseline_fnames = sorted(
        fn for fn in os.listdir(BASELINE_RESULTS)
        if fn.endswith("_endings.json") and fn.startswith("b")
    )
    print(f"\nFound {len(baseline_fnames)} baseline result files")

    for fname in baseline_fnames:
        sid = fname.replace("_endings.json", "")
        if sid in all_results:
            print(f"[{sid}] already done, skipping")
            continue
        endings_path = os.path.join(BASELINE_RESULTS, fname)
        corpus_path  = os.path.join(BASELINE_CORPUS,  f"{sid}.txt")
        if not os.path.exists(corpus_path):
            print(f"[{sid}] corpus file missing, skipping")
            continue
        print(f"\n{'='*60}\n[{sid}] baseline")
        all_results[sid] = process_story(
            sid, endings_path, corpus_path, "baseline",
            tokenizer, model, device
        )
        with open(OUTPUT_FILE, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  → saved checkpoint ({len(all_results)} total)")
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nAll done. {len(all_results)} stories → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
