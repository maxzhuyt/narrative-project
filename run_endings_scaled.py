#!/usr/bin/env python3
"""
Scaled ending prediction: sentence-by-sentence, 100 endings per position.

For all stories with ID < 200 and word count <= 5000:
  1. Split into sentences
  2. At each sentence position i (1..N-1), provide sentences 1..i as context
  3. Generate 100 possible endings (high temperature, diverse writers simulation)
  4. Compare each generated ending to the actual ending via embedding distance

Uses vLLM offline batch inference with Qwen3-32B on H200.
"""

import os
import sys
import re
import json
import time
import gc
import argparse

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = "/project/jevans/maxzhuyt/models/Qwen3-32B"
DATA_DIR = "/project/jevans/maxzhuyt/narrative_project/NEWCORPUS_CLEANED"
RESULTS_DIR = "/project/jevans/maxzhuyt/narrative_project/results_scaled"
EMBEDDING_MODEL_PATH = "/project/jevans/maxzhuyt/models/Qwen3-Embedding-0.6B"

N_ENDINGS = 100
TEMPERATURE = 1.2
TOP_P = 0.95
MAX_TOKENS = 100

MAX_STORY_ID = 200
MAX_WORD_COUNT = 5000


def discover_stories(data_dir, max_id=MAX_STORY_ID, max_words=MAX_WORD_COUNT):
    """Auto-discover all stories with ID < max_id and word count <= max_words."""
    stories = {}
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".txt"):
            continue
        story_id = fname.replace(".txt", "")
        try:
            id_num = int(story_id)
        except ValueError:
            continue
        if id_num >= max_id:
            continue
        fpath = os.path.join(data_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read()
        word_count = len(text.split())
        if word_count > max_words:
            continue
        stories[story_id] = fpath
    return stories

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a fiction writer. Write only story text."

COMPLETION_TEMPLATE = """\
Read the following incomplete story. The ending has been removed.

Write one sentence that could end this story.

Write ONLY that sentence. No commentary, no explanation, no preamble.

STORY SO FAR:
{story_so_far}"""

# ---------------------------------------------------------------------------
# Sentence splitting (reused from close_reading_two_passes.py)
# ---------------------------------------------------------------------------

def split_sentences(text):
    """Split text into sentences at natural boundaries."""
    boundaries = [0]
    for m in re.finditer(
        r'[.!?]["\u201d\u2019)]*\s+(?=[A-Z\u201c"(\[])', text
    ):
        boundaries.append(m.end())
    parts = []
    for i in range(len(boundaries)):
        start = boundaries[i]
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            parts.append(chunk)
    return parts


def get_actual_ending(sentences):
    """Get the actual ending: the last sentence of the story."""
    return [sentences[-1]]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stories", type=str, default=None,
                        help="Comma-separated story IDs to process (default: all)")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip embedding computation phase")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Number of prompts per vLLM batch")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    os.makedirs(RESULTS_DIR, exist_ok=True)

    stories = discover_stories(DATA_DIR)
    print(f"Discovered {len(stories)} stories (ID < {MAX_STORY_ID}, "
          f"<= {MAX_WORD_COUNT} words)")
    if args.stories:
        ids = [s.strip().zfill(5) for s in args.stories.split(",")]
        stories = {k: v for k, v in stories.items() if k in ids}

    # ── Load model ────────────────────────────────────────────────────────
    print(f"Loading model: {MODEL_PATH}")
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=8192,       # stories are short, no need for full 40k
        enable_prefix_caching=True,
        gpu_memory_utilization=0.92,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        n=N_ENDINGS,
    )

    # ── Process each story ────────────────────────────────────────────────
    for story_id, story_path in stories.items():
        output_path = os.path.join(RESULTS_DIR, f"{story_id}_endings.json")
        if os.path.exists(output_path):
            print(f"\n[{story_id}] — already done, skipping")
            continue

        # Load and split story
        with open(story_path, "r", encoding="utf-8") as f:
            full_text = f.read().strip()

        sentences = split_sentences(full_text)
        n_sents = len(sentences)

        # Determine actual ending
        ending_sents = get_actual_ending(sentences)
        n_ending = len(ending_sents)
        actual_ending = " ".join(ending_sents)
        last_context_idx = n_sents - n_ending  # last position before ending

        print(f"\n{'='*70}")
        print(f"[{story_id}]")
        print(f"  {n_sents} sentences, {len(full_text)} chars")
        print(f"  Actual ending ({n_ending} sents): {actual_ending[:120]}...")
        print(f"  Generating for positions 1..{last_context_idx} "
              f"× {N_ENDINGS} = {last_context_idx * N_ENDINGS} endings")

        # Prepare all prompts — apply chat template with thinking disabled
        prompts = []
        positions = []
        for i in range(1, last_context_idx + 1):
            context = " ".join(sentences[:i])
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": COMPLETION_TEMPLATE.format(
                    story_so_far=context)},
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompts.append(prompt)
            positions.append(i)

        # Generate in batches
        t0 = time.time()
        all_results = []
        BATCH = args.batch_size

        for batch_start in range(0, len(prompts), BATCH):
            batch_end = min(batch_start + BATCH, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            batch_positions = positions[batch_start:batch_end]

            outputs = llm.generate(batch_prompts, sampling_params)

            for pos, output in zip(batch_positions, outputs):
                endings = [o.text.strip() for o in output.outputs]
                pct = pos / n_sents * 100
                all_results.append({
                    "position": pos,
                    "n_context_sentences": pos,
                    "pct_story_revealed": round(pct, 1),
                    "endings": endings,
                })

            elapsed = time.time() - t0
            print(f"    Batch {batch_start // BATCH + 1}: "
                  f"positions {batch_positions[0]}-{batch_positions[-1]} "
                  f"({elapsed:.0f}s)")

        # Save results
        result = {
            "story_id": story_id,
            "n_sentences": n_sents,
            "n_ending_sentences": n_ending,
            "actual_ending": actual_ending,
            "n_endings_per_position": N_ENDINGS,
            "temperature": TEMPERATURE,
            "model": MODEL_PATH,
            "positions": all_results,
        }
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        elapsed = time.time() - t0
        print(f"  [{story_id}] Done in {elapsed:.0f}s")
        gc.collect()

    print(f"\n{'='*70}")
    print("Generation complete.")

    # ── Phase 2: Embedding distances ──────────────────────────────────────
    if args.skip_embeddings:
        print("Skipping embeddings (--skip-embeddings).")
        return

    print("\nUnloading LLM, loading embedding model...")
    del llm
    gc.collect()

    import torch
    torch.cuda.empty_cache()

    from sentence_transformers import SentenceTransformer
    from numpy.linalg import norm

    embed_model = SentenceTransformer(EMBEDDING_MODEL_PATH)

    for story_id in stories:
        result_path = os.path.join(RESULTS_DIR, f"{story_id}_endings.json")
        if not os.path.exists(result_path):
            continue

        with open(result_path) as f:
            data = json.load(f)

        # Skip if embeddings already computed
        if data["positions"] and "embedding_mean" in data["positions"][0]:
            print(f"[{story_id}] Embeddings already computed, skipping")
            continue

        actual_ending = data["actual_ending"]
        actual_emb = embed_model.encode(actual_ending, normalize_embeddings=True)

        print(f"[{story_id}] computing embeddings "
              f"({len(data['positions'])} positions)...")

        for pos_data in data["positions"]:
            endings = pos_data["endings"]
            # Batch encode
            gen_embs = embed_model.encode(endings, batch_size=256,
                                          normalize_embeddings=True)
            # Cosine distance = 1 - dot product (for normalized vectors)
            sims = gen_embs @ actual_emb
            dists = 1.0 - sims

            pos_data["embedding_distances"] = dists.tolist()
            pos_data["embedding_mean"] = float(np.mean(dists))
            pos_data["embedding_std"] = float(np.std(dists))
            pos_data["embedding_min"] = float(np.min(dists))
            pos_data["embedding_max"] = float(np.max(dists))

        with open(result_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  [{story_id}] Saved.")

    print("\nAll done.")


if __name__ == "__main__":
    main()
