#!/usr/bin/env python3
"""
SGLang offline engine: Qwen3-32B on BASELINE_CORPUS (35 stories).
Speed test — per-position real-time throughput logging.

Usage: python run_baseline_sglang.py [--stories b0001,b0003] [--skip-embeddings]
"""

import os
import re
import json
import time
import gc
import argparse

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH          = "/project/jevans/maxzhuyt/models/Qwen3-32B"
DATA_DIR            = "/project/jevans/maxzhuyt/narrative_project/BASELINE_CORPUS"
RESULTS_DIR         = "/project/jevans/maxzhuyt/narrative_project/BASELINE_CORPUS/results_sglang"
EMBEDDING_MODEL_PATH = "/project/jevans/maxzhuyt/models/Qwen3-Embedding-0.6B"

N_ENDINGS   = 100
TEMPERATURE = 1.2
TOP_P       = 0.95
MAX_TOKENS  = 100


# ---------------------------------------------------------------------------
# Story discovery — handles b0001.txt … b0035.txt
# ---------------------------------------------------------------------------

def discover_stories(data_dir):
    stories = {}
    for fname in sorted(os.listdir(data_dir)):
        if not (fname.endswith(".txt") and fname.startswith("b")):
            continue
        story_id = fname.replace(".txt", "")
        fpath = os.path.join(data_dir, fname)
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
# Sentence splitting
# ---------------------------------------------------------------------------

def split_sentences(text):
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


# ---------------------------------------------------------------------------
# Phase 1: generation
# ---------------------------------------------------------------------------

def run_generation(args):
    import sglang as sgl

    os.makedirs(RESULTS_DIR, exist_ok=True)

    stories = discover_stories(DATA_DIR)
    print(f"Discovered {len(stories)} stories in BASELINE_CORPUS", flush=True)

    if args.stories:
        ids = [s.strip() for s in args.stories.split(",")]
        stories = {k: v for k, v in stories.items() if k in ids}
        print(f"Filtered to {len(stories)}: {sorted(stories.keys())}", flush=True)

    print(f"Loading model: {MODEL_PATH}", flush=True)
    t_load = time.time()
    engine = sgl.Engine(
        model_path=MODEL_PATH,
        dtype="bfloat16",
        tp_size=1,
        context_length=8192,
        mem_fraction_static=0.92,
        log_level="warning",
    )
    print(f"Engine ready in {time.time() - t_load:.0f}s", flush=True)

    tokenizer = engine.tokenizer_manager.tokenizer
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer

    sampling_params = {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_TOKENS,
        "n": N_ENDINGS,
    }

    t_overall = time.time()

    for story_id, story_path in sorted(stories.items()):
        output_path = os.path.join(RESULTS_DIR, f"{story_id}_endings.json")
        if os.path.exists(output_path):
            print(f"\n[{story_id}] already done, skipping", flush=True)
            continue

        with open(story_path, "r", encoding="utf-8") as f:
            full_text = f.read().strip()

        sentences = split_sentences(full_text)
        n_sents = len(sentences)
        actual_ending = sentences[-1]
        last_context_idx = n_sents - 1

        print(f"\n{'='*70}", flush=True)
        print(
            f"[{story_id}]  {n_sents} sentences  "
            f"{last_context_idx * N_ENDINGS:,} completions",
            flush=True,
        )
        print(f"  Actual ending: {actual_ending[:120]}", flush=True)

        prompts = []
        positions = []
        for i in range(1, last_context_idx + 1):
            context = " ".join(sentences[:i])
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": COMPLETION_TEMPLATE.format(story_so_far=context)},
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompts.append(prompt)
            positions.append(i)

        t0 = time.time()
        all_results = []

        for pos, prompt in zip(positions, prompts):
            t_pos = time.time()
            output = engine.generate(prompt, sampling_params)

            if isinstance(output, dict):
                endings = output["text"] if isinstance(output["text"], list) else [output["text"]]
            elif isinstance(output, list):
                endings = [o["text"] for o in output]
            else:
                endings = [str(output)]
            endings = [e.strip() for e in endings]

            elapsed_pos = time.time() - t_pos
            elapsed_total = time.time() - t0
            pct = pos / n_sents * 100
            est_toks = sum(len(e.split()) * 1.3 for e in endings)
            tok_rate = est_toks / elapsed_pos if elapsed_pos > 0 else 0

            print(
                f"    pos {pos:3d}/{n_sents}  ({pct:4.1f}%)  "
                f"{elapsed_pos:5.1f}s  ~{tok_rate:5.0f} tok/s  "
                f"[{elapsed_total:.0f}s total]",
                flush=True,
            )

            all_results.append({
                "position": pos,
                "n_context_sentences": pos,
                "pct_story_revealed": round(pct, 1),
                "endings": endings,
            })

        elapsed = time.time() - t0
        n_total = last_context_idx * N_ENDINGS
        print(
            f"  [{story_id}] Done: {elapsed:.0f}s total  "
            f"{n_total/elapsed:.1f} completions/s  "
            f"{elapsed/last_context_idx:.1f}s/position",
            flush=True,
        )

        result = {
            "story_id": story_id,
            "n_sentences": n_sents,
            "n_ending_sentences": 1,
            "actual_ending": actual_ending,
            "n_endings_per_position": N_ENDINGS,
            "temperature": TEMPERATURE,
            "model": MODEL_PATH,
            "baseline": True,
            "elapsed_seconds": round(elapsed, 1),
            "positions": all_results,
        }
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  [{story_id}] Saved → {output_path}", flush=True)
        gc.collect()

    total_elapsed = time.time() - t_overall
    print(f"\n{'='*70}", flush=True)
    print(f"All stories done.  Wall time: {total_elapsed:.0f}s", flush=True)

    engine.shutdown()
    del engine
    gc.collect()


# ---------------------------------------------------------------------------
# Phase 2: embeddings
# ---------------------------------------------------------------------------

def run_embeddings(args):
    import torch
    torch.cuda.empty_cache()
    from sentence_transformers import SentenceTransformer

    stories = discover_stories(DATA_DIR)
    if args.stories:
        ids = [s.strip() for s in args.stories.split(",")]
        stories = {k: v for k, v in stories.items() if k in ids}

    print("Loading embedding model...", flush=True)
    embed_model = SentenceTransformer(EMBEDDING_MODEL_PATH)

    for story_id in sorted(stories.keys()):
        result_path = os.path.join(RESULTS_DIR, f"{story_id}_endings.json")
        if not os.path.exists(result_path):
            continue

        with open(result_path) as f:
            data = json.load(f)

        if data["positions"] and "embedding_mean" in data["positions"][0]:
            print(f"[{story_id}] Embeddings already computed, skipping", flush=True)
            continue

        actual_emb = embed_model.encode(data["actual_ending"], normalize_embeddings=True)
        print(f"[{story_id}] computing embeddings ({len(data['positions'])} positions)...", flush=True)

        for pos_data in data["positions"]:
            gen_embs = embed_model.encode(pos_data["endings"], batch_size=256, normalize_embeddings=True)
            dists = 1.0 - (gen_embs @ actual_emb)
            pos_data["embedding_distances"] = dists.tolist()
            pos_data["embedding_mean"]  = float(np.mean(dists))
            pos_data["embedding_std"]   = float(np.std(dists))
            pos_data["embedding_min"]   = float(np.min(dists))
            pos_data["embedding_max"]   = float(np.max(dists))

        with open(result_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  [{story_id}] Embeddings saved.", flush=True)

    print("\nEmbeddings complete.", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stories", type=str, default=None,
                        help="Comma-separated story IDs (e.g. b0001,b0003)")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--embeddings-only", action="store_true")
    args = parser.parse_args()

    if not args.embeddings_only:
        run_generation(args)
    if not args.skip_embeddings:
        run_embeddings(args)

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
