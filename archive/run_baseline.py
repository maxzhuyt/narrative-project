#!/usr/bin/env python3
"""
Baseline run: same pipeline as run_endings_scaled.py but for children's
stories placed in NEWCORPUS_CLEANED with IDs 009xx (no ID < 200 filter).

Usage (on H100 node, no sbatch needed):
  conda activate /project/jevans/maxzhuyt/honest_llama_env
  python run_baseline.py --stories 901,902,903,904

Stories should be placed as:
  NEWCORPUS_CLEANED/00901.txt  (e.g. Tortoise and Hare)
  NEWCORPUS_CLEANED/00902.txt  (e.g. Boy Who Cried Wolf)
  etc.

Results go to results_scaled/ alongside the original 6 stories.
"""

import sys
import os

# ── Reuse everything from the main script ─────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
import run_endings_scaled as _base

# Override: no ID filter, use whatever IDs are passed via --stories
def discover_baseline_stories(data_dir, ids):
    """Load only the explicitly requested story IDs."""
    stories = {}
    for sid in ids:
        sid_padded = sid.zfill(5)
        fpath = os.path.join(data_dir, f"{sid_padded}.txt")
        if not os.path.exists(fpath):
            print(f"  WARNING: {fpath} not found — skipping")
            continue
        stories[sid_padded] = fpath
    return stories


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stories", type=str, required=True,
                        help="Comma-separated story IDs, e.g. 901,902,903")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    ids = [s.strip() for s in args.stories.split(",")]
    stories = discover_baseline_stories(_base.DATA_DIR, ids)

    if not stories:
        print("No valid story files found. Exiting.")
        return

    print(f"Baseline stories to process: {list(stories.keys())}")

    # ── Monkey-patch discover_stories so _base.main() uses our story set ──────
    # Instead of re-implementing the pipeline, we inject directly.
    from vllm import LLM, SamplingParams
    import json, time, gc
    import numpy as np

    os.makedirs(_base.RESULTS_DIR, exist_ok=True)

    print(f"Loading model: {_base.MODEL_PATH}")
    llm = LLM(
        model=_base.MODEL_PATH,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=8192,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.92,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=_base.TEMPERATURE,
        top_p=_base.TOP_P,
        max_tokens=_base.MAX_TOKENS,
        n=_base.N_ENDINGS,
    )

    for story_id, story_path in stories.items():
        output_path = os.path.join(_base.RESULTS_DIR, f"{story_id}_endings.json")
        if os.path.exists(output_path):
            print(f"\n[{story_id}] already done, skipping")
            continue

        with open(story_path, "r", encoding="utf-8") as f:
            full_text = f.read().strip()

        sentences = _base.split_sentences(full_text)
        n_sents = len(sentences)
        ending_sents = _base.get_actual_ending(sentences)
        n_ending = len(ending_sents)
        actual_ending = " ".join(ending_sents)
        last_context_idx = n_sents - n_ending

        print(f"\n{'='*70}")
        print(f"[{story_id}] {n_sents} sentences")
        print(f"  Ending: {actual_ending[:100]}...")
        print(f"  Generating {last_context_idx} × {_base.N_ENDINGS} endings")

        prompts, positions = [], []
        for i in range(1, last_context_idx + 1):
            context = " ".join(sentences[:i])
            messages = [
                {"role": "system", "content": _base.SYSTEM_PROMPT},
                {"role": "user",   "content": _base.COMPLETION_TEMPLATE.format(
                    story_so_far=context)},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            prompts.append(prompt)
            positions.append(i)

        t0 = time.time()
        all_results = []
        BATCH = args.batch_size

        for batch_start in range(0, len(prompts), BATCH):
            batch_end = min(batch_start + BATCH, len(prompts))
            outputs = llm.generate(prompts[batch_start:batch_end], sampling_params)
            for pos, output in zip(positions[batch_start:batch_end], outputs):
                all_results.append({
                    "position": pos,
                    "n_context_sentences": pos,
                    "pct_story_revealed": round(pos / n_sents * 100, 1),
                    "endings": [o.text.strip() for o in output.outputs],
                })
            elapsed = time.time() - t0
            print(f"  batch {batch_start // BATCH + 1}: "
                  f"pos {positions[batch_start]}-{positions[min(batch_end,len(positions))-1]} "
                  f"({elapsed:.0f}s)")

        result = {
            "story_id": story_id,
            "n_sentences": n_sents,
            "n_ending_sentences": n_ending,
            "actual_ending": actual_ending,
            "n_endings_per_position": _base.N_ENDINGS,
            "temperature": _base.TEMPERATURE,
            "model": _base.MODEL_PATH,
            "baseline": True,
            "positions": all_results,
        }
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  [{story_id}] done in {time.time()-t0:.0f}s")
        gc.collect()

    print("\nGeneration complete. Loading embedding model...")
    if args.skip_embeddings:
        return

    del llm
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(_base.EMBEDDING_MODEL_PATH)

    for story_id in stories:
        result_path = os.path.join(_base.RESULTS_DIR, f"{story_id}_endings.json")
        if not os.path.exists(result_path):
            continue
        with open(result_path) as f:
            data = json.load(f)
        if data["positions"] and "embedding_mean" in data["positions"][0]:
            print(f"[{story_id}] embeddings already done")
            continue

        actual_emb = embed_model.encode(data["actual_ending"],
                                        normalize_embeddings=True)
        print(f"[{story_id}] computing embeddings "
              f"({len(data['positions'])} positions)...")

        for pos_data in data["positions"]:
            gen_embs = embed_model.encode(pos_data["endings"], batch_size=256,
                                          normalize_embeddings=True)
            dists = 1.0 - (gen_embs @ actual_emb)
            pos_data["embedding_distances"] = dists.tolist()
            pos_data["embedding_mean"]  = float(np.mean(dists))
            pos_data["embedding_std"]   = float(np.std(dists))
            pos_data["embedding_min"]   = float(np.min(dists))
            pos_data["embedding_max"]   = float(np.max(dists))

        with open(result_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  [{story_id}] saved.")

    print("\nAll done.")


if __name__ == "__main__":
    main()
