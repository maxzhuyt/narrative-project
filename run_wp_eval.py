#!/usr/bin/env python3
"""
WritingPrompts (WP) evaluation: sample 50 human + 50 GPT stories from
ghostbuster-data/wp/, run the same vLLM endings pipeline as run_baseline_eval.py.

Optimisation: if the sentence just added at position i has < MIN_SENT_WORDS words,
skip that position and defer evaluation to the end of the next sentence.

Results saved to BASELINE_CORPUS/results/ as:
  wph{001..050}_endings.json   (human)
  wpg{001..050}_endings.json   (gpt)
"""

import os, re, json, gc, time, random
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH      = "/project/jevans/maxzhuyt/models/Qwen3-32B"
RESULTS_DIR     = "/project/jevans/maxzhuyt/narrative_project/BASELINE_CORPUS/results"
WP_HUMAN_DIR    = "/project/jevans/maxzhuyt/narrative_project/BASELINE_CORPUS/ghostbuster-data/wp/human"
WP_GPT_DIR      = "/project/jevans/maxzhuyt/narrative_project/BASELINE_CORPUS/ghostbuster-data/wp/gpt"
EMBEDDING_MODEL = "/project/jevans/maxzhuyt/models/Qwen3-Embedding-0.6B"

N_SAMPLE        = 50          # stories per group
MAX_WORDS       = 3000
MIN_SENT_WORDS  = 10          # skip position if last-added sentence < this many words
N_ENDINGS       = 100
TEMPERATURE     = 1.2
TOP_P           = 0.95
MAX_TOKENS      = 100
BATCH_SIZE      = 64
RANDOM_SEED     = 42

SYSTEM_PROMPT = "You are a fiction writer. Write only story text."
COMPLETION_TEMPLATE = """\
Read the following incomplete story. The ending has been removed.

Write one sentence that could end this story.

Write ONLY that sentence. No commentary, no explanation, no preamble.

STORY SO FAR:
{story_so_far}"""


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def sample_stories(directory, n, max_words, seed):
    """Return list of (sid_prefix_index, filepath, text) for n sampled stories."""
    random.seed(seed)
    all_files = sorted(f for f in os.listdir(directory) if f.endswith(".txt"))
    eligible = []
    for fname in all_files:
        fpath = os.path.join(directory, fname)
        text  = open(fpath, encoding="utf-8", errors="replace").read().strip()
        if len(text.split()) <= max_words:
            eligible.append((fname, fpath, text))
    sampled = random.sample(eligible, min(n, len(eligible)))
    sampled.sort(key=lambda x: x[0])   # deterministic order after sampling
    return sampled


def active_positions(sentences):
    """
    Return list of positions (1-indexed) to evaluate.
    Position i means: context = sentences[:i], newly added = sentences[i-1].
    Skip position i if sentences[i-1] has < MIN_SENT_WORDS words.
    The last content position is always included regardless.
    """
    n          = len(sentences)
    last_pos   = n - 1   # exclude the final sentence (it's the actual ending)
    positions  = []
    for i in range(1, last_pos + 1):
        words = len(sentences[i - 1].split())
        if words >= MIN_SENT_WORDS or i == last_pos:
            positions.append(i)
    return positions


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    from vllm import LLM, SamplingParams

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Sample stories
    human_stories = sample_stories(WP_HUMAN_DIR, N_SAMPLE, MAX_WORDS, RANDOM_SEED)
    gpt_stories   = sample_stories(WP_GPT_DIR,   N_SAMPLE, MAX_WORDS, RANDOM_SEED + 1)

    print(f"Sampled {len(human_stories)} human + {len(gpt_stories)} GPT stories")

    # Build job list, skip already-done
    jobs = []
    for idx, (fname, fpath, text) in enumerate(human_stories, 1):
        sid = f"wph{idx:03d}"
        out = os.path.join(RESULTS_DIR, f"{sid}_endings.json")
        if not os.path.exists(out):
            jobs.append((sid, "wp_human", fname, text, out))
        else:
            print(f"  [{sid}] already done, skipping")
    for idx, (fname, fpath, text) in enumerate(gpt_stories, 1):
        sid = f"wpg{idx:03d}"
        out = os.path.join(RESULTS_DIR, f"{sid}_endings.json")
        if not os.path.exists(out):
            jobs.append((sid, "wp_gpt", fname, text, out))
        else:
            print(f"  [{sid}] already done, skipping")

    if not jobs:
        print("All stories already done.")
        return

    print(f"\n{len(jobs)} stories to process")

    # Load model
    print(f"\nLoading {MODEL_PATH}...")
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=8192,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.88,
        max_num_seqs=256,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=TEMPERATURE, top_p=TOP_P,
        max_tokens=MAX_TOKENS, n=N_ENDINGS,
    )

    for sid, group, source_fname, full_text, output_path in jobs:
        sentences    = split_sentences(full_text)
        n_sents      = len(sentences)
        actual_ending = sentences[-1]
        positions    = active_positions(sentences)

        skipped = (n_sents - 1) - len(positions)
        print(f"\n{'='*70}")
        print(f"[{sid}] {group}  ← {source_fname}")
        print(f"  {n_sents} sents, {len(full_text.split())} words")
        print(f"  Evaluating {len(positions)}/{n_sents-1} positions "
              f"({skipped} skipped, short sentences)")

        prompts   = []
        pos_batch = []
        for pos in positions:
            context  = " ".join(sentences[:pos])
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": COMPLETION_TEMPLATE.format(
                    story_so_far=context)},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            prompts.append(prompt)
            pos_batch.append(pos)

        t0 = time.time()
        all_results = []
        for i in range(0, len(prompts), BATCH_SIZE):
            batch_p   = prompts[i:i + BATCH_SIZE]
            batch_pos = pos_batch[i:i + BATCH_SIZE]
            outputs   = llm.generate(batch_p, sampling_params)
            for pos, out in zip(batch_pos, outputs):
                endings = [o.text.strip() for o in out.outputs]
                pct     = pos / n_sents * 100
                all_results.append({
                    "position":            pos,
                    "n_context_sentences": pos,
                    "pct_story_revealed":  round(pct, 1),
                    "endings":             endings,
                })
            elapsed = time.time() - t0
            print(f"    batch {i//BATCH_SIZE+1}: pos {batch_pos[0]}-{batch_pos[-1]}"
                  f"  ({elapsed:.0f}s)", flush=True)

        result = {
            "story_id":              sid,
            "group":                 group,
            "source_file":           source_fname,
            "n_sentences":           n_sents,
            "n_positions_evaluated": len(positions),
            "n_positions_skipped":   skipped,
            "actual_ending":         actual_ending,
            "n_endings_per_position": N_ENDINGS,
            "temperature":           TEMPERATURE,
            "model":                 MODEL_PATH,
            "baseline":              True,
            "positions":             all_results,
        }
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  [{sid}] done in {time.time()-t0:.0f}s → {output_path}")
        gc.collect()

    print(f"\n{'='*70}\nGeneration complete.")

    # ── Phase 2: Embedding distances ──────────────────────────────────────────
    print("\nUnloading LLM, loading embedding model...")
    del llm
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    all_sids = ([f"wph{i:03d}" for i in range(1, N_SAMPLE+1)] +
                [f"wpg{i:03d}" for i in range(1, N_SAMPLE+1)])

    for sid in all_sids:
        rpath = os.path.join(RESULTS_DIR, f"{sid}_endings.json")
        if not os.path.exists(rpath):
            continue
        with open(rpath) as f:
            data = json.load(f)
        if data["positions"] and "embedding_mean" in data["positions"][0]:
            print(f"[{sid}] embeddings already done, skipping")
            continue

        actual_emb = embed_model.encode(data["actual_ending"], normalize_embeddings=True)
        print(f"[{sid}] embeddings ({len(data['positions'])} positions)...")
        for p in data["positions"]:
            gen_embs = embed_model.encode(p["endings"], batch_size=256,
                                          normalize_embeddings=True)
            sims = gen_embs @ actual_emb
            dists = 1.0 - sims
            p["embedding_distances"] = dists.tolist()
            p["embedding_mean"]      = float(np.mean(dists))
            p["embedding_std"]       = float(np.std(dists))
        with open(rpath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  [{sid}] saved.")

    print("\nAll done.")


if __name__ == "__main__":
    main()
