"""
Chilling ending evaluation — prediction-based pipeline.

For each of 5 ending lengths:
  1. Truncate the story at sentence boundaries
  2. Independently generate 5 possible endings (high temperature)
  3. Compare generated endings to actual ending via:
     a. Embedding cosine distance (local Qwen3 model)
     b. LLM-as-judge similarity scores (0-10)

Output: per-story curves across ending lengths + graphs.
"""

import os
import re
import json
import requests
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# Configuration
load_dotenv("/home/maxzhuyt/.env")
OPENROUTER_API_KEY = os.getenv("NARRATIVE")
MODEL_NAME = "anthropic/claude-opus-4.5"
EMBEDDING_MODEL_NAME = "/home/maxzhuyt/models/Qwen3-Embedding-0.6B"
DATA_DIR = "/home/maxzhuyt/narrative_project/NEWCORPUS_CLEANED"

N_GENERATIONS = 5
N_CHUNKS = 5
GEN_TEMPERATURE = 1.0
MIN_CHUNK_CHARS = 80  # minimum characters per chunk

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

COMPLETION_PROMPT = """
Read the following incomplete story. The ending has been removed.

Write a natural continuation that completes the story in 1–5 sentences.
Match the style, tone, and voice of the preceding text exactly.

Write ONLY the ending. No commentary, no explanation, no preamble.
Do not begin with "The" or any other filler if the original text
would not. Simply continue the narrative.
"""

SIMILARITY_JUDGE_PROMPT = """
Compare each generated ending to the actual ending of a story.

ACTUAL ENDING:
{actual_ending}

GENERATED ENDINGS:
{generated_endings}

For each generated ending, rate its similarity to the actual ending
on a scale of 0 to 10:
  0 = completely different direction, events, and implications.
  5 = similar themes or emotional register but different events.
  10 = essentially the same ending with the same events and implications.

Consider BOTH surface-level plot similarity AND deeper interpretive
similarity (does the generated ending produce the same effect on the
reader's understanding of the preceding story?).
"""

# ---------------------------------------------------------------------------
# API helper
# ---------------------------------------------------------------------------

def _call_openrouter(messages, temperature=0, seed=42,
                     json_mode=True, json_keys=None, max_tokens=None):
    """Call OpenRouter API."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found.")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    if json_mode and json_keys:
        json_instruction = (
            "\nIMPORTANT: Output EXACTLY a valid JSON object. "
            "No text outside the JSON. "
            f"Keys MUST be: {json.dumps(json_keys)}"
        )
        messages = list(messages)
        messages[-1] = {
            **messages[-1],
            "content": messages[-1]["content"] + json_instruction
        }

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
    }
    if seed is not None:
        payload["seed"] = seed
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if max_tokens:
        payload["max_tokens"] = max_tokens

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers, json=payload
    )
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']


def _parse_json(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return json.loads(cleaned.strip())


# ---------------------------------------------------------------------------
# Sentence splitting and chunking
# ---------------------------------------------------------------------------

def _split_sentences(text):
    """Split text into sentences at natural boundaries."""
    # Find sentence boundaries: .!? (optionally followed by closing
    # quote) then whitespace then uppercase or opening quote.
    # Use finditer instead of lookbehind to avoid fixed-width issues.
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


def _normalize_quotes(s):
    return (s.replace("\u201c", '"').replace("\u201d", '"')
             .replace("\u2018", "'").replace("\u2019", "'"))


def chunk_story_tail(full_story, n_chunks=N_CHUNKS):
    """Split the story's tail into n_chunks at sentence boundaries.

    Returns list of dicts (shortest ending first):
    [{"truncated": str, "ending": str, "ending_pct": float, ...}, ...]
    """
    sentences = _split_sentences(full_story)
    if len(sentences) < n_chunks * 2:
        raise ValueError(f"Story too short ({len(sentences)} sentences) "
                         f"for {n_chunks} chunks")

    # Take enough tail sentences for chunking
    # Target: longest ending ~ 15-20% of story chars
    target_tail_chars = int(len(full_story) * 0.20)
    n_tail = 0
    tail_chars = 0
    for s in reversed(sentences):
        tail_chars += len(s)
        n_tail += 1
        if tail_chars >= target_tail_chars:
            break
    n_tail = max(n_tail, n_chunks * 2)  # at least 2 sents per chunk
    n_tail = min(n_tail, len(sentences) // 2)  # never more than half

    tail = sentences[-n_tail:]

    # Divide tail into n_chunks groups
    chunk_size = len(tail) // n_chunks
    remainder = len(tail) % n_chunks
    groups = []
    idx = 0
    for i in range(n_chunks):
        size = chunk_size + (1 if i >= n_chunks - remainder else 0)
        groups.append(tail[idx:idx + size])
        idx += size

    # Build cumulative endings (shortest = last group only)
    results = []
    for i in range(1, n_chunks + 1):
        ending_sents = []
        for g in groups[-i:]:
            ending_sents.extend(g)

        # Find the ending in the original text
        first_words = " ".join(ending_sents[0].split()[:6])
        pos = full_story.find(first_words)
        if pos == -1:
            pos = _normalize_quotes(full_story).find(
                _normalize_quotes(first_words))
        if pos == -1:
            # Last resort: try fewer words
            first_words = " ".join(ending_sents[0].split()[:3])
            pos = _normalize_quotes(full_story).find(
                _normalize_quotes(first_words))

        if pos > 0:
            ending_text = full_story[pos:].strip()
            truncated = full_story[:pos].strip()
        else:
            ending_text = " ".join(ending_sents)
            body = sentences[:len(sentences) - len(ending_sents)]
            truncated = " ".join(body)

        results.append({
            "truncated": truncated,
            "ending": ending_text,
            "ending_chars": len(ending_text),
            "ending_pct": len(ending_text) / len(full_story) * 100,
            "n_sentences": len(ending_sents),
        })

    return results


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_ending(truncated_story):
    """Generate one possible ending with high temperature."""
    messages = [
        {"role": "system",
         "content": "You are an expert fiction writer. "
                    "Write only story text, no meta-commentary."},
        {"role": "user",
         "content": f"{COMPLETION_PROMPT}\n\nSTORY:\n{truncated_story}"}
    ]
    return _call_openrouter(
        messages,
        temperature=GEN_TEMPERATURE,
        seed=None,
        json_mode=False,
        max_tokens=300
    ).strip()


# ---------------------------------------------------------------------------
# Embedding distance
# ---------------------------------------------------------------------------

_encoder = None
_encoder_lock = __import__('threading').Lock()


def _get_encoder():
    global _encoder
    if _encoder is None:
        with _encoder_lock:
            if _encoder is None:
                _encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _encoder


def compute_embedding_distances(actual_ending, generated_endings):
    """Cosine distance between actual ending and each generated ending."""
    encoder = _get_encoder()
    emb_actual = encoder.encode(actual_ending)
    return [
        float(cosine(emb_actual, encoder.encode(gen)))
        for gen in generated_endings
    ]


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

def judge_similarities(actual_ending, generated_endings):
    """LLM rates similarity of each generated ending to actual (0-10)."""
    gen_text = "\n".join(
        f"{i+1}. {e}" for i, e in enumerate(generated_endings)
    )
    prompt = SIMILARITY_JUDGE_PROMPT.format(
        actual_ending=actual_ending,
        generated_endings=gen_text
    )
    messages = [
        {"role": "system",
         "content": "You are an expert literary critic. "
                    "You only output valid JSON."},
        {"role": "user", "content": prompt}
    ]
    raw = _call_openrouter(
        messages, temperature=0, seed=42,
        json_mode=True, json_keys=["scores"]
    )
    result = _parse_json(raw)
    scores = result.get("scores", [])
    if isinstance(scores, dict):
        scores = list(scores.values())
    return [float(s) for s in scores[:len(generated_endings)]]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def evaluate_story(full_story):
    """Full evaluation: chunk, generate, compare across 5 ending lengths."""
    chunks = chunk_story_tail(full_story, N_CHUNKS)

    all_results = []
    for ci, chunk_info in enumerate(chunks):
        level = ci + 1
        ending = chunk_info["ending"]
        truncated = chunk_info["truncated"]

        print(f"    Chunk {level}/{N_CHUNKS} "
              f"({chunk_info['n_sentences']} sents, "
              f"{chunk_info['ending_pct']:.1f}%)")

        # Generate N independent endings
        generated = []
        for gi in range(N_GENERATIONS):
            try:
                gen = generate_ending(truncated)
                generated.append(gen)
            except Exception as e:
                print(f"      Gen {gi+1} failed: {e}")

        if not generated:
            all_results.append({"level": level, "error": "no generations"})
            continue

        # Embedding distances
        emb_dists = compute_embedding_distances(ending, generated)

        # LLM judge
        try:
            llm_scores = judge_similarities(ending, generated)
        except Exception as e:
            print(f"      Judge failed: {e}")
            llm_scores = []

        all_results.append({
            "level": level,
            "ending_chars": chunk_info["ending_chars"],
            "ending_pct": chunk_info["ending_pct"],
            "n_sentences": chunk_info["n_sentences"],
            "actual_ending": ending[:300],
            "generated_endings": [g[:300] for g in generated],
            "embedding_distances": emb_dists,
            "embedding_mean": float(np.mean(emb_dists)),
            "embedding_min": float(np.min(emb_dists)),
            "llm_scores": llm_scores,
            "llm_mean": float(np.mean(llm_scores)) if llm_scores else 0.0,
            "llm_max": float(np.max(llm_scores)) if llm_scores else 0.0,
        })

    return all_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(all_stories, output_dir="results"):
    """Generate comparison graphs across all stories."""
    os.makedirs(output_dir, exist_ok=True)

    # Style markers by expected category
    style_map = {
        "high": {"marker": "s", "linewidth": 2.5},
        "moderate": {"marker": "D", "linewidth": 2.0},
        "low": {"marker": "o", "linewidth": 1.5},
    }

    # --- Embedding distance plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for story_id, title, expected, results in all_stories:
        if not results or "error" in results[0]:
            continue
        levels = [r["level"] for r in results if "error" not in r]
        emb_means = [r["embedding_mean"] for r in results if "error" not in r]
        llm_means = [r["llm_mean"] for r in results if "error" not in r]
        style = style_map.get(expected, {})
        label = f"{title} ({expected})"

        ax1.plot(levels, emb_means, '-', label=label,
                 marker=style.get("marker", "o"),
                 linewidth=style.get("linewidth", 1.5),
                 markersize=7)
        ax2.plot(levels, llm_means, '-', label=label,
                 marker=style.get("marker", "o"),
                 linewidth=style.get("linewidth", 1.5),
                 markersize=7)

    ax1.set_xlabel("Ending length (chunks removed, 1=shortest)")
    ax1.set_ylabel("Mean embedding distance (cosine)\nhigher = less predictable")
    ax1.set_title("Embedding Distance:\nGenerated vs. Actual Ending")
    ax1.legend(fontsize=7, loc="best")
    ax1.set_xticks(range(1, N_CHUNKS + 1))
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Ending length (chunks removed, 1=shortest)")
    ax2.set_ylabel("Mean LLM similarity (0-10)\nlower = less predictable")
    ax2.set_title("LLM Judge Similarity:\nGenerated vs. Actual Ending")
    ax2.legend(fontsize=7, loc="best")
    ax2.set_xticks(range(1, N_CHUNKS + 1))
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ending_prediction.png"), dpi=150)
    plt.close('all')
    print(f"  Plot saved to {output_dir}/ending_prediction.png")
