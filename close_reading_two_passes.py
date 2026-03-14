## Example usage: python close_reading_two_passes.py --filename 00066.txt --ending_text "No one knows exactly what happened to them—only that, six weeks later, the wife had an abortion, just in case, and a year later she gave birth to a baby girl."

import os
import argparse
import requests
import numpy as np
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# Configuration
load_dotenv("/home/maxzhuyt/.env")
OPENROUTER_API_KEY = os.getenv("NARRATIVE")
MODEL_NAME = "anthropic/claude-opus-4.5"
EMBEDDING_MODEL_NAME = "/home/maxzhuyt/models/Qwen3-Embedding-0.6B"
DATA_DIR = "/home/maxzhuyt/narrative_project/NEWCORPUS_CLEANED"

# ---------------------------------------------------------------------------
# Pass 0: Automatic ending extraction
# ---------------------------------------------------------------------------

ENDING_EXTRACTION_PROMPT = """
Read the following story carefully. Your task is to identify the FINAL
PASSAGE — the shortest span of text at the very end of the story that,
if removed, would most change a reader's interpretation of the preceding
narrative.

This is typically the last 1–3 sentences. It should be the minimal
ending that does the most interpretive work — the part that reframes,
reveals, or recontextualizes what came before. If the story has no such
recontextualizing ending, select the final sentence.

Return a JSON object with:
- "start_words": the first 8–10 words of where this final passage begins,
  copied EXACTLY from the text (for matching purposes).
- "explanation": one sentence explaining what this passage does to the
  reader's understanding of the story.
"""

# ---------------------------------------------------------------------------
# Two aspect-specific reading prompts
# ---------------------------------------------------------------------------

# Aspect A: sensitive to factual realignment — what happens, what it means
CONTENT_READING_PROMPT = """
Perform a close reading of the following story focused on its CONTENT
AND MEANING. Structure your response using EXACTLY these three sections,
in this order. Within each section, write exactly one paragraph of
4–6 sentences.

## 1. Plot and Key Events
Summarize the factual sequence of the main events. Note any revelations,
reversals, or withheld information that reshape earlier events.

## 2. Character Relationships and Motivations
Identify the main characters and describe the power dynamics, desires,
and conflicts between them. What does each character want, and how do
their relationships change?

## 3. Central Argument and Implied Meaning
What does the story ultimately argue or reveal? Commit to a specific
interpretive claim. What is the reader left to understand?

Do NOT speculate about what might come after the text provided.
Read ONLY what is on the page. Do not mention that the story feels
incomplete or truncated, even if it does.
"""

# Aspect B: sensitive to stylistic shift — how it's told, how it feels
FORM_READING_PROMPT = """
Perform a close reading of the following story focused on its FORM
AND AFFECT. Structure your response using EXACTLY these three sections,
in this order. Within each section, write exactly one paragraph of
4–6 sentences.

## 1. Voice, Register, and Word Choice
Characterize the narrator's voice and the story's dominant register.
Note any moments where word choice or diction breaks from or intensifies
the prevailing style. What effect does this create?

## 2. Emotional Arc and Tonal Shifts
Describe the story's dominant emotional register and how it changes from
beginning to end. Where does the tone shift most dramatically, and what
is the effect on the reader?

## 3. Imagery, Symbols, and Figurative Language
Identify the most important image pattern, symbol, or figurative device.
How does its meaning develop across the story? Does its significance
change by the end?

Do NOT speculate about what might come after the text provided.
Read ONLY what is on the page. Do not mention that the story feels
incomplete or truncated, even if it does.
"""

# Shared revision prompt — used for both aspects
REVISION_PROMPT = """
Below is a close reading you previously wrote for a TRUNCATED version of
a story, followed by the COMPLETE story (which includes an ending that
was previously withheld). Your task is to REVISE your prior reading in
light of the full text.

RULES:
- Keep EXACTLY the same section headers, in the same order.
- Within each section, write exactly one paragraph of 4–6 sentences.
- Where the ending does NOT change your interpretation, reproduce your
  original language as closely as possible—only make changes that are
  directly motivated by the new ending.
- Where the ending DOES change your interpretation, explain clearly
  what shifted and why.
- Do NOT comment on the task itself or mention that text was withheld.

YOUR PRIOR READING (based on truncated text):
{prior_reading}

COMPLETE STORY:
{full_story}

Now provide your revised reading:
"""

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _call_openrouter(messages, json_keys):
    """Low-level OpenRouter call. Returns raw content string."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found.")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    json_instruction = (
        "\nIMPORTANT: Output your response EXACTLY as a valid JSON object. "
        "Do not include any text outside the JSON. "
        f"The keys MUST be exactly: {json.dumps(json_keys)}"
    )
    messages = list(messages)  # copy
    messages[-1] = {**messages[-1], "content": messages[-1]["content"] + json_instruction}

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "seed": 42,
        "response_format": {"type": "json_object"}
    }
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload
    )
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']


def get_reading(story_text, prompt):
    """Generate a close reading of a story with a given aspect prompt."""
    keys = _keys_for_prompt(prompt)
    messages = [
        {"role": "system", "content": "You are an expert literary critic. You only output valid JSON."},
        {"role": "user", "content": f"{prompt}\n\nSTORY:\n{story_text}"}
    ]
    return _call_openrouter(messages, keys)


def get_revision(prior_reading, full_story, prompt):
    """Revise a prior reading given the full story."""
    keys = _keys_for_prompt(prompt)
    revision_text = REVISION_PROMPT.format(
        prior_reading=prior_reading,
        full_story=full_story
    )
    messages = [
        {"role": "system", "content": "You are an expert literary critic. You only output valid JSON."},
        {"role": "user", "content": revision_text}
    ]
    return _call_openrouter(messages, keys)


def _keys_for_prompt(prompt):
    """Extract expected JSON keys from a prompt's section headers."""
    if "Plot and Key Events" in prompt:
        return ["1. Plot and Key Events",
                "2. Character Relationships and Motivations",
                "3. Central Argument and Implied Meaning"]
    else:
        return ["1. Voice, Register, and Word Choice",
                "2. Emotional Arc and Tonal Shifts",
                "3. Imagery, Symbols, and Figurative Language"]


def extract_ending(full_story):
    """
    Pass 0: Ask the LLM to identify the minimal recontextualizing ending.
    Returns the ending text (a suffix of the story) or None on failure.
    """
    messages = [
        {"role": "system", "content": "You are an expert literary critic. You only output valid JSON."},
        {"role": "user", "content": f"{ENDING_EXTRACTION_PROMPT}\n\nSTORY:\n{full_story}"}
    ]
    keys = ["start_words", "explanation"]
    try:
        raw = _call_openrouter(messages, keys)
        cleaned = raw.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        result = json.loads(cleaned.strip())
        start_words = result.get("start_words", "")
        explanation = result.get("explanation", "")

        # Find the start_words in the story and extract from there to end
        # Handle smart-quote mismatches: normalize both sides
        def _normalize_quotes(s):
            return (s.replace("\u201c", '"').replace("\u201d", '"')
                     .replace("\u2018", "'").replace("\u2019", "'"))

        idx = full_story.find(start_words)
        if idx == -1:
            # Try with normalized quotes
            norm_story = _normalize_quotes(full_story)
            norm_sw = _normalize_quotes(start_words)
            nidx = norm_story.find(norm_sw)
            if nidx == -1:
                # Fuzzy fallback: try first 5 words
                shorter = " ".join(norm_sw.split()[:5])
                nidx = norm_story.find(shorter)
            idx = nidx  # positions match since replacement is 1:1 in length
        if idx == -1:
            print(f"  Warning: could not locate start_words in text: '{start_words[:50]}'")
            return None, explanation

        ending = full_story[idx:].strip()
        return ending, explanation
    except Exception as e:
        print(f"  Error in ending extraction: {e}")
        return None, ""


def parse_sections(reading_text):
    """Parses JSON structured reading into its named sections."""
    if not reading_text:
        return {}
    try:
        cleaned = reading_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return json.loads(cleaned.strip())
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        return {}


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

_encoder = None
_encoder_lock = __import__('threading').Lock()

def _get_encoder():
    """Lazy-load and cache the embedding model (thread-safe)."""
    global _encoder
    if _encoder is None:
        with _encoder_lock:
            if _encoder is None:
                _encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _encoder


def _compute_distances(reading_a, reading_b):
    """Compute per-section cosine distances and a global distance."""
    encoder = _get_encoder()
    sections_a = parse_sections(reading_a)
    sections_b = parse_sections(reading_b)

    section_dists = {}
    embs_a, embs_b = [], []

    for key in sections_a:
        if key in sections_b:
            ea = encoder.encode(sections_a[key])
            eb = encoder.encode(sections_b[key])
            embs_a.append(ea)
            embs_b.append(eb)
            section_dists[key] = float(cosine(ea, eb))

    if embs_a:
        global_dist = float(cosine(np.mean(embs_a, axis=0),
                                   np.mean(embs_b, axis=0)))
    else:
        global_dist = 0.0

    return global_dist, section_dists


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def calculate_interpretive_divergence(full_story, ending_segment=None):
    """
    Two-aspect, revision-anchored Interpretive Divergence.

    If ending_segment is None, Pass 0 auto-extracts the ending.
    Returns a results dict with content_distance, form_distance,
    global_distance, ending info, and all raw readings.
    """
    # --- Pass 0: extract ending if not provided ---
    ending_explanation = ""
    if ending_segment is None:
        print("  Pass 0: Extracting ending...")
        ending_segment, ending_explanation = extract_ending(full_story)
        if ending_segment is None:
            return {"error": "Failed to extract ending"}
        print(f"  Extracted ending ({len(ending_segment)} chars): "
              f"'{ending_segment[:60]}...'")

    truncated_story = full_story.replace(ending_segment, "").strip()
    ending_fraction = len(ending_segment) / len(full_story)

    # --- Pass 1: read truncated story through both lenses ---
    print("  Pass 1a: Content reading of truncated story...")
    content_trunc = get_reading(truncated_story, CONTENT_READING_PROMPT)

    print("  Pass 1b: Form reading of truncated story...")
    form_trunc = get_reading(truncated_story, FORM_READING_PROMPT)

    if not content_trunc or not form_trunc:
        return {"error": "Failed to generate truncated readings"}

    # --- Pass 2: revise each reading given the full story ---
    print("  Pass 2a: Revising content reading with full story...")
    content_full = get_revision(content_trunc, full_story, CONTENT_READING_PROMPT)

    print("  Pass 2b: Revising form reading with full story...")
    form_full = get_revision(form_trunc, full_story, FORM_READING_PROMPT)

    if not content_full or not form_full:
        return {"error": "Failed to generate revised readings"}

    # --- Compute distances ---
    content_global, content_sections = _compute_distances(content_trunc, content_full)
    form_global, form_sections = _compute_distances(form_trunc, form_full)

    global_distance = (content_global + form_global) / 2

    return {
        "ending_segment": ending_segment,
        "ending_explanation": ending_explanation,
        "ending_fraction": ending_fraction,
        "content_distance": content_global,
        "form_distance": form_global,
        "global_distance": global_distance,
        "content_sections": content_sections,
        "form_sections": form_sections,
        "readings": {
            "content_truncated": content_trunc,
            "content_full": content_full,
            "form_truncated": form_trunc,
            "form_full": form_full,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate Interpretive Divergence for a story file.")
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--ending_text", type=str, required=True)
    args = parser.parse_args()

    file_path = os.path.join(DATA_DIR, args.filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()

        if args.ending_text not in full_text:
            print(f"Error: ending not found in '{args.filename}'.")
        else:
            print(f"Processing '{args.filename}'...")
            results = calculate_interpretive_divergence(full_text, args.ending_text)

            if "error" in results:
                print(results["error"])
            else:
                base = os.path.splitext(args.filename)[0]
                rdir = os.path.join("results", base)
                os.makedirs(rdir, exist_ok=True)
                with open(os.path.join(rdir, "results.json"), "w") as f:
                    json.dump({k: v for k, v in results.items()
                               if k != "readings"}, f, indent=4)
                for name, text in results["readings"].items():
                    with open(os.path.join(rdir, f"{name}.json"), "w") as f:
                        f.write(text)

                ef = results["ending_fraction"]
                print(f"\nEnding fraction: {ef:.1%}")
                print(f"\n{'Aspect':<12} {'Raw':>8} {'Normalized':>10}")
                print("-" * 32)
                print(f"{'Content':<12} {results['content_distance']:>8.4f} {results['content_normalized']:>10.4f}")
                print(f"{'Form':<12} {results['form_distance']:>8.4f} {results['form_normalized']:>10.4f}")
                print(f"{'Global':<12} {results['global_distance']:>8.4f} {results['global_normalized']:>10.4f}")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"Error: {e}")
