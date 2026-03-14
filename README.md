# Evaluating Chilling Endings in Short Fiction

An evaluation pipeline for measuring how much a story's ending reshapes its meaning — what we call **Interpretive Divergence**.

## What is a chilling ending?

A "chilling ending" occurs when the final sentences of a story force a re-reading of the preceding text. We identified three mechanisms (see `1_8 Update.pdf`):

1. **Factual realignment** — the ending introduces a fact that reshapes understanding (e.g., "the wife had an abortion, just in case")
2. **Stylistic shift** — a word or tone breaks from expectations (e.g., "melancholically" reframing a satirical triumph)
3. **Meta-narrative reframing** — the ending comments on the act of storytelling itself

## Pipeline overview

```
Full Story
    |
    v
[Pass 0] Auto-extract minimal ending (1-3 sentences)
    |
    v
[Pass 1a] Content reading of truncated story (plot, character, meaning)
[Pass 1b] Form reading of truncated story (voice, tone, imagery)
    |
    v
[Pass 2a] Revise content reading given full story
[Pass 2b] Revise form reading given full story
    |
    v
Embed each section pair -> cosine distance
    |
    v
Content distance, Form distance, Global distance
```

**Key design choices:**
- **Two aspect-specific readings** instead of one generic reading. Content readings are sensitive to factual realignment; Form readings are sensitive to stylistic shift. The divergence *pattern* across the two reveals the mechanism type without a separate classifier.
- **Revision-anchored** (temperature=0). The model revises its own prior reading, preserving language where interpretation doesn't change. This controls noise while surfacing genuine shifts.
- **Automatic ending extraction** (Pass 0). The LLM identifies the minimal final passage that does the most interpretive work, typically 1-3 sentences. This standardizes ending length across stories (0.2-1.4% of text) and removes a major confound.
- **Local embedding model** (Qwen3-Embedding-0.6B, 1024-dim) for cosine distance computation. Runs on GPU.

## Results on 6 test stories

| Story | Expected | End% | Content | Form | Global |
|-------|----------|------|---------|------|--------|
| Poor Girl (66) | high | 1.4% | **0.107** | **0.070** | **0.088** |
| Two Ruminations (135) | moderate | 1.2% | 0.063 | 0.015 | 0.039 |
| Snowing in Greenwich Village (56) | low | 0.4% | 0.061 | 0.018 | 0.039 |
| The Fellow (15) | low | 0.2% | 0.051 | 0.027 | 0.039 |
| Ladies' Lunch (144) | high | 0.8% | 0.018 | **0.043** | 0.031 |
| Invasion of the Martians (166) | low | 0.7% | 0.010 | 0.028 | 0.019 |

### What works

- **Poor Girl** scores 2x higher than everything else. The ending ("the wife had an abortion, just in case") is a paradigmatic factual realignment — one sentence that reframes the entire story.
- **Invasion of the Martians** correctly scores lowest. The ending confirms the satire without recontextualizing anything.
- **The content/form split reveals mechanism type:**
  - Poor Girl: Content >> Form (factual realignment)
  - Ladies' Lunch: Form >> Content (tonal/emotional shift, not new facts)
  - Two Ruminations: Content >> Form (meta-narrative registers as argument shift)

### Known limitations

- **Negative space endings** (Ladies' Lunch) remain hard. The sentence "Farah and Bridget still mean to figure out some way to go up and see Lotte, maybe in the spring" is devastating to human readers who fill in the absence, but the revision prompt can only work with what's on the page — it doesn't introduce new vocabulary that embeddings can detect.
- **Embedding distance measures lexical novelty of the interpretive shift, not its depth.** Factual realignment introduces new words (abortion, consent, paternity); stylistic shift and negative space don't.
- Scores in the middle range (0.03-0.04) are noisy and hard to separate.

### Ending extraction examples

| Story | Auto-extracted ending | Explanation |
|-------|----------------------|-------------|
| Poor Girl | "No one knows exactly what happened to them — only that, six weeks later, the wife had an abortion, just in case, and a year later she gave birth to a baby girl." | Reframes the story from broken marriage to cyclical entrapment; "just in case" implies lingering suspicion |
| Ladies' Lunch | "Farah and Bridget still mean to figure out some way to go up and see Lotte, maybe in the spring, when the weather is nicer." | Friends' rescue plans devolve into vague intentions, confirming Lotte's abandonment |
| Two Ruminations | "And expressions of discontent — you think in the car... never solve the riddle of the world, or bring the banality of sequential reality to a location of deeper grace." | Reframes narrative as attempt to transform suffering into art |
| Martians | "The Senator moved the switch to the seventh position, as a kind of triumphant salute, and melancholically left the stage to a standing ovation." | "Melancholically" recontextualizes public triumph as hollow victory |

## Data

`NEWCORPUS_CLEANED/` contains ~5000 short stories (numbered .txt files).

## Usage

```bash
# Single story with auto-extracted ending
python close_reading_two_passes.py --filename 00066.txt

# Single story with manual ending
python close_reading_two_passes.py --filename 00066.txt \
  --ending_text "No one knows exactly what happened to them..."

# Batch run on 6 test stories
python run_batch.py
```

Requires:
- Python 3.10+
- `sentence-transformers`, `scipy`, `requests`, `python-dotenv`
- OpenRouter API key (as `NARRATIVE` in `.env`)
- Local embedding model at `/home/maxzhuyt/models/Qwen3-Embedding-0.6B`

## Pipeline evolution

We iterated through several versions:

| Version | Approach | Poor Girl | Ladies' Lunch | Martians |
|---------|----------|-----------|---------------|----------|
| v1 | Single 9-section reading + revision | 0.059 | 0.024 | 0.014 |
| v2 | Single reading, independent (no revision) | 0.049-0.073 | 0.035-0.048 | 0.026-0.031 |
| v3 | Two-aspect + revision + manual endings | 0.082 | 0.063 | 0.030 |
| **v4** | **Two-aspect + revision + auto endings** | **0.088** | **0.031** | **0.019** |

Key lessons:
- Independent readings uncap signal but also uncap noise. Revision at temperature=0 is the better tradeoff.
- Splitting into Content and Form aspects surfaces mechanism type from the divergence pattern itself.
- Normalizing by ending length distorts results (1/x is too steep). Standardizing the *input* via auto-extraction is better than post-hoc correction.
