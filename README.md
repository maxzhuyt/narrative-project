# Narrative Predictability

Measuring how well a language model can predict story endings as more text is revealed. Unpredictability — the model's persistent inability to anticipate the ending — is used as a proxy for narrative surprise.

---

## What we tried first (and why it didn't work)

### Close-reading + embedding distance (v1–v4)

The original approach asked an LLM to do a two-pass "close reading": first interpret the story without the ending, then revise the interpretation after seeing the ending. Cosine distance between the two readings was supposed to capture how much the ending reshapes meaning.


## Current approach: 100-ending evaluation

At **every sentence boundary**, generate 100 independent possible endings with Qwen3-32B (temperature=1.2, top_p=0.95). Compare each generated ending to the **actual continuation** — the rest of the story from that position — using cosine distance with Qwen3-Embedding-0.6B.

```
for each position i = 1..N-1:
    context = sentences[:i]
    generate 100 endings (Qwen3-32B, temperature=1.2)
    continuation = sentences[i:]   ← the actual text that follows
    mean_cosine_dist = avg cosine_dist(each ending, continuation)

→ plot mean_cosine_dist vs. % tokens revealed
```

**Why this works better:**
- 100 samples per position gives stable mean estimates.
- Sentence-level granularity produces a full narrative curve, not 5 points.
- Comparing against the **continuation** (not just the final sentence) rewards endings that naturally resolve the narrative tension — a more meaningful target than the last sentence alone.
- No LLM judge: pure embedding distance is deterministic given the generations.

**Reading the curves:** High distance = the model cannot anticipate where the story goes. A curve that stays high late in the story = persistently unpredictable ending. A falling curve = the model "sees it coming" as context grows.

---

## Corpora

| Corpus | Location | N | Description |
|--------|----------|---|-------------|
| Literary | `NEWCORPUS_CLEANED/` | 6 | New Yorker short fiction (hand-selected) |
| Baseline (simple) | `BASELINE_CORPUS/b*.txt` | 35 | 5 Gutenberg fairy tales + 30 StoryStar short fiction |
| Baseline (WP human) | `BASELINE_CORPUS/ghostbuster-data/wp/human/` | 50 sampled | Human-authored WritingPrompts fiction |
| Baseline (WP GPT) | `BASELINE_CORPUS/ghostbuster-data/wp/gpt/` | 50 sampled | GPT-generated WritingPrompts fiction |

The Gutenberg stories (b0001–b0005) turned out to be near-memorised by Qwen3 (mean cosine dist ≈ 0.14 vs ≈ 0.20 for StoryStar) and are **not** a reliable unpredictability baseline — the model is essentially recalling the ending rather than predicting it.

---

## Scripts

### Generation

**`run_endings_scaled.py` / `run_endings_scaled.sbatch`**
Generates 100 endings per sentence position for the 6 literary stories in `NEWCORPUS_CLEANED/`. Results saved to `results_scaled/{sid}_endings.json`.

**`run_baseline_eval.py` / `run_baseline_eval.sbatch`**
Same pipeline for the 35 baseline stories in `BASELINE_CORPUS/b*.txt`. Results saved to `BASELINE_CORPUS/results/{sid}_endings.json`.

**`run_wp_eval.py` / `run_wp_eval.sbatch`**
Same pipeline for 50 sampled human + 50 sampled GPT stories from the WritingPrompts (WP) subset of the Ghostbuster dataset. Stories ≤3000 words; positions with <10-word sentences are skipped (deferred to next position). Results saved to `BASELINE_CORPUS/results/wph{001-050}_endings.json` and `wpg{001-050}_endings.json`.

All generation scripts use Qwen3-32B via vLLM on H200 (`--constraint=H200`).

### Embedding distances

**`compute_distances.py` / `compute_distances.sbatch`**
Loads all `_endings.json` result files (literary + all baselines), encodes each generated ending and the actual continuation with Qwen3-Embedding-0.6B, computes mean cosine distance per position, and saves everything to `distances/all_distances.json`. Resume-safe: skips already-processed stories.

Run after any new generation job completes:
```bash
sbatch compute_distances.sbatch
```

### Analysis

**`analysis_predictability.ipynb`**
Loads `distances/all_distances.json` — no GPU required. Produces:
- Per-story narrative curves (% tokens revealed vs. mean cosine distance)
- Peak annotations (top 3 local maxima per story, labelled via DeepSeek API)
- Descriptive statistics comparing literary stories to baseline distribution (z-scores, percentile ranks)

---

## Directory structure

```
narrative_project/
├── analysis_predictability.ipynb   # main analysis notebook (CPU only)
├── compute_distances.py/.sbatch    # embedding distance computation (GPU)
├── run_endings_scaled.py/.sbatch   # literary story generation (GPU)
├── run_baseline_eval.py/.sbatch    # baseline story generation (GPU)
├── run_wp_eval.py/.sbatch          # WritingPrompts generation (GPU)
├── NEWCORPUS_CLEANED/              # ~925 New Yorker stories (.txt)
├── BASELINE_CORPUS/
│   ├── b0001–b0035.txt             # 35 simple baseline stories
│   ├── ghostbuster-data/wp/        # WritingPrompts subset
│   └── results/                    # _endings.json per story
├── results_scaled/                 # _endings.json for 6 literary stories
├── distances/                      # all_distances.json (pre-computed)
├── figures/                        # output PNGs
└── archive/                        # superseded scripts
```
