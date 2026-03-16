"""Batch runner: prediction-based chilling ending evaluation."""
import sys
import os
import json
import concurrent.futures

sys.path.insert(0, os.path.dirname(__file__))
from close_reading_two_passes import evaluate_story, plot_results, DATA_DIR

STORIES = {
    "00066": {"title": "Poor Girl",                      "expected": "high"},
    "00144": {"title": "Ladies' Lunch",                  "expected": "high"},
    "00135": {"title": "Two Ruminations",                "expected": "moderate"},
    "00015": {"title": "The Fellow",                     "expected": "low"},
    "00056": {"title": "Snowing in Greenwich Village",   "expected": "low"},
    "00166": {"title": "Invasion of the Martians",       "expected": "low"},
}


def process_story(story_id, info):
    file_path = os.path.join(DATA_DIR, f"{story_id}.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    print(f"\n[{story_id}] === {info['title']} (expected: {info['expected']}) ===")
    results = evaluate_story(full_text)

    # Save per-story
    results_dir = os.path.join("results", story_id)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    return story_id, info["title"], info["expected"], results


def main():
    all_stories = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(process_story, sid, info): sid
            for sid, info in STORIES.items()
        }
        for future in concurrent.futures.as_completed(futures):
            all_stories.append(future.result())

    # Sort by expected category for display
    order = {"high": 0, "moderate": 1, "low": 2}
    all_stories.sort(key=lambda r: (order.get(r[2], 3), r[1]))

    # --- Summary table ---
    print("\n" + "=" * 95)
    print(f"{'Story':<28} {'Exp':<8} "
          f"{'L1-emb':>7} {'L2-emb':>7} {'L3-emb':>7} "
          f"{'L4-emb':>7} {'L5-emb':>7} | "
          f"{'L1-llm':>7} {'L5-llm':>7}")
    print("-" * 95)
    for story_id, title, expected, results in all_stories:
        embs = [r.get("embedding_mean", 0) for r in results]
        llms = [r.get("llm_mean", 0) for r in results]
        emb_str = " ".join(f"{e:>7.3f}" for e in embs)
        print(f"{title:<28} {expected:<8} {emb_str} | "
              f"{llms[0]:>7.1f} {llms[-1]:>7.1f}")
    print("=" * 95)

    # --- Detailed per-story output ---
    for story_id, title, expected, results in all_stories:
        print(f"\n  {title} ({expected}):")
        print(f"  {'Level':<6} {'EndPct':>6} {'Sents':>5} "
              f"{'Emb-Mean':>9} {'Emb-Min':>9} "
              f"{'LLM-Mean':>9} {'LLM-Max':>9}")
        print(f"  {'-'*55}")
        for r in results:
            if "error" in r:
                print(f"  {r['level']:<6} ERROR")
                continue
            print(f"  {r['level']:<6} {r['ending_pct']:>5.1f}% {r['n_sentences']:>5} "
                  f"{r['embedding_mean']:>9.4f} {r['embedding_min']:>9.4f} "
                  f"{r['llm_mean']:>9.1f} {r['llm_max']:>9.1f}")

    # --- Generate plots ---
    print("\nGenerating plots...")
    plot_results(all_stories)

    # --- Save combined summary ---
    os.makedirs("results", exist_ok=True)
    summary = {}
    for story_id, title, expected, results in all_stories:
        summary[story_id] = {
            "title": title,
            "expected": expected,
            "levels": [
                {k: v for k, v in r.items()
                 if k not in ("actual_ending", "generated_endings")}
                for r in results
            ],
        }
    with open("results/batch_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    print("Summary saved to results/batch_summary.json")


if __name__ == "__main__":
    main()
