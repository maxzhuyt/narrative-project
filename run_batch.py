"""Batch runner: auto-extracts endings, then runs two-aspect divergence."""
import sys
import os
import json
import concurrent.futures

sys.path.insert(0, os.path.dirname(__file__))
from close_reading_two_passes import calculate_interpretive_divergence, DATA_DIR

STORIES = {
    "00066": {"title": "Poor Girl",                      "expected": "high"},
    "00144": {"title": "Ladies' Lunch",                  "expected": "high"},
    "00135": {"title": "Two Ruminations",                "expected": "moderate"},
    "00015": {"title": "The Fellow",                     "expected": "low"},
    "00056": {"title": "Snowing in Greenwich Village",   "expected": "low"},
    "00166": {"title": "Invasion of the Martians",       "expected": "low"},
}


def process_story(story_id, info):
    """Process a single story: auto-extract ending, then evaluate."""
    file_path = os.path.join(DATA_DIR, f"{story_id}.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    print(f"[{story_id}] Starting: {info['title']} (expected: {info['expected']})")
    results = calculate_interpretive_divergence(full_text)  # no ending arg

    # Save per-story results
    results_dir = os.path.join("results", story_id)
    os.makedirs(results_dir, exist_ok=True)
    saveable = {k: v for k, v in results.items() if k != "readings"}
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(saveable, f, indent=4)
    for name, text in results.get("readings", {}).items():
        with open(os.path.join(results_dir, f"{name}.json"), "w") as f:
            f.write(text)

    cd = results.get("content_distance", 0)
    fd = results.get("form_distance", 0)
    ef = results.get("ending_fraction", 0)
    print(f"[{story_id}] Done: {info['title']} -> "
          f"content={cd:.4f} form={fd:.4f} end={ef:.1%}")
    return story_id, info, results


def main():
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(process_story, sid, info): sid
            for sid, info in STORIES.items()
        }
        for future in concurrent.futures.as_completed(futures):
            all_results.append(future.result())

    # Sort by global distance descending
    all_results.sort(
        key=lambda r: r[2].get("global_distance", 0), reverse=True)

    # --- Summary table ---
    print("\n" + "=" * 90)
    print(f"{'Story':<28} {'Exp':<8} {'End%':>5} "
          f"{'Content':>8} {'Form':>8} {'Global':>8}")
    print("-" * 90)
    for story_id, info, res in all_results:
        if "error" in res:
            print(f"{info['title']:<28} {info['expected']:<8} {'ERROR'}")
            continue
        ef = res["ending_fraction"]
        print(f"{info['title']:<28} {info['expected']:<8} {ef:>5.1%} "
              f"{res['content_distance']:>8.4f} "
              f"{res['form_distance']:>8.4f} "
              f"{res['global_distance']:>8.4f}")
    print("=" * 90)

    # --- Extracted endings ---
    print("\nExtracted Endings:")
    print("-" * 90)
    for story_id, info, res in all_results:
        if "error" in res:
            continue
        ending = res.get("ending_segment", "")
        expl = res.get("ending_explanation", "")
        preview = ending[:80].replace("\n", " ")
        print(f"  {info['title']:<28} \"{preview}...\"")
        if expl:
            print(f"  {'':28} Why: {expl}")

    # --- Per-section breakdown ---
    print("\nPer-Section Breakdown:")
    print("-" * 90)
    for story_id, info, res in all_results:
        if "error" in res:
            continue
        gd = res["global_distance"]
        print(f"\n  {info['title']} (expected: {info['expected']}, "
              f"global: {gd:.4f}):")
        print(f"    Content:")
        for s, d in res.get("content_sections", {}).items():
            print(f"      {s:<45} {d:.4f}")
        print(f"    Form:")
        for s, d in res.get("form_sections", {}).items():
            print(f"      {s:<45} {d:.4f}")

    # Save
    summary = {}
    for story_id, info, res in all_results:
        summary[story_id] = {
            "title": info["title"],
            "expected": info["expected"],
            **{k: v for k, v in res.items() if k != "readings"},
        }
    os.makedirs("results", exist_ok=True)
    with open("results/batch_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    print("\nSummary saved to results/batch_summary.json")


if __name__ == "__main__":
    main()
