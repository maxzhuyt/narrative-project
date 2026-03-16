#!/usr/bin/env python3
"""
Prepare 25 baseline stories:
  00901-00905: Gutenberg fairy tales (Andersen, Three Little Pigs, Little Red Hen)
  00906-00925: Quotev short fiction (≤10 pages, English, complete)
"""
import os, re, html, time, sys
import urllib.request

CORPUS_DIR = "/project/jevans/maxzhuyt/narrative_project/NEWCORPUS_CLEANED"
BASELINE_DIR = os.path.join(CORPUS_DIR, "baseline")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

def fetch(url, retries=3):
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception as e:
            print(f"    retry {attempt+1}: {e}")
            time.sleep(2 * (attempt + 1))
    return None

def clean_gutenberg(text):
    text = re.sub(r'\[Illustration:?\s*[^\]]*\]', '', text)
    text = re.sub(r'VIDEO FROM THE NEW YORKER.*', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    paragraphs = []
    current = []
    for line in text.split('\n'):
        s = line.strip()
        if not s:
            if current:
                paragraphs.append(' '.join(current))
                current = []
            continue
        current.append(s)
    if current:
        paragraphs.append(' '.join(current))
    return '\n\n'.join(paragraphs).strip()

# ═══════════════════════════════════════════════════════════════════════
# 1. GUTENBERG STORIES
# ═══════════════════════════════════════════════════════════════════════
def extract_gutenberg():
    stories = {}

    # pg1597 — Andersen
    with open(os.path.join(BASELINE_DIR, "pg1597.txt")) as f:
        lines = f.readlines()
    stories["00901"] = ("The Emperor's New Clothes",
                        clean_gutenberg(''.join(lines[68:261])))
    stories["00902"] = ("The Little Match Girl",
                        clean_gutenberg(''.join(lines[5237:5334])))
    stories["00903"] = ("The Red Shoes",
                        clean_gutenberg(''.join(lines[5617:5847])))

    # pg18155 — Three Little Pigs
    with open(os.path.join(BASELINE_DIR, "pg18155.txt")) as f:
        lines = f.readlines()
    stories["00904"] = ("The Three Little Pigs",
                        clean_gutenberg(''.join(lines[62:218])))

    # pg18735 — Little Red Hen
    with open(os.path.join(BASELINE_DIR, "pg18735.txt")) as f:
        lines = f.readlines()
    stories["00905"] = ("The Little Red Hen",
                        clean_gutenberg(''.join(lines[88:377])))

    return stories

# ═══════════════════════════════════════════════════════════════════════
# 2. QUOTEV STORIES
# ═══════════════════════════════════════════════════════════════════════
def parse_listing(page_html):
    """Return list of (url, title, word_count, pages) from a Quotev listing page."""
    entries = re.findall(
        r'href="(https://www\.quotev\.com/story/\d+/[^"]+)"[^>]*>([^<]+)</a></h2>'
        r'.*?title="([\d,]+) words"[^>]*>([\d,]+) pages',
        page_html, re.DOTALL
    )
    results = []
    for url, title, wc, pc in entries:
        title = html.unescape(title.strip())
        wc_int = int(wc.replace(',', ''))
        pc_int = int(pc.replace(',', ''))
        results.append((url, title, wc_int, pc_int))
    return results

def fetch_story_text(story_url):
    """Fetch all chapters of a Quotev story and return concatenated text."""
    h = fetch(story_url)
    if not h:
        return None

    # Get chapter list (multi-chapter stories)
    # Single-page stories have text in #rescontent directly
    all_text = []

    # Extract text from current page
    block = re.search(r'id="rescontent"[^>]*>(.*?)</div>\s*<div', h, re.DOTALL)
    if not block:
        block = re.search(r'id="rescontent"[^>]*>(.*?)</div>', h, re.DOTALL)
    if block:
        txt = re.sub(r'<br\s*/?>', '\n', block.group(1))
        txt = re.sub(r'<[^>]+>', ' ', txt)
        txt = html.unescape(txt).strip()
        all_text.append(txt)

    # Check for chapter links (multi-chapter)
    chapter_links = re.findall(
        r'href="(https://www\.quotev\.com/story/\d+/[^/]+/\d+)"',
        h
    )
    # Deduplicate, skip chapter 1 (already fetched)
    seen = {story_url}
    for cl in chapter_links:
        if cl not in seen:
            seen.add(cl)
            time.sleep(0.5)
            ch = fetch(cl)
            if not ch:
                continue
            block = re.search(r'id="rescontent"[^>]*>(.*?)</div>\s*<div', ch, re.DOTALL)
            if not block:
                block = re.search(r'id="rescontent"[^>]*>(.*?)</div>', ch, re.DOTALL)
            if block:
                txt = re.sub(r'<br\s*/?>', '\n', block.group(1))
                txt = re.sub(r'<[^>]+>', ' ', txt)
                txt = html.unescape(txt).strip()
                all_text.append(txt)

    return '\n\n'.join(all_text)

def clean_quotev(text):
    """Clean Quotev story text."""
    # Remove excessive whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    # Normalize line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove leading/trailing whitespace per line
    lines = [l.strip() for l in text.split('\n')]
    text = '\n'.join(lines)
    return text.strip()

def scrape_quotev(n=20):
    """Scrape n short fiction stories from Quotev."""
    candidates = []
    for page in range(1, 15):  # scan up to 14 pages
        url = f"https://www.quotev.com/stories/c/Fiction?v=created&page={page}"
        print(f"  Listing page {page}...")
        h = fetch(url)
        if not h:
            continue
        entries = parse_listing(h)
        for eurl, title, wc, pc in entries:
            # Filter: ≤10 pages, ≥500 words, English
            if pc <= 10 and wc >= 500:
                candidates.append((eurl, title, wc, pc))
        print(f"    {len(entries)} stories, {len(candidates)} candidates so far")
        if len(candidates) >= n * 2:  # get extra in case some fail
            break
        time.sleep(1)

    print(f"\n  {len(candidates)} candidates total, fetching top {n}...")
    stories = {}
    sid = 906

    for eurl, title, wc, pc in candidates:
        if len(stories) >= n:
            break
        print(f"  [{sid:05d}] {title} ({wc}w, {pc}p)...")
        text = fetch_story_text(eurl)
        if not text or len(text.split()) < 300:
            print(f"    SKIP: too short after extraction")
            continue
        text = clean_quotev(text)
        stories[f"{sid:05d}"] = (title, text)
        print(f"    OK: {len(text.split())} words")
        sid += 1
        time.sleep(1)

    return stories

# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("Preparing 25 baseline stories")
    print("=" * 70)

    print("\n[1/2] Extracting Gutenberg stories...")
    gutenberg = extract_gutenberg()
    for sid, (title, text) in gutenberg.items():
        print(f"  {sid}: {title} ({len(text.split())} words)")

    print("\n[2/2] Scraping Quotev short fiction...")
    quotev = scrape_quotev(20)

    all_stories = {**gutenberg, **quotev}

    print(f"\n{'=' * 70}")
    print(f"Saving {len(all_stories)} stories...")
    for sid, (title, text) in sorted(all_stories.items()):
        outpath = os.path.join(CORPUS_DIR, f"{sid}.txt")
        with open(outpath, 'w') as f:
            f.write(f"{title}\n\n{text}\n")
        print(f"  {outpath} ({len(text.split())} words)")

    ids = ','.join(str(int(s)) for s in sorted(all_stories.keys()))
    print(f"\nDone! {len(all_stories)} stories saved.")
    print(f"\nTo run evaluation:")
    print(f"  python run_baseline.py --stories {ids}")

if __name__ == "__main__":
    main()
