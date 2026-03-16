#!/usr/bin/env python3
"""
Scrape 20 short stories from StoryStar.com (fiction category) into BASELINE_CORPUS/.
Stories saved as b0006.txt - b0025.txt (b0001-b0005 are Gutenberg fairy tales).
"""
import os, re, html, time, sys
import urllib.request

BASELINE_DIR = "/project/jevans/maxzhuyt/narrative_project/BASELINE_CORPUS"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

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

def clean_text(raw):
    """Clean extracted story text."""
    # Normalize whitespace
    raw = raw.replace('\r\n', '\n').replace('\r', '\n')
    # Remove tabs
    raw = raw.replace('\t', ' ')
    # Collapse multiple spaces
    raw = re.sub(r'  +', ' ', raw)
    # Strip each line
    lines = [l.strip() for l in raw.split('\n')]
    # Collapse 3+ blank lines to 2
    cleaned = []
    blank_count = 0
    for l in lines:
        if not l:
            blank_count += 1
            if blank_count <= 1:
                cleaned.append('')
        else:
            blank_count = 0
            cleaned.append(l)
    return '\n\n'.join(p for p in '\n'.join(cleaned).split('\n\n') if p.strip())

def extract_story(page_html):
    """Extract title and story text from a StoryStar story page."""
    # Title from <h1>
    title_m = re.search(r'<h1[^>]*>(.*?)</h1>', page_html, re.DOTALL)
    title = html.unescape(re.sub(r'<[^>]+>', '', title_m.group(1)).strip()) if title_m else None

    # Story text from <p> tags (filter out short nav/UI paragraphs)
    blocks = re.findall(r'<p[^>]*>(.*?)</p>', page_html, re.DOTALL)
    story_parts = []
    for b in blocks:
        # Convert <br> to newlines before stripping tags
        txt = re.sub(r'<br\s*/?>', '\n', b)
        txt = re.sub(r'<[^>]+>', '', txt)
        txt = html.unescape(txt).strip()
        # Keep paragraphs that are actual story content (>50 chars)
        if len(txt) > 50:
            story_parts.append(txt)

    text = '\n\n'.join(story_parts)
    text = clean_text(text)
    return title, text

def get_story_urls(n_pages=8):
    """Scrape listing pages for fiction story URLs."""
    urls = []
    for page in range(1, n_pages + 1):
        listing_url = f"https://www.storystar.com/read-short-stories?page={page}"
        print(f"  Listing page {page}...")
        h = fetch(listing_url)
        if not h:
            continue
        # Extract data-redirect URLs (fiction only)
        redirects = re.findall(r'data-redirect="(https://www\.storystar\.com/story/[^"]+)"', h)
        fiction = [u for u in redirects if '/fiction/' in u]
        urls.extend(fiction)
        print(f"    {len(fiction)} fiction stories on this page")
        if len(urls) >= 40:  # get extras in case some are too short
            break
        time.sleep(0.5)
    # Deduplicate preserving order
    return list(dict.fromkeys(urls))

def main():
    print("=" * 60)
    print("Scraping 20 fiction stories from StoryStar.com")
    print(f"Output: {BASELINE_DIR}/b0006.txt - b0025.txt")
    print("=" * 60)

    urls = get_story_urls()
    print(f"\n{len(urls)} fiction story URLs found. Fetching...\n")

    saved = 0
    sid = 6
    for url in urls:
        if saved >= 20:
            break
        h = fetch(url)
        if not h:
            continue
        title, text = extract_story(h)
        wc = len(text.split())
        if not title or wc < 300:
            print(f"  SKIP ({wc}w): {url.split('/')[-1]}")
            continue

        fpath = os.path.join(BASELINE_DIR, f"b{sid:04d}.txt")
        with open(fpath, 'w') as f:
            f.write(f"{title}\n\n{text}\n")
        print(f"  b{sid:04d}: {title} ({wc}w)")
        sid += 1
        saved += 1
        time.sleep(0.5)

    print(f"\nDone! {saved} stories saved to {BASELINE_DIR}/")

if __name__ == "__main__":
    main()
