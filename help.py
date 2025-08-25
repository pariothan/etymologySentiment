# -*- coding: utf-8 -*-
import sqlite3
import json
import re
import gzip
import os

KAIKKI = "/Users/sam/Documents/Code/review polarity etymology/kaikki.org-dictionary-English.jsonl"  # or .jsonl.gz
DBPATH = "/Users/sam/Documents/Code/review polarity etymology/etymology.sqlite"

# Capture the language phrase after "from ", including Proto+hyphen/space sequences.
LANG_RE = re.compile(
    r"\bfrom\s+((?:Proto(?:[\s-]+[A-Z][a-z]+)+|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*))"
)

def iter_lines(path):
    # Works for .jsonl or .jsonl.gz
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
            for line in f:
                if line.strip():
                    yield line
    else:
        with open(path, "r", encoding="utf-8", newline="") as f:
            for line in f:
                if line.strip():
                    yield line

def first_origin(entry):
    texts = []
    et = entry.get("etymology_text")
    if et:
        texts.append(et)
    ets = entry.get("etymology_texts")
    if ets:
        # Some dumps have lists; ensure strings only
        texts.extend(t for t in ets if isinstance(t, str))
    for t in texts:
        m = LANG_RE.search(t or "")
        if m:
            return m.group(1)
    return None

def build_db():
    if os.path.exists(DBPATH):
        os.remove(DBPATH)
    conn = sqlite3.connect(DBPATH)
    cur = conn.cursor()
    cur.executescript("""
    PRAGMA journal_mode=OFF;
    PRAGMA synchronous=OFF;
    PRAGMA temp_store=MEMORY;
    PRAGMA mmap_size=30000000000;
    CREATE TABLE ety(word TEXT PRIMARY KEY, origin TEXT);
    """)
    batch, seen = [], set()
    BATCH_SIZE = 5000
    for i, line in enumerate(iter_lines(KAIKKI), 1):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        w = e.get("word")
        if not w or w in seen:
            continue
        origin = first_origin(e)
        if origin:
            seen.add(w)
            batch.append((w.lower(), origin))
            if len(batch) >= BATCH_SIZE:
                cur.executemany("INSERT OR IGNORE INTO ety(word, origin) VALUES(?,?)", batch)
                conn.commit()
                batch.clear()
        if i % 500_000 == 0:
            print(f"Processed {i:,} lines…")
    if batch:
        cur.executemany("INSERT OR IGNORE INTO ety(word, origin) VALUES(?,?)", batch)
        conn.commit()
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ety_word ON ety(word);")
    conn.commit()
    conn.close()
    print("Built:", DBPATH)

if __name__ == "__main__":
    build_db()
