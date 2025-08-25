# -*- coding: utf-8 -*-
import nltk
import pandas as pd

# ----------------- Settings -----------------
WORDS_PER_SCALE = 100.0  # scale to "per X tokens"
MIN_MEMBER_TOTAL = 40    # RAW count threshold for synonym significance
MIN_PAIR_RATIO   = 0.01  # min ratio for top2 counts in a synset
TOP_SYNONYM_SETS = 150   # number of synonym sets to show
PARQUET_FILE = "/Users/sam/Documents/Code/review polarity etymology/reviews.parquet"
DBPATH       = "/Users/sam/Documents/Code/review polarity etymology/etymology.sqlite"

# <<< NEW hOPTION: choose unigrams (1) or bigrams (2) >>>
NGRAM_N = 1  # set to 2 to analyze two-word sequences

# <<< NEW OPTION: print full lemma list for this language (or None to skip) >>>
PRINT_LEMMAS_FOR_LANGUAGE = "Arabic"   # e.g. "Latin" or "French" or None

# tqdm (optional)
try:
    from tqdm import tqdm
    def _tqdm(iterable=None, **kw):
        kw.setdefault("mininterval", 0.3)
        kw.setdefault("leave", False)
        return tqdm(iterable, **kw) if iterable is not None else tqdm(**kw)
except Exception:
    def _tqdm(iterable=None, **kw):
        return iterable

# ---- One-time downloads ----
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from functools import lru_cache
from heapq import nlargest
from operator import itemgetter
from scipy.stats import chi2_contingency
import sqlite3
from collections import defaultdict

# ----------------- Utilities -----------------
STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def penn_to_wn(tag):
    t = tag[0].upper() if tag else ''
    if t == 'J': return wn.ADJ
    if t == 'V': return wn.VERB
    if t == 'N': return wn.NOUN
    if t == 'R': return wn.ADV
    return None

def _lemmatized_tokens(text):
    """Tokenize -> POS tag -> lemmatize -> filter punctuation/stopwords/short,
       and exclude proper nouns (NNP, NNPS)."""
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    out = []
    for word, tag in tagged:
        word = word.lower()
        if not word.isalpha():
            continue
        if word in STOPWORDS:
            continue
        if tag in ("NNP", "NNPS"):  # <<< NEW: skip proper nouns
            continue
        wn_pos = penn_to_wn(tag) or wn.NOUN
        lemma = lemmatizer.lemmatize(word, pos=wn_pos)
        if len(lemma) < 2:
            continue
        out.append(lemma)
    return out


def tokenize_units(text, n=NGRAM_N):
    """Emit either unigrams (n=1) or bigrams (n=2) as strings."""
    toks = _lemmatized_tokens(text)
    if n == 1:
        return toks
    elif n == 2:
        return [f"{toks[i]}_{toks[i+1]}" for i in range(len(toks)-1)]
    else:
        raise ValueError("Only n=1 (unigram) or n=2 (bigram) supported.")

def calculate_bar_size(max_val, val):
    if max_val <= 0: return 0
    return int(25 * float(abs(val)) / float(abs(max_val)))

def make_bar(max_val, val, sign_val=None):
    length = calculate_bar_size(max_val, val)
    sign_source = val if sign_val is None else sign_val
    ch = '+' if sign_source >= 0 else '-'
    return ch * length

def fmt_count(x):
    return f"{x:.3f}/({int(WORDS_PER_SCALE)}w)"

def sort_freqs(freqs):
    return sorted(freqs.items(), key=lambda item: item[1], reverse=True)

# ------------- Load and process parquet file -------------
try:
    df = pd.read_parquet(PARQUET_FILE)
    print(f"Loaded parquet file: {PARQUET_FILE}")
    print(f"Dataset shape: {df.shape}")

    if 'text' not in df.columns:
        raise ValueError("Column 'text' not found in parquet file")

    # Find a binary sentiment column
    sentiment_col = None
    for col in df.columns:
        if col != 'text':
            unique_vals = df[col].unique()
            if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                sentiment_col = col
                break
    if sentiment_col is None:
        raise ValueError("No binary sentiment column (0/1 values) found")

    print(f"Using sentiment column: '{sentiment_col}'")

    # Sample for speed
    sample_size = max(1, int(len(df) * 1))
    df_sample = df.sample(n=sample_size, random_state=42)

    positive_reviews = df_sample[df_sample[sentiment_col] == 1]['text'].tolist()
    negative_reviews = df_sample[df_sample[sentiment_col] == 0]['text'].tolist()

except FileNotFoundError:
    print(f"Error: Parquet file '{PARQUET_FILE}' not found. Using sample data.")
    sample_data = {
        'text': [
            "This movie was absolutely fantastic! Great acting and amazing plot.",
            "Terrible film, waste of time. Poor acting and boring story.",
            "I loved every minute of this movie. Highly recommended!",
            "Disappointing movie. Expected much better from this director.",
            "Outstanding performance by the lead actor. Brilliant cinematography.",
        ],
        'sentiment': [1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(sample_data)
    positive_reviews = df[df['sentiment'] == 1]['text'].tolist()
    negative_reviews = df[df['sentiment'] == 0]['text'].tolist()

print(f"There are {len(positive_reviews)} positive reviews and {len(negative_reviews)} negative reviews")
print(f"Normalization: per {int(WORDS_PER_SCALE)} tokens")
print(f"N-gram setting: {NGRAM_N} ({'unigrams' if NGRAM_N==1 else 'bigrams'})")

# ------------- Build raw counts --------------
raw_pos, raw_neg = {}, {}
total_pos = 0
total_neg = 0

print("\nProcessing positive reviews...")
for review_text in _tqdm(positive_reviews, desc="Positive reviews"):
    units = tokenize_units(review_text, n=NGRAM_N)
    total_pos += len(units)
    for u in units:
        raw_pos[u] = raw_pos.get(u, 0) + 1

print("Processing negative reviews...")
for review_text in _tqdm(negative_reviews, desc="Negative reviews"):
    units = tokenize_units(review_text, n=NGRAM_N)
    total_neg += len(units)
    for u in units:
        raw_neg[u] = raw_neg.get(u, 0) + 1


# Normalized per-unit counts
scale_pos = WORDS_PER_SCALE / total_pos if total_pos > 0 else 0
scale_neg = WORDS_PER_SCALE / total_neg if total_neg > 0 else 0
pos_freqs = {w: c * scale_pos for w, c in raw_pos.items()}
neg_freqs = {w: c * scale_neg for w, c in raw_neg.items()}

# ------------- Top tokens ---------------------
def print_freqs(token_freq_pairs):
    if not token_freq_pairs:
        print("(none)")
        return
    max_freq = token_freq_pairs[0][1]
    max_len = max(len(tok) for tok, _ in token_freq_pairs)
    for tok, freq in token_freq_pairs:
        bar = make_bar(max_freq, freq, sign_val=+1)
        print(f"{tok.ljust(max_len)} | {bar} ({fmt_count(freq)})")
        if len(bar) == 0:
            break

label = "Tokens" if NGRAM_N == 2 else "Words"
print(f"\nPositive Review Top {label} (lemmatized & per-token normalized)")
pos_top = sort_freqs(pos_freqs)
print_freqs(pos_top[:25])

print(f"\nNegative Review Top {label} (lemmatized & per-token normalized)")
neg_top = sort_freqs(neg_freqs)
print_freqs(neg_top[:25])

# ------------- Differences -------------------
diff_rows = []
all_units = set(pos_freqs) | set(neg_freqs)
for u in all_units:
    p = pos_freqs.get(u, 0.0)
    n = neg_freqs.get(u, 0.0)
    diff = p - n
    total = p + n
    pct_diff = (diff / total * 100.0) if total > 0 else 0.0
    diff_rows.append((u, p, n, diff, pct_diff))

print(f"\nTop 25 {label} used MORE positively than negatively")
more_pos = [row for row in diff_rows if row[3] > 0]
more_pos.sort(key=lambda x: x[3], reverse=True)
if more_pos:
    max_tok_len = max(len(w) for w, *_ in more_pos[:25])
    max_diff = more_pos[0][3]
    for w, p, n, diff, pct in more_pos[:25]:
        bar = make_bar(max_diff, diff)
        print(f"{w.ljust(max_tok_len)} | {bar} (pos: {fmt_count(p)}, neg: {fmt_count(n)}, abs diff: {fmt_count(diff)}, pct diff: {pct:.1f}%)")
else:
    print("(none)")

print(f"\nTop 25 {label} used MORE negatively than positively")
more_neg = [row for row in diff_rows if row[3] < 0]
more_neg.sort(key=lambda x: x[3])
if more_neg:
    max_neg_mag = abs(more_neg[0][3])
    max_tok_len = max(len(w) for w, *_ in more_neg[:25])
    for w, p, n, diff, pct in more_neg[:25]:
        bar = make_bar(max_neg_mag, diff)
        print(f"{w.ljust(max_tok_len)} | {bar} (pos: {fmt_count(p)}, neg: {fmt_count(n)}, abs diff: {fmt_count(diff)}, pct diff: {pct:.1f}%)")
else:
    print("(none)")

# -------- Synonym disagreement analysis (only for unigrams) -------
if NGRAM_N == 1:
    def primary_synset(word):
        for pos in (wn.NOUN, wn.ADJ, wn.VERB, wn.ADV, None):
            syns = synsets_cached(word, pos)
            if syns:
                return syns[0]
        return None

    @lru_cache(maxsize=None)
    def synsets_cached(word, pos):
        return wn.synsets(word, pos=pos) if pos else wn.synsets(word)

    diff_index = {row[0]: row for row in diff_rows}
    raw_totals = {w: raw_pos.get(w, 0) + raw_neg.get(w, 0) for w in (set(raw_pos) | set(raw_neg))}
    candidate_lemmas = [w for w, tot in raw_totals.items() if tot >= MIN_MEMBER_TOTAL]

    groups = {}
    for lemma in candidate_lemmas:
        row = diff_index.get(lemma)
        if row is None: continue
        syn = primary_synset(lemma)
        if syn is None: continue
        w, p, n, diff, pct = row
        total_raw = raw_totals.get(lemma, 0)
        key = syn.name()
        bucket = groups.setdefault(key, {"syn": syn, "members": []})
        bucket["members"].append((w, p, n, diff, pct, total_raw))

    def has_two_significant_members_nosort(members):
        top1 = top2 = -1
        for m in members:
            t = m[5]
            if t > top1:
                top2 = top1; top1 = t
            elif t > top2:
                top2 = t
        if top2 < 0: return False
        if top1 < MIN_MEMBER_TOTAL or top2 < MIN_MEMBER_TOTAL: return False
        if top2 / max(top1, 1) < MIN_PAIR_RATIO: return False
        return True

    groups = {k: v for k, v in groups.items() if len(v["members"]) >= 2 and has_two_significant_members_nosort(v["members"])}

    ranked = []
    for g in groups.values():
        diffs = (m[3] for m in g["members"])
        it = iter(diffs)
        try:
            first = next(it)
        except StopIteration:
            continue
        dmin = dmax = first
        for d in it:
            if d < dmin: dmin = d
            if d > dmax: dmax = d
        spread = dmax - dmin
        ranked.append((spread, g))

    top_groups = nlargest(TOP_SYNONYM_SETS, ranked, key=itemgetter(0))

    def calculate_statistical_significance(members, total_pos_words, total_neg_words):
        results = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                word1 = members[i]
                word2 = members[j]
                w1_pos = raw_pos.get(word1[0], 0); w1_neg = raw_neg.get(word1[0], 0)
                w2_pos = raw_pos.get(word2[0], 0); w2_neg = raw_neg.get(word2[0], 0)
                contingency_table = [[w1_pos, w1_neg],[w2_pos, w2_neg]]
                if any(cell < 5 for row in contingency_table for cell in row):
                    continue
                try:
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                    n = sum(sum(row) for row in contingency_table)
                    cramers_v = (chi2 / n) ** 0.5
                    results.append({
                        'word1': word1[0], 'word2': word2[0],
                        'chi2': chi2, 'p_value': p_value, 'cramers_v': cramers_v,
                        'w1_pos': w1_pos, 'w1_neg': w1_neg, 'w2_pos': w2_pos, 'w2_neg': w2_neg,
                        'significant': p_value < 0.05
                    })
                except ValueError:
                    continue
        return results

    print(f"\nTop {TOP_SYNONYM_SETS} synonym sets with greatest internal polarity disagreement")
    print(f"(thresholds on RAW totals: MIN_MEMBER_TOTAL={MIN_MEMBER_TOTAL}, MIN_PAIR_RATIO={MIN_PAIR_RATIO})")

# ----------------- Etymology via SQLite (only for unigrams) -----------------
def _open_ety_conn(dbpath=DBPATH):
    try:
        conn = sqlite3.connect(dbpath)
        conn.row_factory = lambda cur, row: row[0]
        return conn
    except Exception:
        return None

_etym_conn = _open_ety_conn()
_etym_cache = {}

def get_origin_language(token: str):
    """Return [origin] from SQLite index, or [] if not found/available.
       For bigrams, returns []."""
    if NGRAM_N != 1:
        return []  # skip for bigrams
    wl = token.lower()
    if wl in _etym_cache:
        return _etym_cache[wl]
    if _etym_conn is None:
        _etym_cache[wl] = []
        return []
    try:
        row = _etym_conn.execute("SELECT origin FROM ety WHERE word=?", (wl,)).fetchone()
        res = [row] if row else []
        _etym_cache[wl] = res
        return res
    except sqlite3.Error:
        _etym_cache[wl] = []
        return []

# ----------------- Main reporting -----------------
all_significant_pairs = []

if NGRAM_N == 1:
    # Only for unigrams: print synset groups + significance, with etymology hints
    # (rebuild 'top_groups' if not already present in scope due to conditional)
    try:
        top_groups
    except NameError:
        top_groups = []

    for spread, g in top_groups:
        syn = g["syn"]
        gloss = syn.definition()
        print(f"\n[{syn.name()}] spread={spread:.3f}  ::  {gloss}")

        members = sorted(g["members"], key=lambda m: m[3], reverse=True)

        max_word_len = 0; max_mag = 1
        for m in members:
            if len(m[0]) > max_word_len: max_word_len = len(m[0])
            if abs(m[3]) > max_mag: max_mag = abs(m[3])

        for w, p, n, d, pct, total_raw in members:
            bar = make_bar(max_mag, d)
            print(f"  {w.ljust(max_word_len)} | {bar} (pos:{fmt_count(p)}, neg:{fmt_count(n)}, diff:{fmt_count(d)}, pct:{pct:.1f}%)")

        sig_results = calculate_statistical_significance(members, total_pos, total_neg)

        if sig_results:
            print(f"\n  Statistical significance tests for [{syn.name()}]:")
            for result in sig_results:
                significance_marker = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['significant'] else ""

                w1_origins = get_origin_language(result['word1'])
                w2_origins = get_origin_language(result['word2'])

                origin_info = ""
                if w1_origins or w2_origins:
                    origin_parts = []
                    origin_parts.append(f"{result['word1']}({w1_origins[0]})" if w1_origins else f"{result['word1']}(?)")
                    origin_parts.append(f"{result['word2']}({w2_origins[0]})" if w2_origins else f"{result['word2']}(?)")
                    origin_info = f" [{' vs '.join(origin_parts)}]"

                print(f"    {result['word1']} vs {result['word2']}: χ²={result['chi2']:.3f}, p={result['p_value']:.4f} {significance_marker}, Cramér's V={result['cramers_v']:.3f}{origin_info}")

            all_significant_pairs.extend([r for r in sig_results if r['significant']])

    # Summary
    print(f"\n{'='*80}")
    print("STATISTICAL SIGNIFICANCE SUMMARY")
    print(f"{'='*80}")
    print(f"Found {len(all_significant_pairs)} statistically significant synonym pairs (p < 0.05)")

    if all_significant_pairs:
        all_significant_pairs.sort(key=lambda x: x['p_value'])
        print("\nMost statistically significant synonym pairs:")
        print("(*** p<0.001, ** p<0.01, * p<0.05)")
        for i, result in enumerate(all_significant_pairs[:20], 1):
            significance_marker = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*"
            w1_origins = get_origin_language(result['word1'])
            w2_origins = get_origin_language(result['word2'])
            origin_info = ""
            if w1_origins or w2_origins:
                origin_parts = []
                origin_parts.append(f"{result['word1']}({w1_origins[0]})" if w1_origins else f"{result['word1']}(?)")
                origin_parts.append(f"{result['word2']}({w2_origins[0]})" if w2_origins else f"{result['word2']}(?)")
                origin_info = f" [{' vs '.join(origin_parts)}]"
            w1_pos_rate = (result['w1_pos'] / total_pos) * WORDS_PER_SCALE if total_pos else 0.0
            w1_neg_rate = (result['w1_neg'] / total_neg) * WORDS_PER_SCALE if total_neg else 0.0
            w2_pos_rate = (result['w2_pos'] / total_pos) * WORDS_PER_SCALE if total_pos else 0.0
            w2_neg_rate = (result['w2_neg'] / total_neg) * WORDS_PER_SCALE if total_neg else 0.0

            print(f"{i:2d}. {result['word1']} vs {result['word2']} {significance_marker}{origin_info}")
            print(f"    χ²={result['chi2']:.3f}, p={result['p_value']:.4f}, Cramér's V={result['cramers_v']:.3f}")
            print(f"    {result['word1']}: pos={w1_pos_rate:.3f}/100w, neg={w1_neg_rate:.3f}/100w")
            print(f"    {result['word2']}: pos={w2_pos_rate:.3f}/100w, neg={w2_neg_rate:.3f}/100w")
            print()

    print("\nInterpretation:")
    print("- χ² (chi-square): Higher values indicate stronger association between word choice and sentiment")
    print("- p-value: Probability that the observed difference is due to chance (< 0.05 = significant)")
    print("- Cramér's V: Effect size measure (0-1, where >0.1 is small, >0.3 is medium, >0.5 is large effect)")

    if _etym_conn is None:
        print("- Etymology: SQLite index not available (DB not opened).")
    else:
        try:
            c = _etym_conn.execute("SELECT COUNT(1) FROM ety").fetchone()
            print(f"- Etymology: SQLite index available (rows: {c:,}).")
        except Exception:
            print("- Etymology: SQLite index available.")

# ----------------- Polarity by Origin Language (only for unigrams) -----------------
if NGRAM_N == 1:
    origin_norm = defaultdict(lambda: {"pos": 0.0, "neg": 0.0, "count_words": 0})
    origin_raw  = defaultdict(lambda: {"pos": 0,   "neg": 0,   "count_words": 0})

    for w in _tqdm(all_units, desc="Aggregating by origin"):
        origins = get_origin_language(w)
        if not origins:
            origins = ["Unknown"]
        p_norm = pos_freqs.get(w, 0.0)
        n_norm = neg_freqs.get(w, 0.0)
        p_raw = raw_pos.get(w, 0)
        n_raw = raw_neg.get(w, 0)
        for lang in origins:
            origin_norm[lang]["pos"] += p_norm
            origin_norm[lang]["neg"] += n_norm
            origin_norm[lang]["count_words"] += 1
            origin_raw[lang]["pos"]  += p_raw
            origin_raw[lang]["neg"]  += n_raw

    grand_pos_raw = sum(raw_pos.values())
    grand_neg_raw = sum(raw_neg.values())
    rows = []

    for lang in _tqdm(list(origin_norm.keys()), desc="χ² by origin"):
        if lang == "Unknown":
            continue  # skip Unknown from final report

        pos_n = origin_norm[lang]["pos"]
        neg_n = origin_norm[lang]["neg"]
        diff_n = pos_n - neg_n
        total_n = pos_n + neg_n
        pct_n = (diff_n / total_n * 100.0) if total_n > 0 else 0.0

        pos_r = origin_raw[lang]["pos"]
        neg_r = origin_raw[lang]["neg"]
        other_pos = max(grand_pos_raw - pos_r, 0)
        other_neg = max(grand_neg_raw - neg_r, 0)

        chi2 = p_value = cramers_v = None
        significant = False
        try:
            table = [[pos_r, neg_r],[other_pos, other_neg]]
            if sum(sum(r) for r in table) >= 10:
                chi2, p_val, dof, expected = chi2_contingency(table)
                n_total = sum(sum(r) for r in table)
                cramers_v = (chi2 / n_total) ** 0.5 if n_total > 0 else 0.0
                p_value = p_val
                significant = p_val < 0.05
        except Exception:
            pass

        rows.append({
            "lang": lang,
            "words": origin_norm[lang]["count_words"],
            "pos_norm": pos_n,
            "neg_norm": neg_n,
            "absdiff_norm": diff_n,
            "pctdiff_norm": pct_n,
            "pos_raw": pos_r,
            "neg_raw": neg_r,
            "chi2": chi2,
            "p_value": p_value,
            "cramers_v": cramers_v,
            "significant": significant,
            "total_words": origin_norm[lang]["count_words"]
        })

    rows.sort(key=lambda r: r["total_words"], reverse=True)
    SHOW_TOP_ORIGINS = 30
    rows = rows[:SHOW_TOP_ORIGINS]

    print("\n" + "="*120)
    print(f"SENTIMENT BREAKDOWN BY WORD ORIGIN (Top {SHOW_TOP_ORIGINS} by total words, normalized per 100 tokens, with χ² vs. rest)")
    print("="*120)
    print(f"{'Origin':20} | {'Lemma':>6} | {'Pos/100w':>10} | {'Neg/100w':>10} | {'AbsDiff':>10} | {'PctDiff':>7} | {'χ²':>7} | {'p':>8} | {'V':>5} | Sig")
    print("-"*120)
    for r in rows:
        chi2_str = f"{r['chi2']:.3f}" if r['chi2'] is not None else "  -  "
        p_str    = f"{r['p_value']:.4f}" if r['p_value'] is not None else "   -   "
        v_str    = f"{r['cramers_v']:.3f}" if r['cramers_v'] is not None else "  -  "
        sig_mark = "***" if (r['p_value'] is not None and r['p_value'] < 0.001) else \
                   ("**" if (r['p_value'] is not None and r['p_value'] < 0.01) else \
                   ("*" if r['significant'] else ""))
        print(f"{r['lang']:20} | {r['words']:6d} | {r['pos_norm']:10.3f} | {r['neg_norm']:10.3f} | {r['absdiff_norm']:10.3f} | {r['pctdiff_norm']:6.1f}% | {chi2_str:>7} | {p_str:>8} | {v_str:>5} | {sig_mark}")
else:
    # If you're in bigram mode, explain what’s intentionally skipped.
    print("\nNOTE:")
    print("- Bigram mode (NGRAM_N=2) reports frequency and polarity differences for two-word tokens.")
    print("- Synonym-set (WordNet) analysis and etymology-by-origin breakdown are skipped for bigrams.")

# ----------------- Optional full lemma dump -----------------
if NGRAM_N == 1 and PRINT_LEMMAS_FOR_LANGUAGE:
    target = PRINT_LEMMAS_FOR_LANGUAGE
    lemmas_for_lang = []
    for w in all_units:
        origins = get_origin_language(w)
        if origins and target in origins:
            pos = raw_pos.get(w, 0)
            neg = raw_neg.get(w, 0)
            total = pos + neg
            lemmas_for_lang.append((w, pos, neg, total))
    if lemmas_for_lang:
        lemmas_for_lang.sort(key=lambda x: x[3], reverse=True)
        print(f"\n{'='*80}")
        print(f"FULL LEMMA LIST FOR LANGUAGE: {target}")
        print(f"{'='*80}")
        print(f"{'Lemma':20} | {'Pos':>6} | {'Neg':>6} | {'Total':>6}")
        print("-"*80)
        for w, pos, neg, total in lemmas_for_lang:
            print(f"{w:20} | {pos:6d} | {neg:6d} | {total:6d}")
    else:
        print(f"\n(No lemmas found for language '{target}')")
