"""
Design overview:
  1. Load cleaned Reddit posts + comments CSVs (repo-relative paths).
  2. Combine title + body into a single 'text' field.
  3. VADER baseline sentiment  ->  vader_sentiment   (primary signal)
  4. TF-IDF auxiliary feature  ->  tfidf_context
       TF-IDF is NOT treated as a sentiment model. It is used two ways:
         (a) Per-document context score: difference in TF-IDF weight mass
             on a small curated positive vs. negative seed vocabulary.
             This is a lightweight directional nudge, not a classifier.
         (b) Top-term analysis: identify the highest TF-IDF terms per team
             for qualitative interpretation (saved to summary).
  5. Sports sentiment placeholder  ->  sports_sentiment
       Returns 0.0 until Kenneth delivers his lexicon.
       Swap in his file at sports_lexicon_path= with zero other changes.
  6. Final score = weighted combination of the three components.
       Weights are constants at the top of combine_sentiment() and are
       labelled as INITIAL DEFAULTS -- tune them after Kenneth's layer
       is integrated and validation results are reviewed.
  7. MLB schedule merged only for validation/sanity checks (win/loss
     direction test). Schedule data never influences sentiment scores.

Outputs:
  results/sentiment_results.csv   -- one row per post/comment, all score cols
  results/sentiment_summary.csv   -- per-team aggregate stats + top TF-IDF terms
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# from nltk.sentiment import SentimentIntensityAnalyzer
from baseball_lexicon import baseball_lexicon

warnings.filterwarnings("ignore")

# =============================================================================
# PATH CONFIG
# =============================================================================
# Root of the repo (directory this script lives in)
# Path of this file (where my .py lives)
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
# paths
REDDIT_DIR = os.path.join(REPO_ROOT, "data", "clean", "reddit")
SCHEDULE_DIR = os.path.join(REPO_ROOT, "data", "clean", "mlb")
OUTPUT_DIR = os.path.join(REPO_ROOT, "results")

print("REPO_ROOT:", REPO_ROOT)                          
print("FILES IN ROOT:", os.listdir(REPO_ROOT))
print("REDDIT_DIR:", REDDIT_DIR)
print("SCHEDULE_DIR:", SCHEDULE_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)
# =============================================================================
# TEAM CONFIG -- all 8 teams
# =============================================================================

TEAM_FILES = {
    "CHW": {
        "posts":    os.path.join(REDDIT_DIR, "CHW_posts.csv"),
        "comments": os.path.join(REDDIT_DIR, "CHW_comments.csv"),
        "schedule": os.path.join(SCHEDULE_DIR, "CHW_2025.csv"),
    },
    "COL": {
        "posts":    os.path.join(REDDIT_DIR, "COL_posts.csv"),
        "comments": os.path.join(REDDIT_DIR, "COL_comments.csv"),
        "schedule": os.path.join(SCHEDULE_DIR, "COL_2025.csv"),
    },
    "MIL": {
        "posts":    os.path.join(REDDIT_DIR, "MIL_posts.csv"),
        "comments": os.path.join(REDDIT_DIR, "MIL_comments.csv"),
        "schedule": os.path.join(SCHEDULE_DIR, "MIL_2025.csv"),
    },
    "MIN": {
        "posts":    os.path.join(REDDIT_DIR, "MIN_posts.csv"),
        "comments": os.path.join(REDDIT_DIR, "MIN_comments.csv"),
        "schedule": os.path.join(SCHEDULE_DIR, "MIN_2025.csv"),
    },
    "NYY": {
        "posts":    os.path.join(REDDIT_DIR, "NYY_posts.csv"),
        "comments": os.path.join(REDDIT_DIR, "NYY_comments.csv"),
        "schedule": os.path.join(SCHEDULE_DIR, "NYY_2025.csv"),
    },
    "PHI": {
        "posts":    os.path.join(REDDIT_DIR, "PHI_posts.csv"),
        "comments": os.path.join(REDDIT_DIR, "PHI_comments.csv"),
        "schedule": os.path.join(SCHEDULE_DIR, "PHI_2025.csv"),
    },
    "TOR": {
        "posts":    os.path.join(REDDIT_DIR, "TOR_posts.csv"),
        "comments": os.path.join(REDDIT_DIR, "TOR_comments.csv"),
        "schedule": os.path.join(SCHEDULE_DIR, "TOR_2025.csv"),
    },
    "WSN": {
        "posts":    os.path.join(REDDIT_DIR, "WSN_posts.csv"),
        "comments": os.path.join(REDDIT_DIR, "WSN_comments.csv"),
        "schedule": os.path.join(SCHEDULE_DIR, "WSN_2025.csv"),
    },
}
# =============================================================================
# SECTION 1 -- DATA LOADING
# =============================================================================
# Candidate column names searched in order, case-insensitively.
# NOTE: Add variants here if Kenneth's files use different names.
_TITLE_CANDIDATES  = ["title"]
_BODY_CANDIDATES   = ["selftext", "body", "text", "content", "comment_body"]
_TIME_CANDIDATES   = ["created_utc", "created", "timestamp", "utc", "time"]
_AUTHOR_CANDIDATES = ["author", "username", "user"]
_UPS_CANDIDATES    = ["ups", "score", "upvotes", "likes"]
_LINK_CANDIDATES   = ["permalink", "url", "link", "post_url"]

def _find_col(df, candidates):
    """Return the first matching column name (case-insensitive). Returns None if not found."""
    lowered = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lowered:
            return lowered[c.lower()]
    return None

def _safe_get(df, candidates, default=""):
    """Extract a Series by candidate column names; return default Series if none found."""
    col = _find_col(df, candidates)
    if col:
        return df[col].fillna(default).astype(str)
    return pd.Series(default, index=df.index, dtype=str)

def load_posts(filepath, team):
    """
    Load cleaned posts CSV, normalize columns defensively.
    Posts have both a title and a body (selftext).
    """
    df = pd.read_csv(filepath, low_memory=False)
    df["team"]        = team
    df["record_type"] = "post"
    df["title_raw"]   = _safe_get(df, _TITLE_CANDIDATES)
    df["body_raw"]    = _safe_get(df, _BODY_CANDIDATES)
    df["created_utc"] = _safe_get(df, _TIME_CANDIDATES, default=np.nan)
    df["author"]      = _safe_get(df, _AUTHOR_CANDIDATES)
    df["ups"]         = pd.to_numeric(_safe_get(df, _UPS_CANDIDATES, default="0"),
                                       errors="coerce").fillna(0).astype(int)
    df["permalink"]   = _safe_get(df, _LINK_CANDIDATES)
    return df[["team", "record_type", "author", "created_utc",
               "title_raw", "body_raw", "ups", "permalink"]]

def load_comments(filepath, team):
    """
    Load cleaned comments CSV, normalize columns defensively.
    Comments have no title -- title_raw is left empty.
    """
    df = pd.read_csv(filepath, low_memory=False)
    df["team"]        = team
    df["record_type"] = "comment"
    df["title_raw"]   = ""
    df["body_raw"]    = _safe_get(df, _BODY_CANDIDATES)
    df["created_utc"] = _safe_get(df, _TIME_CANDIDATES, default=np.nan)
    df["author"]      = _safe_get(df, _AUTHOR_CANDIDATES)
    df["ups"]         = pd.to_numeric(_safe_get(df, _UPS_CANDIDATES, default="0"),
                                       errors="coerce").fillna(0).astype(int)
    df["permalink"]   = _safe_get(df, _LINK_CANDIDATES)
    return df[["team", "record_type", "author", "created_utc",
               "title_raw", "body_raw", "ups", "permalink"]]

def _clean_text(s):
    """Strip URLs, markdown artifacts, and collapse whitespace."""
    s = re.sub(r"http\S+", " ", str(s))           # remove URLs
    s = re.sub(r"\[.*?\]\(.*?\)", " ", s)          # remove markdown links
    s = re.sub(r"[^\w\s.,!?'-]", " ", s)           # strip misc symbols
    s = re.sub(r"\s+", " ", s).strip()
    return s

def combine_title_body(df):
    """
    Concatenate title + body into a single 'text' field, then clean.
    For comments, title_raw is empty so text == body only.
    """
    df = df.copy()
    df["text"] = (df["title_raw"] + " " + df["body_raw"]).str.strip()
    df["text"] = df["text"].apply(_clean_text)
    return df

def utc_to_date(utc_series):
    """Convert UNIX UTC integer timestamps to YYYY-MM-DD strings."""
    numeric = pd.to_numeric(utc_series, errors="coerce")
    return pd.to_datetime(numeric, unit="s", utc=True).dt.date.astype(str)


# =============================================================================
# SECTION 2 -- VADER BASELINE  (primary sentiment signal)
# =============================================================================

_vader = SentimentIntensityAnalyzer()

def compute_vader(texts):
    """
    Apply VADER to each text and return compound scores in [-1, 1].

    VADER interpretation:
      compound >  0.05  -> positive
      compound < -0.05  -> negative
      otherwise         -> neutral

    VADER is used as the primary baseline because it is well-validated
    for short, informal, social-media text and requires no training data.
    """
    return texts.apply(lambda t: _vader.polarity_scores(t)["compound"])


# =============================================================================
# SECTION 3 -- TF-IDF AUXILIARY FEATURE
#
# What TF-IDF is doing here and what it is NOT doing:
#   TF-IDF measures term importance within a document relative to the corpus.
#   It does not inherently encode sentiment polarity.
#
#   We use it in two ways:
#     (a) Per-document context score (tfidf_context column):
#           Sum of TF-IDF weights on a small curated positive seed vocab
#           minus sum on a negative seed vocab, then normalized via tanh.
#           This is a directional nudge, not a classifier. It captures
#           cases where domain-specific positive/negative words appear
#           prominently in a document, which VADER may underweight.
#     (b) Top-term analysis (saved to summary CSV):
#           Mean TF-IDF per term across all documents for each team.
#           Used qualitatively to understand what topics drive each fanbase.
#           Not used in the sentiment score.
# =============================================================================

# Positive and negative seed word sets.
# These are general-purpose starters. Extend them after reviewing top TF-IDF
# terms in the summary output to add baseball-specific vocabulary.
POSITIVE_SEEDS = {
    "win", "won", "great", "good", "best", "amazing", "excellent",
    "love", "awesome", "happy", "excited", "proud", "hope", "nice",
    "solid", "strong", "improved", "comeback", "clutch", "fun",
    "fantastic", "perfect", "beautiful", "brilliant", "incredible",
    "dominant", "dominant", "impressive", "optimistic", "promising"
}

NEGATIVE_SEEDS = {
    "lose", "lost", "bad", "terrible", "awful", "horrible", "worst",
    "hate", "disappointing", "disappointed", "sad", "angry", "mad",
    "pathetic", "fired", "bust", "blew", "collapse", "injury",
    "injured", "sucks", "embarrassing", "frustrating", "frustration",
    "disgrace", "disgraceful", "useless", "brutal", "disaster",
    "struggling", "struggle", "worried", "concerning", "mess"
}

def build_tfidf_matrix(texts):
    """
    Fit TF-IDF on the full corpus across all teams.
    Returns (fitted vectorizer, sparse document-term matrix).

    Parameters chosen for short Reddit text:
      max_features=8000  -- keeps vocabulary manageable
      min_df=3           -- drops hapax legomena and near-hapax
      ngram_range=(1,2)  -- unigrams + bigrams to catch phrases like "walk off"
      sublinear_tf=True  -- log-scales term frequency to dampen very common terms
    """
    vectorizer = TfidfVectorizer(
        max_features=8000,
        min_df=3,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    matrix = vectorizer.fit_transform(texts.fillna(""))
    return vectorizer, matrix

def compute_tfidf_context_score(texts, vectorizer, matrix):
    """
    Per-document context score derived from TF-IDF weights.

    For each document:
      raw = sum(TF-IDF weights for positive seed terms present in vocab)
          - sum(TF-IDF weights for negative seed terms present in vocab)
      tfidf_context = tanh(scale * raw)   ->  in (-1, 1)

    tanh normalization keeps extreme values bounded and maps most typical
    documents near zero, where the score should be when seed terms are absent.

    This is an AUXILIARY feature, not a replacement for VADER.
    It contributes only 10% of the final score by default.
    """
    feature_names = vectorizer.get_feature_names_out()
    pos_idx = [i for i, t in enumerate(feature_names) if t in POSITIVE_SEEDS]
    neg_idx = [i for i, t in enumerate(feature_names) if t in NEGATIVE_SEEDS]

    # Use sparse column slicing to avoid materializing the full dense matrix.
    # matrix[:, idx].sum(axis=1) operates on sparse columns and stays in memory.
    n_docs = matrix.shape[0]
    pos_mass = np.asarray(matrix[:, pos_idx].sum(axis=1)).flatten() if pos_idx \
               else np.zeros(n_docs)
    neg_mass = np.asarray(matrix[:, neg_idx].sum(axis=1)).flatten() if neg_idx \
               else np.zeros(n_docs)
    raw = pos_mass - neg_mass

    # scale=5 spreads values that are typically in [0, 0.3] across the tanh curve
    normed = np.tanh(raw * 5)
    return pd.Series(normed, index=texts.index)

def get_top_tfidf_terms(df, vectorizer, matrix, top_n=20):
    """
    For each team, compute mean TF-IDF weight per term and return
    the top_n terms. Used for qualitative analysis in the summary only.
    Returns a dict: { team -> [(term, mean_weight), ...] }
    """
    feature_names = vectorizer.get_feature_names_out()
    results = {}
    for team in df["team"].unique():
        team_idx = df[df["team"] == team].index.tolist()
        team_matrix = matrix[team_idx]
        # .mean(axis=0) on a sparse matrix returns a (1, n_features) matrix
        mean_weights = np.asarray(team_matrix.mean(axis=0)).flatten()
        top_idx = mean_weights.argsort()[::-1][:top_n]
        results[team] = [(feature_names[i], round(float(mean_weights[i]), 5))
                         for i in top_idx]
    return results

# =============================================================================
# SECTION 4 -- SPORTS SENTIMENT PLACEHOLDER
# Kenneth's layer goes here. Do not modify the function signatures.
# =============================================================================

# nltk.download("vader_lexicon", quiet=True)

# Initialize VADER analyzer
sia = SentimentIntensityAnalyzer()

# Update VADER with custom lexicon
sia.lexicon.update(baseball_lexicon)

def extract_phrases(text, lexicon):
    matched = []
    
    # Sort phrases by length (longest first)
    phrases = sorted(
        [k for k in lexicon if " " in k],
        key=lambda x: len(x.split()),
        reverse=True
    )
    
    for phrase in phrases:
        if phrase in text:
            matched.append(phrase)
            text = text.replace(phrase, "")  # prevent double counting
    
    return matched, text

def score_custom_lexicon(text):
    score = 0.0
    
    # Extract phrases first
    phrases, remaining_text = extract_phrases(text, baseball_lexicon)
    
    for phrase in phrases:
        score += baseball_lexicon.get(phrase, 0.0)
    
    # Token-level scoring
    tokens = re.findall(r"\b\w+\b", remaining_text)
    
    for token in tokens:
        score += baseball_lexicon.get(token, 0.0)
    
    return score

def compute_sports_sentiment(texts):
    """
    Uses custom baseball-aware sentiment logic.
    Returns normalized score in [-1, 1] per text.
    """

    # Lowercase
    texts = texts.astype(str).str.lower()

    # VADER scores
    # vader_scores = texts.apply(lambda t: sia.polarity_scores(t)["compound"])

    # Custom lexicon scores
    custom_scores = texts.apply(lambda t: score_custom_lexicon(t))

    # Normalize custom scores
    custom_scores_norm = custom_scores.apply(lambda s: max(min(s / 10, 1), -1))

    # Clip to valid range just in case
    return custom_scores_norm

# =============================================================================
# SECTION 5 -- FINAL SENTIMENT SCORE
# =============================================================================

def combine_sentiment(vader, sports, tfidf_ctx):
    """
    Weighted combination of all three sentiment components.

    *** INITIAL DEFAULT WEIGHTS -- tune these after validation ***

    W_VADER  = 0.60
      VADER is the most reliable general-purpose signal.
      It handles negation, emphasis, and punctuation well for social text.

    W_SPORTS = 0.30
      Reserved for Kenneth's domain-specific layer.
      Currently contributes 0.0 everywhere (placeholder is neutral).
      RAISE this weight (e.g., to 0.40-0.50) after his layer is integrated
      and validated against the VADER baseline.

    W_TFIDF  = 0.10
      Lightweight auxiliary nudge from seed-word TF-IDF mass.
      Keep low: it is an imprecise proxy, not a trained signal.

    Weights must sum to 1.0.
    Final score is clipped to [-1, 1].
    """
    W_VADER  = 0.60   # <-- tune here
    W_SPORTS = 0.30   # <-- raise after Kenneth's layer is in
    W_TFIDF  = 0.10   # <-- keep low; auxiliary only

    assert abs(W_VADER + W_SPORTS + W_TFIDF - 1.0) < 1e-9, \
        "Weights must sum to 1.0"

    final = W_VADER * vader + W_SPORTS * sports + W_TFIDF * tfidf_ctx
    return final.clip(-1, 1)

# =============================================================================
# SECTION 6 -- SCHEDULE LOADING & MERGE
# Used ONLY for validation. Schedule data does NOT affect sentiment scores.
# =============================================================================

def load_schedule(filepath, team):
    """
    Load game schedule/results CSV for one team.
    Expects at minimum a Date column and a win column (1/0).
    """
    try:
        df = pd.read_csv(filepath, parse_dates=["Date"])
    except Exception as e:
        print(f"  [WARN] Could not load schedule for {team}: {e}")
        return pd.DataFrame()
    df["team"] = team
    df = df.rename(columns={"Date": "date_sched"})
    df["date_sched"] = df["date_sched"].dt.date.astype(str)
    keep = [c for c in ["team", "date_sched", "win", "run_diff", "home", "W/L"]
            if c in df.columns]
    return df[keep]

def merge_with_schedule(reddit_df, schedule_df):
    """
    Left-join Reddit records to schedule on (team, date).
    Only used to check directional validity in validation output.
    """
    if schedule_df.empty:
        return reddit_df
    return reddit_df.merge(
        schedule_df,
        left_on=["team", "date"],
        right_on=["team", "date_sched"],
        how="left"
    )

# =============================================================================
# SECTION 7 -- VALIDATION
# =============================================================================

def validate_sentiment(df, top_terms_by_team=None):
    """
    Checks:
      1. 5 random sample rows with text and all score columns
      2. Mean sentiment by team (all four columns)
      3. Mean final_sentiment by team x week
      4. Most positive and most negative records
      5. (If schedule merged) does sentiment track wins vs. losses?
      6. Score distribution: % positive / neutral / negative per team
    """
    print("\n" + "=" * 65)
    print("VALIDATION REPORT")
    print("=" * 65)

    sample_cols = ["team", "record_type", "date", "text",
                   "vader_sentiment", "sports_sentiment",
                   "tfidf_context", "final_sentiment"]

    # 1. Sample rows
    print("\n[1] 5 random sample records:")
    print(df[sample_cols].sample(min(5, len(df)), random_state=42)
          .to_string(index=False, max_colwidth=55))

    # 2. Mean by team
    print("\n[2] Mean sentiment scores by team:")
    print(df.groupby("team")[
        ["vader_sentiment", "sports_sentiment", "tfidf_context", "final_sentiment"]
    ].mean().round(4).to_string())

    # 3. Mean final_sentiment by team x week
    print("\n[3] Mean final_sentiment by team x week:")
    tmp = df.copy()
    tmp["week"] = pd.to_datetime(tmp["date"], errors="coerce").dt.to_period("W")
    weekly = (tmp.groupby(["team", "week"])["final_sentiment"]
                 .mean().round(4).reset_index())
    print(weekly.to_string(index=False))

    # 4. Extreme records
    print("\n[4a] Most POSITIVE records (top 5):")
    for _, row in df.nlargest(5, "final_sentiment").iterrows():
        print(f"  [{row['team']} | {row['date']} | score={row['final_sentiment']:.3f}]")
        print(f"  {str(row['text'])[:115]}\n")

    print("[4b] Most NEGATIVE records (top 5):")
    for _, row in df.nsmallest(5, "final_sentiment").iterrows():
        print(f"  [{row['team']} | {row['date']} | score={row['final_sentiment']:.3f}]")
        print(f"  {str(row['text'])[:115]}\n")

    # 5. Win/loss direction check
    if "win" in df.columns and df["win"].notna().any():
        print("[5] Mean final_sentiment by game outcome (win=1, loss=0):")
        print(df[df["win"].notna()].groupby(["team", "win"])["final_sentiment"]
              .mean().round(4).unstack(fill_value=np.nan).to_string())
        print("  Expected: sentiment higher on win days than loss days.\n")
    else:
        print("[5] Schedule not merged -- win/loss direction check skipped.\n")

    # 6. Score distribution
    print("[6] Score distribution by team (% of records):")
    dist_rows = []
    for team in sorted(df["team"].unique()):
        t = df[df["team"] == team]["final_sentiment"]
        dist_rows.append({
            "team":         team,
            "n":            len(t),
            "pct_positive": f"{(t > 0.05).mean():.1%}",
            "pct_neutral":  f"{t.between(-0.05, 0.05).mean():.1%}",
            "pct_negative": f"{(t < -0.05).mean():.1%}",
            "mean_final":   f"{t.mean():.4f}",
        })
    print(pd.DataFrame(dist_rows).to_string(index=False))

    # 7. Top TF-IDF terms per team (qualitative; not used in scoring)
    if top_terms_by_team:
        print("\n[7] Top TF-IDF terms per team (qualitative analysis only):")
        for team, terms in sorted(top_terms_by_team.items()):
            term_str = ", ".join(f"{t}({w:.4f})" for t, w in terms[:15])
            print(f"  {team}: {term_str}")

    print("\n" + "=" * 65 + "\n")

# =============================================================================
# SECTION 8 -- MAIN PIPELINE
# =============================================================================

def run_pipeline(
    team_files=None,
    sports_lexicon_path=None,
    run_validation=True
):
    """
    Full pipeline: load -> combine text -> score all components -> validate -> save.

    Args:
        team_files (dict):          Team config dict. Defaults to TEAM_FILES above.
        run_validation (bool):      Whether to print the validation report.

    Returns:
        pd.DataFrame with all sentiment columns for downstream use.
    """
    if team_files is None:
        team_files = TEAM_FILES

    # ---- Load all teams ----
    all_records = []
    for team, paths in team_files.items():
        posts_path    = paths.get("posts", "")
        comments_path = paths.get("comments", "")

        if not os.path.exists(posts_path):
            print(f"[WARN] Posts file not found for {team}: {posts_path} -- skipping.")
        else:
            print(f"[INFO] Loading {team} posts...")
            all_records.append(load_posts(posts_path, team))

        if not os.path.exists(comments_path):
            print(f"[WARN] Comments file not found for {team}: {comments_path} -- skipping.")
        else:
            print(f"[INFO] Loading {team} comments...")
            all_records.append(load_comments(comments_path, team))

    if not all_records:
        raise RuntimeError("No data loaded. Check that REDDIT_DIR points to the cleaned CSVs.")

    df = pd.concat(all_records, ignore_index=True)
    print(f"\n[INFO] Total records loaded: {len(df)}")

    # ---- Combine title + body -> text ----
    df = combine_title_body(df)
    df["date"] = utc_to_date(df["created_utc"])

    # Drop records where text is empty after cleaning
    df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    print(f"[INFO] Records after dropping empty text: {len(df)}")

    # ---- VADER baseline (primary signal) ----
    print("[INFO] Computing VADER sentiment...")
    df["vader_sentiment"] = compute_vader(df["text"])

    # ---- TF-IDF auxiliary feature ----
    print("[INFO] Building TF-IDF matrix...")
    vectorizer, tfidf_matrix = build_tfidf_matrix(df["text"])

    print("[INFO] Computing TF-IDF context scores (auxiliary feature)...")
    df["tfidf_context"] = compute_tfidf_context_score(
        df["text"], vectorizer, tfidf_matrix
    )

    # Compute top terms per team for validation/analysis (not used in scoring)
    top_terms = get_top_tfidf_terms(df, vectorizer, tfidf_matrix, top_n=20)

    # ---- Sports sentiment ----
    print("[INFO] Computing sports sentiment...")
    df["sports_sentiment"] = compute_sports_sentiment(df["text"])

    # ---- Final combined score ----
    print("[INFO] Computing final sentiment score...")
    df["final_sentiment"] = combine_sentiment(
        df["vader_sentiment"],
        df["sports_sentiment"],
        df["tfidf_context"]
    )

    # ---- Merge schedule (validation only) ----
    schedule_frames = []
    for team, paths in team_files.items():
        sched_path = paths.get("schedule", "")
        if sched_path and os.path.exists(sched_path):
            sched = load_schedule(sched_path, team)
            if not sched.empty:
                schedule_frames.append(sched)
    if schedule_frames:
        sched_df = pd.concat(schedule_frames, ignore_index=True)
        df = merge_with_schedule(df, sched_df)
        print(f"[INFO] Schedule merged for validation ({len(schedule_frames)} team(s)).")
    else:
        print("[INFO] No schedule files found -- win/loss validation check will be skipped.")

    # ---- Validation ----
    if run_validation:
        validate_sentiment(df, top_terms_by_team=top_terms)

    # ---- Save: per-record results ----
    base_cols = [
        "team", "record_type", "author", "date", "created_utc",
        "text", "ups", "permalink",
        "vader_sentiment", "sports_sentiment", "tfidf_context", "final_sentiment"
    ]
    extra_cols = [c for c in ["win", "run_diff", "home", "W/L"]
                  if c in df.columns]
    out_cols = base_cols + extra_cols

    out_records = os.path.join(OUTPUT_DIR, "sentiment_results.csv")
    df[[c for c in out_cols if c in df.columns]].to_csv(out_records, index=False)
    print(f"[SAVED] Per-record results  ->  {out_records}")

    # ---- Save: validation summary ----
    summary_rows = []
    for team in sorted(df["team"].unique()):
        t = df[df["team"] == team]
        fs = t["final_sentiment"]
        row = {
            "team":          team,
            "n_records":     len(t),
            "n_posts":       (t["record_type"] == "post").sum(),
            "n_comments":    (t["record_type"] == "comment").sum(),
            "mean_vader":    round(t["vader_sentiment"].mean(),  4),
            "mean_sports":   round(t["sports_sentiment"].mean(), 4),
            "mean_tfidf":    round(t["tfidf_context"].mean(),    4),
            "mean_final":    round(fs.mean(),                    4),
            "std_final":     round(fs.std(),                     4),
            "median_final":  round(fs.median(),                  4),
            "pct_positive":  round((fs >  0.05).mean(),          4),
            "pct_neutral":   round(fs.between(-0.05, 0.05).mean(), 4),
            "pct_negative":  round((fs < -0.05).mean(),          4),
            "top_tfidf_terms": "; ".join(
                f"{term}({w:.4f})" for term, w in top_terms.get(team, [])[:10]
            ),
        }
        summary_rows.append(row)

    out_summary = os.path.join(OUTPUT_DIR, "sentiment_summary.csv")
    pd.DataFrame(summary_rows).to_csv(out_summary, index=False)
    print(f"[SAVED] Validation summary  ->  {out_summary}")

    return df

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    results = run_pipeline(
        team_files=TEAM_FILES,
        run_validation=True
    )
    print(f"\n[DONE] Pipeline complete.")
    print(f"  Total records scored : {len(results)}")
    print(f"  Teams processed      : {sorted(results['team'].unique())}")
    print(f"  Output columns       : {[c for c in results.columns if 'sentiment' in c or c == 'tfidf_context']}")
