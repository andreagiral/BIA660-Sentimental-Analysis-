# preprocess_data.py

"""
Cleans and preprocesses data files:
- data/mlb/{TEAM}_2025.csv -> data/clean/mlb/{TEAM}_2025.csv
- data/sampled/{TEAM}_posts.jsonl -> data/clean/reddit/{TEAM}_posts.csv
- data/sampled/{TEAM}_comments.jsonl -> data/clean/reddit/{TEAM}_comments.csv

CSV (MLB game logs):
- Parse and normalize dates to ISO format (YYYY-MM-DD)
- Normalize W/L column: strip walk-off markers, handle ties/postponements
- Cast numeric columns (R, RA, Inn, Attendance, cLI) to correct types
- Derive useful columns: win (0/1), run_diff, home (0/1), walkoff (0/1)
- Drop columns the pipeline never uses
- Flag and drop duplicate game rows (doubleheader deduplication)
- Drop rows with missing Date, W/L, R, or RA

JSONL (posts + comments):
- Drop rows missing required fields (author, created_utc, body/title)
- Drop deleted/removed authors and body text ('[deleted]', '[removed]')
- Unescape permalink field (raw JSONL backslash-escapes forward slashes)
- Normalize created_utc to integer Unix timestamp
- Clip ups to >= 0 (no negative upvote counts)
- Deduplicate on permalink
- Filter to season window (configurable)
- For posts: clean selftext whitespace, drop empty-title rows
- For comments: drop very short comments (< MIN_COMMENT_CHARS chars)
- Save only the columns the pipeline uses
"""

import os
import re
import json
import string
import pandas as pd
import numpy as np
from datetime import datetime

# CONFIG

TEAM_IDS = ["WSN", "TOR", "CHW", "NYY", "PHI", "MIL", "MIN", "COL"]

MLB_DIR = "data/mlb"
REDDIT_DIR = "data/sampled"
OUT_DIR = "data/clean"

# Season window — rows outside this range are dropped from Reddit data.
# A week before opening day, a week after end of regular season.
SEASON_START_UTC = int(datetime(2025, 3, 20).timestamp())
SEASON_END_UTC = int(datetime(2025, 10, 6).timestamp())

# Comments shorter than this (after stripping whitespace) are noise
MIN_COMMENT_CHARS = 15

# Columns to keep in output files
MLB_KEEP_COLS = [
    "Date", "W/L", "R", "RA",
    "win", "run_diff", "home", "walkoff",
    "Opp", "Inn", "Attendance",
]

POST_KEEP_COLS = ["author", "created_utc", "permalink", "selftext", "title", "ups"]
COMMENT_KEEP_COLS = ["author", "created_utc", "body", "permalink", "ups"]

# MLB CSV CLEANING

# All known W/L values
WIN_VALUES  = {"W", "W-wo"} # wins
LOSS_VALUES = {"L", "L-wo"} # losses

def clean_mlb_csv(team: str):
    path = os.path.join(MLB_DIR, f"{team}_2025.csv")
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path, dtype=str).fillna("")

    # Parse Date
    # Use regex to strip doubleheader markers like "(1)" or "(2)" from dates
    raw_dates = df["Date"].str.replace(re.compile(r"\s*\(\d+\)"), "", regex=True).str.strip()
    df["Date"] = pd.to_datetime(raw_dates + " 2025", format="%A %b %d %Y", errors="coerce").dt.strftime("%Y-%m-%d")

    # Drop rows where date couldn't be parsed
    df.dropna(subset=["Date"], inplace=True)

    # Normalize W/L
    df["walkoff"] = df["W/L"].str.endswith("-wo").astype(int)
    df["win"] = df["W/L"].apply(
        lambda x: 1 if x in WIN_VALUES else (0 if x in LOSS_VALUES else np.nan)
    )
    df.dropna(subset=["win"], inplace=True)
    df["win"] = df["win"].astype(int)

    # Numeric columns
    for col in ["R", "RA", "Inn"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["R", "RA"], inplace=True)
    df["R"]  = df["R"].astype(int)
    df["RA"] = df["RA"].astype(int)

    for col in ["Attendance", "cLI"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived columns
    df["run_diff"] = df["R"] - df["RA"]
    df["home"] = (~df.get("Home/Away", pd.Series("", index=df.index)).str.strip().str.upper().eq("@")).astype(int)

    # Deduplicate (handles doubleheader CSV quirks)
    df = df[~df.duplicated(subset=["Date", "R", "RA", "win"], keep="first")].copy()

    keep = [c for c in MLB_KEEP_COLS if c in df.columns]
    return df[keep].sort_values("Date").reset_index(drop=True)

# REDDIT JSONL CLEANING

DELETED = {"[deleted]", "[removed]", ""}
ALLOWED = set(string.ascii_letters + string.digits + string.punctuation + ' ')

def clean_chars(text):
    return ''.join(c for c in text if c in ALLOWED)

def clean_text(text: str):
    if not isinstance(text, str):
        return ""
    text = text.replace("\\/", "/") # \/r\/ -> /r/
    text = re.sub(r"\s+", " ", text).strip() # collapse whitespace
    text = clean_chars(text) # remove weird characters
    return text

def load_jsonl(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records

def clean_posts(team: str):
    path = os.path.join(REDDIT_DIR, f"{team}_posts.jsonl")
    if not os.path.exists(path):
        return None

    df = pd.DataFrame(load_jsonl(path))

    for col in POST_KEEP_COLS:
        if col not in df.columns:
            df[col] = "" if col in ("selftext", "title", "permalink") else 0
    df = df[POST_KEEP_COLS].copy()

    # Drop missing / deleted
    df["author"] = df["author"].astype(str).str.strip()
    df["title"] = df["title"].astype(str).str.strip()
    df = df[~df["author"].isin(DELETED)]
    df = df[df["title"].str.len() > 0]

    # Normalize text fields
    df["title"] = df["title"].apply(clean_text)
    df["selftext"] = df["selftext"].astype(str).apply(clean_text)
    df["selftext"] = df["selftext"].apply(lambda x: "" if x.strip() in DELETED else x)
    df["permalink"] = df["permalink"].astype(str).apply(clean_text)

    # Normalize created_utc
    df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")
    df.dropna(subset=["created_utc"], inplace=True)
    df["created_utc"] = df["created_utc"].astype(int)

    # Season filter
    df = df[(df["created_utc"] >= SEASON_START_UTC) & (df["created_utc"] <= SEASON_END_UTC)]

    # Clip upvotes
    df["ups"] = pd.to_numeric(df["ups"], errors="coerce").fillna(0).clip(lower=0).astype(int)

    # Deduplicate on permalink
    df = df.drop_duplicates(subset=["permalink"], keep="first")

    return df.sort_values("created_utc").reset_index(drop=True)

def clean_comments(team: str):
    path = os.path.join(REDDIT_DIR, f"{team}_comments.jsonl")
    if not os.path.exists(path):
        return None

    df = pd.DataFrame(load_jsonl(path))

    for col in COMMENT_KEEP_COLS:
        if col not in df.columns:
            df[col] = "" if col in ("body", "permalink") else 0
    df = df[COMMENT_KEEP_COLS].copy()

    # Drop missing / deleted
    df["author"] = df["author"].astype(str).str.strip()
    df["body"] = df["body"].astype(str).str.strip()
    df = df[~df["author"].isin(DELETED)]
    df = df[~df["body"].isin(DELETED)]

    # Normalize text
    df["body"] = df["body"].apply(clean_text)
    df["permalink"] = df["permalink"].astype(str).apply(clean_text)

    # Drop very short comments (noise)
    df = df[df["body"].str.len() >= MIN_COMMENT_CHARS]

    # Normalize created_utc
    df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")
    df.dropna(subset=["created_utc"], inplace=True)
    df["created_utc"] = df["created_utc"].astype(int)

    # Season filter
    df = df[(df["created_utc"] >= SEASON_START_UTC) & (df["created_utc"] <= SEASON_END_UTC)]

    # Clip upvotes
    df["ups"] = pd.to_numeric(df["ups"], errors="coerce").fillna(0).clip(lower=0).astype(int)

    # Deduplicate on permalink
    df = df.drop_duplicates(subset=["permalink"], keep="first")

    return df.sort_values("created_utc").reset_index(drop=True)


# SAVE HELPERS

def save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def save_jsonl(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)

# MAIN

def main():
    for team in TEAM_IDS:
        mlb_df = clean_mlb_csv(team)
        if mlb_df is not None:
            save_csv(mlb_df, os.path.join(OUT_DIR, "mlb", f"{team}_2025.csv"))

        posts_df = clean_posts(team)
        if posts_df is not None:
            # save_jsonl(posts_df, os.path.join(OUT_DIR, "reddit", f"{team}_posts.jsonl"))
            save_csv(posts_df, os.path.join(OUT_DIR, "reddit", f"{team}_posts.csv"))

        comments_df = clean_comments(team)
        if comments_df is not None:
            # save_jsonl(comments_df, os.path.join(OUT_DIR, "reddit", f"{team}_comments.jsonl"))
            save_csv(comments_df, os.path.join(OUT_DIR, "reddit", f"{team}_comments.csv"))

if __name__ == "__main__":
    main()
