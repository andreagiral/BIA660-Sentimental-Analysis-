
# preprocess_reddit.py

"""
For each team, this script:
1. Loads the raw posts + comments JSONL files
2. Filters to the 2025 MLB season window
3. Keeps only the columns the pipeline actually uses
4. Samples proportionally within each calendar week to preserve the temporal distribution of discussion.
5. Saves trimmed files to data/sampled/<TEAM>_posts.jsonl, data/sampled/<TEAM>_comments.jsonl
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# CONFIG

OUTPUT_DIR = "data/sampled"

RANDOM_STATE = 42

# Date window (2025 MLB season)
START_DATE = pd.Timestamp("2025-03-20")
END_DATE   = pd.Timestamp("2025-10-05")

# Post/comment caps - Set to approximately the smallest counts among all teams
MAX_COMMENTS = 50_000
MAX_POSTS = 2_500

# Fields to keep from raw data
POST_COLS    = ["author", "created_utc", "permalink", "selftext", "title", "ups"]
COMMENT_COLS = ["author", "created_utc", "body", "permalink", "ups"]

# Filenames for each team
TEAM_CONFIGS = [
    {
        "team": "WSN",
        "posts_fname": "data/r_Nationals_posts.jsonl",
        "comments_fname": "data/r_Nationals_comments.jsonl",
    },
    {
        "team": "CHW",
        "posts_fname": "data/r_whitesox_posts.jsonl",
        "comments_fname": "data/r_whitesox_comments.jsonl",
    },
    {
        "team": "COL",
        "posts_fname": "data/r_ColoradoRockies_posts.jsonl",
        "comments_fname": "data/r_ColoradoRockies_comments.jsonl",
    },
    {
        "team": "MIN",
        "posts_fname": "data/r_minnesotatwins_posts.jsonl",
        "comments_fname": "data/r_minnesotatwins_comments.jsonl",
    },
    {
        "team": "TOR",
        "posts_fname": "data/r_Torontobluejays_posts.jsonl",
        "comments_fname": "data/r_Torontobluejays_comments.jsonl",
    },
    {
        "team": "NYY",
        "posts_fname": "data/r_NYYankees_posts.jsonl",
        "comments_fname": "data/r_NYYankees_comments.jsonl",
    },
    {
        "team": "MIL",
        "posts_fname": "data/r_Brewers_posts.jsonl",
        "comments_fname": "data/r_Brewers_comments.jsonl",
    },
    {
        "team": "PHI",
        "posts_fname": "data/r_phillies_posts.jsonl",
        "comments_fname": "data/r_phillies_comments.jsonl",
    },
]

# HELPER FUNCTIONS

# Load JSONL file in chunks to avoid loading it all into memory
# Returns pandas df of only desired columns
def load_jsonl_chunked(fname, cols, chunksize=100_000):
    chunks = []
    with open(fname, "r", encoding="utf-8") as f:
        batch = []
        for i, line in enumerate(f):
            try:
                batch.append(json.loads(line))
            except json.JSONDecodeError:
                continue

            if len(batch) == chunksize:
                df = pd.DataFrame(batch)
                # Keep only columns that exist in this file
                keep = [c for c in cols if c in df.columns]
                chunks.append(df[keep])
                batch = []

        if batch:
            df = pd.DataFrame(batch)
            keep = [c for c in cols if c in df.columns]
            chunks.append(df[keep])

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

# Filters out rows where created_utc is not within the specified window
def filter_by_date(df, start=START_DATE, end=END_DATE):
    # created_utc is a Unix timestamp
    start_ts = start.timestamp()
    end_ts = end.timestamp()
    mask = (df["created_utc"] >= start_ts) & (df["created_utc"] <= end_ts)
    return df[mask].copy()

# Randolmy sample n rows, with an even distribution by week
def sample_data(df, n, random_state=RANDOM_STATE):
    # If df has less than n rows, return df
    if len(df) <= n:
        return df

    df = df.copy()
    df["_week"] = pd.to_datetime(df["created_utc"], unit="s").dt.isocalendar().week

    week_counts = df["_week"].value_counts()
    total = len(df)
    rng = np.random.default_rng(random_state)
    sampled_parts = []

    for week, count in week_counts.items():
        week_df = df[df["_week"] == week]
        week_target = max(1, round(n * count / total))
        week_target = min(week_target, len(week_df))
        sampled_parts.append(
            week_df.sample(week_target, random_state=random_state)
        )

    result = pd.concat(sampled_parts).drop(columns=["_week"])

    # Trim or top-up to exactly n (rounding can drift by a few rows)
    if len(result) > n:
        result = result.sample(n, random_state=random_state)

    return result.reset_index(drop=True)

def save_jsonl(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_json(path, orient="records", lines=True)

# PIPELINE

def process_team(cfg):
    team = cfg["team"]
    print(f"\nProcessing team: {team}")

    # Posts
    print(f"  Posts:")
    posts_raw = load_jsonl_chunked(cfg["posts_fname"], POST_COLS)
    print(f"    Raw count: {len(posts_raw)}")
    posts_by_date = filter_by_date(posts_raw)
    posts_sampled = sample_data(posts_by_date, MAX_POSTS)
    print(f"    Final count: {len(posts_sampled)}")

    # Comments
    print(f"  Comments:")
    comments_raw = load_jsonl_chunked(cfg["comments_fname"], COMMENT_COLS)
    print(f"    Raw count: {len(comments_raw)}")
    comments_by_date = filter_by_date(comments_raw)
    comments_sampled = sample_data(comments_by_date, MAX_COMMENTS)
    print(f"    Final count: {len(comments_sampled)}")

    # Save to jsonl
    save_jsonl(posts_sampled, os.path.join(OUTPUT_DIR, f"{team}_posts.jsonl"))
    save_jsonl(comments_sampled, os.path.join(OUTPUT_DIR, f"{team}_comments.jsonl"))

    return {
        "team": team,
        "raw_posts": len(posts_raw),
        "raw_comments": len(comments_raw),
        "sampled_posts": len(posts_sampled),
        "sampled_comments": len(comments_sampled),
    }

def main():
    print("Preprocessing Reddit data:")
    print(f"Date window : {START_DATE.date()} - {END_DATE.date()}")
    print(f"Comment cap : {MAX_COMMENTS} per team")
    print(f"Post cap : {MAX_POSTS} per team")
    print(f"Output directory : {OUTPUT_DIR}/")

    summaries = []
    for team_cfg in TEAM_CONFIGS:
        summary = process_team(team_cfg)
        summaries.append(summary)

    # Print comparison table
    print("SUMMARY:")
    for team_sum in summaries:
        print(f"Team {team_sum['team']}")
        print(f"  Raw posts: {team_sum['raw_posts']}")
        print(f"  Raw comments: {team_sum['raw_comments']}")
        print(f"  Sampled posts: {team_sum['sampled_posts']}")
        print(f"  Sampled comments: {team_sum['sampled_comments']}")
        print()

    print(f"\nSampled files saved to:")
    print(f"  {OUTPUT_DIR}/<TEAM>_posts.jsonl")
    print(f"  {OUTPUT_DIR}/<TEAM>_comments.jsonl")


if __name__ == "__main__":
    main()
