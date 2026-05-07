# BIA660-Sentimental-Analysis-

## Pipeline

### Data Collection

#### 2025 MLB Game Data

Retrieved game outcomes in CSV format from [Baseball-Reference.com](https://www.baseball-reference.com/).

Each team's data saved to `<TEAM_ID>_2025.csv`.

#### Reddit Data

Used [Arctic Shift](https://arctic-shift.photon-reddit.com/) to obtain Reddit posts and comments from the following subreddits:

- [Chicago White Sox](https://www.reddit.com/r/whitesox/)
- [Colorado Rockies](https://www.reddit.com/r/ColoradoRockies/)
- [Milwaukee Brewers](https://www.reddit.com/r/Brewers/)
- [Minnesota Twins](https://www.reddit.com/r/minnesotatwins/)
- [New York Yankees](https://www.reddit.com/r/NYYankees/)
- [Philadelphia Phillies](https://www.reddit.com/r/phillies/)
- [Toronto Blue Jays](https://www.reddit.com/r/Torontobluejays/)
- [Washington Nationals](https://www.reddit.com/r/Nationals/)

Data saved to `jsonl` files as `data/r_<SUBREDDIT>_posts.jsonl` and `data/r_<SUBREDDIT>_comments.jsonl`.
- e.g., `data/r_minnesotatwins_posts.jsonl`, `data/r_minnesotatwins_comments.jsonl`

### Data Preprocessing

#### 1. Run `preprocess_reddit.py`

Samples data from the raw `jsonl` files (posts and comments) and keeps only relevant data.

Sampled data is saved to `data/sampled/<TEAM_ID>_posts.jsonl` and `data/sampled/<TEAM_ID>_comments.jsonl`.

#### 2. Run `clean_data.py`

Preprocesses and cleans MLB csv data and Reddit jsonl data. All data is saved as new CSV files.

Cleaned data is sorted into two directories and saved as follows:
- `data/clean/mlb/`
    - `data/clean/mlb/<TEAM_ID>_2025.csv`
- `data/clean/reddit/`.
    - `data/clean/reddit/<TEAM_ID>_posts.csv`
    - `data/clean/reddit/<TEAM_ID>_comments.csv`.

## Notes

### Raw data fields

#### Reddit Post & Comment Data - JSONL

- `ups` / `score`
    - Both seem to indicate upvotes, but perhaps `score` considers downvotes and `ups` does not
- `created` / `created_utc`
    - Both seem to be the same - a unix timestamp of creation date
    - Can access with datetime.fromtimestamp()

#### MLB Statistics - CSV

- `W/L` - Win/Loss
    - Possible values are 'W', 'L', 'W-wo', and 'L-wo'
    - '-wo' indicates a walk-off
    - There may be additional options, like for ties (which are rare), but I haven't confirmed
- `R` - Runs scored
- `RA` - Runs allowed

## Sentiment Modeling & Validation

### Run `sentiment_modeling_&_validation.py`

Constructs the final sentiment indicator used throughout the project.

The pipeline:
- Loads cleaned Reddit posts and comments for all MLB teams
- Combines title + body into a single text field
- Computes baseline sentiment using VADER
- Computes contextual TF-IDF features
- Integrates a custom baseball sentiment lexicon
- Produces a final weighted sentiment score

### Sentiment Components

#### 1. VADER Baseline
Uses the VADER sentiment analyzer to generate a compound sentiment score between `-1` and `1`.

Designed for:
- social media text
- punctuation emphasis
- capitalization
- negation handling

#### 2. Baseball-Specific Sentiment Lexicon
Adds domain-specific baseball language and fan discourse not captured well by generic sentiment tools.

Examples:
- positive:
  - `filthy`
  - `walk-off`
  - `winning`
- negative:
  - `washed`
  - `dumpster fire`
  - `bullpen collapse`

The lexicon was expanded using high-frequency terms identified during corpus analysis.

#### 3. TF-IDF Context Features
Uses TF-IDF as a lightweight contextual feature.

Purpose:
- identify important words within fan discussions
- provide directional/contextual adjustment
- NOT used as a standalone sentiment classifier

### Final Sentiment Score

The final score combines all three components:

```python
final_sentiment =
    0.60 * vader_sentiment +
    0.30 * sports_sentiment +
    0.10 * tfidf_context
```

Final scores are clipped to the range `[-1, 1]`.

### Output Files

Generated files:

#### Results
- `results/sentiment_results.csv`
- `results/sentiment_summary.csv`
- `results/pipeline_output.txt`

#### Statistical Analysis / Visualizations
Saved under `stats/`

- `fig1_sentiment_over_time.png`
- `fig2_sentiment_vs_winpct.png`
- `fig3_wins_vs_losses.png`
- `fig4_recent_performance.png`
- `fig6a_cross_team_reactivity.png`
- `fig6b_alignment_vs_reactivity.png`
- `fig7_components.png`

### Validation

Validation checks include:
- sentiment differences between wins and losses
- score distribution analysis
- cross-team consistency
- temporal trend analysis
- extreme positive/negative case checks
- TF-IDF qualitative term inspection

Results showed:
- sentiment was consistently higher after wins than losses
- realistic distributions of positive, neutral, and negative sentiment
- stable trends across teams and time periods

---

# Repository Structure

```text
BIA660-Sentimental-Analysis-/
│
├── data/
│   ├── clean/
│   │   ├── mlb/
│   │   └── reddit/
│   │
│   └── raw/
│       ├── mlb/
│       └── reddit/
│
├── results/
│   ├── pipeline_output.txt
│   ├── sentiment_results.csv
│   └── sentiment_summary.csv
│
├── stats/
│   ├── fig1_sentiment_over_time.png
│   ├── fig2_sentiment_vs_winpct.png
│   ├── fig3_wins_vs_losses.png
│   ├── fig4_recent_performance.png
│   ├── fig6a_cross_team_reactivity.png
│   ├── fig6b_alignment_vs_reactivity.png
│   ├── fig7_components.png
│   └── sentiment_analytics.ipynb
│
├── baseball_lexicon.py
├── preprocess_reddit.py
├── clean_data.py
├── sentiment_modeling_&_validation.py
├── README.md
└── Project Workflow.pdf

```