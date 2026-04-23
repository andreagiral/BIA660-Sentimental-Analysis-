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
