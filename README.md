# BIA660-Sentimental-Analysis-

## Notes on raw data fields

### Reddit - JSON

- "ups" / "score"
    - both seem to indicate upvotes, but perhaps "score" considers downvotes and "ups" does not
- "created" / "created_utc"
    - Both seem to be the same - a unix timestamp of creation date
    - access with datetime.fromtimestamp()

### MLB Statistics - CSV

- "W/L" - Win/Loss
    - Options are 'W', 'L', 'W-wo', and 'L-wo'
    - '-wo' indicates a walk-off
    - There may be additional options, like for ties (which are rare)
- "R" - Runs scored
- "RA" - Runs allowed
