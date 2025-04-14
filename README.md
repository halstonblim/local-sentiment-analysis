# Daily Reddit Sentiment Analyzer

A modular, config-driven tool for daily sentiment analysis of Reddit posts.

## Features

- Daily scraping of Reddit posts from specified subreddits
- Sentiment analysis using Hugging Face models
- Configurable through YAML
- Structured data storage
- Comprehensive logging
- Rate limit awareness
- Dry-run mode for testing

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Reddit API credentials:
   ```
   REDDIT_CLIENT_ID=your_client_id
   REDDIT_CLIENT_SECRET=your_client_secret
   REDDIT_USER_AGENT=your_user_agent
   ```

## Configuration

Edit `config.yaml` to specify:
- Subreddits to track
- Post limits
- Output directory
- Model configuration
- Timezone

Example config:
```yaml
subreddits:
  - name: apple
    include_comments: false
    post_limit: 50
  - name: GooglePixel
    include_comments: false
    post_limit: 30

output_dir: "data"
model_name: "distilbert-base-uncased-finetuned-sst-2-english"
timezone: "US/Central"
```

## Usage

Run the script:
```bash
python src/main.py
```

Options:
- `--config`: Specify a different config file (default: config.yaml)
- `--dry-run`: Run without saving files

## Output Structure

```
/data
  /subreddit_name
    posts_YYYY-MM-DD.csv
/logs
  reddit_sentiment_YYYY-MM-DD_HH-MM-SS.log
```

## Data Format

CSV files contain:
- Post metadata (title, score, etc.)
- Text content
- Sentiment scores and labels
- Model version
- Retrieval timestamp

## Logging

A single log file is created for each script run in the `/logs` directory, containing:
- Processing start/end times
- Number of posts processed for each subreddit
- Rate limit information
- Any errors encountered
- Both file and console output

## License

MIT
