import os
import argparse
import yaml
from dotenv import load_dotenv
from reddit_scraper import RedditScraper
from sentiment_analyzer import SentimentAnalyzer
from file_manager import FileManager
import logging

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load environment variables
    load_dotenv()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Daily Reddit Sentiment Analyzer")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--dry-run", action="store_true", help="Run without saving files")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    file_manager = FileManager(config)
    scraper = RedditScraper(config)
    analyzer = SentimentAnalyzer(config['model_name'])

    # Process each subreddit
    for subreddit_config in config['subreddits']:
        subreddit_name = subreddit_config['name']
        logging.info(f"Starting processing for r/{subreddit_name}")
        
        # Print initial rate limit info
        scraper.print_rate_limit_info()
        
        try:
            # Get posts
            df = scraper.get_posts(subreddit_config)
            logging.info(f"Retrieved {len(df)} posts from r/{subreddit_name}")
            
            # Analyze sentiment
            df = analyzer.analyze_dataframe(df)
            logging.info(f"Analyzed sentiment for {len(df)} posts")
            
            # Save results
            if file_manager.save_data(df, subreddit_name, args.dry_run):
                logging.info(f"Successfully processed r/{subreddit_name}")
            else:
                logging.info(f"Skipped r/{subreddit_name} (data already exists)")
            
            # Print final rate limit info
            scraper.print_rate_limit_info()
            
        except Exception as e:
            logging.error(f"Error processing r/{subreddit_name}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 