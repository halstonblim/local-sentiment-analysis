import os
import sys
import logging
from datetime import datetime
import pandas as pd

class FileManager:
    def __init__(self, config):
        self.config = config
        self.output_dir = config['output_dir']
        self.timezone = config['timezone']
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_file = self.setup_logging()

    def get_output_path(self, subreddit_name):
        """Get the output path for a subreddit's data."""
        date_str = datetime.now().strftime('%Y-%m-%d')
        subreddit_dir = os.path.join(self.output_dir, subreddit_name)
        os.makedirs(subreddit_dir, exist_ok=True)
        return os.path.join(subreddit_dir, f"posts_{date_str}.csv")

    def save_data(self, df, subreddit_name, dry_run=False):
        """Save DataFrame to CSV file."""
        output_path = self.get_output_path(subreddit_name)
        
        if os.path.exists(output_path):
            logging.info(f"File already exists: {output_path}")
            return False

        if dry_run:
            logging.info(f"[DRY RUN] Would save to: {output_path}")
            return True

        df.to_csv(output_path, index=False)
        logging.info(f"Saved data to: {output_path}")
        return True

    def setup_logging(self):
        """Setup logging for the entire script run with UTF-8-safe output."""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        date_str = datetime.now().strftime('%Y-%m-%d')
        timestamp = datetime.now().strftime('%H-%M-%S')
        log_file = os.path.join(log_dir, f"reddit_sentiment_{date_str}_{timestamp}.log")

        # Clear existing handlers to avoid duplicates
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Configure UTF-8-safe logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(stream=open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1))
            ]
        )

        logging.info("Starting Reddit Sentiment Analysis")
        logging.info(f"Log file: {log_file}")
        return log_file
