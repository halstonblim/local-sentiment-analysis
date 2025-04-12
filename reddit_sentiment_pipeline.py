import os
import argparse
import pytz
import torch
import pandas as pd
from datetime import datetime
from torch.nn.functional import softmax
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
from tqdm import tqdm
import praw

def get_reddit_posts(subreddit_name, time_filter="year", limit=1000, output_dir="data"):
    ct = pytz.timezone("US/Central")
    retrieved_at = datetime.now(ct)
    records = []

    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT")
    )
    subreddit = reddit.subreddit(subreddit_name)

    for submission in tqdm(subreddit.top(time_filter=time_filter, limit=limit), total=limit, desc="Processing posts"):
        records.append({
            "subreddit": subreddit_name,
            "created_at": datetime.fromtimestamp(submission.created_utc, tz=ct),
            "retrieved_at": retrieved_at,
            "type": "post",
            "text": submission.title + "\n\n" + submission.selftext,
            "score": submission.score,
            "num_comments": submission.num_comments
        })

    df = pd.DataFrame(records)
    os.makedirs(output_dir, exist_ok=True)
    date_str = retrieved_at.strftime('%Y-%m-%d')
    filename = f"{output_dir}/{subreddit_name}_tf-{time_filter}_n-{limit}_{date_str}.csv"
    df.to_csv(filename, index=False)
    print_rate_limit_info(reddit)
    return filename

def analyze_sentiment(csv_file, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    df = pd.read_csv(csv_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    texts = df["text"].astype(str).tolist()
    logits_neg, logits_pos, preds = [], [], []

    batch_size = 32
    for i in tqdm(range(0, len(texts), batch_size), desc="Running batches"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=100).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            batch_logits = outputs.logits
            batch_probs = softmax(batch_logits, dim=1)
            batch_preds = torch.argmax(batch_probs, dim=1)

            logits_neg.extend(batch_logits[:, 0].cpu().tolist())
            logits_pos.extend(batch_logits[:, 1].cpu().tolist())
            preds.extend(batch_preds.cpu().tolist())

    df["logit_negative"] = logits_neg
    df["logit_positive"] = logits_pos
    df["predicted_sentiment"] = preds
    df["predicted_label"] = df["predicted_sentiment"].replace({0: "negative", 1: "positive"})
    df["date"] = pd.to_datetime(df["created_at"], utc=True).dt.date
    df["year_month"] = pd.to_datetime(df["created_at"], utc=True).dt.strftime("%Y-%m")
    df.to_csv(csv_file.replace(".csv", "_scored.csv"), index=False)

    ax = df.groupby("year_month").predicted_sentiment.mean().plot(marker=".")
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid()
    ax.set_title(f"r/{df.subreddit.max()} monthly average sentiment classification\n{model_name}")
    fig_path = csv_file.replace(".csv", ".png")
    plt.tight_layout()
    plt.savefig(fig_path)
    print(f"Plot saved to {fig_path}")


def print_rate_limit_info(reddit):
    print("\n🔄 Reddit API Rate Limit Info")
    print(f"Requests used:      {reddit.auth.limits.get('used')}")
    print(f"Requests remaining: {reddit.auth.limits.get('remaining')}")
    print(f"Resets at:          {reddit.auth.limits.get('reset_timestamp')}\n")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Reddit Sentiment Analyzer")
    parser.add_argument("--subreddit", type=str, required=True, help="Subreddit name to analyze")
    parser.add_argument("--time_filter", type=str, default="year", help="Time filter: day, week, month, year, all")
    parser.add_argument("--limit", type=int, default=1000, help="Number of posts to fetch")
    args = parser.parse_args()

    csv_file = get_reddit_posts(args.subreddit, args.time_filter, args.limit)
    analyze_sentiment(csv_file)

if __name__ == "__main__":
    main()
