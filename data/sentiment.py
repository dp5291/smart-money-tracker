# ============================================================
# Smart Money Tracker
# Copyright (c) 2026 Dhruv Patel. All rights reserved.
#
# This software is proprietary and confidential.
# Unauthorized copying, distribution, or modification
# of this file, via any medium, is strictly prohibited.
#
# Author:  Dhruv Patel
# GitHub:  github.com/dhruvpatel29
# Email:   dhruvkumarp79@gmail.com
# ============================================================

"""
data/sentiment.py — Real-time sentiment analysis using FinBERT.

FinBERT is a BERT model fine-tuned on financial text.
It outputs: positive / negative / neutral with probabilities.

We aggregate scores from:
  1. NewsAPI financial headlines (100 req/day free)
  2. Reddit r/stocks, r/wallstreetbets (completely free via PRAW)

Final score: -1.0 (very bearish) to +1.0 (very bullish)
"""

import os
import praw
import datetime
from typing import Optional

from newsapi import NewsApiClient
from transformers import pipeline

from config import (
    NEWSAPI_KEY, REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
)

# ── Load FinBERT once at startup (takes ~10s first time) ──────
# Downloads ~400MB model on first run, cached after that
print("Loading FinBERT model...")
_finbert = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    top_k=None,         # return all 3 label probabilities
    truncation=True,
    max_length=512,
)
print("FinBERT ready.")


def score_text(text: str) -> float:
    """
    Score a single piece of text with FinBERT.

    Returns:
        float between -1.0 and +1.0
        +1.0 = extremely bullish
        -1.0 = extremely bearish
         0.0 = neutral

    How the score is computed:
        score = P(positive) - P(negative)
        Neutral probability is ignored — it reduces the magnitude
        but doesn't flip direction.
    """
    if not text or len(text.strip()) < 10:
        return 0.0

    try:
        result = _finbert(text[:512])[0]   # truncate to 512 tokens
        scores = {r["label"]: r["score"] for r in result}
        return scores.get("positive", 0) - scores.get("negative", 0)
    except Exception:
        return 0.0


def fetch_news_sentiment(
    ticker: str,
    company_name: str,
    hours_back: int = 6
) -> dict:
    """
    Fetch recent financial news and score with FinBERT.

    Args:
        ticker:       e.g. "AAPL"
        company_name: e.g. "Apple"  (used as search query)
        hours_back:   look back this many hours

    Returns:
        {
            "score":         float (-1 to 1),
            "article_count": int,
            "headlines":     list[str],
        }

    Free tier limits: 100 requests/day on newsapi.org
    """
    if not NEWSAPI_KEY:
        print("No NEWSAPI_KEY set — skipping news sentiment")
        return {"score": 0.0, "article_count": 0, "headlines": []}

    client = NewsApiClient(api_key=NEWSAPI_KEY)
    from_date = (datetime.datetime.utcnow()
                - datetime.timedelta(hours=hours_back)).strftime("%Y-%m-%d")

    try:
        response = client.get_everything(
            q=f'{company_name} stock',
            language="en",
            sort_by="publishedAt",
            page_size=30,
        )
        articles = response.get("articles", [])
    except Exception as e:
        print(f"NewsAPI error: {e}")
        return {"score": 0.0, "article_count": 0, "headlines": []}

    headlines = [
        a["title"] for a in articles
        if a.get("title") and a["title"] != "[Removed]"
    ]

    if not headlines:
        return {"score": 0.0, "article_count": 0, "headlines": []}

    scores = [score_text(h) for h in headlines]
    avg_score = sum(scores) / len(scores)

    return {
        "score":         avg_score,
        "article_count": len(headlines),
        "headlines":     headlines[:5],   # return top 5 for display
    }


def fetch_reddit_sentiment(
    ticker: str,
    subreddits: list = None,
    hours_back: int = 6,
    post_limit: int = 50
) -> dict:
    """
    Fetch Reddit posts mentioning the ticker and score with FinBERT.

    Searches: r/stocks, r/investing, r/wallstreetbets, r/StockMarket

    How to get Reddit API credentials (takes 2 minutes):
      1. Go to reddit.com/prefs/apps
      2. Click "create another app"
      3. Name: SmartMoneyTracker, Type: script
      4. Copy client_id (below app name) and secret
      5. Put them in your .env file

    Returns:
        {
            "score":      float (-1 to 1),
            "post_count": int,
        }
    """
    if not REDDIT_CLIENT_ID:
        print("No REDDIT_CLIENT_ID set — skipping Reddit sentiment")
        return {"score": 0.0, "post_count": 0}

    if subreddits is None:
        subreddits = ["stocks", "investing", "StockMarket", "wallstreetbets"]

    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
        )
    except Exception as e:
        print(f"Reddit init error: {e}")
        return {"score": 0.0, "post_count": 0}

    texts = []
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=hours_back)
    cutoff_ts = cutoff.timestamp()

    for sub_name in subreddits:
        try:
            sub = reddit.subreddit(sub_name)
            for post in sub.search(ticker, sort="new", limit=post_limit // len(subreddits)):
                if post.created_utc < cutoff_ts:
                    continue
                # Score title + first 200 chars of body
                text = post.title
                if post.selftext:
                    text += " " + post.selftext[:200]
                texts.append(text)
        except Exception as e:
            print(f"Reddit error on r/{sub_name}: {e}")
            continue

    if not texts:
        return {"score": 0.0, "post_count": 0}

    scores = [score_text(t) for t in texts]
    return {
        "score":      sum(scores) / len(scores),
        "post_count": len(texts),
    }


def get_combined_sentiment(
    ticker: str,
    company_name: str,
    hours_back: int = 6
) -> dict:
    """
    Combine news and Reddit sentiment into a single score.

    Weighting:
      - News: 60% (more reliable, curated sources)
      - Reddit: 40% (faster signal, noisier)

    Returns:
        {
            "combined_score": float (-1 to 1),
            "news_score":     float,
            "reddit_score":   float,
            "article_count":  int,
            "post_count":     int,
            "label":          "bullish" / "bearish" / "neutral",
        }
    """
    news   = fetch_news_sentiment(ticker, company_name, hours_back)
    reddit = fetch_reddit_sentiment(ticker, hours_back=hours_back)

    # Weighted average (handle case where data is missing)
    n_score = news["score"]
    r_score = reddit["score"]
    n_count = news["article_count"]
    r_count = reddit["post_count"]

    if n_count == 0 and r_count == 0:
        combined = 0.0
    elif n_count == 0:
        combined = r_score
    elif r_count == 0:
        combined = n_score
    else:
        combined = 0.60 * n_score + 0.40 * r_score

    # Determine label
    if combined > 0.15:
        label = "bullish"
    elif combined < -0.15:
        label = "bearish"
    else:
        label = "neutral"

    return {
        "combined_score": round(combined, 4),
        "news_score":     round(n_score, 4),
        "reddit_score":   round(r_score, 4),
        "article_count":  n_count,
        "post_count":     r_count,
        "label":          label,
    }


# ── Ticker → company name mapping ─────────────────────────────
TICKER_NAMES = {
    "AAPL":    "Apple",
    "NVDA":    "Nvidia",
    "TSLA":    "Tesla",
    "MSFT":    "Microsoft",
    "AMZN":    "Amazon",
    "BTC-USD": "Bitcoin",
    "META":    "Meta Facebook",
    "GOOGL":   "Google Alphabet",
}


if __name__ == "__main__":
    # Quick test — run: python data/sentiment.py
    ticker = "AAPL"
    company = TICKER_NAMES.get(ticker, ticker)
    print(f"\nScoring sentiment for {ticker} ({company})...")
    result = get_combined_sentiment(ticker, company, hours_back=24)
    print("\nResult:")
    for k, v in result.items():
        print(f"  {k}: {v}")
