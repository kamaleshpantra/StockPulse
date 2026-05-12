import praw
from yahooquery import Ticker as YQTicker
import pandas as pd
import os
import logging
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

import requests
import random
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of realistic User-Agents
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0',
    'Mozilla/5.0 (AppleWebKit/537.36; KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
]

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

class DataFetcher:
    def __init__(self, reddit_client_id=None, reddit_client_secret=None, reddit_user_agent=None):
        self.reddit = None
        if reddit_client_id and reddit_client_secret:
            try:
                self.reddit = praw.Reddit(
                    client_id=reddit_client_id,
                    client_secret=reddit_client_secret,
                    user_agent=reddit_user_agent or "StockPulse/1.0",
                    read_only=True
                )
                logger.info("Reddit API initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Reddit API: {e}")
        
        self.sid = SentimentIntensityAnalyzer()

    def fetch_reddit_sentiment(self, company, subreddits=["wallstreetbets", "stocks", "investing"], days=10):
        """Fetch Reddit posts and calculate daily average sentiment."""
        if not self.reddit:
            logger.warning("Reddit API not initialized. Returning neutral sentiment.")
            return pd.DataFrame()

        all_posts = []
        end_timestamp = datetime.now().timestamp()
        start_timestamp = (datetime.now() - timedelta(days=days)).timestamp()

        for sub in subreddits:
            try:
                submissions = self.reddit.subreddit(sub).search(
                    query=company, sort="new", limit=100, time_filter="month"
                )
                for post in submissions:
                    if start_timestamp <= post.created_utc <= end_timestamp:
                        text = f"{post.title} {post.selftext}"
                        sentiment = self.sid.polarity_scores(text)["compound"]
                        date = datetime.fromtimestamp(post.created_utc).date()
                        all_posts.append({"date": date, "sentiment": sentiment})
            except Exception as e:
                logger.error(f"Error fetching from r/{sub}: {e}")

        if not all_posts:
            return pd.DataFrame()

        df = pd.DataFrame(all_posts)
        daily_sentiment = df.groupby("date")["sentiment"].agg(["mean", "count"]).reset_index()
        daily_sentiment.columns = ["date", "avg_sentiment", "post_count"]
        return daily_sentiment

    def fetch_stock_data(self, ticker, days=365):
        """Fetch historical stock data using yahooquery."""
        print(f"[DEBUG] Fetching stock data for: {ticker} via yahooquery")
        
        try:
            # yahooquery handles sessions and headers internally
            yq = YQTicker(ticker)
            hist = yq.history(period='1y', interval='1d')
            
            if isinstance(hist, pd.DataFrame) and not hist.empty:
                print(f"[DEBUG] Successfully fetched {len(hist)} rows for {ticker}")
                # yahooquery returns a multi-index (symbol, date)
                hist = hist.reset_index()
                
                # Check for common column names
                cols = hist.columns.tolist()
                date_col = 'date' if 'date' in cols else cols[1]
                
                hist = hist[[date_col, 'close', 'volume']]
                hist.columns = ['Date', 'Close', 'Volume']
                hist["Date"] = pd.to_datetime(hist["Date"]).dt.date
                return hist
            else:
                print(f"[DEBUG] yahooquery returned empty or error: {hist}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"[DEBUG] ERROR in yahooquery: {e}")
            logger.error(f"Error fetching stock data for {ticker}: {e}")
            return pd.DataFrame()

    def get_merged_data(self, ticker, company_name=None, stock_days=365, reddit_days=30):
        """Merge stock and sentiment data."""
        print(f"[DEBUG] Starting pipeline for {ticker}")
        stock_df = self.fetch_stock_data(ticker, days=stock_days)
        
        if stock_df.empty:
            print("[DEBUG] Pipeline stopped: stock_df is empty")
            return pd.DataFrame()

        print(f"[DEBUG] Fetching Reddit sentiment for: {company_name or ticker}")
        sentiment_df = self.fetch_reddit_sentiment(company_name or ticker, days=reddit_days)
        
        if sentiment_df.empty:
            print("[DEBUG] No Reddit data found, continuing with neutral sentiment")
            merged = stock_df.copy()
            merged["avg_sentiment"] = 0.0
            merged["post_count"] = 0
        else:
            print(f"[DEBUG] Merging {len(sentiment_df)} sentiment rows with stock data")
            merged = stock_df.merge(sentiment_df, left_on="Date", right_on="date", how="left")
            merged["avg_sentiment"] = merged["avg_sentiment"].fillna(0.0)
            merged["post_count"] = merged["post_count"].fillna(0)
            merged = merged.drop(columns=["date"])

        # Add trend column (1 if next day close > current day close)
        merged["trend"] = (merged["Close"].shift(-1) > merged["Close"]).astype(int)
        print("[DEBUG] Pipeline complete")
        return merged.dropna(subset=["Close"])
