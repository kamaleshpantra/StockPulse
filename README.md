# StockPulse: Stock Trend Prediction with Social Sentiment

<img src="Screenshots/Logo.jpg" alt="StockPulse Logo" width="200" height="200">

## Overview
**StockPulse** is a machine learning project that predicts stock price trends (up or down) by integrating historical stock data from Yahoo Finance with social media sentiment from Reddit. It uses a Long Short-Term Memory (LSTM) neural network to analyze 10-day sequences of closing prices and daily sentiment counts (positive, negative, neutral) from subreddits like `r/wallstreetbets`, `r/stocks`, and `r/investing`. Built in Google Colab with a Streamlit interface, it’s designed to run privately without public hosting.

## Features
- **Data Sources**:
  - **Stock Data**: 2 years of daily closing prices via `yfinance`.
  - **Reddit Data**: Posts from the last 10 days (or customizable to 1 month) via `praw`.
- **Sectors**: Technology (AAPL, MSFT, GOOGL, TSLA), Finance (JPM, BAC, WFC, GS), Healthcare (PFE, JNJ, MRK, GILD).
- **Model**: LSTM (2 layers, 50 units each) trained on price and sentiment sequences.
- **Outputs**: 
  - Predictions (e.g., “AAPL: Predicted up”).
  - Metrics (accuracy, precision, recall, F1).
  - Plotly graphs (stock trends with predictions).

## Prerequisites
- **Environment**: Google Colab (free tier, GPU recommended).
- **Python Libraries**:
  - `streamlit`
  - `pandas`
  - `plotly`
  - `nltk` (with `vader_lexicon`)
  - `pyflink`
  - `tensorflow`
  - `yfinance`
  - `praw`
- **Credentials**: Reddit API `client_id` and `client_secret` ([reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)).

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/[your-username]/StockPulse.git
   cd StockPulse
