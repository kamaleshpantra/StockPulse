import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.data_fetcher import DataFetcher
from src.ml_logic import MLLogic
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="StockPulse | AI Sentiment & Trends", layout="wide", page_icon="📈")

# Initialize components
fetcher = DataFetcher(
    reddit_client_id=os.getenv("REDDIT_CLIENT_ID"),
    reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    reddit_user_agent=os.getenv("REDDIT_USER_AGENT")
)
ml = MLLogic()

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .stButton>button { width: 100%; border-radius: 5px; background-color: #238636; color: white; border: none; }
    .prediction-box { padding: 20px; border-radius: 10px; text-align: center; margin: 10px 0; border: 2px solid #30363d; }
    .up { background-color: rgba(35, 134, 54, 0.2); border-color: #238636; }
    .down { background-color: rgba(219, 68, 55, 0.2); border-color: #db4437; }
    </style>
""", unsafe_allow_html=True)

def main():
    st.title("📊 StockPulse: AI Trend Prediction")
    st.write("Combining market data with social sentiment for smarter predictions.")

    # Sidebar
    st.sidebar.header("Configuration")
    ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL").upper()
    company_name = st.sidebar.text_input("Company Name (for Reddit)", value="Apple")
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Market Trend: {ticker}")
        with st.spinner("Fetching data..."):
            data = fetcher.get_merged_data(ticker, company_name)
            
        if not data.empty:
            # Price chart
            fig = px.line(data, x='Date', y='Close', title=f"{ticker} Historical Close Price")
            fig.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Price (USD)", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Improved Sentiment chart (only last 30 days)
            st.markdown("---")
            sentiment_data = data[data['avg_sentiment'] != 0].tail(30)
            if not sentiment_data.empty:
                sentiment_fig = px.bar(sentiment_data, x='Date', y='avg_sentiment', 
                                     title="Recent Reddit Sentiment (Last 30 Days)",
                                     color='avg_sentiment',
                                     color_continuous_scale='RdYlGn',
                                     range_color=[-1, 1])
                sentiment_fig.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Compound Sentiment", coloraxis_showscale=False)
                st.plotly_chart(sentiment_fig, use_container_width=True)
            else:
                st.info("No recent Reddit sentiment data found for this period.")
        else:
            st.error(f"Failed to fetch data for {ticker}. Check your internet connection or ticker symbol.")
            st.info("Check the console logs for detailed error messages from yfinance or Reddit API.")

    with col2:
        st.subheader("Signal Intelligence")
        
        if not data.empty:
            # 1. Sentiment Gauge
            avg_sent = data[data['avg_sentiment'] != 0]['avg_sentiment'].tail(7).mean()
            if pd.isna(avg_sent): avg_sent = 0.0
            
            gauge_fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = avg_sent,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "7-Day Sentiment Mood", 'font': {'size': 18}},
                gauge = {
                    'axis': {'range': [-1, 1], 'tickwidth': 1},
                    'bar': {'color': "#238636" if avg_sent > 0 else "#db4437"},
                    'steps': [
                        {'range': [-1, -0.3], 'color': "rgba(219, 68, 55, 0.3)"},
                        {'range': [-0.3, 0.3], 'color': "rgba(255, 255, 255, 0.1)"},
                        {'range': [0.3, 1], 'color': "rgba(35, 134, 54, 0.3)"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': avg_sent
                    }
                }
            ))
            gauge_fig.update_layout(template="plotly_dark", height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(gauge_fig, use_container_width=True)

            # 2. Prediction & Recommendation
            st.markdown("### AI Analysis")
            model_path = os.path.join("models", f"{ticker}_model.h5")
            
            if os.path.exists(model_path):
                prediction = ml.predict_trend(ticker, data)
                if prediction:
                    prob = prediction['probability']
                    trend = prediction['trend']
                    
                    # Recommendation Logic
                    # Fusion of LSTM trend and Sentiment
                    if trend == "UP" and avg_sent > 0.2:
                        rec = "STRONG BUY"
                        color = "#238636"
                        desc = "AI predicts growth and social sentiment is bullish."
                    elif trend == "UP" or avg_sent > 0.4:
                        rec = "BUY / LONG"
                        color = "#2ea043"
                        desc = "Positive signals detected in price or sentiment."
                    elif trend == "DOWN" and avg_sent < -0.2:
                        rec = "STRONG SELL"
                        color = "#db4437"
                        desc = "AI predicts a drop and social sentiment is bearish."
                    else:
                        rec = "HOLD / NEUTRAL"
                        color = "#8b949e"
                        desc = "Mixed signals. Market waiting for direction."

                    st.markdown(f"""
                        <div style="background-color: {color}22; padding: 20px; border-radius: 10px; border: 2px solid {color}; text-align: center;">
                            <p style="margin: 0; font-size: 0.9em; opacity: 0.8;">Unified Recommendation</p>
                            <h2 style="margin: 10px 0; color: {color};">{rec}</h2>
                            <p style="font-size: 0.85em; font-style: italic;">{desc}</p>
                            <hr style="border: 0.5px solid {color}44;">
                            <div style="display: flex; justify-content: space-around;">
                                <div><small>AI Confidence</small><br><b>{prob*100:.1f}%</b></div>
                                <div><small>Social Mood</small><br><b>{avg_sent:.2f}</b></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No model found.")
                if st.button("Train AI Model for " + ticker):
                    with st.spinner("Training..."):
                        ml.train_and_save(ticker, data)
                        st.success("Trained!")
                        st.rerun()
        
        st.markdown("---")
        st.info("💡 **Insight:** This dashboard fuses technical AI analysis (LSTM) with real-time crowd psychology (Reddit) for a unified market view.")

if __name__ == "__main__":
    main()
