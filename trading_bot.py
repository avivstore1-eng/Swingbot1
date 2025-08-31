#!/usr/bin/env python3
"""
Advanced Swing Trading Bot 10/10 – Full Version
Free, GitHub Actions ready, no TA-Lib, no Alpha Vantage, JSON tickers
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import time
import os
import logging
import json
from typing import Dict, List, Optional
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from textblob import TextBlob
import matplotlib.pyplot as plt
import pytz

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('trading_bot.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class AdvancedSwingTradingBot:
    def __init__(self):
        self.tickers = self.load_tickers()  # Load tickers from JSON or default
        self.model = None
        self.scaler = StandardScaler()
        self.portfolio = {}
        self.performance_history = []
        self.api_keys = {
            'news_api': os.getenv('NEWS_API_KEY'),
            'telegram_bot': os.getenv('TELEGRAM_BOT_TOKEN'),
            'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID')
        }
        logger.info("Bot initialized. Training ML model...")
        self.train_ml_model_all_tickers()

    # ----------------- Ticker Management -----------------
    def load_tickers(self) -> List[str]:
        """Load tickers from tickers.json if exists, else use default."""
        if os.path.exists('tickers.json'):
            try:
                with open('tickers.json', 'r') as f:
                    tickers = json.load(f)
                if not isinstance(tickers, list) or not all(isinstance(t, str) for t in tickers):
                    raise ValueError("tickers.json must contain a list of strings")
                logger.info(f"Loaded {len(tickers)} tickers from tickers.json")
                return tickers
            except Exception as e:
                logger.error(f"Error loading tickers.json: {e}. Using default tickers.")
        logger.info("tickers.json not found. Using default tickers.")
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

    def add_ticker(self, ticker: str):
        if ticker not in self.tickers:
            self.tickers.append(ticker)
            logger.info(f"Added ticker: {ticker}")

    def remove_ticker(self, ticker: str):
        if ticker in self.tickers:
            self.tickers.remove(ticker)
            logger.info(f"Removed ticker: {ticker}")

    # ----------------- Fetch Data -----------------
    def fetch_stock_data(self, ticker: str, period: str = '6mo', interval: str = '1d') -> Optional[pd.DataFrame]:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)
            if data.empty or len(data) < 50:
                logger.warning(f"Insufficient data for {ticker}")
                return None
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None

    # ----------------- Indicators -----------------
    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        df['RSI'] = self.compute_RSI(df['Close'], 14)
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = self.compute_MACD(df['Close'])
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(20).std()
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(20).std()
        df['SlowK'], df['SlowD'] = self.compute_stochastic(df)
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        df['ATR'] = self.compute_ATR(df)
        df['ADX'] = self.compute_ADX(df)
        df['Price_vs_EMA20'] = df['Close'] / df['EMA_20'] - 1
        df['Price_vs_EMA50'] = df['Close'] / df['EMA_50'] - 1
        df = df.fillna(0)  # Handle NaNs
        return df

    # ----------------- Helper Indicator Functions -----------------
    def compute_RSI(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def compute_MACD(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        fast_ema = series.ewm(span=fast, adjust=False).mean()
        slow_ema = series.ewm(span=slow, adjust=False).mean()
        macd = fast_ema - slow_ema
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd.fillna(0), macd_signal.fillna(0), macd_hist.fillna(0)

    def compute_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
        low_min = df['Low'].rolling(k_period).min()
        high_max = df['High'].rolling(k_period).max()
        k = 100 * (df['Close'] - low_min) / (high_max - low_min)
        d = k.rolling(d_period).mean()
        return k.fillna(50), d.fillna(50)

    def compute_ATR(self, df: pd.DataFrame, period: int = 14):
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr.fillna(0)

    def compute_ADX(self, df: pd.DataFrame, period: int = 14):
        plus_dm = df['High'].diff()
        minus_dm = df['Low'].diff().abs()
        tr = self.compute_ATR(df, period)
        dx = 100 * (plus_dm - minus_dm).abs() / tr
        adx = dx.rolling(period).mean()
        return adx.fillna(0)

    # ----------------- News Sentiment -----------------
    def get_news_sentiment(self, ticker: str) -> float:
        try:
            if not self.api_keys['news_api']:
                logger.warning(f"News API key missing for {ticker}. Using neutral sentiment.")
                return 0.0
            url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={self.api_keys['news_api']}"
            response = requests.get(url)
            response.raise_for_status()
            articles = response.json().get('articles', [])[:10]
            sentiments = []
            for art in articles:
                if art.get('title') and art.get('description'):
                    text = f"{art['title']}. {art['description']}"
                    sentiments.append(TextBlob(text).sentiment.polarity)
            return np.mean(sentiments) if sentiments else 0.0
        except Exception as e:
            logger.error(f"Error in news sentiment for {ticker}: {e}")
            return 0.0

    # ----------------- ML Features & Training -----------------
    def prepare_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        latest = df.iloc[-1]
        features = {
            'EMA_20_vs_50': latest['EMA_20'] / latest['EMA_50'] - 1,
            'EMA_20_vs_200': latest['EMA_20'] / latest['EMA_200'] - 1,
            'Price_vs_EMA20': latest['Price_vs_EMA20'],
            'Price_vs_EMA50': latest['Price_vs_EMA50'],
            'RSI': latest['RSI'],
            'MACD_Hist': latest['MACD_Hist'],
            'BB_Position': (latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower']),
            'Volume_Ratio': latest['Volume_Ratio'],
            'ATR': latest['ATR'],
            'ADX': latest['ADX'],
            'Stochastic': latest['SlowK'],
            'Price_Change_1d': df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1 if len(df) > 1 else 0,
            'Price_Change_5d': df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1 if len(df) > 5 else 0
        }
        return pd.DataFrame([features])

    def prepare_ml_data(self, df: pd.DataFrame):
        features_df = pd.DataFrame()
        targets = []
        for i in range(50, len(df) - 5):
            window = df.iloc[i-50:i]
            latest = window.iloc[-1]
            features = {
                'EMA_20_vs_50': latest['EMA_20'] / latest['EMA_50'] - 1,
                'EMA_20_vs_200': latest['EMA_20'] / latest['EMA_200'] - 1,
                'Price_vs_EMA20': latest['Price_vs_EMA20'],
                'Price_vs_EMA50': latest['Price_vs_EMA50'],
                'RSI': latest['RSI'],
                'MACD_Hist': latest['MACD_Hist'],
                'BB_Position': (latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower']),
                'Volume_Ratio': latest['Volume_Ratio'],
                'ATR': latest['ATR'],
                'ADX': latest['ADX'],
                'Stochastic': latest['SlowK'],
                'Price_Change_1d': window['Close'].iloc[-1] / window['Close'].iloc[-2] - 1 if len(window) > 1 else 0,
                'Price_Change_5d': window['Close'].iloc[-1] / window['Close'].iloc[-6] - 1 if len(window) > 5 else 0
            }
            features_df = pd.concat([features_df, pd.DataFrame([features])], ignore_index=True)
            future_close = df['Close'].iloc[i + 5]
            targets.append(1 if future_close > latest['Close'] else 0)
        return features_df, pd.Series(targets)

    def train_ml_model_all_tickers(self):
        all_features = pd.DataFrame()
        all_targets = pd.Series()
        for ticker in self.tickers:
            df = self.fetch_stock_data(ticker)
            if df is None:
                continue
            df = self.calculate_advanced_indicators(df)
            features, targets = self.prepare_ml_data(df)
            all_features = pd.concat([all_features, features], ignore_index=True)
            all_targets = pd.concat([all_targets, targets], ignore_index=True)
        if all_features.empty or len(all_targets) < 10:
            logger.warning("Insufficient data for ML training. Skipping.")
            return
        try:
            X_train, X_test, y_train, y_test = train_test_split(all_features, all_targets, test_size=0.2, random_state=42)
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            self.model.fit(X_train_scaled, y_train)
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            logger.info(f"ML Model trained - Train: {train_score:.2f}, Test: {test_score:.2f}")
        except Exception as e:
            logger.error(f"Error training ML: {e}")
            self.model = None

    def get_ml_signal(self, df: pd.DataFrame) -> str:
        if self.model is None:
            return "HOLD"
        try:
            features = self.prepare_ml_features(df)
            features_scaled = self.scaler.transform(features)
            pred = self.model.predict(features_scaled)[0]
            return "BUY" if pred == 1 else "SELL"
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            return "HOLD"

    # ----------------- Strategies -----------------
    def strategy_trend_following(self, df: pd.DataFrame) -> str:
        latest = df.iloc[-1]
        if latest['EMA_20'] > latest['EMA_50'] > latest['EMA_200'] and latest['ADX'] > 25:
            return "BUY"
        elif latest['EMA_20'] < latest['EMA_50'] < latest['EMA_200']:
            return "SELL"
        return "HOLD"

    def strategy_mean_reversion(self, df: pd.DataFrame) -> str:
        latest = df.iloc[-1]
        if latest['RSI'] < 30 and latest['Close'] < latest['BB_Lower']:
            return "BUY"
        elif latest['RSI'] > 70 and latest['Close'] > latest['BB_Upper']:
            return "SELL"
        return "HOLD"

    def strategy_breakout(self, df: pd.DataFrame) -> str:
        latest = df.iloc[-1]
        high_20 = df['High'].rolling(20).max().iloc[-2]
        low_20 = df['Low'].rolling(20).min().iloc[-2]
        if latest['Close'] > high_20:
            return "BUY"
        elif latest['Close'] < low_20:
            return "SELL"
        return "HOLD"

    # ----------------- Portfolio Management -----------------
    def update_portfolio(self, ticker: str, signal: str):
        if ticker not in self.portfolio:
            self.portfolio[ticker] = {"position": 0, "last_signal": "HOLD"}
        current = self.portfolio[ticker]
        if sum(abs(p['position']) for p in self.portfolio.values()) >= 5 and signal != "HOLD":
            logger.warning(f"Portfolio limit reached. Skipping {signal} for {ticker}")
            return
        current['last_signal'] = signal
        if signal == "BUY":
            current['position'] = 1
        elif signal == "SELL":
            current['position'] = -1
        else:
            current['position'] = 0
        self.portfolio[ticker] = current
        logger.info(f"Portfolio updated: {ticker} -> {current}")

    # ----------------- Telegram Alerts -----------------
    def send_telegram_message(self, message: str):
        try:
            token = self.api_keys['telegram_bot']
            chat_id = self.api_keys['telegram_chat_id']
            if not token or not chat_id:
                logger.warning("Telegram keys missing. Skipping message.")
                return
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
            response = requests.post(url, data=payload)
            response.raise_for_status()
            logger.info("Telegram message sent successfully")
        except Exception as e:
            logger.error(f"Telegram send error: {e}")

    # ----------------- Visualization -----------------
    def visualize_signals(self, df: pd.DataFrame, ticker: str):
        try:
            if df['Close'].isna().all() or df['EMA_20'].isna().all() or df['EMA_50'].isna().all():
                logger.warning(f"Invalid data for visualization in {ticker}")
                return
            plt.figure(figsize=(12, 6))
            plt.plot(df['Close'], label='Close', color='blue')
            plt.plot(df['EMA_20'], label='EMA 20', color='green')
            plt.plot(df['EMA_50'], label='EMA 50', color='red')
            plt.title(f"{ticker} Price & EMA")
            plt.legend()
            plt.savefig(f"{ticker}_chart.png")
            plt.close()
            logger.info(f"Chart saved: {ticker}_chart.png")
        except Exception as e:
            logger.error(f"Error visualizing {ticker}: {e}")

    # ----------------- Backtesting -----------------
    def backtest_strategy(self, df: pd.DataFrame, strategy_func) -> Dict[str, float]:
        df = df.copy()
        signals = []
        for i in range(len(df)):
            signal = strategy_func(df.iloc[:i+1])
            signals.append(signal)
        df['Signal'] = signals
        df['Returns'] = df['Close'].pct_change().shift(-1) * df['Signal'].map({"BUY": 1, "SELL": -1, "HOLD": 0})
        total_return = df['Returns'].sum()
        win_rate = (df['Returns'] > 0).mean() if len(df['Returns']) > 0 else 0
        return {"Total_Return": total_return, "Win_Rate": win_rate}

    # ----------------- Run Bot -----------------
    def run_once(self):
        logger.info("Starting Advanced Swing Trading Bot - Single Run")
        for ticker in self.tickers:
            df = self.fetch_stock_data(ticker)
            if df is None:
                continue
            df = self.calculate_advanced_indicators(df)
            signal_trend = self.strategy_trend_following(df)
            signal_mean = self.strategy_mean_reversion(df)
            signal_breakout = self.strategy_breakout(df)
            signal_ml = self.get_ml_signal(df)
            signals = [signal_trend, signal_mean, signal_breakout, signal_ml]
            final_signal = max(set(signals), key=signals.count)
            self.update_portfolio(ticker, final_signal)
            sentiment_news = self.get_news_sentiment(ticker)
            backtest_trend = self.backtest_strategy(df, self.strategy_trend_following)
            backtest_mean = self.backtest_strategy(df, self.strategy_mean_reversion)
            backtest_breakout = self.backtest_strategy(df, self.strategy_breakout)
            logger.info(f"Backtest for {ticker}: Trend {backtest_trend}, Mean {backtest_mean}, Breakout {backtest_breakout}")
            message = (
                f"*{ticker}*\n"
                f"Signal: {final_signal} (ML: {signal_ml})\n"
                f"Trend: {signal_trend}, Mean: {signal_mean}, Breakout: {signal_breakout}\n"
                f"News Sentiment: {sentiment_news:.2f}\n"
                f"Backtest: Trend {backtest_trend['Total_Return']:.2f}, Mean {backtest_mean['Total_Return']:.2f}, "
                f"Breakout {backtest_breakout['Total_Return']:.2f}"
            )
            self.send_telegram_message(message)
            self.visualize_signals(df, ticker)
            time.sleep(0.1)  # Tiny delay to avoid overwhelming yfinance/NewsAPI
        logger.info("Single run completed.")

    def run_forever(self):
        logger.info("Starting bot in continuous mode - Checks every 60 minutes")
        try:
            while True:
                tz = pytz.timezone('Asia/Jerusalem')
                now = datetime.now(tz)
                if now.weekday() in [0, 1, 2, 3, 4] and 15 <= now.hour <= 23:
                    logger.info(f"Trading hours: {now}. Running bot...")
                    self.run_once()
                else:
                    logger.info(f"Not in trading hours: {now}. Skipping.")
                logger.info("Waiting 60 minutes for next check...")
                time.sleep(3600)  # 60 minutes
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Unexpected error in run_forever: {e}")
            time.sleep(60)  # Retry after 1 min

if __name__ == "__main__":
    bot = AdvancedSwingTradingBot()
    bot.run_forever()
