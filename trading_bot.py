#!/usr/bin/env python3
"""
Advanced Swing Trading Bot 10/10 – Full Version with Top Picks
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
        self.failed_tickers = []  # Track failed tickers
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
        default_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        if os.path.exists('tickers.json'):
            try:
                with open('tickers.json', 'r') as f:
                    data = json.load(f)
                if isinstance(data, dict) and 'tickers' in data:
                    tickers = data['tickers']
                elif isinstance(data, list):
                    tickers = data
                else:
                    raise ValueError("tickers.json must contain a list or an object with 'tickers' key")
                if not all(isinstance(t, str) for t in tickers):
                    raise ValueError("All items in tickers.json must be strings")
                if not tickers:
                    raise ValueError("tickers.json is empty")
                logger.info(f"Loaded {len(tickers)} tickers from tickers.json: {tickers[:5]}{'...' if len(tickers) > 5 else ''}")
                return tickers
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON format in tickers.json: {e}. Using default tickers: {default_tickers}")
            except Exception as e:
                logger.error(f"Error loading tickers.json: {e}. Using default tickers: {default_tickers}")
        else:
            logger.warning(f"tickers.json not found. Using default tickers: {default_tickers}")
        return default_tickers

    # ----------------- Fetch Data -----------------
    def fetch_stock_data(self, ticker: str, period: str = '6mo', interval: str = '1d') -> Optional[pd.DataFrame]:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)
            if data.empty or len(data) < 50:
                logger.warning(f"Insufficient data for {ticker} (rows: {len(data)})")
                self.failed_tickers.append(ticker)
                return None
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                logger.warning(f"Missing required columns for {ticker}: {data.columns}")
                self.failed_tickers.append(ticker)
                return None
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            self.failed_tickers.append(ticker)
            return None

    # ----------------- Indicators -----------------
    def calculate_advanced_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            if len(df) < 50:
                logger.warning(f"Insufficient rows ({len(df)}) for indicators calculation")
                return None
            df = df.copy()
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
            df['RSI'] = self.compute_RSI(df['Close'], 14)
            df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = self.compute_MACD(df['Close'])
            df['BB_Middle'] = df['Close'].rolling(20).mean()
            df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(20).std()
            df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(20).std()
            if df['BB_Upper'].isna().all() or df['BB_Lower'].isna().all() or df['RSI'].isna().all():
                logger.warning("Invalid Bollinger Bands or RSI values")
                return None
            df['SlowK'], df['SlowD'] = self.compute_stochastic(df)
            df['Volume_MA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            df['ATR'] = self.compute_ATR(df)
            df['ADX'] = self.compute_ADX(df)
            df['Price_vs_EMA20'] = df['Close'] / df['EMA_20'] - 1
            df['Price_vs_EMA50'] = df['Close'] / df['EMA_50'] - 1
            df = df.fillna(0)
            return df
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return None

    # ----------------- Helper Indicator Functions -----------------
    def compute_RSI(self, series: pd.Series, period: int = 14) -> pd.Series:
        try:
            delta = series.diff()
            gain = delta.clip(lower=0)
            loss = -1 * delta.clip(upper=0)
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        except Exception as e:
            logger.error(f"Error in RSI calculation: {e}")
            return pd.Series(50, index=series.index)

    def compute_MACD(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        try:
            fast_ema = series.ewm(span=fast, adjust=False).mean()
            slow_ema = series.ewm(span=slow, adjust=False).mean()
            macd = fast_ema - slow_ema
            macd_signal = macd.ewm(span=signal, adjust=False).mean()
            macd_hist = macd - macd_signal
            return macd.fillna(0), macd_signal.fillna(0), macd_hist.fillna(0)
        except Exception as e:
            logger.error(f"Error in MACD calculation: {e}")
            return pd.Series(0, index=series.index), pd.Series(0, index=series.index), pd.Series(0, index=series.index)

    def compute_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
        try:
            low_min = df['Low'].rolling(k_period).min()
            high_max = df['High'].rolling(k_period).max()
            k = 100 * (df['Close'] - low_min) / (high_max - low_min)
            d = k.rolling(d_period).mean()
            return k.fillna(50), d.fillna(50)
        except Exception as e:
            logger.error(f"Error in stochastic calculation: {e}")
            return pd.Series(50, index=df.index), pd.Series(50, index=df.index)

    def compute_ATR(self, df: pd.DataFrame, period: int = 14):
        try:
            high_low = df['High'] - df['Low']
            high_close = (df['High'] - df['Close'].shift()).abs()
            low_close = (df['Low'] - df['Close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            return atr.fillna(0)
        except Exception as e:
            logger.error(f"Error in ATR calculation: {e}")
            return pd.Series(0, index=df.index)

    def compute_ADX(self, df: pd.DataFrame, period: int = 14):
        try:
            plus_dm = df['High'].diff()
            minus_dm = df['Low'].diff().abs()
            tr = self.compute_ATR(df, period)
            dx = 100 * (plus_dm - minus_dm).abs() / tr
            adx = dx.rolling(period).mean()
            return adx.fillna(0)
        except Exception as e:
            logger.error(f"Error in ADX calculation: {e}")
            return pd.Series(0, index=df.index)

    # ----------------- News Sentiment -----------------
    def get_news_sentiment(self, ticker: str) -> float:
        try:
            if not self.api_keys['news_api']:
                logger.warning(f"News API key missing for {ticker}. Using neutral sentiment. Get a key from https://newsapi.org/")
                return 0.0
            url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={self.api_keys['news_api']}&language=en&from={datetime.now().date() - timedelta(days=14)}"
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
    def prepare_ml_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            if len(df) < 6:
                logger.warning(f"Insufficient data rows ({len(df)}) for ML features")
                return None
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
        except Exception as e:
            logger.error(f"Error preparing ML features: {e}")
            return None

    def prepare_ml_data(self, df: pd.DataFrame):
        try:
            if len(df) < 50:
                logger.warning(f"Insufficient data rows ({len(df)}) for ML data")
                return pd.DataFrame(), pd.Series(dtype='int64')
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
            return features_df, pd.Series(targets, dtype='int64')
        except Exception as e:
            logger.error(f"Error preparing ML data: {e}")
            return pd.DataFrame(), pd.Series(dtype='int64')

    def train_ml_model_all_tickers(self):
        all_features = pd.DataFrame()
        all_targets = pd.Series(dtype='int64')
        self.failed_tickers = []
        for ticker in self.tickers:
            df = self.fetch_stock_data(ticker)
            if df is None:
                continue
            df = self.calculate_advanced_indicators(df)
            if df is None:
                self.failed_tickers.append(ticker)
                continue
            features, targets = self.prepare_ml_data(df)
            if not features.empty and not targets.empty:
                all_features = pd.concat([all_features, features], ignore_index=True)
                all_targets = pd.concat([all_targets, targets], ignore_index=True)
            else:
                logger.warning(f"No valid ML data for {ticker}")
                self.failed_tickers.append(ticker)
        if all_features.empty or len(all_targets) < 10:
            logger.warning("Insufficient data for ML training")
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
            if self.failed_tickers:
                logger.info(f"Failed tickers during ML training: {self.failed_tickers}")
        except Exception as e:
            logger.error(f"Error training ML: {e}")
            self.model = None

    def get_ml_signal(self, df: pd.DataFrame) -> str:
        if self.model is None:
            return "HOLD"
        try:
            features = self.prepare_ml_features(df)
            if features is None:
                return "HOLD"
            features_scaled = self.scaler.transform(features)
            pred = self.model.predict(features_scaled)[0]
            return "BUY" if pred == 1 else "SELL"
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            return "HOLD"

    # ----------------- Strategies -----------------
    def strategy_trend_following(self, df: pd.DataFrame) -> str:
        try:
            if len(df) < 50 or df['EMA_20'].isna().all() or df['EMA_50'].isna().all() or df['EMA_200'].isna().all():
                return "HOLD"
            latest = df.iloc[-1]
            if latest['EMA_20'] > latest['EMA_50'] > latest['EMA_200'] and latest['ADX'] > 25:
                return "BUY"
            elif latest['EMA_20'] < latest['EMA_50'] < latest['EMA_200']:
                return "SELL"
            return "HOLD"
        except Exception as e:
            logger.error(f"Error in trend following strategy: {e}")
            return "HOLD"

    def strategy_mean_reversion(self, df: pd.DataFrame) -> str:
        try:
            if len(df) < 50 or df['RSI'].isna().all() or df['BB_Lower'].isna().all() or df['BB_Upper'].isna().all():
                return "HOLD"
            latest = df.iloc[-1]
            if latest['RSI'] < 30 and latest['Close'] < latest['BB_Lower']:
                return "BUY"
            elif latest['RSI'] > 70 and latest['Close'] > latest['BB_Upper']:
                return "SELL"
            return "HOLD"
        except Exception as e:
            logger.error(f"Error in mean reversion strategy: {e}")
            return "HOLD"

    def strategy_breakout(self, df: pd.DataFrame) -> str:
        try:
            if len(df) < 50 or df['High'].isna().all() or df['Low'].isna().all():
                return "HOLD"
            latest = df.iloc[-1]
            high_20 = df['High'].rolling(20).max().iloc[-2]
            low_20 = df['Low'].rolling(20).min().iloc[-2]
            if latest['Close'] > high_20:
                return "BUY"
            elif latest['Close'] < low_20:
                return "SELL"
            return "HOLD"
        except Exception as e:
            logger.error(f"Error in breakout strategy: {e}")
            return "HOLD"

    # ----------------- Portfolio Management -----------------
    def update_portfolio(self, ticker: str, signal: str):
        try:
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
        except Exception as e:
            logger.error(f"Error updating portfolio for {ticker}: {e}")

    # ----------------- Telegram Alerts -----------------
    def send_telegram_message(self, message: str):
        try:
            token = self.api_keys['telegram_bot']
            chat_id = self.api_keys['telegram_chat_id']
            if not token or not chat_id:
                logger.warning("Telegram keys missing. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in GitHub Secrets.")
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
        try:
            if len(df) < 50:
                logger.warning("Insufficient data for backtesting")
                return {"Total_Return": 0.0, "Win_Rate": 0.0}
            df = df.copy()
            signals = []
            failed_logged = False
            for i in range(len(df)):
                window = df.iloc[:i+1]
                if len(window) < 50:
                    signals.append("HOLD")
                    continue
                signal = strategy_func(window)
                if signal == "HOLD" and not failed_logged:
                    if len(window) < 50 or window['RSI'].isna().all() or window['BB_Lower'].isna().all() or window['BB_Upper'].isna().all():
                        logger.warning(f"Invalid data for {strategy_func.__name__} during backtest")
                        failed_logged = True
                signals.append(signal)
            df['Signal'] = signals
            df['Returns'] = df['Close'].pct_change().shift(-1) * df['Signal'].map({"BUY": 1, "SELL": -1, "HOLD": 0})
            total_return = df['Returns'].sum()
            win_rate = (df['Returns'] > 0).mean() if len(df['Returns']) > 0 else 0
            return {"Total_Return": total_return, "Win_Rate": win_rate}
        except Exception as e:
            logger.error(f"Error in backtesting: {e}")
            return {"Total_Return": 0.0, "Win_Rate": 0.0}

    # ----------------- Calculate Trade Details -----------------
    def calculate_trade_details(self, df: pd.DataFrame, signal: str) -> Dict[str, float]:
        try:
            latest = df.iloc[-1]
            entry_price = latest['Close']
            atr = latest['ATR']
            if signal == "BUY":
                take_profit = entry_price + 2 * atr
                stop_loss = entry_price - atr
            elif signal == "SELL":
                take_profit = entry_price - 2 * atr
                stop_loss = entry_price + atr
            else:
                take_profit = stop_loss = entry_price
            return {
                "Entry_Price": entry_price,
                "Take_Profit": take_profit,
                "Stop_Loss": stop_loss
            }
        except Exception as e:
            logger.error(f"Error calculating trade details: {e}")
            return {"Entry_Price": 0.0, "Take_Profit": 0.0, "Stop_Loss": 0.0}

    # ----------------- Run Bot -----------------
    def run_once(self):
        logger.info("Starting Advanced Swing Trading Bot - Single Run")
        self.failed_tickers = []
        results = []
        for ticker in self.tickers:
            try:
                df = self.fetch_stock_data(ticker)
                if df is None:
                    continue
                df = self.calculate_advanced_indicators(df)
                if df is None:
                    self.failed_tickers.append(ticker)
                    continue
                signal_trend = self.strategy_trend_following(df)
                signal_mean = self.strategy_mean_reversion(df)
                signal_breakout = self.strategy_breakout(df)
                signal_ml = self.get_ml_signal(df)
                signals = [signal_trend, signal_mean, signal_breakout, signal_ml]
                final_signal = max(set(signals), key=signals.count)
                sentiment_news = self.get_news_sentiment(ticker)
                backtest_trend = self.backtest_strategy(df, self.strategy_trend_following)
                backtest_mean = self.backtest_strategy(df, self.strategy_mean_reversion)
                backtest_breakout = self.backtest_strategy(df, self.strategy_breakout)
                trade_details = self.calculate_trade_details(df, final_signal)
                logger.info(
                    f"Processing {ticker}:\n"
                    f"  Signal: {final_signal} (ML: {signal_ml})\n"
                    f"  Trend: {signal_trend}, Mean: {signal_mean}, Breakout: {signal_breakout}\n"
                    f"  News Sentiment: {sentiment_news:.2f}\n"
                    f"  Backtest: Trend {backtest_trend['Total_Return']:.2f} (Win: {backtest_trend['Win_Rate']:.2f}), "
                    f"Mean {backtest_mean['Total_Return']:.2f} (Win: {backtest_mean['Win_Rate']:.2f}), "
                    f"Breakout {backtest_breakout['Total_Return']:.2f} (Win: {backtest_breakout['Win_Rate']:.2f})\n"
                    f"  Trade: Entry ${trade_details['Entry_Price']:.2f}, TP ${trade_details['Take_Profit']:.2f}, SL ${trade_details['Stop_Loss']:.2f}"
                )
                # Collect result for summary and filtering
                result = {
                    'Ticker': ticker,
                    'Final_Signal': final_signal,
                    'ML_Signal': signal_ml,
                    'Trend_Signal': signal_trend,
                    'Mean_Signal': signal_mean,
                    'Breakout_Signal': signal_breakout,
                    'News_Sentiment': sentiment_news,
                    'Backtest_Trend': backtest_trend['Total_Return'],
                    'Backtest_Trend_Win': backtest_trend['Win_Rate'],
                    'Backtest_Mean': backtest_mean['Total_Return'],
                    'Backtest_Mean_Win': backtest_mean['Win_Rate'],
                    'Backtest_Breakout': backtest_breakout['Total_Return'],
                    'Backtest_Breakout_Win': backtest_breakout['Win_Rate'],
                    'Entry_Price': trade_details['Entry_Price'],
                    'Take_Profit': trade_details['Take_Profit'],
                    'Stop_Loss': trade_details['Stop_Loss']
                }
                results.append(result)
                # Send Telegram message only for top picks
                if final_signal != "HOLD":
                    weighted_score = (
                        0.4 * max(backtest_trend['Total_Return'], backtest_mean['Total_Return'], backtest_breakout['Total_Return']) +
                        0.3 * max(backtest_trend['Win_Rate'], backtest_mean['Win_Rate'], backtest_breakout['Win_Rate']) +
                        0.3 * abs(sentiment_news)
                    )
                    if (final_signal == "BUY" and sentiment_news > 0.1) or (final_signal == "SELL" and sentiment_news < -0.1):
                        if max(backtest_trend['Total_Return'], backtest_mean['Total_Return'], backtest_breakout['Total_Return']) > 0.05:
                            if max(backtest_trend['Win_Rate'], backtest_mean['Win_Rate'], backtest_breakout['Win_Rate']) > 0.6:
                                win_rate = max(
                                    (backtest_trend['Win_Rate'] if signal_trend == final_signal else 0),
                                    (backtest_mean['Win_Rate'] if signal_mean == final_signal else 0),
                                    (backtest_breakout['Win_Rate'] if signal_breakout == final_signal else 0)
                                )
                                message = (
                                    f"*{ticker}* - *{final_signal}*\n"
                                    f"Entry: ${trade_details['Entry_Price']:.2f}\n"
                                    f"Take Profit: ${trade_details['Take_Profit']:.2f}\n"
                                    f"Stop Loss: ${trade_details['Stop_Loss']:.2f}\n"
                                    f"Win Rate: {win_rate*100:.1f}%\n"
                                    f"News Sentiment: {sentiment_news:.2f}\n"
                                    f"Backtest: Trend {backtest_trend['Total_Return']:.2f}, Mean {backtest_mean['Total_Return']:.2f}, "
                                    f"Breakout {backtest_breakout['Total_Return']:.2f}\n"
                                    f"Score: {weighted_score:.2f}"
                                )
                                self.send_telegram_message(message)
                self.visualize_signals(df, ticker)
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                self.failed_tickers.append(ticker)
        # Create and save summary
        if results:
            results_df = pd.DataFrame(results)
            # Sort by weighted score for summary
            results_df['Weighted_Score'] = (
                0.4 * results_df[['Backtest_Trend', 'Backtest_Mean', 'Backtest_Breakout']].max(axis=1) +
                0.3 * results_df[['Backtest_Trend_Win', 'Backtest_Mean_Win', 'Backtest_Breakout_Win']].max(axis=1) +
                0.3 * results_df['News_Sentiment'].abs()
            )
            top_picks = results_df[results_df['Final_Signal'] != "HOLD"].sort_values(by='Weighted_Score', ascending=False).head(5)
            summary = results_df.to_string(index=False)
            logger.info(f"Run Summary:\n{summary}")
            if not top_picks.empty:
                top_picks_summary = top_picks.to_string(index=False)
                self.send_telegram_message(f"*Top 5 Picks*\n```\n{top_picks_summary}\n```")
            results_df.to_csv('trading_bot_summary.csv', index=False)
            logger.info("Summary saved to trading_bot_summary.csv")
        if self.failed_tickers:
            logger.info(f"Failed tickers during run: {self.failed_tickers}")
            self.send_telegram_message(f"*Failed Tickers*\n{self.failed_tickers}")
        logger.info("Single run completed.")

    def run_forever(self):
        logger.info("Starting bot in continuous mode - Checks every 60 minutes")
        try:
            while True:
                tz = pytz.timezone('Asia/Jerusalem')
                now = datetime.now(tz)
                is_manual_run = os.getenv('GITHUB_EVENT_NAME') == 'workflow_dispatch'
                if is_manual_run:
                    logger.info(f"Manual run triggered at {now}. Running bot immediately...")
                    self.run_once()
                elif now.weekday() in [0, 1, 2, 3, 4] and 15 <= now.hour <= 23:
                    logger.info(f"Trading hours: {now}. Running bot...")
                    self.run_once()
                else:
                    logger.info(f"Not in trading hours: {now}. Skipping.")
                logger.info("Waiting 60 minutes for next check...")
                time.sleep(3600)
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Unexpected error in run_forever: {e}")
            time.sleep(60)

if __name__ == "__main__":
    bot = AdvancedSwingTradingBot()
    bot.run_forever()
