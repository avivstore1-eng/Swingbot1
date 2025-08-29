#!/usr/bin/env python3
"""
Swing Trading Bot - בוט מתקדם למסחר סווינג חודשי
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import time
import json
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import logging
from typing import Dict, List, Optional, Any
import talib
from textblob import TextBlob
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

# טעינת environment variables
load_dotenv()

# הגדרת logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedSwingTradingBot:
    def __init__(self):
        self.tickers = self.load_custom_tickers()
        self.model = None
        self.scaler = StandardScaler()
        self.portfolio = {}
        self.performance_history = []
        
        # טעינת API keys מ-environment variables
        self.api_keys = {
            'news_api': os.getenv('NEWS_API_KEY'),
            'alpha_vantage': os.getenv('ALPHA_VANTAGE_KEY'),
            'telegram_bot': os.getenv('TELEGRAM_BOT_TOKEN'),
            'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID')
        }
        
    def load_custom_tickers(self) -> List[str]:
        """טעינת רשימת מניות מותאמת אישית מקובץ JSON"""
        try:
            with open('tickers.json', 'r') as f:
                data = json.load(f)
                return data.get('tickers', [])
        except FileNotFoundError:
            logger.warning("tickers.json not found. Using default tickers.")
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    def save_custom_tickers(self, tickers: List[str]):
        """שמירת רשימת מניות מותאמת אישית"""
        os.makedirs(os.path.dirname('tickers.json'), exist_ok=True)
        with open('tickers.json', 'w') as f:
            json.dump({'tickers': tickers}, f, indent=2)
    
    def add_ticker(self, ticker: str):
        """הוספת טיקר לרשימה"""
        if ticker not in self.tickers:
            self.tickers.append(ticker)
            self.save_custom_tickers(self.tickers)
            logger.info(f"Added ticker: {ticker}")
    
    def remove_ticker(self, ticker: str):
        """הסרת טיקר מהרשימה"""
        if ticker in self.tickers:
            self.tickers.remove(ticker)
            self.save_custom_tickers(self.tickers)
            logger.info(f"Removed ticker: {ticker}")
    
    def fetch_stock_data(self, ticker: str, period: str = '6mo', interval: str = '1d') -> Optional[pd.DataFrame]:
        """הורדת נתוני מניה מ-Yahoo Finance"""
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
    
    def calculate_advanced_indicators(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """חישוב אינדיקטורים טכניים מתקדמים עם TA-Lib"""
        if data is None or len(data) < 50:
            return None
            
        df = data.copy()
        
        # ממוצעים נעים
        df['EMA_20'] = talib.EMA(df['Close'], timeperiod=20)
        df['EMA_50'] = talib.EMA(df['Close'], timeperiod=50)
        df['EMA_200'] = talib.EMA(df['Close'], timeperiod=200)
        
        # RSI
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        
        # MACD
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(
            df['Close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # Bollinger Bands
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(
            df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        
        # Stochastic
        df['SlowK'], df['SlowD'] = talib.STOCH(
            df['High'], df['Low'], df['Close'], 
            fastk_period=14, slowk_period=3, slowd_period=3
        )
        
        # Volume indicators
        df['Volume_MA'] = talib.SMA(df['Volume'], timeperiod=20)
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # תנודתיות
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # מחיר יחסי ל-EMA
        df['Price_vs_EMA20'] = df['Close'] / df['EMA_20'] - 1
        df['Price_vs_EMA50'] = df['Close'] / df['EMA_50'] - 1
        
        return df
    
    def get_news_sentiment(self, ticker: str) -> float:
        """קבלת סנטימנט מחדשות עם News API"""
        try:
            if not self.api_keys['news_api']:
                logger.warning("News API key not configured. Using demo sentiment.")
                return np.random.uniform(-0.5, 0.5)
            
            # שימוש ב-News API אמיתי
            url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={self.api_keys['news_api']}"
            response = requests.get(url)
            
            if response.status_code == 200:
                articles = response.json().get('articles', [])
                sentiments = []
                
                for article in articles[:10]:  # ניתוח 10 המאמרים הראשונים
                    if article['title'] and article['description']:
                        text = f"{article['title']}. {article['description']}"
                        analysis = TextBlob(text)
                        sentiments.append(analysis.sentiment.polarity)
                
                if sentiments:
                    return sum(sentiments) / len(sentiments)
                else:
                    return 0
            else:
                logger.error(f"News API error: {response.status_code}")
                return 0
                
        except Exception as e:
            logger.error(f"Error getting news sentiment for {ticker}: {e}")
            return 0
    
    def get_social_sentiment(self, ticker: str) -> float:
        """קבלת סנטימנט מרשתות חברתיות"""
        try:
            # כאן יש לשלב API של Twitter או Reddi
            # לדוגמה, שימוש ב-Alpha Vantage עבור sentiment
            if not self.api_keys['alpha_vantage']:
                logger.warning("Alpha Vantage API key not configured. Using demo sentiment.")
                return np.random.uniform(-0.3, 0.3)
            
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={self.api_keys['alpha_vantage']}"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                feed = data.get('feed', [])
                
                if feed:
                    sentiments = [float(item.get('overall_sentiment_score', 0)) for item in feed[:5]]
                    return sum(sentiments) / len(sentiments) if sentiments else 0
                else:
                    return 0
            else:
                logger.error(f"Alpha Vantage API error: {response.status_code}")
                return 0
                
        except Exception as e:
            logger.error(f"Error getting social sentiment for {ticker}: {e}")
            return 0
    
    def prepare_ml_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """הכנת מאפיינים למודל ML"""
        if df is None or len(df) < 50:
            return None
            
        latest = df.iloc[-1]
        
        features = {
            'EMA_20_vs_50': latest['EMA_20'] / latest['EMA_50'] - 1,
            'EMA_20_vs_200': latest['EMA_20'] / latest['EMA_200'] - 1,
            'Price_vs_EMA20': latest['Close'] / latest['EMA_20'] - 1,
            'Price_vs_EMA50': latest['Close'] / latest['EMA_50'] - 1,
            'RSI': latest['RSI'],
            'MACD_Hist': latest['MACD_Hist'],
            'BB_Position': (latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower']),
            'Volume_Ratio': latest['Volume_Ratio'],
            'ATR': latest['ATR'],
            'ADX': latest['ADX'],
            'Stochastic': latest['SlowK'],
            'Price_Change_1d': df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1,
            'Price_Change_5d': df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1,
        }
        
        return pd.DataFrame([features])
    
    def train_ml_model(self, X: pd.DataFrame, y: pd.Series):
        """אימון מודל ML מתקדם"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # מודל Gradient Boosting
            model = GradientBoostingClassifier(
                n_estimators=100, 
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
            
            # דיווח דיוק
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            logger.info(f"ML Model trained - Train accuracy: {train_score:.2f}, Test accuracy: {test_score:.2f}")
            
            return model
        except Exception as e:
            logger.error(f"Error training ML model: {e}")
            return None
    
    def trend_following_strategy(self, df: pd.DataFrame, ticker: str) -> Optional[Dict]:
        """אסטרטגיית מעקב מגמה"""
        try:
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # תנאי כניסה
            buy_condition = (
                latest['EMA_20'] > latest['EMA_50'] and
                latest['EMA_50'] > latest['EMA_200'] and
                latest['ADX'] > 25 and  # מגמה חזקה
                latest['RSI'] > 40 and latest['RSI'] < 70 and
                latest['Volume_Ratio'] > 1.2 and
                latest['Close'] > latest['Open']  # יום ירוק
            )
            
            if buy_condition:
                stop_loss = min(
                    latest['Close'] * 0.92,
                    latest['EMA_20'] * 0.98
                )
                
                take_profit = latest['Close'] * 1.15
                
                return {
                    'ticker': ticker,
                    'action': 'BUY',
                    'price': latest['Close'],
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'timestamp': datetime.now().isoformat(),
                    'confidence': 0.7,
                    'strategy': 'Trend Following',
                    'risk_reward_ratio': (take_profit - latest['Close']) / (latest['Close'] - stop_loss)
                }
        except Exception as e:
            logger.error(f"Error in trend strategy for {ticker}: {e}")
        
        return None
    
    def mean_reversion_strategy(self, df: pd.DataFrame, ticker: str) -> Optional[Dict]:
        """אסטרטגיית mean reversion"""
        try:
            latest = df.iloc[-1]
            
            # תנאי כניסה
            buy_condition = (
                latest['RSI'] < 35 and  # oversold
                latest['Close'] <= latest['BB_Lower'] * 1.02 and  # near lower band
                latest['Volume_Ratio'] > 1.5 and  # high volume
                latest['ADX'] < 20  # no strong trend
            )
            
            if buy_condition:
                stop_loss = latest['BB_Lower'] * 0.98
                take_profit = latest['BB_Middle']  # target middle band
                
                return {
                    'ticker': ticker,
                    'action': 'BUY',
                    'price': latest['Close'],
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'timestamp': datetime.now().isoformat(),
                    'confidence': 0.6,
                    'strategy': 'Mean Reversion',
                    'risk_reward_ratio': (take_profit - latest['Close']) / (latest['Close'] - stop_loss)
                }
        except Exception as e:
            logger.error(f"Error in mean reversion strategy for {ticker}: {e}")
        
        return None
    
    def breakout_strategy(self, df: pd.DataFrame, ticker: str) -> Optional[Dict]:
        """אסטרטגיית breakout"""
        try:
            latest = df.iloc[-1]
            
            # חישוב מתח מסחר
            resistance = df['High'].rolling(20).max().iloc[-1]
            support = df['Low'].rolling(20).min().iloc[-1]
            
            # תנאי כניסה
            buy_condition = (
                latest['Close'] > resistance and  # breakout above resistance
                latest['Volume_Ratio'] > 1.8 and  # high volume confirmation
                latest['ADX'] > 20  # trend strength
            )
            
            if buy_condition:
                stop_loss = support * 0.99
                take_profit = latest['Close'] + (latest['Close'] - support)  # measured move
                
                return {
                    'ticker': ticker,
                    'action': 'BUY',
                    'price': latest['Close'],
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'timestamp': datetime.now().isoformat(),
                    'confidence': 0.65,
                    'strategy': 'Breakout',
                    'risk_reward_ratio': (take_profit - latest['Close']) / (latest['Close'] - stop_loss)
                }
        except Exception as e:
            logger.error(f"Error in breakout strategy for {ticker}: {e}")
        
        return None
    
    def generate_signals(self) -> List[Dict]:
        """יצירת אותות מסחר עם אסטרטגיות מרובות"""
        signals = []
        
        for ticker in self.tickers:
            try:
                # הורדת נתונים
                data = self.fetch_stock_data(ticker)
                if data is None:
                    continue
                
                # חישוב אינדיקטורים
                df_with_indicators = self.calculate_advanced_indicators(data)
                if df_with_indicators is None:
                    continue
                
                # הרצת כל האסטרטגיות
                strategies = [
                    self.trend_following_strategy(df_with_indicators, ticker),
                    self.mean_reversion_strategy(df_with_indicators, ticker),
                    self.breakout_strategy(df_with_indicators, ticker)
                ]
                
                # הוספת האותות החיוביים
                for signal in strategies:
                    if signal and signal['risk_reward_ratio'] > 1.5:  # רק עם יחס סיכון/תגמול טוב
                        # הוספת סנטימנט
                        news_sentiment = self.get_news_sentiment(ticker)
                        social_sentiment = self.get_social_sentiment(ticker)
                        
                        signal['news_sentiment'] = news_sentiment
                        signal['social_sentiment'] = social_sentiment
                        signal['combined_sentiment'] = (news_sentiment + social_sentiment) / 2
                        
                        # התאמת ביטחון based on סנטימנט
                        if signal['combined_sentiment'] > 0.2:
                            signal['confidence'] = min(signal['confidence'] * 1.2, 0.9)
                        elif signal['combined_sentiment'] < -0.2:
                            signal['confidence'] = signal['confidence'] * 0.8
                        
                        signals.append(signal)
                        
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                continue
        
        # מיון האותות based on ביטחון ויחס סיכון/תגמול
        signals.sort(key=lambda x: x['confidence'] * x['risk_reward_ratio'], reverse=True)
        
        return signals
    
    def create_visualization(self, signals: List[Dict]):
        """יצירת ויזואליזציה של האותות"""
        if not signals:
            return
        
        # יצירת גרף עם plotly
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Top Trading Signals', 'Signal Confidence vs Risk/Reward'),
            vertical_spacing=0.15
        )
        
        # גרף 1: מחיר ו-EMAs עבור האות המוביל
        top_ticker = signals[0]['ticker']
        data = self.fetch_stock_data(top_ticker, period='3mo')
        df_with_indicators = self.calculate_advanced_indicators(data)
        
        fig.add_trace(
            go.Candlestick(
                x=df_with_indicators.index,
                open=df_with_indicators['Open'],
                high=df_with_indicators['High'],
                low=df_with_indicators['Low'],
                close=df_with_indicators['Close'],
                name=top_ticker
            ),
            row=1, col=1
        )
        
        # הוספת ממוצעים נעים
        fig.add_trace(
            go.Scatter(
                x=df_with_indicators.index,
                y=df_with_indicators['EMA_20'],
                name='EMA 20',
                line=dict(color='orange')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_with_indicators.index,
                y=df_with_indicators['EMA_50'],
                name='EMA 50',
                line=dict(color='red')
            ),
            row=1, col=1
        )
        
        # גרף 2: ביטחון vs יחס סיכון/תגמול
        tickers = [s['ticker'] for s in signals]
        confidence = [s['confidence'] for s in signals]
        risk_reward = [s['risk_reward_ratio'] for s in signals]
        
        fig.add_trace(
            go.Bar(
                x=tickers,
                y=confidence,
                name='Confidence',
                marker_color='lightblue'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=tickers,
                y=risk_reward,
                name='Risk/Reward',
                mode='lines+markers',
                line=dict(color='green'),
                yaxis='y2'
            ),
            row=2, col=1
        )
        
        # עיצוב
        fig.update_layout(
            title=f'Swing Trading Signals - {datetime.now().strftime("%Y-%m-%d")}',
            height=800,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Confidence", row=2, col=1)
        fig.update_yaxes(title_text="Risk/Reward", secondary_y=True, row=2, col=1)
        
        fig.write_html('trading_signals.html')
        logger.info("Visualization created: trading_signals.html")
    
    def send_telegram_alert(self, signals: List[Dict]):
        """שליחת התראות ל-Telegram"""
        try:
            if not self.api_keys['telegram_bot'] or not self.api_keys['telegram_chat_id']:
                logger.warning("Telegram credentials not configured. Skipping alerts.")
                return
                
            if not signals:
                message = "📊 No trading signals generated today."
            else:
                message = "🚀 Trading Signals Generated:\n\n"
                
                for i, signal in enumerate(signals[:5]):  # רק 5 האותות המובילים
                    message += (
                        f"{i+1}. {signal['ticker']} - {signal['strategy']}\n"
                        f"   Price: ${signal['price']:.2f}\n"
                        f"   SL: ${signal['stop_loss']:.2f}, TP: ${signal['take_profit']:.2f}\n"
                        f"   Confidence: {signal['confidence']:.2f}, R/R: {signal['risk_reward_ratio']:.2f}\n\n"
                    )
            
            # שליחה ל-Telegram
            url = f"https://api.telegram.org/bot{self.api_keys['telegram_bot']}/sendMessage"
            payload = {
                "chat_id": self.api_keys['telegram_chat_id'],
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                logger.info("Telegram alert sent successfully")
            else:
                logger.error(f"Failed to send Telegram alert: {response.status_code}")
            
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")
    
    def run(self):
        """הרצת הבוט המלאה"""
        logger.info("Starting Advanced Swing Trading Bot")
        
        # יצירת אותות
        signals = self.generate_signals()
        
        # יצירת ויזואליזציה
        self.create_visualization(signals)
        
        # שליחת התראות
        self.send_telegram_alert(signals)
        
        # שמירת התוצאות
        results = {
            'timestamp': datetime.now().isoformat(),
            'signals_generated': len(signals),
            'signals': signals
        }
        
        with open('trading_signals.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Generated {len(signals)} signals. Results saved to trading_signals.json")
        
        return signals

def main():
    """פונקציה ראשית"""
    bot = AdvancedSwingTradingBot()
    
    # דוגמה להוספת טיקרים
    # bot.add_ticker("TSLA")
    # bot.add_ticker("NVDA")
    
    signals = bot.run()
    
    # הדפסת התוצאות
    if signals:
        print(f"\n📈 Generated {len(signals)} trading signals:")
        for signal in signals[:5]:  # רק 5 האותות המובילים
            print(f"{signal['ticker']}: {signal['action']} at ${signal['price']:.2f}")
            print(f"  Strategy: {signal['strategy']}, Confidence: {signal['confidence']:.2f}")
            print(f"  SL: ${signal['stop_loss']:.2f}, TP: ${signal['take_profit']:.2f}")
            print(f"  Risk/Reward: {signal['risk_reward_ratio']:.2f}\n")
    else:
        print("No trading signals generated today.")

if __name__ == "__main__":
    main()
