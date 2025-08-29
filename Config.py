"""
קובץ הגדרות מתקדם עבור Swing Trading Bot
"""

import os
from datetime import time
from typing import List, Dict, Any

# רשימת מניות default (ניתן לשנות דרך הקוד או הקובץ)
DEFAULT_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 
    'TSLA', 'NVDA', 'JPM', 'V', 'JNJ'
]

# הגדרות ניתוח טכני
TECHNICAL_SETTINGS = {
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'bb_std_dev': 2,
    'ema_periods': [20, 50, 200],
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'atr_period': 14,
    'adx_period': 14,
    'stochastic_period': 14
}

# הגדרות ניהול סיכונים מתקדמות
RISK_MANAGEMENT = {
    'max_position_size': 0.1,  # 10% מהתיק לכל מניה
    'stop_loss_pct': 0.08,     # 8% stop loss
    'trailing_stop_pct': 0.05, # 5% trailing stop
    'take_profit_pct': 0.15,   # 15% take profit
    'max_drawdown': 0.2,       # 20% max drawdown
    'consecutive_losses': 3,   # stop after 3 consecutive losses
    'min_risk_reward': 1.5     # min risk/reward ratio
}

# הגדרות אסטרטגיות
STRATEGY_SETTINGS = {
    'trend_following': {
        'enabled': True,
        'min_adx': 25,
        'min_volume_ratio': 1.2
    },
    'mean_reversion': {
        'enabled': True,
        'max_adx': 20,
        'min_volume_ratio': 1.5
    },
    'breakout': {
        'enabled': True,
        'min_volume_ratio': 1.8,
        'min_adx': 20
    }
}

# הגדרות ML
ML_SETTINGS = {
    'enabled': True,
    'model_type': 'gradient_boosting',
    'test_size': 0.2,
    'random_state': 42,
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 3
}

# הגדרות נתונים
DATA_SETTINGS = {
    'default_period': '6mo',
    'default_interval': '1d',
    'max_retries': 3,
    'retry_delay': 1,
    'cache_expiry': 300  # 5 minutes
}

# הגדרות API (יש להחליף במפתחות אמיתיים)
API_KEYS = {
    'news_api': os.getenv('NEWS_API_KEY', 'your_news_api_key_here'),
    'alpha_vantage': os.getenv('ALPHA_VANTAGE_KEY', 'your_alpha_vantage_key_here'),
    'telegram_bot': os.getenv('TELEGRAM_BOT_TOKEN', 'your_telegram_bot_token_here'),
    'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID', 'your_telegram_chat_id_here')
}

# הגדרות Telegram
TELEGRAM_SETTINGS = {
    'enabled': True,
    'chat_id': API_KEYS['telegram_chat_id'],
    'alert_time': time(9, 30),  # זמן שליחת ההתראות
    'max_signals_to_send': 5    # מקסימום אותות לשליחה
}

# הגדרות backtesting
BACKTEST_SETTINGS = {
    'initial_capital': 100000,
    'commission': 0.001,  # 0.1% commission
    'slippage': 0.001,    # 0.1% slippage
    'test_start_date': '2020-01-01',
    'test_end_date': '2023-12-31'
}

# הגדרות logging
LOGGING_SETTINGS = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': 'trading_bot.log',
    'max_bytes': 10485760,  # 10MB
    'backup_count': 5
}
