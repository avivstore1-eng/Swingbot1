#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Swing Trading Bot – גרסת 10/10 בעברית
- חינמי: yfinance + טלגרם אופציונלי
- בטוח להרצה ב-GitHub Actions (matplotlib Agg)
- משמר את כל היכולות של הגרסה הקודמת + תיקונים (ADX מדויק, BB בטוח, קאש, התמדה של מודל)
- הודעות ותקצירים בעברית, כולל מצב ריצה ברור (פתוח/סגור/מוגבל)
"""

import os
import time
import json
import math
import pickle
import random
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytz
import requests
import yfinance as yf

# Headless plotting ל-CI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# שחזוריות
random.seed(42)
np.random.seed(42)

# נסה xgboost (חינמי). אם חסר—הבוט עדיין ירוץ בלי ה-ML.
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# לוגים
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("trading_bot.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ===== תצורה =====
FORCE_ANALYSIS_WHEN_CLOSED = os.getenv("FORCE_ANALYSIS_WHEN_CLOSED", "true").lower() == "true"  # אמת=לנתח גם כשהשוק סגור
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
CACHE_PREFIX = "cache_"
RESULTS_CSV = "trading_bot_summary.csv"
RESULTS_CSV_LIMITED = "trading_bot_summary_limited.csv"

# טלגרם (אופציונלי; חינמי)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


# ===================== עזר =====================

def send_telegram_message(message: str):
    """שליחה בטוחה לטלגרם (בעברית)."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.info("סודות טלגרם לא מוגדרים. מדלג על שליחה.")
        return
    try:
        if len(message) > 3500:
            message = message[:3490] + "\n…(הודעה קוצרה)"
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        r = requests.post(url, data=payload, timeout=10)
        r.raise_for_status()
        logger.info("הודעת טלגרם נשלחה.")
    except Exception as e:
        logger.error(f"שגיאת שליחת טלגרם: {e}")


def load_tickers() -> List[str]:
    """טעינת טיקרים מ-tickers.json (או ברירת מחדל)."""
    blacklist = {"ANSS"}  # דוגמה לטיקר בעייתי
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    if os.path.exists("tickers.json"):
        try:
            with open("tickers.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            tickers = data.get("tickers", data) if isinstance(data, dict) else data
            valid = []
            for t in tickers:
                if isinstance(t, str) and t.strip() and len(t.strip()) <= 10 and t.isascii() and t not in blacklist:
                    valid.append(t.strip().upper())
            if not valid:
                raise ValueError("לא נמצאו טיקרים תקינים.")
            logger.info(f"נטענו {len(valid)} טיקרים מ-tickers.json.")
            return valid
        except Exception as e:
            logger.error(f"שגיאה בקריאת tickers.json: {e}. משתמש בברירת מחדל.")
            return default_tickers
    else:
        logger.warning("tickers.json לא נמצא. משתמש ברשימת בדיקה.")
        return test_tickers


def is_nyse_holiday(date_obj: datetime.date) -> bool:
    """חגים עיקריים של NYSE לשנים 2024–2026 (קירוב ללא תלות חיצונית)."""
    def nth_weekday(year, month, weekday, n):
        # weekday: שני=0 ... ראשון=6
        d = datetime(year, month, 1)
        offset = (weekday - d.weekday()) % 7
        day = 1 + offset + (n - 1) * 7
        return datetime(year, month, day).date()

    y = date_obj.year
    holidays = set()

    for year in [2024, 2025, 2026]:
        # New Year's Day (observed)
        new_year = datetime(year, 1, 1).date()
        if new_year.weekday() == 5:     # שבת -> שישי
            holidays.add(new_year - timedelta(days=1))
        elif new_year.weekday() == 6:   # ראשון -> שני
            holidays.add(new_year + timedelta(days=1))
        else:
            holidays.add(new_year)

        # MLK – שני שלישי בינואר
        holidays.add(nth_weekday(year, 1, 0, 3))
        # Presidents' Day – שני שלישי בפברואר
        holidays.add(nth_weekday(year, 2, 0, 3))
        # Memorial Day – שני אחרון במאי
        last_may = datetime(year, 5, 31).date()
        holidays.add(last_may - timedelta(days=(last_may.weekday() - 0) % 7))
        # Juneteenth – 19 ביוני (observed)
        jun19 = datetime(year, 6, 19).date()
        holidays.add(jun19 - timedelta(days=1) if jun19.weekday() == 5 else (jun19 + timedelta(days=1) if jun19.weekday() == 6 else jun19))
        # Independence Day – 4 ביולי (observed)
        july4 = datetime(year, 7, 4).date()
        holidays.add(july4 - timedelta(days=1) if july4.weekday() == 5 else (july4 + timedelta(days=1) if july4.weekday() == 6 else july4))
        # Labor Day – שני ראשון בספטמבר
        holidays.add(nth_weekday(year, 9, 0, 1))
        # Thanksgiving – חמישי רביעי בנובמבר
        holidays.add(nth_weekday(year, 11, 3, 4))
        # Christmas – 25 בדצמבר (observed)
        xmas = datetime(year, 12, 25).date()
        holidays.add(xmas - timedelta(days=1) if xmas.weekday() == 5 else (xmas + timedelta(days=1) if xmas.weekday() == 6 else xmas))

    return date_obj in holidays


def is_nyse_open_now() -> bool:
    """בדיקת פתיחת שוק פשוטה (ללא תלות חיצונית)."""
    try:
        tz_ny = pytz.timezone("America/New_York")
        now = datetime.now(tz_ny)
        if now.weekday() >= 5:
            return False
        if is_nyse_holiday(now.date()):
            return False
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now <= market_close
    except Exception as e:
        logger.error(f"שגיאה בבדיקת פתיחת NYSE: {e}")
        return False


def safe_cache_key(*parts: str) -> str:
    return hashlib.md5("::".join(parts).encode()).hexdigest()


# ===================== אינדיקטורים =====================

def compute_RSI(series: pd.Series, period: int = 14) -> pd.Series:
    try:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    except Exception as e:
        logger.error(f"שגיאת RSI: {e}")
        return pd.Series(50, index=series.index)


def compute_MACD(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    try:
        fast_ema = series.ewm(span=fast, adjust=False).mean()
        slow_ema = series.ewm(span=slow, adjust=False).mean()
        macd = fast_ema - slow_ema
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd.fillna(0), macd_signal.fillna(0), macd_hist.fillna(0)
    except Exception as e:
        logger.error(f"שגיאת MACD: {e}")
        idx = series.index
        return pd.Series(0, index=idx), pd.Series(0, index=idx), pd.Series(0, index=idx)


def compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    try:
        low_min = df["Low"].rolling(k_period).min()
        high_max = df["High"].rolling(k_period).max()
        denom = (high_max - low_min).replace(0, np.nan)
        k = 100 * (df["Close"] - low_min) / denom
        d = k.rolling(d_period).mean()
        return k.fillna(50), d.fillna(50)
    except Exception as e:
        logger.error(f"שגיאת סטוכסטי: {e}")
        idx = df.index
        return pd.Series(50, index=idx), pd.Series(50, index=idx)


def compute_ATR(df: pd.DataFrame, period: int = 14):
    try:
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr.fillna(0)
    except Exception as e:
        logger.error(f"שגיאת ATR: {e}")
        return pd.Series(0, index=df.index)


def compute_ADX(df: pd.DataFrame, period: int = 14):
    """ADX לפי ווילדר (מדויק)."""
    try:
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        high_low = high - low
        high_prev_close = (high - close.shift()).abs()
        low_prev_close = (low - close.shift()).abs()
        tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)

        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr_smooth = pd.Series(tr).rolling(period).sum()
        plus_dm_smooth = pd.Series(plus_dm).rolling(period).sum()
        minus_dm_smooth = pd.Series(minus_dm).rolling(period).sum()

        plus_di = 100 * (plus_dm_smooth / tr_smooth.replace(0, np.nan))
        minus_di = 100 * (minus_dm_smooth / tr_smooth.replace(0, np.nan))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)

        adx = dx.rolling(period).mean().fillna(0)
        return adx.fillna(0)
    except Exception as e:
        logger.error(f"שגיאת ADX: {e}")
        return pd.Series(0, index=df.index)


def calculate_indicators(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    try:
        if len(df) < 50:
            logger.warning(f"פחות מ-50 שורות—אין מספיק נתונים לאינדיקטורים.")
            return None
        df = df.copy()
        df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
        df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()

        df["RSI"] = compute_RSI(df["Close"], 14)
        df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = compute_MACD(df["Close"])
        df["BB_Middle"] = df["Close"].rolling(20).mean()
        bb_std = df["Close"].rolling(20).std()
        df["BB_Upper"] = df["BB_Middle"] + 2 * bb_std
        df["BB_Lower"] = df["BB_Middle"] - 2 * bb_std
        eps = 1e-8
        band_w = (df["BB_Upper"] - df["BB_Lower"]).replace(0, eps)
        df["BB_Position"] = ((df["Close"] - df["BB_Lower"]) / band_w).clip(0, 1)

        df["SlowK"], df["SlowD"] = compute_stochastic(df)
        df["Volume_MA"] = df["Volume"].rolling(20).mean().replace(0, np.nan)
        df["Volume_Ratio"] = (df["Volume"] / df["Volume_MA"]).fillna(1.0)
        df["ATR"] = compute_ATR(df)
        df["ADX"] = compute_ADX(df)

        df["Price_vs_EMA20"] = df["Close"] / df["EMA_20"] - 1
        df["Price_vs_EMA50"] = df["Close"] / df["EMA_50"] - 1

        return df.replace([np.inf, -np.inf], 0).fillna(0)
    except Exception as e:
        logger.error(f"שגיאה בחישוב אינדיקטורים: {e}")
        return None


# ===================== נתונים =====================

_data_cache: Dict[str, Dict] = {}

def fetch_stock_data(ticker: str, period: str = "6mo", interval: str = "1d") -> Optional[pd.DataFrame]:
    """משיכת נתוני yfinance עם קאש יומי (חינמי)."""
    try:
        cache_key = safe_cache_key(ticker, period, interval)
        cache_file = f"{CACHE_PREFIX}{cache_key}.pkl"
        current_date = datetime.utcnow().date().isoformat()

        if cache_key in _data_cache and _data_cache[cache_key].get("date") == current_date:
            return _data_cache[cache_key]["data"]

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    disk_obj = pickle.load(f)
                if disk_obj.get("date") == current_date:
                    _data_cache[cache_key] = disk_obj
                    return disk_obj["data"]
            except Exception as e:
                logger.warning(f"שגיאת טעינת קאש עבור {ticker}: {e}")

        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if df is None or df.empty or len(df) < 50:
            logger.warning(f"לא מספיק נתונים ל-{ticker}.")
            return None

        obj = {"date": current_date, "data": df}
        _data_cache[cache_key] = obj
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(obj, f)
        except Exception as e:
            logger.warning(f"שגיאת שמירת קאש עבור {ticker}: {e}")

        return df
    except Exception as e:
        logger.error(f"שגיאה בשליפת נתונים ל-{ticker}: {e}")
        return None


# ===================== ML =====================

def prepare_ml_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    try:
        if len(df) < 6:
            return None
        latest = df.iloc[-1]
        features = {
            "EMA_20_vs_50": latest["EMA_20"] / latest["EMA_50"] - 1,
            "EMA_20_vs_200": latest["EMA_20"] / latest["EMA_200"] - 1,
            "Price_vs_EMA20": latest["Price_vs_EMA20"],
            "Price_vs_EMA50": latest["Price_vs_EMA50"],
            "RSI": latest["RSI"],
            "MACD_Hist": latest["MACD_Hist"],
            "BB_Position": latest["BB_Position"],
            "Volume_Ratio": latest["Volume_Ratio"],
            "ATR": latest["ATR"],
            "ADX": latest["ADX"],
            "Stochastic": latest["SlowK"],
            "Price_Change_1d": df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1,
            "Price_Change_5d": (df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1) if len(df) > 5 else 0,
        }
        return pd.DataFrame([features])
    except Exception as e:
        logger.error(f"שגיאת הכנת מאפייני ML: {e}")
        return None


def prepare_ml_dataset(df: pd.DataFrame):
    try:
        if len(df) < 60:
            return pd.DataFrame(), pd.Series(dtype="int64")
        feats = []
        targets = []
        for i in range(50, len(df) - 5):
            win = df.iloc[i - 50: i]
            latest = win.iloc[-1]
            f = {
                "EMA_20_vs_50": latest["EMA_20"] / latest["EMA_50"] - 1,
                "EMA_20_vs_200": latest["EMA_20"] / latest["EMA_200"] - 1,
                "Price_vs_EMA20": latest["Price_vs_EMA20"],
                "Price_vs_EMA50": latest["Price_vs_EMA50"],
                "RSI": latest["RSI"],
                "MACD_Hist": latest["MACD_Hist"],
                "BB_Position": latest["BB_Position"],
                "Volume_Ratio": latest["Volume_Ratio"],
                "ATR": latest["ATR"],
                "ADX": latest["ADX"],
                "Stochastic": latest["SlowK"],
                "Price_Change_1d": win["Close"].iloc[-1] / win["Close"].iloc[-2] - 1 if len(win) > 1 else 0,
                "Price_Change_5d": win["Close"].iloc[-1] / win["Close"].iloc[-6] - 1 if len(win) > 5 else 0,
            }
            feats.append(f)
            future_close = df["Close"].iloc[i + 5]
            targets.append(1 if future_close > latest["Close"] else 0)

        X = pd.DataFrame(feats)
        y = pd.Series(targets, dtype="int64")
        return X, y
    except Exception as e:
        logger.error(f"שגיאת בניית דסייט ML: {e}")
        return pd.DataFrame(), pd.Series(dtype="int64")


def load_or_train_model(tickers: List[str]):
    """שומר/טוען מודל כדי לחסוך זמן ב-CI. נשאר חינמי."""
    model = None
    scaler = None

    # נסה לטעון
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and XGB_AVAILABLE:
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            with open(SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)
            logger.info("מודל וסקיילר נטענו מקובץ.")
            return model, scaler
        except Exception as e:
            logger.warning(f"כשל טעינת מודל/סקיילר: {e}")

    # אימון
    if not XGB_AVAILABLE:
        send_telegram_message("*שגיאה קריטית*: לא מותקן xgboost. התקן עם: `pip install xgboost`.\nהבוט ימשיך בלי ML.")
        logger.error("xgboost חסר. ML יושבת.")
        return None, StandardScaler()

    all_X = pd.DataFrame()
    all_y = pd.Series(dtype="int64")
    training_tickers = random.sample(tickers, min(25, len(tickers)))
    logger.info(f"אימון ML על {len(training_tickers)} טיקרים...")

    for t in training_tickers:
        df = fetch_stock_data(t)
        if df is None:
            continue
        df = calculate_indicators(df)
        if df is None:
            continue
        X, y = prepare_ml_dataset(df)
        if not X.empty and not y.empty:
            all_X = pd.concat([all_X, X], ignore_index=True)
            all_y = pd.concat([all_y, y], ignore_index=True)

    if all_X.empty or len(all_y) < 10:
        send_telegram_message("*שגיאה*: אין מספיק נתונים לאימון ML.")
        logger.warning("אין מספיק נתונים לאימון ML.")
        return None, StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
        n_jobs=2,
    )
    model.fit(X_train_s, y_train)
    train_score = model.score(X_train_s, y_train)
    test_score = model.score(X_test_s, y_test)
    logger.info(f"מודל XGB אומן. Train={train_score:.3f}, Test={test_score:.3f}")

    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
    except Exception as e:
        logger.warning(f"שגיאה בשמירת מודל/סקיילר: {e}")

    return model, scaler


# ===================== אסטרטגיות =====================

def strategy_trend_following(df: pd.DataFrame) -> str:
    try:
        if len(df) < 50:
            return "HOLD"
        latest = df.iloc[-1]
        if latest["EMA_20"] > latest["EMA_50"] > latest["EMA_200"] and latest["ADX"] > 25:
            return "BUY"
        if latest["EMA_20"] < latest["EMA_50"] < latest["EMA_200"] and latest["ADX"] > 25:
            return "SELL"
        return "HOLD"
    except Exception as e:
        logger.error(f"שגיאה באסטרטגיית טרנד: {e}")
        return "HOLD"


def strategy_mean_reversion(df: pd.DataFrame) -> str:
    try:
        if len(df) < 50:
            return "HOLD"
        latest = df.iloc[-1]
        if latest["RSI"] < 30 and latest["Close"] < latest["BB_Lower"] and latest["ADX"] > 20:
            return "BUY"
        if latest["RSI"] > 70 and latest["Close"] > latest["BB_Upper"] and latest["ADX"] > 20:
            return "SELL"
        return "HOLD"
    except Exception as e:
        logger.error(f"שגיאה באסטרטגיית ממוצע חוזר: {e}")
        return "HOLD"


def strategy_breakout(df: pd.DataFrame) -> str:
    try:
        if len(df) < 50:
            return "HOLD"
        latest = df.iloc[-1]
        high_20 = df["High"].rolling(20).max().iloc[-2]
        low_20 = df["Low"].rolling(20).min().iloc[-2]
        if latest["Close"] > high_20 and latest["ADX"] > 25:
            return "BUY"
        if latest["Close"] < low_20 and latest["ADX"] > 25:
            return "SELL"
        return "HOLD"
    except Exception as e:
        logger.error(f"שגיאה באסטרטגיית פריצה: {e}")
        return "HOLD"


def vote_final_signal(signals: List[str]) -> str:
    """הצבעה משופרת: צריך לפחות שני BUY/SELL כדי לנצח HOLD."""
    votes = {"BUY": signals.count("BUY"), "SELL": signals.count("SELL"), "HOLD": signals.count("HOLD")}
    if max(votes["BUY"], votes["SELL"]) >= 2:
        return "BUY" if votes["BUY"] > votes["SELL"] else "SELL"
    return "HOLD"


# ===================== Backtest ודירוג =====================

def backtest_day_ahead(df: pd.DataFrame, strategy_func) -> Dict[str, float]:
    """מודל פשוט: תשואת יום קדימה לפי האיתות."""
    try:
        if len(df) < 50:
            return {"Total_Return": 0.0, "Win_Rate": 0.0}
        rows = []
        for i in range(50, len(df) - 1):
            window = df.iloc[: i + 1]
            sig = strategy_func(window)
            day_ret = df["Close"].iloc[i + 1] / df["Close"].iloc[i] - 1
            pos = 1 if sig == "BUY" else (-1 if sig == "SELL" else 0)
            rows.append(pos * day_ret)
        if not rows:
            return {"Total_Return": 0.0, "Win_Rate": 0.0}
        returns = pd.Series(rows)
        total_return = returns.sum()
        win_rate = (returns > 0).mean()
        return {"Total_Return": float(total_return), "Win_Rate": float(win_rate)}
    except Exception as e:
        logger.error(f"שגיאת Backtest: {e}")
        return {"Total_Return": 0.0, "Win_Rate": 0.0}


def weighted_score(row: pd.Series) -> float:
    """ציון משוקלל מנורמל: מקס' תשואה של אסטרטגיה + win rate תואם."""
    ret_max = max(abs(row["Backtest_Trend"]), abs(row["Backtest_Mean"]), abs(row["Backtest_Breakout"]), 1e-9)
    ret_norm = max(row["Backtest_Trend"], row["Backtest_Mean"], row["Backtest_Breakout"]) / ret_max
    win_norm = max(row["Backtest_Trend_Win"], row["Backtest_Mean_Win"], row["Backtest_Breakout_Win"])
    return 0.6 * ret_norm + 0.4 * win_norm


# ===================== ויזואליזציה =====================

def visualize_signals(df: pd.DataFrame, ticker: str, final_signal: str):
    try:
        if df["Close"].isna().all() or df["EMA_20"].isna().all() or df["EMA_50"].isna().all() or df["RSI"].isna().all():
            logger.warning(f"נתונים לא תקינים לתרשים עבור {ticker}")
            return
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})
        ax1.plot(df.index, df["Close"], label="סגירה")
        ax1.plot(df.index, df["EMA_20"], label="EMA 20")
        ax1.plot(df.index, df["EMA_50"], label="EMA 50")
        if final_signal == "BUY":
            ax1.scatter(df.index[-1], df["Close"].iloc[-1], marker="^", s=100, label="איתות קנייה")
        elif final_signal == "SELL":
            ax1.scatter(df.index[-1], df["Close"].iloc[-1], marker="v", s=100, label="איתות מכירה")
        ax1.set_title(f"{ticker} – מחיר ו-EMA + איתות")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(df.index, df["RSI"], label="RSI")
        ax2.axhline(70, linestyle="--", alpha=0.5)
        ax2.axhline(30, linestyle="--", alpha=0.5)
        ax2.set_title("RSI")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        out = f"{ticker}_chart.png"
        plt.savefig(out)
        plt.close()
        logger.info(f"נשמר תרשים: {out}")
    except Exception as e:
        logger.error(f"שגיאה בשרטוט {ticker}: {e}")


# ===================== הלוגיקה הראשית =====================

class AdvancedSwingTradingBot:
    def __init__(self):
        self.tickers = load_tickers()
        self.model, self.scaler = load_or_train_model(self.tickers)
        self.portfolio: Dict[str, Dict] = {}
        self.failed_tickers: List[str] = []

    def get_ml_signal(self, df: pd.DataFrame) -> str:
        if self.model is None:
            return "HOLD"
        try:
            feats = prepare_ml_features(df)
            if feats is None:
                return "HOLD"
            Xs = self.scaler.transform(feats)
            pred = self.model.predict(Xs)[0]
            return "BUY" if int(pred) == 1 else "SELL"
        except Exception as e:
            logger.error(f"שגיאת ML Prediction: {e}")
            return "HOLD"

    def calc_trade_details(self, df: pd.DataFrame, signal: str) -> Dict[str, float]:
        try:
            latest = df.iloc[-1]
            entry = float(latest["Close"])
            atr = float(latest["ATR"])
            if signal == "BUY":
                tp = entry + 2 * atr
                sl = entry - 1 * atr
            elif signal == "SELL":
                tp = entry - 2 * atr
                sl = entry + 1 * atr
            else:
                tp = sl = entry
            return {"Entry_Price": entry, "Take_Profit": tp, "Stop_Loss": sl}
        except Exception as e:
            logger.error(f"שגיאה בחישוב פרטי עסקה: {e}")
            return {"Entry_Price": 0.0, "Take_Profit": 0.0, "Stop_Loss": 0.0}

    def update_portfolio(self, ticker: str, signal: str):
        try:
            pos_count = sum(1 for p in self.portfolio.values() if p.get("position", 0) != 0)
            if ticker not in self.portfolio:
                self.portfolio[ticker] = {"position": 0, "last_signal": "HOLD"}
            if pos_count >= 5 and signal in ("BUY", "SELL"):
                logger.warning(f"הגבלת פורטפוליו—מדלג על {ticker} {signal}.")
                return
            current = self.portfolio[ticker]
            current["last_signal"] = signal
            current["position"] = 1 if signal == "BUY" else (-1 if signal == "SELL" else 0)
            self.portfolio[ticker] = current
        except Exception as e:
            logger.error(f"שגיאה בעדכון פורטפוליו: {e}")

    def process_ticker(self, ticker: str) -> Optional[Dict]:
        try:
            df = fetch_stock_data(ticker)
            if df is None:
                self.failed_tickers.append(ticker)
                return None
            df = calculate_indicators(df)
            if df is None:
                self.failed_tickers.append(ticker)
                return None

            s_trend = strategy_trend_following(df)
            s_mean = strategy_mean_reversion(df)
            s_break = strategy_breakout(df)
            s_ml = self.get_ml_signal(df)

            signals = [s_trend, s_mean, s_break, s_ml]
            final_signal = vote_final_signal(signals)

            bt_trend = backtest_day_ahead(df, strategy_func=strategy_trend_following)
            bt_mean = backtest_day_ahead(df, strategy_func=strategy_mean_reversion)
            bt_break = backtest_day_ahead(df, strategy_func=strategy_breakout)

            trade = self.calc_trade_details(df, final_signal)

            result = {
                "Ticker": ticker,
                "Final_Signal": final_signal,
                "ML_Signal": s_ml,
                "Trend_Signal": s_trend,
                "Mean_Signal": s_mean,
                "Breakout_Signal": s_break,
                "Backtest_Trend": bt_trend["Total_Return"],
                "Backtest_Trend_Win": bt_trend["Win_Rate"],
                "Backtest_Mean": bt_mean["Total_Return"],
                "Backtest_Mean_Win": bt_mean["Win_Rate"],
                "Backtest_Breakout": bt_break["Total_Return"],
                "Backtest_Breakout_Win": bt_break["Win_Rate"],
                "Entry_Price": trade["Entry_Price"],
                "Take_Profit": trade["Take_Profit"],
                "Stop_Loss": trade["Stop_Loss"],
            }
            return result
        except Exception as e:
            logger.error(f"שגיאה בעיבוד {ticker}: {e}")
            self.failed_tickers.append(ticker)
            return None

    def _send_no_picks_summary(self, analyzed: int, limited: bool):
        mode = "מצומצם (דגימה)" if limited else "מלא"
        send_telegram_message(
            f"*סיכום {datetime.utcnow().date()}*\n"
            f"מצב ריצה: {mode}\n"
            f"נסקרו {analyzed} טיקרים.\n"
            f"לא נמצאו בחירות פעולה (כולם HOLD). קבצים נשמרו כארטיפקטים."
        )

    def run_full_strategy(self):
        results = []
        self.failed_tickers = []

        logger.info(f"מעבד {len(self.tickers)} טיקרים...")
        for t in self.tickers:
            res = self.process_ticker(t)
            if res:
                results.append(res)

        if not results:
            logger.warning("לא הופקו תוצאות כלל.")
            self._send_no_picks_summary(0, limited=False)
            return

        df = pd.DataFrame(results)
        df["Weighted_Score"] = df.apply(weighted_score, axis=1)
        top = df[df["Final_Signal"] != "HOLD"].sort_values("Weighted_Score", ascending=False).head(5)

        if not top.empty:
            for _, row in top.iterrows():
                d = fetch_stock_data(row["Ticker"])
                if d is None:
                    continue
                di = calculate_indicators(d)
                if di is None:
                    continue
                visualize_signals(di, row["Ticker"], row["Final_Signal"])

            avg_ret = top[["Backtest_Trend", "Backtest_Mean", "Backtest_Breakout"]].max(axis=1).mean()
            avg_win = top[["Backtest_Trend_Win", "Backtest_Mean_Win", "Backtest_Breakout_Win"]].max(axis=1).mean()
            market_status = "פתוחה" if is_nyse_open_now() else "סגורה – משתמש בנתונים העדכניים האחרונים"
            msg = f"*5 הבחירות המובילות ל-{datetime.utcnow().date()}*\n" \
                  f"*סטטוס שוק*: NYSE {market_status}\n" \
                  f"*תשואה ממוצעת (Backtest)*: {avg_ret*100:.1f}%\n" \
                  f"*שיעור הצלחה ממוצע*: {avg_win*100:.1f}%\n\n"
            for i, (_, row) in enumerate(top.iterrows(), start=1):
                win_rate = max(
                    row["Backtest_Trend_Win"] if row["Trend_Signal"] == row["Final_Signal"] else 0,
                    row["Backtest_Mean_Win"] if row["Mean_Signal"] == row["Final_Signal"] else 0,
                    row["Backtest_Breakout_Win"] if row["Breakout_Signal"] == row["Final_Signal"] else 0,
                )
                msg += (
                    f"{i}. *{row['Ticker']}* — *{row['Final_Signal']}*\n"
                    f"   כניסה: ${row['Entry_Price']:.2f}\n"
                    f"   טייק פרופיט: ${row['Take_Profit']:.2f}\n"
                    f"   סטופ-לוס: ${row['Stop_Loss']:.2f}\n"
                    f"   שיעור הצלחה מוערך: {win_rate*100:.1f}%\n\n"
                )
            send_telegram_message(msg)
        else:
            self._send_no_picks_summary(len(df), limited=False)

        df.to_csv(RESULTS_CSV, index=False)
        logger.info(f"נשמר סיכום ל-{RESULTS_CSV}")

        if self.failed_tickers:
            send_telegram_message(f"*טיקרים שנכשלו בעיבוד*\n{self.failed_tickers[:50]}{'...' if len(self.failed_tickers) > 50 else ''}")

    def run_single_run(self):
        """ריצה מצומצמת כשהשוק סגור (אם לא כופים ניתוח מלא)."""
        sample = random.sample(self.tickers, min(15, len(self.tickers)))
        results = []
        self.failed_tickers = []

        for t in sample:
            res = self.process_ticker(t)
            if res:
                results.append(res)

        if not results:
            self._send_no_picks_summary(0, limited=True)
            return

        df = pd.DataFrame(results)
        df["Weighted_Score"] = df.apply(weighted_score, axis=1)
        top = df[df["Final_Signal"] != "HOLD"].sort_values("Weighted_Score", ascending=False).head(3)

        if not top.empty:
            msg = f"*3 בחירות מובילות ל-{datetime.utcnow().date()} (ריצה מצומצמת)*\n" \
                  f"*סטטוס שוק*: NYSE סגורה – משתמש בנתונים העדכניים האחרונים\n\n"
            for i, (_, row) in enumerate(top.iterrows(), start=1):
                win_rate = max(
                    row["Backtest_Trend_Win"] if row["Trend_Signal"] == row["Final_Signal"] else 0,
                    row["Backtest_Mean_Win"] if row["Mean_Signal"] == row["Final_Signal"] else 0,
                    row["Backtest_Breakout_Win"] if row["Breakout_Signal"] == row["Final_Signal"] else 0,
                )
                msg += (
                    f"{i}. *{row['Ticker']}* — *{row['Final_Signal']}*\n"
                    f"   כניסה: ${row['Entry_Price']:.2f}\n"
                    f"   טייק פרופיט: ${row['Take_Profit']:.2f}\n"
                    f"   סטופ-לוס: ${row['Stop_Loss']:.2f}\n"
                    f"   שיעור הצלחה מוערך: {win_rate*100:.1f}%\n\n"
                )
            send_telegram_message(msg)
        else:
            self._send_no_picks_summary(len(df), limited=True)

        df.to_csv(RESULTS_CSV_LIMITED, index=False)
        logger.info(f"נשמר סיכום מצומצם ל-{RESULTS_CSV_LIMITED}")

        if self.failed_tickers:
            send_telegram_message(f"*טיקרים שנכשלו בעיבוד (מצומצם)*\n{self.failed_tickers[:50]}{'...' if len(self.failed_tickers) > 50 else ''}")

    def run_once(self):
        tz_ist = pytz.timezone("Asia/Jerusalem")
        now_ist = datetime.now(tz_ist)
        nyse_open = is_nyse_open_now()
        mode = ("אסטרטגיה מלאה (שוק פתוח)" if nyse_open
                else ("אסטרטגיה מלאה על נתונים עדכניים אחרונים" if FORCE_ANALYSIS_WHEN_CLOSED
                      else "ריצה מצומצמת (דגימה)"))

        send_telegram_message(
            f"*בוט מסחר – תחילת ריצה*\n"
            f"זמן מקומי: {now_ist}\n"
            f"סטטוס NYSE: {'פתוחה' if nyse_open else 'סגורה'}\n"
            f"מצב ריצה: {mode}"
        )

        start = time.time()
        try:
            if nyse_open:
                self.run_full_strategy()
            else:
                if FORCE_ANALYSIS_WHEN_CLOSED:
                    logger.info("שוק סגור – מריץ אסטרטגיה מלאה על הנתון האחרון.")
                    self.run_full_strategy()
                else:
                    logger.info("שוק סגור – מריץ ריצה מצומצמת (דגימה).")
                    self.run_single_run()
        finally:
            elapsed = time.time() - start
            if elapsed > 600:
                send_telegram_message(f"*אזהרה*: זמן ריצה {elapsed/60:.1f} דקות (>10 דק'). בדוק לוגים.")


if __name__ == "__main__":
    bot = AdvancedSwingTradingBot()
    bot.run_once()
