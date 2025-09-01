#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Swing Trading Bot – גרסת "מקסימום" בעברית (חינמי)
- ריצה מלאה על *כל* הטיקרים תמיד (גם כשהשוק סגור)
- אימון ML על *כל* הטיקרים אם xgboost זמין (אחרת ממשיך בלי ML)
- סינון נזילות/מחיר, פילטר מצב שוק SPY, דירוג EV, גודל פוזיציה ATR, Trailing Stop, Cooldown
- מקביליות בטוחה ל-500 טיקרים (ThreadPool), קאש יומי, גרפים CI-safe
- הודעות טלגרם בעברית עם פירוט גודל פוזיציה ו-Trailing Stop
"""

import os
import time
import json
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

from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================== תצורה גלובלית =====================

# כופה ניתוח מלא תמיד (גם כשהשוק סגור)
FORCE_ANALYSIS_WHEN_CLOSED = True

# הון וירטואלי (לדמו/סימולציה של גודל עסקה)
EQUITY = float(os.getenv("EQUITY", "100000"))             # ברירת מחדל: 100K$
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))  # 1% סיכון לעסקה
COOLDOWN_DAYS = int(os.getenv("COOLDOWN_DAYS", "2"))         # מניעת היפוך מהיר

# מקביליות (איזון מול רייט-לימיט של yfinance)
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))

# קבצים
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
CACHE_PREFIX = "cache_"
RESULTS_CSV = "trading_bot_summary.csv"
RESULTS_CSV_LIMITED = "trading_bot_summary_limited.csv"  # נשמר לשקיפות

# טלגרם (אופציונלי; חינמי)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# שחזוריות
random.seed(42)
np.random.seed(42)

# נסה xgboost (חינמי). אם חסר—הבוט ממשיך בלי ML.
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
    blacklist = {"ANSS"}  # דוגמה לטיקר בעייתי; אפשר להסיר אם תרצה
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
        d = datetime(year, month, 1)
        offset = (weekday - d.weekday()) % 7
        day = 1 + offset + (n - 1) * 7
        return datetime(year, month, day).date()

    holidays = set()
    for year in [2024, 2025, 2026]:
        new_year = datetime(year, 1, 1).date()
        holidays.add(new_year - timedelta(days=1) if new_year.weekday() == 5
                     else new_year + timedelta(days=1) if new_year.weekday() == 6
                     else new_year)
        holidays.add(nth_weekday(year, 1, 0, 3))             # MLK
        holidays.add(nth_weekday(year, 2, 0, 3))             # Presidents' Day
        last_may = datetime(year, 5, 31).date()              # Memorial Day
        holidays.add(last_may - timedelta(days=(last_may.weekday() - 0) % 7))
        jun19 = datetime(year, 6, 19).date()                 # Juneteenth observed
        holidays.add(jun19 - timedelta(days=1) if jun19.weekday() == 5
                     else jun19 + timedelta(days=1) if jun19.weekday() == 6
                     else jun19)
        july4 = datetime(year, 7, 4).date()                  # Independence Day observed
        holidays.add(july4 - timedelta(days=1) if july4.weekday() == 5
                     else july4 + timedelta(days=1) if july4.weekday() == 6
                     else july4)
        holidays.add(nth_weekday(year, 9, 0, 1))             # Labor Day
        holidays.add(nth_weekday(year, 11, 3, 4))            # Thanksgiving
        xmas = datetime(year, 12, 25).date()                 # Christmas observed
        holidays.add(xmas - timedelta(days=1) if xmas.weekday() == 5
                     else xmas + timedelta(days=1) if xmas.weekday() == 6
                     else xmas)

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
            logger.warning("פחות מ-50 שורות—אין מספיק נתונים לאינדיקטורים.")
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


# ===================== נתונים + קאש =====================

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
    """שומר/טוען מודל כדי לחסוך זמן ב-CI. אימון על *כל* הטיקרים (חינמי)."""
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

    if not XGB_AVAILABLE:
        send_telegram_message("*שגיאה קריטית*: לא מותקן xgboost. התקן עם: `pip install xgboost`.\nהבוט ימשיך בלי ML.")
        logger.error("xgboost חסר. ML יושבת.")
        return None, StandardScaler()

    # אימון על כל הטיקרים
    training_tickers = tickers
    logger.info(f"אימון ML על {len(training_tickers)} טיקרים...")

    all_X = pd.DataFrame()
    all_y = pd.Series(dtype="int64")

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
    # Early stopping (חינם): מונע אוברפיטינג
    model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], early_stopping_rounds=30, verbose=False)
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


# ===================== פילטרי "רווח מקסימלי" =====================

def market_regime_long_allowed() -> bool:
    """לונגים רק כשה-SPY במגמת עליה בסיסית (EMA20>EMA50)."""
    spy = fetch_stock_data("SPY", period="6mo", interval="1d")
    if spy is None or len(spy) < 50:
        return True
    spy = calculate_indicators(spy)
    if spy is None:
        return True
    latest = spy.iloc[-1]
    return bool(latest["EMA_20"] > latest["EMA_50"])


def market_regime_short_allowed() -> bool:
    """שורטים רק כשה-SPY במגמת ירידה בסיסית (EMA20<EMA50)."""
    spy = fetch_stock_data("SPY", period="6mo", interval="1d")
    if spy is None or len(spy) < 50:
        return True
    spy = calculate_indicators(spy)
    if spy is None:
        return True
    latest = spy.iloc[-1]
    return bool(latest["EMA_20"] < latest["EMA_50"])


def passes_liquidity_filter(df: pd.DataFrame) -> bool:
    """נזילות/מחיר: מחיר >$5 ונפח ממוצע 20 יום > 300k."""
    try:
        price_ok = df["Close"].iloc[-1] >= 5
        vol_ma = df["Volume"].rolling(20).mean().iloc[-1]
        vol_ok = (vol_ma is not None) and (vol_ma >= 300_000)
        return bool(price_ok and vol_ok)
    except Exception:
        return False


# ===================== Backtest מתקדם + דירוג EV =====================

def backtest_day_ahead(df: pd.DataFrame, strategy_func) -> Dict[str, float]:
    """
    Backtest יומי קדימה: מחזיר WinRate, ממוצע חיובי/שלילי וה-EV (חישוב גס).
    EV ≈ p*avg_pos - (1-p)*avg_neg
    """
    try:
        if len(df) < 50:
            return {"Total_Return": 0.0, "Win_Rate": 0.0, "Avg_Pos": 0.0, "Avg_Neg": 0.0, "EV": 0.0}
        pnl = []
        pos_moves = []
        neg_moves = []
        for i in range(50, len(df) - 1):
            window = df.iloc[: i + 1]
            sig = strategy_func(window)
            day_ret = df["Close"].iloc[i + 1] / df["Close"].iloc[i] - 1
            pos = 1 if sig == "BUY" else (-1 if sig == "SELL" else 0)
            trade_ret = pos * day_ret
            if pos != 0:
                pnl.append(trade_ret)
                if trade_ret > 0:
                    pos_moves.append(trade_ret)
                elif trade_ret < 0:
                    neg_moves.append(abs(trade_ret))
        if not pnl:
            return {"Total_Return": 0.0, "Win_Rate": 0.0, "Avg_Pos": 0.0, "Avg_Neg": 0.0, "EV": 0.0}
        s = pd.Series(pnl)
        win_rate = (s > 0).mean()
        avg_pos = float(np.mean(pos_moves)) if pos_moves else 0.0
        avg_neg = float(np.mean(neg_moves)) if neg_moves else 0.0
        ev = float(win_rate * avg_pos - (1 - win_rate) * avg_neg)
        return {"Total_Return": float(s.sum()), "Win_Rate": float(win_rate), "Avg_Pos": avg_pos, "Avg_Neg": avg_neg, "EV": ev}
    except Exception as e:
        logger.error(f"שגיאת Backtest: {e}")
        return {"Total_Return": 0.0, "Win_Rate": 0.0, "Avg_Pos": 0.0, "Avg_Neg": 0.0, "EV": 0.0}


def weighted_score(row: pd.Series) -> float:
    """דירוג לפי EV (80%) + WinRate (20%)."""
    return 0.8 * float(row.get("Best_EV", 0.0)) + 0.2 * float(row.get("Best_WinRate", 0.0))


# ===================== ניהול פוזיציה: ATR Sizing + Trailing =====================

def position_size_by_atr(entry_price: float, atr: float) -> int:
    """
    גודל פוזיציה לפי סיכון קבוע: כמה מניות כדי לסכן ~RISK_PER_TRADE מההון כאשר סטופ=ATR אחד.
    """
    if entry_price <= 0 or atr <= 0:
        return 0
    risk_dollars = EQUITY * RISK_PER_TRADE
    per_share_risk = atr
    shares = int(max(0, risk_dollars // per_share_risk))
    # אל תגזים: לא יותר מ-20% הון לפוזיציה
    max_shares_by_cap = int((EQUITY * 0.2) // entry_price)
    return max(0, min(shares, max_shares_by_cap))


def trailing_stop_levels(entry: float, atr: float, side: str, multiple: float = 1.5) -> float:
    """
    Trailing Stop לפי ATR: לונג – סטופ מתחת לכניסה במרחק multiple*ATR, שורט – מעל.
    (הניהול בפועל יתבצע בסימולטור/דמו; כאן מחושב ערך התחלתי להצגה.)
    """
    if atr <= 0:
        return entry
    if side == "BUY":
        return max(0.0, entry - multiple * atr)
    elif side == "SELL":
        return entry + multiple * atr
    return entry


# מעקב אחר איתות אחרון לטיקר (למנגנון COOLDOWN)
_last_signal_date: Dict[str, Dict[str, datetime]] = {}


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
            size = position_size_by_atr(entry, atr)
            return {"Entry_Price": entry, "Take_Profit": tp, "Stop_Loss": sl, "Size": size}
        except Exception as e:
            logger.error(f"שגיאה בחישוב פרטי עסקה: {e}")
            return {"Entry_Price": 0.0, "Take_Profit": 0.0, "Stop_Loss": 0.0, "Size": 0}

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

            # סינון נזילות/מחיר – מפחית רעש
            if not passes_liquidity_filter(df):
                return None

            s_trend = strategy_trend_following(df)
            s_mean = strategy_mean_reversion(df)
            s_break = strategy_breakout(df)
            s_ml = self.get_ml_signal(df)

            signals = [s_trend, s_mean, s_break, s_ml]
            final_signal = vote_final_signal(signals)

            # פילטר מצב שוק: לא לונגים כש-SPY חלש, לא שורטים כש-SPY חזק
            if final_signal == "BUY" and not market_regime_long_allowed():
                final_signal = "HOLD"
            if final_signal == "SELL" and not market_regime_short_allowed():
                final_signal = "HOLD"

            # Backtests + EV
            bt_trend = backtest_day_ahead(df, strategy_func=strategy_trend_following)
            bt_mean  = backtest_day_ahead(df, strategy_func=strategy_mean_reversion)
            bt_break = backtest_day_ahead(df, strategy_func=strategy_breakout)

            best_ev = max(bt_trend["EV"], bt_mean["EV"], bt_break["EV"])
            best_wr = max(bt_trend["Win_Rate"], bt_mean["Win_Rate"], bt_break["Win_Rate"])

            # קירור אותות – מניעת היפוך מהיר
            today = datetime.utcnow().date()
            ls = _last_signal_date.get(ticker, {})
            if final_signal in ("BUY", "SELL"):
                other = "SELL" if final_signal == "BUY" else "BUY"
                if ls.get(other) and (today - ls[other].date()).days < COOLDOWN_DAYS:
                    final_signal = "HOLD"
            if final_signal in ("BUY", "SELL"):
                _last_signal_date[ticker] = {final_signal: datetime.utcnow()}

            trade = self.calc_trade_details(df, final_signal)
            trailing = trailing_stop_levels(trade["Entry_Price"], df.iloc[-1]["ATR"], final_signal)

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
                "Best_EV": best_ev,
                "Best_WinRate": best_wr,
                "Entry_Price": trade["Entry_Price"],
                "Take_Profit": trade["Take_Profit"],
                "Stop_Loss": trade["Stop_Loss"],
                "Trailing_Stop": trailing,
                "Size": trade.get("Size", 0),
            }
            return result
        except Exception as e:
            logger.error(f"שגיאה בעיבוד {ticker}: {e}")
            self.failed_tickers.append(ticker)
            return None

    def _send_no_picks_summary(self, analyzed: int, mode: str):
        send_telegram_message(
            f"*סיכום {datetime.utcnow().date()}*\n"
            f"מצב ריצה: {mode}\n"
            f"נסקרו {analyzed} טיקרים (כולם).\n"
            f"לא נמצאו בחירות פעולה (כולם HOLD). קבצים נשמרו כארטיפקטים."
        )

    def run_full_strategy(self):
        """ריצה מלאה – על *כל* הטיקרים תמיד, במקביל."""
        results = []
        self.failed_tickers = []

        logger.info(f"מעבד {len(self.tickers)} טיקרים במקביל ({MAX_WORKERS} חוטים)...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            fut = {ex.submit(self.process_ticker, t): t for t in self.tickers}
            for f in as_completed(fut):
                r = f.result()
                if r:
                    results.append(r)

        if not results:
            logger.warning("לא הופקו תוצאות כלל.")
            self._send_no_picks_summary(0, mode="מלא (כל הטיקרים)")
            return

        df = pd.DataFrame(results)
        df["Weighted_Score"] = df.apply(weighted_score, axis=1)

        # בחר טופ רק עם EV חיובי וסיגנל פעיל
        picks = df[(df["Final_Signal"] != "HOLD") & (df["Best_EV"] > 0)].copy()
        top = picks.sort_values("Weighted_Score", ascending=False).head(5)

        if not top.empty:
            for _, row in top.iterrows():
                d = fetch_stock_data(row["Ticker"])
                if d is None:
                    continue
                di = calculate_indicators(d)
                if di is None:
                    continue
                visualize_signals(di, row["Ticker"], row["Final_Signal"])

            avg_ev = float(top["Best_EV"].mean())
            avg_wr = float(top["Best_WinRate"].mean())
            market_status = "פתוחה" if is_nyse_open_now() else "סגורה – משתמש בנתונים העדכניים האחרונים"
            msg = f"*5 הבחירות המובילות ל-{datetime.utcnow().date()}*\n" \
                  f"*סטטוס שוק*: NYSE {market_status}\n" \
                  f"*EV ממוצע*: {avg_ev*100:.2f}‰  (אחוזים אלפיים לטרייד)\n" \
                  f"*Win Rate ממוצע (Backtest)*: {avg_wr*100:.1f}%\n\n"
            for i, (_, row) in enumerate(top.iterrows(), start=1):
                msg += (
                    f"{i}. *{row['Ticker']}* — *{row['Final_Signal']}*\n"
                    f"   כניסה: ${row['Entry_Price']:.2f} | גודל מומלץ: {int(row['Size'])} מניות\n"
                    f"   TP: ${row['Take_Profit']:.2f} | SL: ${row['Stop_Loss']:.2f} | Trailing: ${row['Trailing_Stop']:.2f}\n"
                    f"   EV: {row['Best_EV']*100:.2f}‰ | WinRate: {row['Best_WinRate']*100:.1f}%\n\n"
                )
            send_telegram_message(msg)
        else:
            self._send_no_picks_summary(len(df), mode="מלא (כל הטיקרים)")

        df.to_csv(RESULTS_CSV, index=False)
        logger.info(f"נשמר סיכום ל-{RESULTS_CSV}")

        if self.failed_tickers:
            send_telegram_message(f"*טיקרים שנכשלו בעיבוד*\n{self.failed_tickers[:50]}{'...' if len(self.failed_tickers) > 50 else ''}")

    def run_single_run(self):
        """גם 'מצומצם' ירוץ על *כל* הטיקרים – לשקיפות והתנהגות אחידה."""
        results = []
        self.failed_tickers = []

        logger.info(f"מעבד {len(self.tickers)} טיקרים במקביל ({MAX_WORKERS} חוטים)...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            fut = {ex.submit(self.process_ticker, t): t for t in self.tickers}
            for f in as_completed(fut):
                r = f.result()
                if r:
                    results.append(r)

        if not results:
            self._send_no_picks_summary(0, mode="מצומצם (על כל הטיקרים)")
            return

        df = pd.DataFrame(results)
        df["Weighted_Score"] = df.apply(weighted_score, axis=1)
        picks = df[(df["Final_Signal"] != "HOLD") & (df["Best_EV"] > 0)].copy()
        top = picks.sort_values("Weighted_Score", ascending=False).head(3)

        if not top.empty:
            msg = f"*3 בחירות מובילות ל-{datetime.utcnow().date()} (ריצה מצומצמת)*\n" \
                  f"*סטטוס שוק*: NYSE סגורה – משתמש בנתונים העדכניים האחרונים\n\n"
            for i, (_, row) in enumerate(top.iterrows(), start=1):
                msg += (
                    f"{i}. *{row['Ticker']}* — *{row['Final_Signal']}*\n"
                    f"   כניסה: ${row['Entry_Price']:.2f} | גודל מומלץ: {int(row['Size'])} מניות\n"
                    f"   TP: ${row['Take_Profit']:.2f} | SL: ${row['Stop_Loss']:.2f} | Trailing: ${row['Trailing_Stop']:.2f}\n"
                    f"   EV: {row['Best_EV']*100:.2f}‰ | WinRate: {row['Best_WinRate']*100:.1f}%\n\n"
                )
            send_telegram_message(msg)
        else:
            self._send_no_picks_summary(len(df), mode="מצומצם (על כל הטיקרים)")

        df.to_csv(RESULTS_CSV_LIMITED, index=False)
        logger.info(f"נשמר סיכום מצומצם ל-{RESULTS_CSV_LIMITED}")

        if self.failed_tickers:
            send_telegram_message(f"*טיקרים שנכשלו בעיבוד (מצומצם)*\n{self.failed_tickers[:50]}{'...' if len(self.failed_tickers) > 50 else ''}")

    def run_once(self):
        tz_ist = pytz.timezone("Asia/Jerusalem")
        now_ist = datetime.now(tz_ist)
        nyse_open = is_nyse_open_now()
        mode = ("אסטרטגיה מלאה על *כל* הטיקרים (שוק פתוח)" if nyse_open
                else "אסטרטגיה מלאה על *כל* הטיקרים (שוק סגור, נתונים עדכניים אחרונים)")

        send_telegram_message(
            f"*בוט מסחר – תחילת ריצה*\n"
            f"זמן מקומי: {now_ist}\n"
            f"סטטוס NYSE: {'פתוחה' אם nyse_open else 'סגורה'}\n"
            f"מצב ריצה: {mode}\n"
            f"פרמטרים: הון={EQUITY:,.0f}$ | סיכון לעסקה={RISK_PER_TRADE*100:.1f}% | Cooldown={COOLDOWN_DAYS} ימים | Workers={MAX_WORKERS}"
        )

        start = time.time()
        try:
            if nyse_open or FORCE_ANALYSIS_WHEN_CLOSED:
                logger.info("מריץ אסטרטגיה מלאה על כל הטיקרים.")
                self.run_full_strategy()
            else:
                logger.info("שוק סגור – מריץ ריצה מצומצמת (אך על כל הטיקרים).")
                self.run_single_run()
        finally:
            elapsed = time.time() - start
            if elapsed > 600:
                send_telegram_message(f"*אזהרה*: זמן ריצה {elapsed/60:.1f} דקות (>10 דק'). בדוק לוגים.")


if __name__ == "__main__":
    bot = AdvancedSwingTradingBot()
    bot.run_once()
