#!/usr/bin/env python3
"""
PROFESSIONAL SWING TRADING SCANNER WITH SQGLP FRAMEWORK
========================================================
- Fixed $10K position sizing with ATR-based stop-loss
- SQGLP flagging for long-term compounders
- Evidence-based multi-factor scoring
- Clean, robust signal generation
- KEY LEVELS: Support/Resistance analysis for each stock

SQGLP Framework:
- Small: Market cap < $10B (room to grow)
- Quality: High ROIC, owner-operators, strong margins
- Growth: 15%+ revenue growth, predictable
- Longevity: Real competitive moats, low beta
- Price: Markets misprice quality (reasonable valuations)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import talib
import time
import pickle
import concurrent.futures
from tqdm import tqdm
import warnings
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scipy.stats import percentileofscore
from scipy.signal import argrelextrema

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    # Position Sizing (Fixed $10K per trade)
    "position_size_dollars": 10000,
    "max_risk_percent": 2.0,  # Max 2% loss per trade ($200 on $10K)
    "atr_stoploss_multiplier": 2.0,  # 2 ATR stop is industry standard
    "min_reward_risk_ratio": 2.5,
    
    # Technical Parameters
    "atr_period": 14,
    "min_avg_volume": 500000,
    "min_relative_volume": 1.2,
    "rsi_period": 14,
    "rsi_optimal_min": 40,
    "rsi_optimal_max": 70,
    
    # Market Structure Filters
    "market_health_ticker": "SPY",
    "min_market_cap": 500_000_000,  # $500M minimum
    "min_price": 5.00,
    "max_price": 500.00,
    "min_data_points": 252,
    
    # Scoring Thresholds
    "min_conviction_score": 50,
    "high_conviction_threshold": 70,
    
    # SQGLP Thresholds
    "sqglp_small_cap_max": 10_000_000_000,  # $10B
    "sqglp_quality_roe_min": 0.15,
    "sqglp_quality_margin_min": 0.10,
    "sqglp_growth_revenue_min": 0.15,
    "sqglp_longevity_beta_max": 1.2,
    "sqglp_price_ps_max": 10,
    "sqglp_min_score": 12,  # Out of ~25 possible
    
    # Performance Settings
    "max_workers": 10,
    "batch_size": 150,
    "ticker_file": "data/tickers.pkl",
    "max_results": 20,
    
    # Key Levels Settings
    "pivot_lookback": 20,  # Days to look back for pivot points
    "level_tolerance": 0.02,  # 2% tolerance for level clustering
}

# ==============================================================================
# ROBUST HTTP SESSION
# ==============================================================================
def create_robust_session():
    """Create HTTP session with retry logic"""
    session = Session()
    retries = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        respect_retry_after_header=True
    )
    session.mount("https://", HTTPAdapter(max_retries=retries, pool_maxsize=20))
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    })
    return session

SHARED_SESSION = create_robust_session()

# ==============================================================================
# KEY LEVELS CALCULATION
# ==============================================================================
def calculate_key_levels(high_series, low_series, close_series, current_price):
    """
    Calculate key support and resistance levels for a stock.
    
    Returns dict with:
    - strong_support: Major support level
    - support_1: Nearest support
    - current_zone: Where price sits
    - resistance_1: Nearest resistance
    - resistance_2: Next resistance
    - ath: All-time high (52-week)
    - atl: All-time low (52-week)
    - ma_levels: Key moving average levels
    """
    try:
        high = high_series.dropna().values
        low = low_series.dropna().values
        close = close_series.dropna().values
        
        if len(close) < 50:
            return None
        
        levels = {
            'supports': [],
            'resistances': [],
            'ma_levels': {},
            'ath': None,
            'atl': None,
            'pct_from_ath': None,
            'pct_from_atl': None,
        }
        
        # 52-week high/low
        lookback_252 = min(252, len(high))
        levels['ath'] = float(np.max(high[-lookback_252:]))
        levels['atl'] = float(np.min(low[-lookback_252:]))
        levels['pct_from_ath'] = ((current_price - levels['ath']) / levels['ath']) * 100
        levels['pct_from_atl'] = ((current_price - levels['atl']) / levels['atl']) * 100
        
        # Moving Averages as dynamic S/R
        if len(close) >= 20:
            levels['ma_levels']['sma_20'] = float(talib.SMA(close, 20)[-1])
        if len(close) >= 50:
            levels['ma_levels']['sma_50'] = float(talib.SMA(close, 50)[-1])
        if len(close) >= 200:
            levels['ma_levels']['sma_200'] = float(talib.SMA(close, 200)[-1])
        
        # Find pivot highs and lows using scipy
        order = CONFIG['pivot_lookback']
        
        # Pivot highs (resistance)
        if len(high) > order * 2:
            pivot_high_idx = argrelextrema(high, np.greater, order=order)[0]
            pivot_highs = high[pivot_high_idx]
            
            # Filter to relevant levels (within reasonable range of current price)
            relevant_highs = [h for h in pivot_highs if h > current_price * 0.9]
            if relevant_highs:
                # Cluster nearby levels
                clustered_res = cluster_levels(relevant_highs, CONFIG['level_tolerance'])
                levels['resistances'] = sorted(clustered_res)[:5]  # Top 5 resistance levels
        
        # Pivot lows (support)
        if len(low) > order * 2:
            pivot_low_idx = argrelextrema(low, np.less, order=order)[0]
            pivot_lows = low[pivot_low_idx]
            
            # Filter to relevant levels
            relevant_lows = [l for l in pivot_lows if l < current_price * 1.1]
            if relevant_lows:
                clustered_sup = cluster_levels(relevant_lows, CONFIG['level_tolerance'])
                levels['supports'] = sorted(clustered_sup, reverse=True)[:5]  # Top 5 support levels
        
        # Add recent swing points (last 60 days)
        recent_high = float(np.max(high[-60:]))
        recent_low = float(np.min(low[-60:]))
        
        if recent_high > current_price and recent_high not in levels['resistances']:
            levels['resistances'].insert(0, recent_high)
        if recent_low < current_price and recent_low not in levels['supports']:
            levels['supports'].insert(0, recent_low)
        
        # Gap detection (significant gaps can act as S/R)
        gaps = detect_gaps(close, high, low)
        for gap in gaps:
            if gap['type'] == 'up' and gap['bottom'] > current_price:
                if gap['bottom'] not in levels['resistances']:
                    levels['resistances'].append(gap['bottom'])
            elif gap['type'] == 'down' and gap['top'] < current_price:
                if gap['top'] not in levels['supports']:
                    levels['supports'].append(gap['top'])
        
        # Sort and clean
        levels['supports'] = sorted(list(set([round(s, 2) for s in levels['supports'] if s < current_price])), reverse=True)[:4]
        levels['resistances'] = sorted(list(set([round(r, 2) for r in levels['resistances'] if r > current_price])))[:4]
        
        # Classify levels
        classified = classify_levels(levels, current_price)
        
        return classified
        
    except Exception as e:
        return None


def cluster_levels(levels, tolerance):
    """Cluster nearby price levels into single levels"""
    if not levels:
        return []
    
    levels = sorted(levels)
    clusters = []
    current_cluster = [levels[0]]
    
    for level in levels[1:]:
        if abs(level - current_cluster[-1]) / current_cluster[-1] <= tolerance:
            current_cluster.append(level)
        else:
            # Save the mean of the cluster
            clusters.append(np.mean(current_cluster))
            current_cluster = [level]
    
    clusters.append(np.mean(current_cluster))
    return clusters


def detect_gaps(close, high, low):
    """Detect unfilled gaps that can act as S/R"""
    gaps = []
    
    for i in range(1, len(close)):
        # Gap up: today's low > yesterday's high
        if low[i] > high[i-1]:
            gaps.append({
                'type': 'up',
                'top': float(low[i]),
                'bottom': float(high[i-1]),
                'size': float(low[i] - high[i-1])
            })
        # Gap down: today's high < yesterday's low
        elif high[i] < low[i-1]:
            gaps.append({
                'type': 'down',
                'top': float(low[i-1]),
                'bottom': float(high[i]),
                'size': float(low[i-1] - high[i])
            })
    
    # Return only significant gaps (> 1% of price) from last 60 days
    recent_gaps = gaps[-60:] if len(gaps) > 60 else gaps
    avg_price = np.mean(close[-60:])
    significant_gaps = [g for g in recent_gaps if g['size'] / avg_price > 0.01]
    
    return significant_gaps[-5:]  # Return last 5 significant gaps


def classify_levels(levels, current_price):
    """Classify levels by significance and proximity"""
    classified = {
        'strong_support': None,
        'support_1': None,
        'support_2': None,
        'current_price': current_price,
        'resistance_1': None,
        'resistance_2': None,
        'resistance_3': None,
        'ath': levels['ath'],
        'atl': levels['atl'],
        'pct_from_ath': levels['pct_from_ath'],
        'pct_from_atl': levels['pct_from_atl'],
        'ma_levels': levels['ma_levels'],
        'all_supports': levels['supports'],
        'all_resistances': levels['resistances'],
    }
    
    # Assign support levels
    supports = levels['supports']
    if supports:
        classified['support_1'] = supports[0]  # Nearest support
        if len(supports) > 1:
            classified['support_2'] = supports[1]
        # Strong support is the lowest tested level or 200 SMA
        if len(supports) > 2:
            classified['strong_support'] = supports[-1]
        elif 'sma_200' in levels['ma_levels']:
            sma200 = levels['ma_levels']['sma_200']
            if sma200 < current_price:
                classified['strong_support'] = sma200
    
    # Assign resistance levels
    resistances = levels['resistances']
    if resistances:
        classified['resistance_1'] = resistances[0]  # Nearest resistance
        if len(resistances) > 1:
            classified['resistance_2'] = resistances[1]
        if len(resistances) > 2:
            classified['resistance_3'] = resistances[2]
    
    # Add ATH as final resistance if not already included
    if classified['ath'] and (not classified['resistance_3'] or classified['ath'] > classified['resistance_3']):
        if classified['pct_from_ath'] < -5:  # Only if we're meaningfully below ATH
            classified['resistance_3'] = classified['ath']
    
    return classified


# ==============================================================================
# MARKET REGIME DETECTION
# ==============================================================================
def check_market_health(ticker='SPY'):
    """
    Analyze market regime for position sizing adjustments.
    Returns: (status_string, regime_score 0-100, volatility_percentile)
    """
    print(f"\n{'='*70}")
    print(f"📊 MARKET REGIME ANALYSIS: {ticker}")
    print(f"{'='*70}")
    
    try:
        data = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
        if data.empty or len(data) < 252:
            return "Unknown", 50, 50
        
        close = data['Close'].values.flatten().astype(np.float64)
        high = data['High'].values.flatten().astype(np.float64)
        low = data['Low'].values.flatten().astype(np.float64)
        
        # Moving averages
        sma50 = talib.SMA(close, 50)
        sma200 = talib.SMA(close, 200)
        atr = talib.ATR(high, low, close, 14)
        
        price = close[-1]
        sma50_val, sma200_val = sma50[-1], sma200[-1]
        
        # Trend scoring (0-100)
        trend_score = 0
        if price > sma200_val: trend_score += 25
        if price > sma50_val: trend_score += 25
        if sma50_val > sma200_val: trend_score += 25
        
        # Momentum (20-day ROC)
        roc_20 = ((price - close[-20]) / close[-20]) * 100
        if roc_20 > 0: trend_score += 25
        
        # Volatility percentile
        vol_percentile = percentileofscore(atr[-60:], atr[-1])
        
        # Status determination
        if trend_score >= 75 and vol_percentile < 70:
            status = "🟢 STRONG BULL (Low Vol)"
        elif trend_score >= 50:
            status = "🟡 BULL (Cautious)"
        elif trend_score >= 25:
            status = "🟠 NEUTRAL"
        else:
            status = "🔴 BEAR"
        
        print(f"Status: {status}")
        print(f"  SPY: ${price:.2f} | 50-SMA: ${sma50_val:.2f} | 200-SMA: ${sma200_val:.2f}")
        print(f"  Trend Score: {trend_score}/100 | Volatility: {vol_percentile:.0f}th percentile")
        print(f"  20-Day Momentum: {roc_20:+.2f}%")
        print(f"{'='*70}\n")
        
        return status, trend_score, vol_percentile
        
    except Exception as e:
        print(f"❌ Error: {e}. Using defaults.")
        return "Unknown", 50, 50

# ==============================================================================
# DATA VALIDATION
# ==============================================================================
def validate_and_clean_data(hist_data, batch_tickers):
    """Validate downloaded data quality"""
    if hist_data.empty:
        return None, []
    
    # Handle single ticker case
    if not isinstance(hist_data.columns, pd.MultiIndex):
        if len(batch_tickers) == 1:
            ticker = batch_tickers[0]
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            if all(col in hist_data.columns for col in required):
                if len(hist_data) >= CONFIG['min_data_points']:
                    hist_data.columns = pd.MultiIndex.from_product([[ticker], hist_data.columns])
                    return hist_data, [ticker]
        return None, []
    
    valid_tickers = []
    required_fields = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    for ticker in hist_data.columns.get_level_values(0).unique():
        try:
            if not all((ticker, f) in hist_data.columns for f in required_fields):
                continue
            
            close_data = hist_data[ticker, 'Close']
            if close_data.isnull().all() or close_data.count() < CONFIG['min_data_points']:
                continue
            
            if close_data.mean() <= 0:
                continue
            
            # Skip stocks with too many extreme moves (data quality issue)
            if (close_data.pct_change().abs() > 0.5).sum() > 5:
                continue
            
            valid_tickers.append(ticker)
        except:
            continue
    
    if not valid_tickers:
        return None, []
    
    return hist_data[valid_tickers], valid_tickers

# ==============================================================================
# TECHNICAL ANALYSIS
# ==============================================================================
def safe_indicator(func, *args, **kwargs):
    """Safely calculate indicator, return last value"""
    try:
        result = func(*args, **kwargs)
        return result[-1] if len(result) > 0 else np.nan
    except:
        return np.nan

def calculate_slope(series, periods):
    """Linear regression slope normalized by price"""
    try:
        if len(series) < periods:
            return np.nan
        y = series.iloc[-periods:].values
        x = np.arange(periods)
        slope = np.polyfit(x, y, 1)[0]
        return slope / np.mean(y) if np.mean(y) != 0 else np.nan
    except:
        return np.nan

def run_technical_analysis(df):
    """
    Multi-factor technical analysis.
    Returns DataFrame with scores and indicators.
    """
    if df.empty:
        return pd.DataFrame(), {}
    
    try:
        if isinstance(df.columns, pd.MultiIndex):
            tickers = df.columns.get_level_values(0).unique()
            close = pd.DataFrame({t: df[t]['Close'] for t in tickers})
            high = pd.DataFrame({t: df[t]['High'] for t in tickers})
            low = pd.DataFrame({t: df[t]['Low'] for t in tickers})
            volume = pd.DataFrame({t: df[t]['Volume'] for t in tickers})
        else:
            return pd.DataFrame(), {}
    except Exception as e:
        print(f"⚠️ Data extraction error: {e}")
        return pd.DataFrame(), {}
    
    results = pd.DataFrame(index=close.columns)
    key_levels_dict = {}  # Store key levels for each ticker
    
    # === PRICE & TREND ===
    results['current_price'] = close.iloc[-1]
    results['sma_20'] = close.apply(lambda x: safe_indicator(talib.SMA, x.dropna(), 20))
    results['sma_50'] = close.apply(lambda x: safe_indicator(talib.SMA, x.dropna(), 50))
    results['sma_200'] = close.apply(lambda x: safe_indicator(talib.SMA, x.dropna(), 200))
    results['ema_21'] = close.apply(lambda x: safe_indicator(talib.EMA, x.dropna(), 21))
    
    # ATR
    def calc_atr(ticker):
        try:
            h = high[ticker].dropna().values
            l = low[ticker].dropna().values
            c = close[ticker].dropna().values
            min_len = min(len(h), len(l), len(c))
            if min_len < CONFIG['atr_period'] + 1:
                return np.nan
            return safe_indicator(talib.ATR, h[-min_len:], l[-min_len:], c[-min_len:], CONFIG['atr_period'])
        except:
            return np.nan
    
    results['atr'] = results.index.map(calc_atr)
    results['atr_percent'] = (results['atr'] / results['current_price']) * 100
    
    # === CALCULATE KEY LEVELS FOR EACH TICKER ===
    for ticker in tickers:
        try:
            current_price = results.loc[ticker, 'current_price']
            levels = calculate_key_levels(
                high[ticker],
                low[ticker],
                close[ticker],
                current_price
            )
            if levels:
                key_levels_dict[ticker] = levels
        except Exception as e:
            continue
    
    # === MOMENTUM ===
    results['rsi'] = close.apply(lambda x: safe_indicator(talib.RSI, x.dropna(), CONFIG['rsi_period']))
    
    # MACD
    def calc_macd(series):
        try:
            macd, signal, hist = talib.MACD(series.dropna(), 12, 26, 9)
            if len(macd) < 2:
                return pd.Series([np.nan] * 4)
            return pd.Series([macd[-1], signal[-1], hist[-1], hist[-2]])
        except:
            return pd.Series([np.nan] * 4)
    
    macd_data = close.apply(calc_macd)
    results['macd'] = macd_data.iloc[0]
    results['macd_signal'] = macd_data.iloc[1]
    results['macd_hist'] = macd_data.iloc[2]
    results['macd_hist_prev'] = macd_data.iloc[3]
    
    # Rate of Change
    results['roc_10'] = close.apply(lambda x: safe_indicator(talib.ROC, x.dropna(), 10))
    results['roc_20'] = close.apply(lambda x: safe_indicator(talib.ROC, x.dropna(), 20))
    
    # === TREND STRENGTH ===
    results['slope_20'] = close.apply(lambda x: calculate_slope(x.dropna(), 20))
    results['slope_60'] = close.apply(lambda x: calculate_slope(x.dropna(), 60))
    
    # ADX
    def calc_adx(ticker):
        try:
            h = high[ticker].dropna().values
            l = low[ticker].dropna().values
            c = close[ticker].dropna().values
            min_len = min(len(h), len(l), len(c))
            if min_len < 14:
                return np.nan
            return safe_indicator(talib.ADX, h[-min_len:], l[-min_len:], c[-min_len:], 14)
        except:
            return np.nan
    
    results['adx'] = results.index.map(calc_adx)
    
    # === VOLUME ===
    results['avg_vol_20d'] = volume.apply(lambda x: x.iloc[-20:].mean() if len(x) >= 20 else x.mean())
    results['vol_5d'] = volume.apply(lambda x: x.iloc[-5:].mean() if len(x) >= 5 else x.mean())
    results['rel_volume'] = (results['vol_5d'] / results['avg_vol_20d']).fillna(0)
    
    # OBV trend
    def calc_obv_slope(ticker):
        try:
            obv = talib.OBV(close[ticker].dropna().values, volume[ticker].dropna().values)
            if len(obv) < 20:
                return np.nan
            return calculate_slope(pd.Series(obv), 20)
        except:
            return np.nan
    
    results['obv_slope'] = results.index.map(calc_obv_slope)
    
    # === PRICE POSITION ===
    results['high_52w'] = high.apply(lambda x: x.iloc[-252:].max() if len(x) >= 252 else x.max())
    results['low_52w'] = low.apply(lambda x: x.iloc[-252:].min() if len(x) >= 252 else x.min())
    results['pct_from_high'] = ((results['current_price'] - results['high_52w']) / results['high_52w']) * 100
    results['pct_from_low'] = ((results['current_price'] - results['low_52w']) / results['low_52w']) * 100
    
    # === CONVICTION SCORING (100 points max) ===
    score = pd.Series(0.0, index=results.index)
    
    # 1. TREND ALIGNMENT (30 points)
    score += np.where(
        (results['current_price'] > results['ema_21']) & 
        (results['ema_21'] > results['sma_50']) & 
        (results['sma_50'] > results['sma_200']),
        20, 0
    )
    score += np.where(
        (results['current_price'] > results['sma_50']) & 
        (results['sma_50'] > results['sma_200']) &
        ~((results['ema_21'] > results['sma_50'])),
        12, 0
    )
    score += np.where(results['adx'] > 25, 10, np.where(results['adx'] > 20, 5, 0))
    
    # 2. MOMENTUM (25 points)
    score += np.where(
        (results['rsi'] >= CONFIG['rsi_optimal_min']) & 
        (results['rsi'] <= CONFIG['rsi_optimal_max']),
        10, 0
    )
    score += np.where(results['macd'] > results['macd_signal'], 8, 0)
    score += np.where(
        (results['macd_hist'] > 0) & (results['macd_hist'] > results['macd_hist_prev']),
        7, 0
    )
    
    # 3. PRICE STRUCTURE (20 points)
    score += np.where(results['slope_60'] > 0.002, 12, np.where(results['slope_60'] > 0.001, 6, 0))
    score += np.where(
        (results['pct_from_high'] > -15) & (results['pct_from_high'] < -3),
        8, 0
    )
    
    # 4. VOLUME CONFIRMATION (15 points)
    score += np.where(results['rel_volume'] > 1.5, 10, np.where(results['rel_volume'] > 1.2, 6, 0))
    score += np.where(results['obv_slope'] > 0, 5, 0)
    
    # 5. VOLATILITY (10 points)
    score += np.where(
        (results['atr_percent'] > 1.5) & (results['atr_percent'] < 5),
        10, np.where((results['atr_percent'] >= 5) & (results['atr_percent'] < 8), 5, 0)
    )
    
    results['conviction_score'] = score
    
    # === APPLY FILTERS ===
    filters = (
        (results['current_price'] >= CONFIG['min_price']) &
        (results['current_price'] <= CONFIG['max_price']) &
        (results['avg_vol_20d'] >= CONFIG['min_avg_volume']) &
        (results['atr'] > 0) & (~results['atr'].isna()) &
        (results['rel_volume'] >= CONFIG['min_relative_volume']) &
        (results['conviction_score'] >= CONFIG['min_conviction_score']) &
        (~results['adx'].isna())
    )
    
    # Filter key_levels_dict to only include passing tickers
    filtered_tickers = results[filters].index.tolist()
    filtered_key_levels = {t: key_levels_dict[t] for t in filtered_tickers if t in key_levels_dict}
    
    return results[filters], filtered_key_levels

# ==============================================================================
# FUNDAMENTAL ANALYSIS + SQGLP
# ==============================================================================
def fetch_fundamentals(tickers):
    """Fetch fundamental data with fallbacks for rate limiting"""
    def get_info(ticker):
        try:
            stock = yf.Ticker(ticker, session=SHARED_SESSION)
            
            try:
                info = stock.info
                if not info or not isinstance(info, dict):
                    info = {}
            except Exception:
                info = {}
            
            if len(info) < 5:
                try:
                    fast = stock.fast_info
                    info = {
                        'marketCap': getattr(fast, 'market_cap', 0),
                        'sector': 'Unknown',
                        'industry': 'Unknown',
                        'shortName': ticker,
                        'beta': 1.0,
                    }
                except:
                    info = {'shortName': ticker}
            
            data = {
                'market_cap': info.get('marketCap', 0) or 0,
                'revenue_growth': info.get('revenueGrowth', 0) or 0,
                'profit_margin': info.get('profitMargins', 0) or 0,
                'operating_margin': info.get('operatingMargins', 0) or 0,
                'roe': info.get('returnOnEquity', 0) or 0,
                'roic': info.get('returnOnEquity', 0) or 0,
                'debt_to_equity': info.get('debtToEquity', 0) or 0,
                'current_ratio': info.get('currentRatio', 0) or 0,
                'free_cash_flow': info.get('freeCashflow', 0) or 0,
                'price_to_sales': info.get('priceToSalesTrailing12Months', None),
                'price_to_earnings': info.get('trailingPE', None),
                'price_to_book': info.get('priceToBook', None),
                'earnings_growth': info.get('earningsGrowth', 0) or 0,
                'industry': info.get('industry', 'Unknown'),
                'sector': info.get('sector', 'Unknown'),
                'beta': info.get('beta', 1.0) or 1.0,
                'insider_ownership': info.get('heldPercentInsiders', 0) or 0,
                'short_name': info.get('shortName', ticker),
            }
            
            return ticker, data
            
        except Exception as e:
            return ticker, {
                'market_cap': 0, 'revenue_growth': 0, 'profit_margin': 0,
                'operating_margin': 0, 'roe': 0, 'roic': 0, 'debt_to_equity': 0,
                'current_ratio': 0, 'free_cash_flow': 0, 'price_to_sales': None,
                'price_to_earnings': None, 'price_to_book': None, 'earnings_growth': 0,
                'industry': 'Unknown', 'sector': 'Unknown', 'beta': 1.0,
                'insider_ownership': 0, 'short_name': ticker,
            }
    
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(get_info, t): t for t in tickers}
        
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(tickers), desc="📊 Fundamentals"):
            ticker, data = future.result()
            if data:
                results[ticker] = data
    
    return results

def safe_float(val, default=0.0):
    """Safely convert value to float, handling None and strings"""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def compute_fundamental_quality(fund):
    """Score fundamental quality and track which criteria are met."""
    criteria = []
    met = []
    
    mkt_cap = safe_float(fund.get('market_cap', 0))
    rev_growth = safe_float(fund.get('revenue_growth', 0))
    profit_margin = safe_float(fund.get('profit_margin', 0))
    roe = safe_float(fund.get('roe', 0))
    debt_to_equity = safe_float(fund.get('debt_to_equity', 0))
    fcf = safe_float(fund.get('free_cash_flow', 0))
    
    criteria.append('market_cap')
    if mkt_cap >= 500_000_000:
        met.append('market_cap')
    
    criteria.append('revenue_growth')
    if rev_growth > 0.10:
        met.append('revenue_growth')
    
    criteria.append('profitable')
    if profit_margin > 0:
        met.append('profitable')
    
    criteria.append('roe')
    if roe > 0.10:
        met.append('roe')
    
    criteria.append('low_debt')
    if debt_to_equity < 1.5:
        met.append('low_debt')
    
    criteria.append('fcf_positive')
    if fcf > 0:
        met.append('fcf_positive')
    
    score = len(met)
    total = len(criteria)
    
    return {
        'score': score,
        'total': total,
        'ratio': f"{score}/{total}",
        'met': met,
        'missing': [c for c in criteria if c not in met],
        'quality_grade': 'Strong' if score >= 5 else 'Good' if score >= 3 else 'Weak' if score >= 1 else 'N/A'
    }


def compute_sqglp_analysis(fund):
    """SQGLP Framework Analysis for Long-Term Compounder Identification"""
    sqglp = {
        'small': {'pass': False, 'score': 0, 'detail': ''},
        'quality': {'pass': False, 'score': 0, 'detail': ''},
        'growth': {'pass': False, 'score': 0, 'detail': ''},
        'longevity': {'pass': False, 'score': 0, 'detail': ''},
        'price': {'pass': False, 'score': 0, 'detail': ''},
    }
    
    total_score = 0
    
    mkt_cap = safe_float(fund.get('market_cap', 0))
    roe = safe_float(fund.get('roe', 0))
    profit_margin = safe_float(fund.get('profit_margin', 0))
    debt_to_equity = safe_float(fund.get('debt_to_equity', 0))
    rev_growth = safe_float(fund.get('revenue_growth', 0))
    earn_growth = safe_float(fund.get('earnings_growth', 0))
    beta = safe_float(fund.get('beta', 1.0), 1.0)
    op_margin = safe_float(fund.get('operating_margin', 0))
    fcf = safe_float(fund.get('free_cash_flow', 0))
    ps = fund.get('price_to_sales')
    pe = fund.get('price_to_earnings')
    
    # === S: SMALL ===
    mkt_cap_b = mkt_cap / 1e9
    if mkt_cap_b < 2:
        sqglp['small']['pass'] = True
        sqglp['small']['score'] = 5
        sqglp['small']['detail'] = f"${mkt_cap_b:.1f}B - Micro/Small cap"
    elif mkt_cap_b < 5:
        sqglp['small']['pass'] = True
        sqglp['small']['score'] = 4
        sqglp['small']['detail'] = f"${mkt_cap_b:.1f}B - Small cap"
    elif mkt_cap_b < 10:
        sqglp['small']['pass'] = True
        sqglp['small']['score'] = 3
        sqglp['small']['detail'] = f"${mkt_cap_b:.1f}B - Mid cap"
    elif mkt_cap_b < 25:
        sqglp['small']['score'] = 1
        sqglp['small']['detail'] = f"${mkt_cap_b:.1f}B - Large cap"
    else:
        sqglp['small']['detail'] = f"${mkt_cap_b:.1f}B - Mega cap"
    total_score += sqglp['small']['score']
    
    # === Q: QUALITY ===
    quality_points = 0
    quality_notes = []
    
    if roe > 0.25:
        quality_points += 3
        quality_notes.append(f"ROE {roe*100:.0f}%")
    elif roe > 0.15:
        quality_points += 2
        quality_notes.append(f"ROE {roe*100:.0f}%")
    elif roe > 0.10:
        quality_points += 1
        quality_notes.append(f"ROE {roe*100:.0f}%")
    
    if profit_margin > 0.20:
        quality_points += 2
        quality_notes.append(f"Margin {profit_margin*100:.0f}%")
    elif profit_margin > 0.10:
        quality_points += 1
        quality_notes.append(f"Margin {profit_margin*100:.0f}%")
    
    if debt_to_equity < 0.3:
        quality_points += 2
        quality_notes.append(f"D/E {debt_to_equity:.1f}")
    elif debt_to_equity < 0.8:
        quality_points += 1
        quality_notes.append(f"D/E {debt_to_equity:.1f}")
    
    sqglp['quality']['score'] = quality_points
    sqglp['quality']['pass'] = quality_points >= 4
    sqglp['quality']['detail'] = " | ".join(quality_notes) if quality_notes else "Limited data"
    total_score += quality_points
    
    # === G: GROWTH ===
    growth_points = 0
    growth_notes = []
    
    if rev_growth > 0.30:
        growth_points += 4
        growth_notes.append(f"Rev +{rev_growth*100:.0f}%")
    elif rev_growth > 0.20:
        growth_points += 3
        growth_notes.append(f"Rev +{rev_growth*100:.0f}%")
    elif rev_growth > 0.15:
        growth_points += 2
        growth_notes.append(f"Rev +{rev_growth*100:.0f}%")
    elif rev_growth > 0.08:
        growth_points += 1
        growth_notes.append(f"Rev +{rev_growth*100:.0f}%")
    
    if earn_growth > 0.15:
        growth_points += 1
        growth_notes.append(f"EPS +{earn_growth*100:.0f}%")
    
    sqglp['growth']['score'] = min(growth_points, 5)
    sqglp['growth']['pass'] = rev_growth >= CONFIG['sqglp_growth_revenue_min']
    sqglp['growth']['detail'] = " | ".join(growth_notes) if growth_notes else "Low/no growth"
    total_score += sqglp['growth']['score']
    
    # === L: LONGEVITY ===
    longevity_points = 0
    longevity_notes = []
    
    if beta < 0.8:
        longevity_points += 2
        longevity_notes.append(f"Beta {beta:.2f}")
    elif beta < 1.1:
        longevity_points += 1
        longevity_notes.append(f"Beta {beta:.2f}")
    
    if op_margin > 0.25:
        longevity_points += 2
        longevity_notes.append(f"Op Margin {op_margin*100:.0f}%")
    elif op_margin > 0.15:
        longevity_points += 1
        longevity_notes.append(f"Op Margin {op_margin*100:.0f}%")
    
    if fcf > 0:
        longevity_points += 1
        longevity_notes.append("FCF+")
    
    sqglp['longevity']['score'] = longevity_points
    sqglp['longevity']['pass'] = longevity_points >= 3
    sqglp['longevity']['detail'] = " | ".join(longevity_notes) if longevity_notes else "Limited moat"
    total_score += longevity_points
    
    # === P: PRICE ===
    price_points = 0
    price_notes = []
    
    ps_val = safe_float(ps) if ps is not None else None
    pe_val = safe_float(pe) if pe is not None else None
    
    if ps_val is not None and ps_val > 0:
        if ps_val < 3:
            price_points += 2
            price_notes.append(f"P/S {ps_val:.1f}")
        elif ps_val < 6:
            price_points += 1
            price_notes.append(f"P/S {ps_val:.1f}")
        else:
            price_notes.append(f"P/S {ps_val:.1f}")
    
    if pe_val is not None and pe_val > 0:
        if pe_val < 20:
            price_points += 2
            price_notes.append(f"P/E {pe_val:.0f}")
        elif pe_val < 35:
            price_points += 1
            price_notes.append(f"P/E {pe_val:.0f}")
        else:
            price_notes.append(f"P/E {pe_val:.0f}")
    
    sqglp['price']['score'] = min(price_points, 4)
    sqglp['price']['pass'] = price_points >= 2
    sqglp['price']['detail'] = " | ".join(price_notes) if price_notes else "Valuation unclear"
    total_score += sqglp['price']['score']
    
    # === OVERALL ===
    pillars_passed = sum([
        sqglp['small']['pass'],
        sqglp['quality']['pass'],
        sqglp['growth']['pass'],
        sqglp['longevity']['pass'],
        sqglp['price']['pass']
    ])
    
    sqglp['total_score'] = total_score
    sqglp['pillars_passed'] = pillars_passed
    sqglp['is_compounder'] = total_score >= CONFIG['sqglp_min_score'] and pillars_passed >= 3
    
    if total_score >= 20:
        sqglp['grade'] = "A+ (Elite)"
    elif total_score >= 16:
        sqglp['grade'] = "A (Strong)"
    elif total_score >= 12:
        sqglp['grade'] = "B+ (Potential)"
    elif total_score >= 8:
        sqglp['grade'] = "B (Some Quality)"
    else:
        sqglp['grade'] = "C (Trading Only)"
    
    return sqglp

# ==============================================================================
# POSITION SIZING ($10K Fixed)
# ==============================================================================
def calculate_position_size(price, atr, score, market_regime_score):
    """Fixed $10K position sizing with ATR-based stop-loss."""
    position_value = CONFIG['position_size_dollars']
    
    stop_distance = atr * CONFIG['atr_stoploss_multiplier']
    stop_loss = price - stop_distance
    
    if stop_distance <= 0 or stop_loss <= 0:
        return None
    
    risk_per_share = stop_distance
    risk_percent = (risk_per_share / price) * 100
    
    max_risk_dollars = position_value * (CONFIG['max_risk_percent'] / 100)
    
    shares_from_capital = int(position_value / price)
    shares_from_risk = int(max_risk_dollars / risk_per_share)
    
    shares = min(shares_from_capital, shares_from_risk)
    
    if shares <= 0:
        return None
    
    actual_position_value = shares * price
    actual_risk = shares * risk_per_share
    
    target = price + (risk_per_share * CONFIG['min_reward_risk_ratio'])
    potential_gain = shares * (target - price)
    
    return {
        'shares': shares,
        'entry': price,
        'stop_loss': stop_loss,
        'stop_distance': stop_distance,
        'stop_percent': risk_percent,
        'target': target,
        'position_value': actual_position_value,
        'max_loss': actual_risk,
        'max_loss_percent': (actual_risk / actual_position_value) * 100,
        'potential_gain': potential_gain,
        'reward_risk': CONFIG['min_reward_risk_ratio'],
    }

# ==============================================================================
# RESULTS ASSEMBLY
# ==============================================================================
def assemble_results(tech_df, fund_dict, key_levels_dict, market_regime_score):
    """Combine technical, fundamental, SQGLP analysis, and key levels"""
    results = []
    
    for ticker in tech_df.index:
        tech = tech_df.loc[ticker]
        base_score = tech['conviction_score']
        
        if ticker in fund_dict:
            fundamentals = fund_dict[ticker]
        else:
            fundamentals = {
                'market_cap': 0, 'revenue_growth': 0, 'profit_margin': 0,
                'operating_margin': 0, 'roe': 0, 'roic': 0, 'debt_to_equity': 0,
                'current_ratio': 0, 'free_cash_flow': 0, 'price_to_sales': None,
                'price_to_earnings': None, 'price_to_book': None, 'earnings_growth': 0,
                'industry': 'Unknown', 'sector': 'Unknown', 'beta': 1.0,
                'insider_ownership': 0, 'short_name': ticker,
            }
        
        fund_quality = compute_fundamental_quality(fundamentals)
        sqglp = compute_sqglp_analysis(fundamentals)
        
        # Get key levels
        key_levels = key_levels_dict.get(ticker, None)
        
        if fund_quality['score'] >= 5:
            base_score += 8
        elif fund_quality['score'] >= 4:
            base_score += 5
        elif fund_quality['score'] >= 3:
            base_score += 3
        
        if sqglp['is_compounder']:
            base_score += 5
        
        position = calculate_position_size(
            tech['current_price'],
            tech['atr'],
            base_score,
            market_regime_score
        )
        
        if position is None:
            continue
        
        results.append({
            'ticker': ticker,
            'name': fundamentals['short_name'],
            'conviction_score': min(base_score, 100),
            'technicals': {
                'price': tech['current_price'],
                'rsi': tech['rsi'],
                'adx': tech['adx'],
                'macd_hist': tech['macd_hist'],
                'rel_volume': tech['rel_volume'],
                'pct_from_high': tech['pct_from_high'],
                'atr': tech['atr'],
                'atr_percent': tech['atr_percent'],
                'slope_60': tech['slope_60'],
                'sma_20': tech['sma_20'],
                'sma_50': tech['sma_50'],
                'sma_200': tech['sma_200'],
            },
            'fundamentals': fundamentals,
            'fund_quality': fund_quality,
            'sqglp': sqglp,
            'position': position,
            'key_levels': key_levels,
        })
    
    return results

# ==============================================================================
# OUTPUT FORMATTING
# ==============================================================================
def print_key_levels_table(key_levels, current_price):
    """Print formatted key levels table"""
    if not key_levels:
        print("   ⚠️ Key levels data unavailable")
        return
    
    print("\n🎯 KEY LEVELS")
    print("   ┌─────────────────────────┬────────────┬─────────────────────────────────┐")
    print("   │ Level                   │ Price      │ Significance                    │")
    print("   ├─────────────────────────┼────────────┼─────────────────────────────────┤")
    
    # Resistances (from nearest to furthest)
    if key_levels.get('ath'):
        pct = key_levels['pct_from_ath']
        print(f"   │ 🔴 ATH (52-Week High)   │ ${key_levels['ath']:>8.2f} │ {pct:+.1f}% from current              │")
    
    if key_levels.get('resistance_3') and key_levels['resistance_3'] != key_levels.get('ath'):
        pct = ((key_levels['resistance_3'] - current_price) / current_price) * 100
        print(f"   │ 🟠 Resistance 3         │ ${key_levels['resistance_3']:>8.2f} │ +{pct:.1f}% - Previous high           │")
    
    if key_levels.get('resistance_2'):
        pct = ((key_levels['resistance_2'] - current_price) / current_price) * 100
        print(f"   │ 🟡 Resistance 2         │ ${key_levels['resistance_2']:>8.2f} │ +{pct:.1f}% - Key resistance          │")
    
    if key_levels.get('resistance_1'):
        pct = ((key_levels['resistance_1'] - current_price) / current_price) * 100
        print(f"   │ 🟡 Resistance 1         │ ${key_levels['resistance_1']:>8.2f} │ +{pct:.1f}% - Nearest resistance      │")
    
    # Current price
    print(f"   ├─────────────────────────┼────────────┼─────────────────────────────────┤")
    print(f"   │ ➡️  CURRENT PRICE        │ ${current_price:>8.2f} │ << YOU ARE HERE                 │")
    print(f"   ├─────────────────────────┼────────────┼─────────────────────────────────┤")
    
    # Supports (from nearest to furthest)
    if key_levels.get('support_1'):
        pct = ((key_levels['support_1'] - current_price) / current_price) * 100
        print(f"   │ 🟢 Support 1            │ ${key_levels['support_1']:>8.2f} │ {pct:.1f}% - Nearest support          │")
    
    if key_levels.get('support_2'):
        pct = ((key_levels['support_2'] - current_price) / current_price) * 100
        print(f"   │ 🟢 Support 2            │ ${key_levels['support_2']:>8.2f} │ {pct:.1f}% - Secondary support        │")
    
    if key_levels.get('strong_support'):
        pct = ((key_levels['strong_support'] - current_price) / current_price) * 100
        print(f"   │ 💪 Strong Support       │ ${key_levels['strong_support']:>8.2f} │ {pct:.1f}% - Major accumulation zone │")
    
    if key_levels.get('atl'):
        pct = key_levels['pct_from_atl']
        print(f"   │ ⬇️  ATL (52-Week Low)    │ ${key_levels['atl']:>8.2f} │ +{pct:.1f}% from current             │")
    
    print("   └─────────────────────────┴────────────┴─────────────────────────────────┘")
    
    # Moving Average context
    ma_levels = key_levels.get('ma_levels', {})
    if ma_levels:
        print("\n   📊 Moving Average Context:")
        ma_status = []
        if 'sma_20' in ma_levels:
            pos = "above" if current_price > ma_levels['sma_20'] else "below"
            ma_status.append(f"20-SMA: ${ma_levels['sma_20']:.2f} ({pos})")
        if 'sma_50' in ma_levels:
            pos = "above" if current_price > ma_levels['sma_50'] else "below"
            ma_status.append(f"50-SMA: ${ma_levels['sma_50']:.2f} ({pos})")
        if 'sma_200' in ma_levels:
            pos = "above" if current_price > ma_levels['sma_200'] else "below"
            ma_status.append(f"200-SMA: ${ma_levels['sma_200']:.2f} ({pos})")
        
        print(f"   {' | '.join(ma_status)}")


def print_detailed_analysis(res, idx):
    """Print comprehensive analysis for a single stock"""
    score = res['conviction_score']
    sqglp = res['sqglp']
    tech = res['technicals']
    fund = res['fundamentals']
    pos = res['position']
    fq = res['fund_quality']
    key_levels = res.get('key_levels')
    
    if score >= 80:
        grade = "🏆 A+"
    elif score >= 70:
        grade = "🥇 A"
    elif score >= 60:
        grade = "🥈 B+"
    else:
        grade = "🥉 B"
    
    if sqglp['is_compounder']:
        lt_badge = "⭐ LONG-TERM COMPOUNDER CANDIDATE"
    else:
        lt_badge = ""
    
    fq_badge = f"📋 Fundamentals: {fq['ratio']} ({fq['quality_grade']})"
    
    print(f"\n{'─'*70}")
    print(f"#{idx}  {res['ticker']} - {res['name']}")
    print(f"     {grade} | Swing Score: {score:.0f}/100 | {fq_badge}")
    if lt_badge:
        print(f"     {lt_badge}")
    print(f"{'─'*70}")
    
    # Sector & Market Cap
    mkt_cap_str = f"${fund['market_cap']/1e9:.2f}B" if fund['market_cap'] >= 1e9 else f"${fund['market_cap']/1e6:.0f}M" if fund['market_cap'] > 0 else "N/A"
    print(f"\n📍 {fund['sector']} > {fund['industry']}")
    print(f"   Market Cap: {mkt_cap_str} | Beta: {fund['beta']:.2f}")
    
    # === KEY LEVELS TABLE ===
    print_key_levels_table(key_levels, tech['price'])
    
    # Technical Profile
    print(f"\n📊 TECHNICAL PROFILE")
    print(f"   Price: ${tech['price']:.2f} ({tech['pct_from_high']:+.1f}% from 52w high)")
    print(f"   RSI: {tech['rsi']:.1f} | ADX: {tech['adx']:.1f} | MACD Hist: {tech['macd_hist']:.3f}")
    print(f"   Rel Volume: {tech['rel_volume']:.2f}x | 60d Slope: {tech['slope_60']*1000:.2f}")
    print(f"   ATR: ${tech['atr']:.2f} ({tech['atr_percent']:.1f}% of price)")
    
    # Fundamental Quality
    print(f"\n💼 FUNDAMENTAL QUALITY ({fq['ratio']} criteria met)")
    fund_checks = []
    fund_checks.append(f"{'✅' if 'market_cap' in fq['met'] else '❌'} Mkt Cap $500M+")
    fund_checks.append(f"{'✅' if 'revenue_growth' in fq['met'] else '❌'} Rev Growth >10%")
    fund_checks.append(f"{'✅' if 'profitable' in fq['met'] else '❌'} Profitable")
    fund_checks.append(f"{'✅' if 'roe' in fq['met'] else '❌'} ROE >10%")
    fund_checks.append(f"{'✅' if 'low_debt' in fq['met'] else '❌'} D/E <1.5")
    fund_checks.append(f"{'✅' if 'fcf_positive' in fq['met'] else '❌'} FCF+")
    print(f"   {' | '.join(fund_checks[:3])}")
    print(f"   {' | '.join(fund_checks[3:])}")
    
    print(f"   Revenue Growth: {fund['revenue_growth']*100:+.1f}% | Profit Margin: {fund['profit_margin']*100:.1f}%")
    print(f"   ROE: {fund['roe']*100:.1f}% | D/E: {fund['debt_to_equity']:.2f}")
    if fund['price_to_sales']:
        print(f"   P/S: {fund['price_to_sales']:.1f}", end="")
    if fund['price_to_earnings']:
        print(f" | P/E: {fund['price_to_earnings']:.1f}", end="")
    print()
    
    # SQGLP Analysis
    print(f"\n🧠 SQGLP ANALYSIS ({sqglp['grade']})")
    print(f"   Total: {sqglp['total_score']}/26 | Pillars Passed: {sqglp['pillars_passed']}/5")
    print(f"   ┌─ S (Small):    {'✅' if sqglp['small']['pass'] else '❌'} +{sqglp['small']['score']} | {sqglp['small']['detail']}")
    print(f"   ├─ Q (Quality):  {'✅' if sqglp['quality']['pass'] else '❌'} +{sqglp['quality']['score']} | {sqglp['quality']['detail']}")
    print(f"   ├─ G (Growth):   {'✅' if sqglp['growth']['pass'] else '❌'} +{sqglp['growth']['score']} | {sqglp['growth']['detail']}")
    print(f"   ├─ L (Longevity):{'✅' if sqglp['longevity']['pass'] else '❌'} +{sqglp['longevity']['score']} | {sqglp['longevity']['detail']}")
    print(f"   └─ P (Price):    {'✅' if sqglp['price']['pass'] else '❌'} +{sqglp['price']['score']} | {sqglp['price']['detail']}")
    
    # Trade Setup
    print(f"\n📈 TRADE SETUP ($10K Position)")
    print(f"   Entry:  ${pos['entry']:.2f}")
    print(f"   Stop:   ${pos['stop_loss']:.2f} (-{pos['stop_percent']:.1f}%)")
    print(f"   Target: ${pos['target']:.2f} (+{((pos['target']-pos['entry'])/pos['entry'])*100:.1f}%)")
    print(f"   R:R Ratio: {pos['reward_risk']:.1f}:1")
    
    print(f"\n💰 POSITION DETAILS")
    print(f"   Shares: {pos['shares']:,} @ ${pos['entry']:.2f}")
    print(f"   Position Value: ${pos['position_value']:,.0f}")
    print(f"   Max Loss: ${pos['max_loss']:.0f} ({pos['max_loss_percent']:.1f}%)")
    print(f"   Potential Gain: ${pos['potential_gain']:.0f}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    start_time = time.time()
    
    print("="*70)
    print("🎯 SWING TRADING SCANNER + SQGLP FRAMEWORK")
    print("   $10K Position Sizing | Multi-Factor | Key Levels | Long-Term Flagging")
    print("="*70)
    
    market_status, regime_score, vol_percentile = check_market_health(CONFIG['market_health_ticker'])
    
    try:
        with open(CONFIG['ticker_file'], 'rb') as f:
            all_tickers = pickle.load(f)
        print(f"✅ Loaded {len(all_tickers)} tickers\n")
    except Exception as e:
        print(f"❌ Error loading tickers: {e}")
        return
    
    final_results = []
    batches = [all_tickers[i:i + CONFIG['batch_size']] 
               for i in range(0, len(all_tickers), CONFIG['batch_size'])]
    
    print(f"🚀 Scanning {len(all_tickers)} tickers in {len(batches)} batches\n")
    
    for i, batch_tickers in enumerate(batches):
        print(f"--- Batch {i+1}/{len(batches)} ({len(batch_tickers)} tickers) ---")
        
        try:
            if i > 0:
                time.sleep(1)
            
            hist_data = yf.download(
                batch_tickers,
                period="2y",
                auto_adjust=True,
                progress=False,
                group_by='ticker',
                threads=True
            )
            
            cleaned_data, valid_tickers = validate_and_clean_data(hist_data, batch_tickers)
            
            if cleaned_data is None or not valid_tickers:
                print(f"⚠️ No valid data. Skipping.\n")
                continue
            
            print(f"   ✓ {len(valid_tickers)} valid tickers")
            
            tech_results, key_levels_dict = run_technical_analysis(cleaned_data)
            promising = tech_results.index.tolist()
            print(f"   ✓ {len(promising)} passed technical screening")
            print(f"   ✓ {len(key_levels_dict)} with key levels calculated")
            
            if not promising:
                print()
                continue
            
            fund_data = fetch_fundamentals(promising)
            with_real_data = sum(1 for t, d in fund_data.items() if d.get('market_cap', 0) > 0)
            print(f"   ✓ {len(fund_data)}/{len(promising)} fetched ({with_real_data} with market cap)")
            
            batch_results = assemble_results(tech_results, fund_data, key_levels_dict, regime_score)
            final_results.extend(batch_results)
            print(f"   ✓ {len(batch_results)} added to results\n")
            
        except Exception as e:
            print(f"❌ Batch error: {e}\n")
            continue
    
    if not final_results:
        print("\n" + "="*70)
        print("❌ No qualifying stocks found today.")
        print("="*70)
        return
    
    final_results.sort(key=lambda x: x['conviction_score'], reverse=True)
    
    compounders = [r for r in final_results if r['sqglp']['is_compounder']]
    
    print("\n" + "="*70)
    print(f"🏆 TOP {min(CONFIG['max_results'], len(final_results))} SWING TRADE OPPORTUNITIES")
    print("="*70)
    
    for idx, res in enumerate(final_results[:CONFIG['max_results']], 1):
        print_detailed_analysis(res, idx)
    
    if compounders:
        print("\n" + "="*70)
        print("⭐ LONG-TERM COMPOUNDER CANDIDATES (SQGLP Qualified)")
        print("="*70)
        
        compounders.sort(key=lambda x: x['sqglp']['total_score'], reverse=True)
        
        for res in compounders[:10]:
            sqglp = res['sqglp']
            fund = res['fundamentals']
            print(f"\n   {res['ticker']} - {res['name']}")
            print(f"   └─ SQGLP: {sqglp['total_score']}/26 ({sqglp['grade']})")
            print(f"      Market Cap: ${fund['market_cap']/1e9:.1f}B | Rev Growth: {fund['revenue_growth']*100:+.0f}%")
            print(f"      ROE: {fund['roe']*100:.0f}% | Margin: {fund['profit_margin']*100:.0f}% | Beta: {fund['beta']:.2f}")
    
    # Portfolio summary
    print("\n" + "="*70)
    print("📋 PORTFOLIO SUMMARY")
    print("="*70)
    
    top_n = final_results[:CONFIG['max_results']]
    total_value = sum(r['position']['position_value'] for r in top_n)
    total_risk = sum(r['position']['max_loss'] for r in top_n)
    total_potential = sum(r['position']['potential_gain'] for r in top_n)
    
    print(f"\nTop {len(top_n)} Opportunities:")
    print(f"  • Total Position Value: ${total_value:,.0f}")
    print(f"  • Total Max Risk: ${total_risk:,.0f}")
    print(f"  • Total Potential Gain: ${total_potential:,.0f}")
    print(f"  • Aggregate R:R: 1:{(total_potential/total_risk):.2f}" if total_risk > 0 else "  • Aggregate R:R: N/A")
    print(f"  • SQGLP Compounders: {len(compounders)} identified")
    
    strong_fund = sum(1 for r in top_n if r['fund_quality']['score'] >= 5)
    good_fund = sum(1 for r in top_n if 3 <= r['fund_quality']['score'] < 5)
    weak_fund = sum(1 for r in top_n if r['fund_quality']['score'] < 3)
    
    print(f"\n  Fundamental Quality Distribution:")
    print(f"    • Strong (5-6/6): {strong_fund} stocks")
    print(f"    • Good (3-4/6): {good_fund} stocks")
    print(f"    • Weak/Unknown (0-2/6): {weak_fund} stocks")
    
    sectors = {}
    for r in top_n:
        s = r['fundamentals']['sector']
        sectors[s] = sectors.get(s, 0) + 1
    
    print(f"\n  Sector Distribution:")
    for sector, count in sorted(sectors.items(), key=lambda x: x[1], reverse=True):
        print(f"    • {sector}: {count}")
    
    print("\n" + "="*70)
    print(f"✅ Analysis Complete!")
    print(f"   • Market Regime: {market_status}")
    print(f"   • Total Qualifying: {len(final_results)}")
    print(f"   • High Conviction (70+): {sum(1 for r in final_results if r['conviction_score'] >= 70)}")
    print(f"   • SQGLP Compounders: {len(compounders)}")
    print(f"   • Execution Time: {time.time() - start_time:.1f}s")
    print("="*70 + "\n")
    
    if regime_score < 50:
        print("⚠️  WARNING: Weak market regime. Consider smaller positions or staying cash.")
    if vol_percentile > 80:
        print("⚠️  WARNING: High volatility. Widen stops or reduce exposure.")

if __name__ == "__main__":
    main()