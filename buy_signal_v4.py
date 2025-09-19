import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import talib

def fetch_data(ticker, days=100):
    end_date = datetime.now() + timedelta(1)
    start_date = end_date - timedelta(days=days)
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    return df

def add_technical_indicators(df):
    df['SMA_10'] = talib.SMA(df['Close'], timeperiod=10)
    df['SMA_30'] = talib.SMA(df['Close'], timeperiod=30)
    df['RSI'] = talib.RSI(df['Close'])
    df['MACD'], df['MACD_Signal'], _ = talib.MACD(df['Close'])
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(df['Close'])
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'])
    df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['High'], df['Low'], df['Close'])
    df['OBV'] = talib.OBV(df['Close'], df['Volume'])
    df['ROC'] = talib.ROC(df['Close'])
    df['MOM'] = talib.MOM(df['Close'])
    df['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'])
    return df

def analyze_indicators(df):
    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    signals = {
        'bullish': 0,
        'bearish': 0,
        'neutral': 0
    }
    
    # RSI
    if 40 < current['RSI'] < 70 and current['RSI'] > prev['RSI']:
        signals['bullish'] += 1
    elif 30 < current['RSI'] < 60 and current['RSI'] < prev['RSI']:
        signals['bearish'] += 1
    else:
        signals['neutral'] += 1
    
    # MACD
    if current['MACD'] > current['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
        signals['bullish'] += 1
    elif current['MACD'] < current['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
        signals['bearish'] += 1
    else:
        signals['neutral'] += 1
    
    # Bollinger Bands
    if current['Close'] > current['BB_Upper']:
        signals['bullish'] += 1
    elif current['Close'] < current['BB_Lower']:
        signals['bearish'] += 1
    else:
        signals['neutral'] += 1
    
    # ADX
    if current['ADX'] > 25 and current['ADX'] > prev['ADX']:
        signals['bullish'] += 1
    elif current['ADX'] > 25 and current['ADX'] < prev['ADX']:
        signals['bearish'] += 1
    else:
        signals['neutral'] += 1
    
    # Stochastic Oscillator
    if current['STOCH_K'] < 80 and current['STOCH_K'] > current['STOCH_D'] and prev['STOCH_K'] <= prev['STOCH_D']:
        signals['bullish'] += 1
    elif current['STOCH_K'] > 20 and current['STOCH_K'] < current['STOCH_D'] and prev['STOCH_K'] >= prev['STOCH_D']:
        signals['bearish'] += 1
    else:
        signals['neutral'] += 1
    
    # Moving Averages
    if current['Close'] > current['SMA_10'] > current['SMA_30']:
        signals['bullish'] += 1
    elif current['Close'] < current['SMA_10'] < current['SMA_30']:
        signals['bearish'] += 1
    else:
        signals['neutral'] += 1
    
    # OBV
    if current['OBV'] > prev['OBV']:
        signals['bullish'] += 1
    elif current['OBV'] < prev['OBV']:
        signals['bearish'] += 1
    else:
        signals['neutral'] += 1
    
    # ROC and Momentum
    if current['ROC'] > 0 and current['MOM'] > prev['MOM']:
        signals['bullish'] += 1
    elif current['ROC'] < 0 and current['MOM'] < prev['MOM']:
        signals['bearish'] += 1
    else:
        signals['neutral'] += 1
    
    # Williams %R
    if current['WILLR'] > prev['WILLR'] and current['WILLR'] < -50:
        signals['bullish'] += 1
    elif current['WILLR'] < prev['WILLR'] and current['WILLR'] > -50:
        signals['bearish'] += 1
    else:
        signals['neutral'] += 1
    
    return signals

def generate_recommendation(signals):
    if signals['bullish'] >= 6:
        return "Buy"
    elif signals['bearish'] >= 6:
        return "Sell"
    else:
        return "Hold"

def main():
    import pickle
    pickle_file = 'data/tickers.pkl'
    with open(pickle_file, 'rb') as f:
        tickers = pickle.load(f)
    for ticker in tickers:
        print(f"\n--- Swing Trading Recommendation for {ticker} ---")
        
        try:
            df = fetch_data(ticker)
            df = add_technical_indicators(df)
            df.dropna(inplace=True)
            
            signals = analyze_indicators(df)
            recommendation = generate_recommendation(signals)
            
            print(f"Stock: {ticker}")
            print(f"Current Price: ${df['Close'].iloc[-1]:.2f}")
            print(f"Recommendation: {recommendation}")
            print(f"\nSignal Breakdown:")
            print(f"Bullish Signals: {signals['bullish']} - Indicators suggesting price may rise")
            print(f"Bearish Signals: {signals['bearish']} - Indicators suggesting price may fall")
            print(f"Neutral Signals: {signals['neutral']} - Indicators suggesting price may remain stable")
            
            print("\nDetailed Signal Analysis:")
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            print(f"RSI: {current['RSI']:.2f} ({'Bullish' if 40 < current['RSI'] < 70 and current['RSI'] > prev['RSI'] else 'Bearish' if 30 < current['RSI'] < 60 and current['RSI'] < prev['RSI'] else 'Neutral'})")
            print(f"MACD: {current['MACD']:.2f} vs Signal {current['MACD_Signal']:.2f} ({'Bullish' if current['MACD'] > current['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal'] else 'Bearish' if current['MACD'] < current['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal'] else 'Neutral'})")
            print(f"Bollinger Bands: Close {current['Close']:.2f} vs Upper {current['BB_Upper']:.2f}, Lower {current['BB_Lower']:.2f} ({'Bullish' if current['Close'] > current['BB_Upper'] else 'Bearish' if current['Close'] < current['BB_Lower'] else 'Neutral'})")
            print(f"ADX: {current['ADX']:.2f} ({'Bullish' if current['ADX'] > 25 and current['ADX'] > prev['ADX'] else 'Bearish' if current['ADX'] > 25 and current['ADX'] < prev['ADX'] else 'Neutral'})")
            print(f"Stochastic Oscillator: K {current['STOCH_K']:.2f} vs D {current['STOCH_D']:.2f} ({'Bullish' if current['STOCH_K'] < 80 and current['STOCH_K'] > current['STOCH_D'] and prev['STOCH_K'] <= prev['STOCH_D'] else 'Bearish' if current['STOCH_K'] > 20 and current['STOCH_K'] < current['STOCH_D'] and prev['STOCH_K'] >= prev['STOCH_D'] else 'Neutral'})")
            print(f"Moving Averages: Close {current['Close']:.2f} vs SMA10 {current['SMA_10']:.2f} vs SMA30 {current['SMA_30']:.2f} ({'Bullish' if current['Close'] > current['SMA_10'] > current['SMA_30'] else 'Bearish' if current['Close'] < current['SMA_10'] < current['SMA_30'] else 'Neutral'})")
            print(f"OBV: {current['OBV']:.2f} ({'Bullish' if current['OBV'] > prev['OBV'] else 'Bearish' if current['OBV'] < prev['OBV'] else 'Neutral'})")
            print(f"ROC: {current['ROC']:.2f}, Momentum: {current['MOM']:.2f} ({'Bullish' if current['ROC'] > 0 and current['MOM'] > prev['MOM'] else 'Bearish' if current['ROC'] < 0 and current['MOM'] < prev['MOM'] else 'Neutral'})")
            print(f"Williams %R: {current['WILLR']:.2f} ({'Bullish' if current['WILLR'] > prev['WILLR'] and current['WILLR'] < -50 else 'Bearish' if current['WILLR'] < prev['WILLR'] and current['WILLR'] > -50 else 'Neutral'})")
            
        except Exception as e:
            print(f"An error occurred while processing {ticker}: {str(e)}")
            print("Skipping to the next stock.")

if __name__ == "__main__":
    main()
