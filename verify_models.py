"""
Comprehensive Model Verification Script
Tests if models are actually broken or just correctly predicting HOLD
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import yaml
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def connect_mt5(config):
    if not mt5.initialize(
        path=config.get('mt5_path'),
        login=config['mt5_login'],
        password=config['mt5_password'],
        server=config['mt5_server']
    ):
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return False
    print(f"✅ Connected to MT5: {config['mt5_server']}")
    return True

def load_model(symbol):
    model_path = Path('ml_models_simple') / f"{symbol}_ensemble.pkl"
    scaler_path = Path('ml_models_simple') / f"{symbol}_scaler.pkl"
    
    if not model_path.exists():
        return None, None
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model_data, scaler

def calculate_indicators(df):
    """Same as trading bot"""
    # Price changes
    df['price_change'] = df['close'].pct_change()
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    
    # Moving averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
    
    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # RSI
    for period in [7, 14, 21]:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    for period in [20, 50]:
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        df[f'bb_upper_{period}'] = sma + (2 * std)
        df[f'bb_lower_{period}'] = sma - (2 * std)
        df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
    
    # Bollinger Band position
    bb_range = df['bb_upper_20'] - df['bb_lower_20']
    df['bb_position'] = (df['close'] - df['bb_lower_20']) / (bb_range + 1e-10)
    
    # ATR
    for period in [7, 14, 21]:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df[f'atr_{period}'] = true_range.rolling(period).mean()
    
    # Stochastic
    for period in [14, 21]:
        low_min = df['low'].rolling(period).min()
        high_max = df['high'].rolling(period).max()
        df[f'stoch_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min)
    
    # Volume indicators
    df['volume_sma_20'] = df['tick_volume'].rolling(20).mean()
    df['volume_ratio'] = df['tick_volume'] / df['volume_sma_20']
    
    # Momentum
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = df['close'].pct_change(period)
    
    # Rate of Change
    for period in [5, 10, 20]:
        df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
    
    # Williams %R
    for period in [14, 21]:
        high_max = df['high'].rolling(period).max()
        low_min = df['low'].rolling(period).min()
        df[f'williams_r_{period}'] = -100 * (high_max - df['close']) / (high_max - low_min)
    
    # CCI
    for period in [14, 20]:
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        df[f'cci_{period}'] = (tp - sma_tp) / (0.015 * mad)
    
    # ADX
    period = 14
    high_diff = df['high'].diff()
    low_diff = -df['low'].diff()
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    tr = df[f'atr_{period}'] * period
    plus_di = 100 * plus_dm.rolling(period).mean() / tr
    minus_di = 100 * minus_dm.rolling(period).mean() / tr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    df['adx'] = dx.rolling(period).mean()
    
    # Price patterns
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
    df['inside_bar'] = ((df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))).astype(int)
    df['outside_bar'] = ((df['high'] > df['high'].shift(1)) & (df['low'] < df['low'].shift(1))).astype(int)
    
    df = df.dropna()
    return df

def test_model_on_data(symbol, model_data, scaler, df):
    """Test model predictions on data"""
    df_features = calculate_indicators(df)
    
    if len(df_features) == 0:
        return None
    
    # Get feature columns
    feature_cols = model_data['feature_columns']
    rf_model = model_data['rf']
    xgb_model = model_data.get('xgb')
    
    # Prepare features
    X = df_features[feature_cols].values
    X_scaled = scaler.transform(X)
    
    # Get predictions
    rf_pred_proba = rf_model.predict_proba(X_scaled)
    rf_pred = rf_model.predict(X_scaled)
    
    if xgb_model:
        xgb_pred_proba = xgb_model.predict_proba(X_scaled)
        pred_proba = (rf_pred_proba + xgb_pred_proba) / 2
    else:
        pred_proba = rf_pred_proba
    
    # Count predictions
    buy_count = (rf_pred == 1).sum()
    sell_count = (rf_pred == -1).sum()
    hold_count = (rf_pred == 0).sum()
    total = len(rf_pred)
    
    return {
        'buy_pct': buy_count / total * 100,
        'sell_pct': sell_count / total * 100,
        'hold_pct': hold_count / total * 100,
        'latest_pred': rf_pred[-1],
        'latest_proba': pred_proba[-1]
    }

def main():
    print("=" * 80)
    print("COMPREHENSIVE MODEL VERIFICATION")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    config = load_config()
    
    if not connect_mt5(config):
        return
    
    symbols = config.get('symbols', [])[:3]  # Test first 3 symbols
    
    print(f"\n{'='*80}")
    print(f"Testing {len(symbols)} symbols...")
    print(f"{'='*80}\n")
    
    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"SYMBOL: {symbol}")
        print(f"{'='*80}")
        
        # Load model
        model_data, scaler = load_model(symbol)
        if model_data is None:
            print(f"❌ Model not found")
            continue
        
        print(f"✅ Model loaded")
        
        # Get recent data (last 2000 bars)
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M30, 0, 2000)
        if rates is None:
            print(f"❌ No data available")
            continue
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        print(f"✅ Retrieved {len(df)} bars")
        
        # Test on recent data
        result = test_model_on_data(symbol, model_data, scaler, df)
        
        if result:
            print(f"\n📊 Prediction Distribution (last 2000 bars):")
            print(f"   BUY:  {result['buy_pct']:.1f}%")
            print(f"   SELL: {result['sell_pct']:.1f}%")
            print(f"   HOLD: {result['hold_pct']:.1f}%")
            
            print(f"\n📍 Latest Prediction:")
            pred_map = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
            print(f"   Signal: {pred_map[result['latest_pred']]}")
            print(f"   Probabilities: {result['latest_proba']}")
            
            # Analyze if HOLD is too dominant
            if result['hold_pct'] > 80:
                print(f"\n⚠️  WARNING: Model predicts HOLD {result['hold_pct']:.1f}% of the time!")
                print(f"   This suggests:")
                print(f"   1. Training data had too many HOLD labels (class imbalance)")
                print(f"   2. Thresholds (0.5-1.0%) are too high for current market")
                print(f"   3. Market is genuinely ranging (low volatility)")
    
    print(f"\n{'='*80}")
    print("VERIFICATION COMPLETE")
    print(f"{'='*80}\n")
    
    mt5.shutdown()

if __name__ == "__main__":
    main()

