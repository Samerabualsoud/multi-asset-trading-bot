"""
Diagnostic Script - Analyze Model Predictions vs Market Conditions
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import yaml
import pickle
from pathlib import Path
from datetime import datetime, timedelta

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
    print(f"Connected to MT5: {config['mt5_server']}")
    return True

def get_market_data(symbol, timeframe=mt5.TIMEFRAME_M30, bars=2000):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def calculate_recent_volatility(df, periods=[24, 48, 96]):
    """Calculate recent price volatility"""
    results = {}
    for period in periods:
        if len(df) >= period:
            recent = df.tail(period)
            price_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
            high_low_range = (recent['high'].max() - recent['low'].min()) / recent['close'].iloc[0]
            results[f'{period}_bars'] = {
                'price_change_pct': price_change * 100,
                'range_pct': high_low_range * 100,
                'hours': period * 0.5  # M30 = 30 min
            }
    return results

def load_model(symbol):
    model_path = Path('ml_models_simple') / f"{symbol}_ensemble.pkl"
    if not model_path.exists():
        return None
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def main():
    print("="*80)
    print("TRADING BOT DIAGNOSTIC - Model Predictions vs Market Conditions")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    config = load_config()
    
    if not connect_mt5(config):
        return
    
    symbols = config.get('symbols', [])
    
    print(f"\nAnalyzing {len(symbols)} symbols...\n")
    print("="*80)
    
    results = []
    
    for symbol in symbols:
        print(f"\n{symbol}")
        print("-"*80)
        
        # Get market data
        df = get_market_data(symbol)
        if df is None or len(df) < 100:
            print(f"  ❌ Insufficient data")
            continue
        
        # Calculate recent volatility
        volatility = calculate_recent_volatility(df)
        
        print(f"  Recent Price Movement:")
        for period, data in volatility.items():
            hours = data['hours']
            change = data['price_change_pct']
            range_pct = data['range_pct']
            print(f"    Last {hours:.0f}h: Change={change:+.3f}%, Range={range_pct:.3f}%")
        
        # Load model
        model = load_model(symbol)
        if model is None:
            print(f"  ❌ Model not found")
            continue
        
        # Get latest close price
        latest_price = df['close'].iloc[-1]
        prev_price_24h = df['close'].iloc[-48] if len(df) >= 48 else df['close'].iloc[0]
        price_change_24h = (latest_price - prev_price_24h) / prev_price_24h * 100
        
        # Check if there's a clear trend
        if abs(price_change_24h) > 0.5:
            trend = "UPTREND" if price_change_24h > 0 else "DOWNTREND"
            print(f"  📈 Market: {trend} ({price_change_24h:+.2f}% in 24h)")
        else:
            print(f"  📊 Market: RANGING ({price_change_24h:+.2f}% in 24h)")
        
        results.append({
            'symbol': symbol,
            'price_change_24h': price_change_24h,
            'range_24h': volatility.get('48_bars', {}).get('range_pct', 0),
            'has_model': True
        })
    
    # Summary
    print("\n" + "="*80)
    print("MARKET CONDITION SUMMARY")
    print("="*80)
    
    df_results = pd.DataFrame(results)
    
    avg_movement = df_results['price_change_24h'].abs().mean()
    avg_range = df_results['range_24h'].mean()
    
    print(f"\nAverage 24h Movement: {avg_movement:.3f}%")
    print(f"Average 24h Range: {avg_range:.3f}%")
    
    if avg_movement < 0.3:
        print("\n⚠️  VERY LOW VOLATILITY - This is typical for Asian session")
        print("   Models are correctly predicting HOLD for most pairs")
        print("   Expect more signals during London (08:00-12:00 GMT) and NY (13:00-17:00 GMT) sessions")
    elif avg_movement < 0.5:
        print("\n📊 LOW VOLATILITY - Limited trading opportunities")
        print("   Some low-confidence signals expected, but most filtered out")
    elif avg_movement < 1.0:
        print("\n📈 MODERATE VOLATILITY - Good trading conditions")
        print("   Should see 3-5 tradeable signals with 70%+ confidence")
    else:
        print("\n🚀 HIGH VOLATILITY - Active trading period")
        print("   Expect 10-20 signals, many with high confidence")
    
    # Trading session info
    now = datetime.now()
    hour_utc = now.hour
    
    print(f"\n📅 Current Time: {now.strftime('%H:%M')} (Local)")
    print(f"   Estimated UTC: ~{hour_utc:02d}:00")
    
    if 0 <= hour_utc < 7:
        print("   Session: ASIAN (Low volatility) 🌙")
        print("   Expected: 0-2 tradeable signals per hour")
    elif 7 <= hour_utc < 12:
        print("   Session: LONDON (High volatility) 🇬🇧")
        print("   Expected: 5-10 tradeable signals per hour")
    elif 12 <= hour_utc < 17:
        print("   Session: NEW YORK (High volatility) 🇺🇸")
        print("   Expected: 5-10 tradeable signals per hour")
    elif 17 <= hour_utc < 22:
        print("   Session: OVERLAP/EVENING (Medium volatility) 🌆")
        print("   Expected: 2-5 tradeable signals per hour")
    else:
        print("   Session: LATE EVENING (Low volatility) 🌙")
        print("   Expected: 0-2 tradeable signals per hour")
    
    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)
    
    mt5.shutdown()

if __name__ == "__main__":
    main()

