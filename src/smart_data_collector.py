#!/usr/bin/env python3
"""
Smart Data Collector
Automatically detects how much historical data is available and collects the maximum possible
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_config():
    """Load configuration"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def initialize_mt5(config):
    """Initialize MT5 connection"""
    if not mt5.initialize(
        path=config['mt5_path'],
        login=config['mt5_login'],
        password=config['mt5_password'],
        server=config['mt5_server']
    ):
        logger.error(f"MT5 initialization failed: {mt5.last_error()}")
        return False
    
    logger.info(f"✅ Connected to MT5: {config['mt5_server']}")
    return True


def detect_available_data(symbol, max_years=10):
    """
    Detect how much historical data is actually available
    
    Args:
        symbol: Trading symbol
        max_years: Maximum years to try
        
    Returns:
        (start_date, end_date, years_available)
    """
    logger.info(f"\n🔍 Detecting available data for {symbol}...")
    
    end_date = datetime.now()
    
    # Try different time ranges
    for years in [max_years, 7, 5, 3, 2, 1]:
        start_date = end_date - timedelta(days=years*365)
        
        logger.info(f"Trying {years} years: {start_date.date()} to {end_date.date()}")
        
        # Try to fetch data
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start_date, end_date)
        
        if rates is not None and len(rates) > 0:
            actual_start = datetime.fromtimestamp(rates[0]['time'])
            actual_end = datetime.fromtimestamp(rates[-1]['time'])
            actual_years = (actual_end - actual_start).days / 365
            
            logger.info(f"✅ Found {len(rates)} bars ({actual_years:.1f} years)")
            logger.info(f"   Actual range: {actual_start.date()} to {actual_end.date()}")
            
            return actual_start, actual_end, actual_years
    
    logger.error(f"❌ No data available for {symbol}")
    return None, None, 0


def collect_maximum_data(symbol):
    """
    Collect maximum available historical data for a symbol
    
    Args:
        symbol: Trading symbol
        
    Returns:
        DataFrame with all available data
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"COLLECTING MAXIMUM DATA FOR {symbol}")
    logger.info(f"{'='*80}")
    
    # Detect available data
    start_date, end_date, years = detect_available_data(symbol, max_years=10)
    
    if start_date is None:
        return None
    
    # Collect all available data
    logger.info(f"\n📊 Collecting {years:.1f} years of data...")
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start_date, end_date)
    
    if rates is None or len(rates) == 0:
        logger.error(f"Failed to collect data for {symbol}")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    logger.info(f"✅ Collected {len(df)} bars ({years:.1f} years)")
    logger.info(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
    
    return df


def main():
    """Main function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    config = load_config()
    
    # Initialize MT5
    if not initialize_mt5(config):
        logger.error("Failed to initialize MT5")
        return
    
    # Symbols
    symbols = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD',
        'EURJPY', 'GBPJPY', 'AUDJPY',
        'BTCUSD', 'ETHUSD',
        'XAUUSD', 'XAGUSD',
        'USOIL', 'UKOIL'
    ]
    
    logger.info("\n" + "="*80)
    logger.info("SMART DATA COLLECTOR")
    logger.info("="*80)
    logger.info("Automatically detects and collects maximum available historical data")
    logger.info("="*80)
    
    results = {}
    
    for symbol in symbols:
        try:
            df = collect_maximum_data(symbol)
            if df is not None:
                results[symbol] = df
                
                # Save raw data
                output_dir = Path('ml_data_raw')
                output_dir.mkdir(exist_ok=True)
                output_file = output_dir / f'{symbol}_raw.csv'
                df.to_csv(output_file)
                logger.info(f"💾 Saved: {output_file}")
                
        except Exception as e:
            logger.error(f"Failed to collect {symbol}: {e}", exc_info=True)
            continue
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("COLLECTION SUMMARY")
    logger.info("="*80)
    
    if results:
        logger.info(f"Successfully collected: {len(results)} symbols\n")
        
        logger.info(f"{'Symbol':<10} {'Bars':<10} {'Years':<10} {'Start Date':<15} {'End Date'}")
        logger.info("-" * 80)
        
        for symbol, df in results.items():
            years = (df.index[-1] - df.index[0]).days / 365
            logger.info(f"{symbol:<10} {len(df):<10} {years:<10.1f} {df.index[0].date()!s:<15} {df.index[-1].date()}")
        
        avg_years = sum((df.index[-1] - df.index[0]).days / 365 for df in results.values()) / len(results)
        logger.info("-" * 80)
        logger.info(f"Average: {avg_years:.1f} years of data")
        
        if avg_years >= 5:
            logger.info("\n🎯 EXCELLENT! 5+ years of data available")
        elif avg_years >= 3:
            logger.info("\n✅ GOOD! 3+ years of data available")
        elif avg_years >= 2:
            logger.info("\n⚠️ FAIR. Only 2+ years available (may need more)")
        else:
            logger.info("\n❌ WARNING! Less than 2 years available (not ideal)")
        
        logger.info(f"\nRaw data saved to: ml_data_raw/")
        logger.info("Next step: Run ml_data_collector.py to add indicators")
        
    else:
        logger.error("❌ No data collected!")
    
    logger.info("="*80)
    
    mt5.shutdown()


if __name__ == "__main__":
    main()

