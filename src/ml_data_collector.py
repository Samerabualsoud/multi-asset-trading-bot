#!/usr/bin/env python3
"""
ML Data Collector for Forex Trading
Collects historical data and prepares it for machine learning
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class MLDataCollector:
    """Collect and prepare data for ML training"""
    
    def __init__(self, symbols=['EURUSD', 'GBPUSD', 'USDJPY'], timeframe=mt5.TIMEFRAME_H1):
        """
        Initialize data collector
        
        Args:
            symbols: List of symbols to collect data for
            timeframe: MT5 timeframe (default H1 for better signal/noise ratio)
        """
        self.symbols = symbols
        self.timeframe = timeframe
        self.data_dir = Path('ml_data')
        self.data_dir.mkdir(exist_ok=True)
        
    def connect_mt5(self):
        """Connect to MT5"""
        if not mt5.initialize():
            logger.error("MT5 initialization failed")
            return False
        return True
    
    def collect_historical_data(self, symbol, years=3):
        """
        Collect historical data for a symbol
        
        Args:
            symbol: Symbol to collect data for
            years: Number of years of historical data
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Collecting {years} years of data for {symbol}...")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        # Get data from MT5
        rates = mt5.copy_rates_range(symbol, self.timeframe, start_date, end_date)
        
        if rates is None or len(rates) == 0:
            logger.error(f"Failed to get data for {symbol}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        logger.info(f"Collected {len(df)} candles for {symbol}")
        
        return df
    
    def add_technical_indicators(self, df):
        """
        Add technical indicators as features
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        logger.info("Adding technical indicators...")
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        for period in [10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # Price relative to MAs
        df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['price_vs_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
        df['price_vs_ema20'] = (df['close'] - df['ema_20']) / df['ema_20']
        
        # MA crossovers
        df['sma_20_50_cross'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
        df['ema_10_20_cross'] = np.where(df['ema_10'] > df['ema_20'], 1, -1)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)
        
        # Bollinger Bands
        for period in [20]:
            sma = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            df[f'bb_upper_{period}'] = sma + (2 * std)
            df[f'bb_lower_{period}'] = sma - (2 * std)
            df[f'bb_middle_{period}'] = sma
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / df[f'bb_middle_{period}']
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        df['atr_percent'] = df['atr'] / df['close']
        
        # ADX (Average Directional Index)
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = true_range
        atr = tr.rolling(14).mean()
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(14).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        # Stochastic Oscillator
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Volume-based features (if available)
        if 'tick_volume' in df.columns:
            df['volume_sma'] = df['tick_volume'].rolling(20).mean()
            df['volume_ratio'] = df['tick_volume'] / df['volume_sma']
        
        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'].pct_change(period)
        
        # Volatility
        for period in [10, 20, 30]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std()
        
        # Price patterns
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)
        df['lower_high'] = (df['high'] < df['high'].shift(1)).astype(int)
        
        # Candle patterns
        df['body'] = df['close'] - df['open']
        df['body_percent'] = df['body'] / df['open']
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        
        logger.info(f"Added {len(df.columns)} features")
        
        return df
    
    def add_time_features(self, df):
        """Add time-based features"""
        logger.info("Adding time features...")
        
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        
        # Trading sessions
        df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['newyork_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
        df['overlap_session'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)
        
        # Good morning session (8:30-11:30 AM Saudi = ~5:30-8:30 UTC)
        df['good_morning_session'] = ((df['hour'] >= 5) & (df['hour'] < 9)).astype(int)
        
        return df
    
    def create_labels(self, df, forward_periods=4, profit_threshold=0.002):
        """
        Create labels for supervised learning
        
        Args:
            df: DataFrame with features
            forward_periods: Number of periods to look forward
            profit_threshold: Minimum profit threshold (0.2% = 20 pips for most pairs)
            
        Returns:
            DataFrame with labels
        """
        logger.info(f"Creating labels (forward_periods={forward_periods}, threshold={profit_threshold})...")
        
        # Calculate forward returns
        df['forward_return'] = df['close'].shift(-forward_periods) / df['close'] - 1
        
        # Create classification labels
        # 1 = BUY (price will go up by threshold)
        # 0 = HOLD (price won't move significantly)
        # -1 = SELL (price will go down by threshold)
        
        conditions = [
            df['forward_return'] > profit_threshold,
            df['forward_return'] < -profit_threshold
        ]
        choices = [1, -1]
        df['label'] = np.select(conditions, choices, default=0)
        
        # Binary labels for simpler models
        df['label_binary_long'] = (df['forward_return'] > profit_threshold).astype(int)
        df['label_binary_short'] = (df['forward_return'] < -profit_threshold).astype(int)
        
        # Regression target (actual forward return)
        df['target_return'] = df['forward_return']
        
        logger.info(f"Label distribution:")
        logger.info(f"  BUY (1): {(df['label'] == 1).sum()} ({(df['label'] == 1).sum() / len(df) * 100:.1f}%)")
        logger.info(f"  HOLD (0): {(df['label'] == 0).sum()} ({(df['label'] == 0).sum() / len(df) * 100:.1f}%)")
        logger.info(f"  SELL (-1): {(df['label'] == -1).sum()} ({(df['label'] == -1).sum() / len(df) * 100:.1f}%)")
        
        return df
    
    def prepare_dataset(self, symbol, years=3):
        """
        Prepare complete dataset for a symbol
        
        Args:
            symbol: Symbol to prepare data for
            years: Number of years of historical data
            
        Returns:
            DataFrame with features and labels
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Preparing dataset for {symbol}")
        logger.info(f"{'='*80}")
        
        # Collect historical data
        df = self.collect_historical_data(symbol, years)
        if df is None:
            return None
        
        # Add features
        df = self.add_technical_indicators(df)
        df = self.add_time_features(df)
        
        # Create labels
        df = self.create_labels(df)
        
        # Drop NaN values
        initial_len = len(df)
        df = df.dropna()
        logger.info(f"Dropped {initial_len - len(df)} rows with NaN values")
        logger.info(f"Final dataset size: {len(df)} rows")
        
        # Save dataset
        output_file = self.data_dir / f'{symbol}_dataset.csv'
        df.to_csv(output_file)
        logger.info(f"Saved dataset to {output_file}")
        
        return df
    
    def collect_all_data(self, years=10):
        """Collect data for all symbols (10 years for better ML performance)"""
        logger.info("\n" + "="*100)
        logger.info("COLLECTING DATA FOR ALL SYMBOLS")
        logger.info("="*100)
        
        if not self.connect_mt5():
            logger.error("Failed to connect to MT5")
            return None
        
        datasets = {}
        
        for symbol in self.symbols:
            df = self.prepare_dataset(symbol, years)
            if df is not None:
                datasets[symbol] = df
        
        mt5.shutdown()
        
        logger.info(f"\n✅ Data collection complete for {len(datasets)} symbols")
        
        return datasets


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # All 15 symbols: major forex, crypto, metals, oil
    all_symbols = [
        # Major Forex
        'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD',
        'EURJPY', 'GBPJPY', 'AUDJPY',
        # Major Crypto
        'BTCUSD', 'ETHUSD',
        # Metals
        'XAUUSD',  # Gold
        'XAGUSD',  # Silver
        # Oil
        'USOIL',   # WTI Crude
        'UKOIL'    # Brent Crude
    ]
    
    collector = MLDataCollector(symbols=all_symbols)
    datasets = collector.collect_all_data(years=3)

