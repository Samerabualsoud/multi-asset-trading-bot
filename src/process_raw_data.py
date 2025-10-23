#!/usr/bin/env python3
"""
Process Raw Data
Reads raw price data and adds all technical indicators
Uses the data already collected by smart_data_collector.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class RawDataProcessor:
    """Process raw price data and add technical indicators"""
    
    def __init__(self, input_dir='ml_data_raw', output_dir='ml_data'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def calculate_indicators(self, df):
        """Calculate all technical indicators"""
        logger.info("Calculating technical indicators...")
        
        # Make a copy
        data = df.copy()
        
        # Basic price features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['hl_ratio'] = (data['high'] - data['low']) / data['close']
        data['co_ratio'] = (data['close'] - data['open']) / data['open']
        
        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            data[f'sma_{period}'] = data['close'].rolling(period).mean()
            data[f'ema_{period}'] = data['close'].ewm(span=period).mean()
        
        # MA relationships
        data['sma_20_50_ratio'] = data['sma_20'] / data['sma_50']
        data['price_sma_20_ratio'] = data['close'] / data['sma_20']
        data['price_sma_50_ratio'] = data['close'] / data['sma_50']
        
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(20).mean()
        data['bb_std'] = data['close'].rolling(20).std()
        data['bb_upper'] = data['bb_middle'] + (data['bb_std'] * 2)
        data['bb_lower'] = data['bb_middle'] - (data['bb_std'] * 2)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        data['rsi_sma'] = data['rsi'].rolling(14).mean()
        
        # MACD
        data['ema_12'] = data['close'].ewm(span=12).mean()
        data['ema_26'] = data['close'].ewm(span=26).mean()
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_diff'] = data['macd'] - data['macd_signal']
        
        # Stochastic
        low_14 = data['low'].rolling(14).min()
        high_14 = data['high'].rolling(14).max()
        data['stoch_k'] = 100 * (data['close'] - low_14) / (high_14 - low_14)
        data['stoch_d'] = data['stoch_k'].rolling(3).mean()
        
        # ATR
        data['tr1'] = data['high'] - data['low']
        data['tr2'] = abs(data['high'] - data['close'].shift())
        data['tr3'] = abs(data['low'] - data['close'].shift())
        data['tr'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
        data['atr'] = data['tr'].rolling(14).mean()
        data['atr_percent'] = data['atr'] / data['close']
        
        # ADX
        data['plus_dm'] = data['high'].diff()
        data['minus_dm'] = -data['low'].diff()
        data['plus_dm'] = data['plus_dm'].where((data['plus_dm'] > data['minus_dm']) & (data['plus_dm'] > 0), 0)
        data['minus_dm'] = data['minus_dm'].where((data['minus_dm'] > data['plus_dm']) & (data['minus_dm'] > 0), 0)
        data['plus_di'] = 100 * (data['plus_dm'].rolling(14).mean() / data['atr'])
        data['minus_di'] = 100 * (data['minus_dm'].rolling(14).mean() / data['atr'])
        data['dx'] = 100 * abs(data['plus_di'] - data['minus_di']) / (data['plus_di'] + data['minus_di'])
        data['adx'] = data['dx'].rolling(14).mean()
        
        # Volume features
        data['volume_sma'] = data['tick_volume'].rolling(20).mean()
        data['volume_ratio'] = data['tick_volume'] / data['volume_sma']
        data['volume_price_trend'] = (data['close'].diff() / data['close'].shift()) * data['tick_volume']
        
        # Momentum
        for period in [5, 10, 20]:
            data[f'momentum_{period}'] = data['close'].pct_change(period)
            data[f'roc_{period}'] = ((data['close'] - data['close'].shift(period)) / data['close'].shift(period)) * 100
        
        # Volatility
        for period in [10, 20, 50]:
            data[f'volatility_{period}'] = data['returns'].rolling(period).std()
        
        # Price channels
        data['high_20'] = data['high'].rolling(20).max()
        data['low_20'] = data['low'].rolling(20).min()
        data['channel_position'] = (data['close'] - data['low_20']) / (data['high_20'] - data['low_20'])
        
        # Fibonacci levels (from recent high/low)
        data['fib_high'] = data['high'].rolling(100).max()
        data['fib_low'] = data['low'].rolling(100).min()
        data['fib_range'] = data['fib_high'] - data['fib_low']
        data['fib_0'] = data['fib_high']
        data['fib_236'] = data['fib_high'] - (data['fib_range'] * 0.236)
        data['fib_382'] = data['fib_high'] - (data['fib_range'] * 0.382)
        data['fib_500'] = data['fib_high'] - (data['fib_range'] * 0.500)
        data['fib_618'] = data['fib_high'] - (data['fib_range'] * 0.618)
        data['fib_1000'] = data['fib_low']
        
        # Time features
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        data['day_of_month'] = data.index.day
        data['month'] = data.index.month
        
        # Session indicators
        data['asian_session'] = ((data['hour'] >= 0) & (data['hour'] < 8)).astype(int)
        data['london_session'] = ((data['hour'] >= 8) & (data['hour'] < 16)).astype(int)
        data['ny_session'] = ((data['hour'] >= 13) & (data['hour'] < 21)).astype(int)
        data['overlap_session'] = ((data['hour'] >= 13) & (data['hour'] < 16)).astype(int)
        
        # Trend indicators
        data['trend_20'] = np.where(data['close'] > data['sma_20'], 1, -1)
        data['trend_50'] = np.where(data['close'] > data['sma_50'], 1, -1)
        data['trend_strength'] = abs(data['close'] - data['sma_50']) / data['sma_50']
        
        # Generate labels (BUY=1, SELL=-1, HOLD=0)
        # Look ahead 24 hours (1 day) to determine if price goes up or down
        future_return = data['close'].shift(-24).pct_change(24)
        threshold = 0.002  # 0.2% threshold
        
        data['label'] = 0  # Default: HOLD
        data.loc[future_return > threshold, 'label'] = 1  # BUY
        data.loc[future_return < -threshold, 'label'] = -1  # SELL
        
        # Drop intermediate columns
        cols_to_drop = ['tr1', 'tr2', 'tr3', 'tr', 'plus_dm', 'minus_dm', 'dx',
                       'ema_12', 'ema_26', 'bb_std']
        data = data.drop(columns=[c for c in cols_to_drop if c in data.columns])
        
        # Drop NaN rows
        data = data.dropna()
        
        logger.info(f"✅ Calculated {len(data.columns)} features")
        logger.info(f"   Rows after cleaning: {len(data)}")
        
        return data
    
    def process_symbol(self, symbol):
        """Process a single symbol"""
        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESSING {symbol}")
        logger.info(f"{'='*80}")
        
        # Load raw data
        input_file = self.input_dir / f'{symbol}_raw.csv'
        
        if not input_file.exists():
            logger.error(f"Raw data not found: {input_file}")
            return None
        
        logger.info(f"Loading raw data from {input_file}")
        df = pd.read_csv(input_file, index_col=0, parse_dates=True)
        
        logger.info(f"Raw data: {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
        
        # Calculate indicators
        processed = self.calculate_indicators(df)
        
        # Save
        output_file = self.output_dir / f'{symbol}_dataset.csv'
        processed.to_csv(output_file)
        logger.info(f"✅ Saved: {output_file}")
        
        return processed
    
    def process_all(self):
        """Process all symbols"""
        logger.info("\n" + "="*80)
        logger.info("RAW DATA PROCESSOR")
        logger.info("="*80)
        logger.info(f"Input: {self.input_dir}/")
        logger.info(f"Output: {self.output_dir}/")
        logger.info("="*80)
        
        # Find all raw data files
        raw_files = list(self.input_dir.glob('*_raw.csv'))
        
        if not raw_files:
            logger.error(f"No raw data files found in {self.input_dir}/")
            logger.error("Run smart_data_collector.py first!")
            return
        
        logger.info(f"Found {len(raw_files)} raw data files")
        
        results = {}
        
        for raw_file in raw_files:
            symbol = raw_file.stem.replace('_raw', '')
            
            try:
                df = self.process_symbol(symbol)
                if df is not None:
                    results[symbol] = df
            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}", exc_info=True)
                continue
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("PROCESSING COMPLETE")
        logger.info("="*80)
        logger.info(f"Successfully processed: {len(results)} symbols")
        logger.info(f"Output directory: {self.output_dir}/")
        logger.info("="*80)
        
        if results:
            logger.info(f"\n{'Symbol':<10} {'Rows':<10} {'Features':<10} {'Years':<10}")
            logger.info("-" * 80)
            
            for symbol, df in results.items():
                years = (df.index[-1] - df.index[0]).days / 365
                logger.info(f"{symbol:<10} {len(df):<10} {len(df.columns):<10} {years:<10.1f}")
            
            logger.info("\n✅ Ready for ML training!")
            logger.info("Next step: python src/ml_model_trainer_v2.py")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    processor = RawDataProcessor()
    processor.process_all()

