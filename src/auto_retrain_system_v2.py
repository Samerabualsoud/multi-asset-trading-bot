#!/usr/bin/env python3
"""
Auto-Retraining System V2
Automatically retrains ML models with auto-discovered symbols
- Discovers all tradeable symbols from MT5
- Trains models every 12 hours with fresh market data
- Supports dynamic symbol list
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import yaml
import logging
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))
from symbol_discovery_enhanced import EnhancedSymbolDiscovery

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_retrain_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoRetrainSystemV2:
    def __init__(self, config_path='config.yaml'):
        self.config = self.load_config(config_path)
        self.retrain_interval = 12 * 3600  # 12 hours in seconds
        self.running = True
        self.last_retrain = None
        self.discovery = EnhancedSymbolDiscovery(config_path)
        
    def load_config(self, config_path):
        """Load configuration"""
        config_file = Path(config_path)
        if not config_file.exists():
            config_file = Path('config') / config_path
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def connect_mt5(self):
        """Connect to MT5"""
        if not mt5.initialize(
            path=self.config.get('mt5_path'),
            login=self.config['mt5_login'],
            password=self.config['mt5_password'],
            server=self.config['mt5_server']
        ):
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        return True
    
    def get_symbols_to_train(self):
        """Get list of symbols to train (auto-discover or from config)"""
        # Check if auto-discovery is enabled
        auto_discover = self.config.get('auto_discover_symbols', False)
        discovery_mode = self.config.get('discovery_mode', 'balanced')
        
        if auto_discover:
            logger.info(f"Auto-discovering symbols in '{discovery_mode}' mode...")
            symbols = self.discovery.get_recommended_symbols(mode=discovery_mode)
            symbol_names = [s['name'] for s in symbols]
            logger.info(f"Discovered {len(symbol_names)} symbols")
            
            # Save to config for reference
            self.discovery.save_to_config(symbols)
            
            return symbol_names
        else:
            # Use symbols from config
            symbols = self.config.get('symbols', [])
            logger.info(f"Using {len(symbols)} symbols from config")
            return symbols
    
    def collect_fresh_data(self, symbol, days=None):
        """Collect fresh market data with smart M5/H1 hybrid approach"""
        logger.info(f"Collecting fresh data for {symbol}...")
        
        # Target: Get as much data as possible (up to 10 years)
        if days is None:
            days = 3650  # 10 years
        
        # First, try M5 data
        logger.info(f"Attempting M5 data collection for {symbol}...")
        df_m5 = self._collect_data_chunked(symbol, mt5.TIMEFRAME_M5, days, bars_per_day=288)
        
        if df_m5 is not None:
            m5_days = len(df_m5) / 288
            m5_years = m5_days / 365
            
            # If we got at least 2 years of M5 data, use it
            if m5_years >= 2.0:
                logger.info(f"✓ Using M5 data: {len(df_m5):,} bars ({m5_years:.1f} years)")
                return df_m5
            else:
                logger.warning(f"⚠ M5 data limited to {m5_years:.1f} years, trying H1 for more history...")
        
        # Fallback to H1 if M5 is limited or unavailable
        logger.info(f"Attempting H1 data collection for {symbol}...")
        df_h1 = self._collect_data_chunked(symbol, mt5.TIMEFRAME_H1, days, bars_per_day=24)
        
        if df_h1 is not None:
            h1_days = len(df_h1) / 24
            h1_years = h1_days / 365
            logger.info(f"✓ Using H1 data: {len(df_h1):,} bars ({h1_years:.1f} years)")
            return df_h1
        
        # If both failed, return M5 data if we have any
        if df_m5 is not None:
            logger.warning(f"⚠ H1 also failed, using limited M5 data ({m5_years:.1f} years)")
            return df_m5
        
        logger.error(f"✗ No data available for {symbol}")
        return None
    
    def _collect_data_chunked(self, symbol, timeframe, days, bars_per_day):
        """Helper function to collect data in chunks for any timeframe"""
        max_bars_per_request = 99999  # Stay under MT5's limit
        total_bars_needed = days * bars_per_day
        
        timeframe_name = "M5" if bars_per_day == 288 else "H1"
        
        all_rates = []
        bars_collected = 0
        chunk_num = 0
        
        while bars_collected < total_bars_needed:
            chunk_num += 1
            bars_to_request = min(max_bars_per_request, total_bars_needed - bars_collected)
            
            # Request data starting from position bars_collected
            rates = mt5.copy_rates_from_pos(symbol, timeframe, bars_collected, bars_to_request)
            
            if rates is None or len(rates) == 0:
                # No more data available
                if chunk_num == 1:
                    return None
                else:
                    break
            
            all_rates.append(rates)
            bars_collected += len(rates)
            
            # If we got fewer bars than requested, we've reached the end
            if len(rates) < bars_to_request:
                break
        
        # Combine all chunks
        if len(all_rates) == 0:
            return None
        
        combined_rates = np.concatenate(all_rates)
        df = pd.DataFrame(combined_rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Remove duplicates (in case of overlap)
        df = df.drop_duplicates(subset=['time'], keep='first')
        df = df.sort_values('time').reset_index(drop=True)
        
        return df
    
    def calculate_indicators(self, df):
        """Calculate all technical indicators (94 features)"""
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
        
        # CCI (Commodity Channel Index)
        for period in [14, 20]:
            tp = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = tp.rolling(period).mean()
            mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
            df[f'cci_{period}'] = (tp - sma_tp) / (0.015 * mad)
        
        # ADX (Average Directional Index)
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
        
        return df
    
    def create_labels(self, df, future_bars=24):
        """Create trading labels (1=buy, -1=sell, 0=hold)"""
        df['future_return'] = df['close'].pct_change(future_bars).shift(-future_bars)
        
        # Define thresholds based on volatility
        volatility = df['close'].pct_change().std()
        buy_threshold = volatility * 1.5
        sell_threshold = -volatility * 1.5
        
        df['label'] = 0
        df.loc[df['future_return'] > buy_threshold, 'label'] = 1
        df.loc[df['future_return'] < sell_threshold, 'label'] = -1
        
        return df
    
    def train_model(self, symbol):
        """Train ML model for a symbol"""
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training model for {symbol}")
            logger.info(f"{'='*60}")
            
            # Collect data
            df = self.collect_fresh_data(symbol)
            if df is None or len(df) < 1000:
                logger.error(f"Insufficient data for {symbol}")
                return False
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            # Create labels
            df = self.create_labels(df)
            
            # Drop NaN
            df = df.dropna()
            
            if len(df) < 500:
                logger.error(f"Insufficient clean data for {symbol}")
                return False
            
            # Prepare features
            feature_columns = [col for col in df.columns if col not in 
                             ['time', 'label', 'future_return', 'open', 'high', 'low', 'close', 
                              'tick_volume', 'spread', 'real_volume']]
            
            X = df[feature_columns].values
            y = df['label'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train ensemble
            logger.info("Training Random Forest...")
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train_scaled, y_train)
            
            logger.info("Training Gradient Boosting...")
            gb = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            )
            gb.fit(X_train_scaled, y_train)
            
            # Evaluate
            rf_pred = rf.predict(X_test_scaled)
            gb_pred = gb.predict(X_test_scaled)
            
            rf_acc = accuracy_score(y_test, rf_pred)
            gb_acc = accuracy_score(y_test, gb_pred)
            
            logger.info(f"Random Forest Accuracy: {rf_acc:.4f}")
            logger.info(f"Gradient Boosting Accuracy: {gb_acc:.4f}")
            
            # Save models
            models_dir = Path('ml_models_simple')
            models_dir.mkdir(exist_ok=True)
            
            ensemble = {'rf': rf, 'gb': gb, 'feature_columns': feature_columns}
            
            with open(models_dir / f"{symbol}_ensemble.pkl", 'wb') as f:
                pickle.dump(ensemble, f)
            
            with open(models_dir / f"{symbol}_scaler.pkl", 'wb') as f:
                pickle.dump(scaler, f)
            
            logger.info(f"[SUCCESS] Model saved for {symbol}")
            logger.info(f"Average Accuracy: {(rf_acc + gb_acc) / 2:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to train {symbol}: {e}")
            return False
    
    def retrain_all_models(self):
        """Retrain all models"""
        if not self.connect_mt5():
            logger.error("Failed to connect to MT5")
            return
        
        # Get symbols to train
        symbols = self.get_symbols_to_train()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"STARTING AUTO-RETRAIN CYCLE")
        logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Symbols to train: {len(symbols)}")
        logger.info(f"{'='*80}\n")
        
        success_count = 0
        failed_symbols = []
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"\nProgress: {i}/{len(symbols)}")
            
            if self.train_model(symbol):
                success_count += 1
            else:
                failed_symbols.append(symbol)
            
            # Small delay between symbols
            time.sleep(2)
        
        mt5.shutdown()
        
        # Summary
        logger.info(f"\n{'='*80}")
        logger.info(f"RETRAIN CYCLE COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Successful: {success_count}/{len(symbols)}")
        logger.info(f"Failed: {len(failed_symbols)}")
        if failed_symbols:
            logger.info(f"Failed symbols: {', '.join(failed_symbols)}")
        logger.info(f"Next retrain in 12 hours")
        logger.info(f"{'='*80}\n")
        
        self.last_retrain = datetime.now()
    
    def run(self):
        """Run auto-retrain system"""
        logger.info("="*80)
        logger.info("AUTO-RETRAIN SYSTEM V2 STARTED")
        logger.info("="*80)
        logger.info(f"Retrain interval: 12 hours")
        logger.info(f"Auto-discovery: {self.config.get('auto_discover_symbols', False)}")
        logger.info(f"Discovery mode: {self.config.get('discovery_mode', 'balanced')}")
        logger.info("="*80 + "\n")
        
        # Initial training
        logger.info("Starting initial training cycle...")
        self.retrain_all_models()
        
        # Continuous retraining
        while self.running:
            try:
                # Wait for next retrain
                next_retrain = self.last_retrain + timedelta(seconds=self.retrain_interval)
                time_until_retrain = (next_retrain - datetime.now()).total_seconds()
                
                if time_until_retrain > 0:
                    logger.info(f"Next retrain at: {next_retrain.strftime('%Y-%m-%d %H:%M:%S')}")
                    time.sleep(min(time_until_retrain, 300))  # Check every 5 minutes
                else:
                    logger.info("Starting scheduled retrain cycle...")
                    self.retrain_all_models()
                    
            except KeyboardInterrupt:
                logger.info("\nShutdown signal received, stopping...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Auto-Retrain System V2')
    parser.add_argument('--once', action='store_true', help='Run once and exit (no continuous retraining)')
    args = parser.parse_args()
    
    system = AutoRetrainSystemV2()
    
    if args.once:
        logger.info("Running single training cycle...")
        system.retrain_all_models()
    else:
        system.run()

if __name__ == "__main__":
    main()

