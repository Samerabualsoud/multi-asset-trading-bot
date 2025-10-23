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
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
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
        """Collect maximum available M30 data (200K bars = ~11 years)"""
        logger.info(f"Collecting fresh data for {symbol}...")
        
        # Target: 200,000 bars (2 chunks of 100K each)
        # M30: 48 bars/day, so 200K bars = 4,166 days = 11.4 years
        target_bars = 200000
        bars_per_day = 48  # M30 timeframe
        
        # Try M30 data first
        logger.info(f"Attempting M30 data collection for {symbol} ({target_bars:,} bars = ~11 years)...")
        df_m30 = self._collect_data_by_bars(symbol, mt5.TIMEFRAME_M30, target_bars, bars_per_day)
        
        if df_m30 is not None:
            m30_days = len(df_m30) / 48
            m30_years = m30_days / 365
            logger.info(f"✓ Using M30 data: {len(df_m30):,} bars ({m30_years:.1f} years)")
            return df_m30
        
        # Only fall back to H1 if M30 completely failed
        logger.warning(f"⚠ M30 data unavailable, trying H1 as fallback...")
        # For H1, request 10 years = 87,600 bars (under 100K limit, single chunk)
        df_h1 = self._collect_data_by_bars(symbol, mt5.TIMEFRAME_H1, 87600, 24)
        
        if df_h1 is not None:
            h1_days = len(df_h1) / 24
            h1_years = h1_days / 365
            logger.info(f"✓ Using H1 data: {len(df_h1):,} bars ({h1_years:.1f} years)")
            return df_h1
        
        logger.error(f"✗ No data available for {symbol}")
        return None
    
    def _collect_data_by_bars(self, symbol, timeframe, total_bars_needed, bars_per_day):
        """Helper function to collect data in chunks for any timeframe"""
        max_bars_per_request = 99999  # Stay under MT5's limit
        
        if bars_per_day == 48:
            timeframe_name = "M30"
        elif bars_per_day == 24:
            timeframe_name = "H1"
        else:
            timeframe_name = f"Unknown({bars_per_day})"
        
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
        
        # Bollinger Band position (0-1, where price is between bands)
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
        
        # Advanced features for better accuracy
        # Candlestick patterns
        body = abs(df['close'] - df['open'])
        range_bar = df['high'] - df['low']
        df['body_ratio'] = body / (range_bar + 1e-10)  # Avoid division by zero
        df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        
        # Volume analysis
        df['volume_ma_5'] = df['tick_volume'].rolling(5).mean()
        df['volume_ma_20'] = df['tick_volume'].rolling(20).mean()
        df['volume_ratio'] = df['tick_volume'] / (df['volume_ma_20'] + 1)
        
        # Price momentum over multiple periods
        for period in [3, 6, 12, 24]:
            df[f'momentum_{period}'] = df['close'].pct_change(period)
            df[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std()
        
        # Interaction features (RSI × MACD)
        df['rsi_macd'] = df['rsi_14'] * df['macd']
        if 'bb_position' in df.columns:
            df['rsi_bb_position'] = df['rsi_14'] * df['bb_position']
        else:
            df['rsi_bb_position'] = 0
        
        # Lagged features (previous bars)
        for lag in [1, 2, 3]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['tick_volume'].shift(lag)
            df[f'rsi_lag_{lag}'] = df['rsi_14'].shift(lag)
        
        # Rolling statistics
        for window in [10, 20, 50]:
            df[f'close_mean_{window}'] = df['close'].rolling(window).mean()
            df[f'close_std_{window}'] = df['close'].rolling(window).std()
            df[f'close_min_{window}'] = df['close'].rolling(window).min()
            df[f'close_max_{window}'] = df['close'].rolling(window).max()
        
        return df
    
    def create_labels(self, df, future_bars=48):
        """Create trading labels (1=buy, -1=sell, 0=hold)
        
        For M30 timeframe: 48 bars = 24 hours (1 day)
        For H1 timeframe: 24 bars = 24 hours (1 day)
        """
        # Calculate future return over next 24 hours
        df['future_return'] = df['close'].pct_change(future_bars).shift(-future_bars)
        
        # Use fixed percentage thresholds (more reliable than volatility-based)
        # 0.5% move in 24 hours is a clear directional signal
        buy_threshold = 0.005  # 0.5% gain
        sell_threshold = -0.005  # 0.5% loss
        
        # Create labels
        df['label'] = 0  # Default: HOLD
        df.loc[df['future_return'] > buy_threshold, 'label'] = 1  # BUY
        df.loc[df['future_return'] < sell_threshold, 'label'] = -1  # SELL
        
        # Log label distribution for debugging
        if len(df) > 0:
            buy_count = (df['label'] == 1).sum()
            sell_count = (df['label'] == -1).sum()
            hold_count = (df['label'] == 0).sum()
            total = len(df.dropna(subset=['label']))
            if total > 0:
                logger.info(f"Label distribution: BUY={buy_count} ({buy_count/total*100:.1f}%), "
                          f"SELL={sell_count} ({sell_count/total*100:.1f}%), "
                          f"HOLD={hold_count} ({hold_count/total*100:.1f}%)")
        
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
            
            # Calculate class weights to handle imbalance
            classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight_dict = dict(zip(classes, class_weights))
            logger.info(f"Class weights: {class_weight_dict}")
            
            # Train ensemble with optimized models (balanced speed vs accuracy)
            logger.info("Training Random Forest...")
            rf = RandomForestClassifier(
                n_estimators=150,  # Reduced from 200 for speed
                max_depth=12,  # Reduced from 15 for speed
                min_samples_split=15,  # Increased for speed
                min_samples_leaf=8,  # Increased for speed
                class_weight='balanced',  # Handle imbalance
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            rf.fit(X_train_scaled, y_train)
            
            logger.info("Training XGBoost...")
            # XGBoost requires labels to be 0, 1, 2 (not -1, 0, 1)
            # Map: -1 (SELL) -> 0, 0 (HOLD) -> 1, 1 (BUY) -> 2
            label_map = {-1: 0, 0: 1, 1: 2}
            label_map_reverse = {0: -1, 1: 0, 2: 1}
            
            y_train_xgb = np.array([label_map[label] for label in y_train])
            y_test_xgb = np.array([label_map[label] for label in y_test])
            
            # Convert class weights for XGBoost (using mapped labels)
            class_weight_dict_xgb = {label_map[k]: v for k, v in class_weight_dict.items()}
            sample_weights = np.array([class_weight_dict_xgb[label] for label in y_train_xgb])
            
            xgb = XGBClassifier(
                n_estimators=150,  # Reduced from 200 for speed
                max_depth=6,  # Reduced from 8 for speed
                learning_rate=0.1,  # Increased from 0.05 for faster convergence
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss',
                verbosity=0
            )
            xgb.fit(X_train_scaled, y_train_xgb, sample_weight=sample_weights, verbose=False)
            
            # Gradient Boosting removed for speed (XGBoost is better anyway)
            # GB is single-threaded and very slow
            
            # Evaluate models
            rf_pred = rf.predict(X_test_scaled)
            xgb_pred_mapped = xgb.predict(X_test_scaled)
            # Convert XGBoost predictions back to -1, 0, 1
            xgb_pred = np.array([label_map_reverse[label] for label in xgb_pred_mapped])
            
            rf_acc = accuracy_score(y_test, rf_pred)
            xgb_acc = accuracy_score(y_test, xgb_pred)
            
            logger.info(f"Random Forest Accuracy: {rf_acc:.4f}")
            logger.info(f"XGBoost Accuracy: {xgb_acc:.4f}")
            
            # Save models
            models_dir = Path('ml_models_simple')
            models_dir.mkdir(exist_ok=True)
            
            ensemble = {'rf': rf, 'xgb': xgb, 'feature_columns': feature_columns}
            
            with open(models_dir / f"{symbol}_ensemble.pkl", 'wb') as f:
                pickle.dump(ensemble, f)
            
            with open(models_dir / f"{symbol}_scaler.pkl", 'wb') as f:
                pickle.dump(scaler, f)
            
            logger.info(f"[SUCCESS] Model saved for {symbol}")
            avg_acc = (rf_acc + xgb_acc) / 2
            logger.info(f"Average Accuracy: {avg_acc:.4f}")
            logger.info(f"Best Model: {'RF' if rf_acc > xgb_acc else 'XGB'}")
            logger.info(f"Best Accuracy: {max(rf_acc, xgb_acc):.4f}")
            
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

