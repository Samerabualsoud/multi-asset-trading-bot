#!/usr/bin/env python3
"""
Auto-Retraining System
Automatically retrains ML models every 12 hours with fresh market data
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_retrain.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoRetrainSystem:
    def __init__(self, config_path='config.yaml'):
        self.config = self.load_config(config_path)
        self.retrain_interval = 12 * 3600  # 12 hours in seconds
        self.running = True
        self.last_retrain = None
        
    def load_config(self, config_path):
        """Load configuration"""
        with open(config_path, 'r') as f:
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
        """Calculate all technical indicators"""
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['atr'] = ranges.max(axis=1).rolling(14).mean()
        
        # Stochastic
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # ADX
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr = df['atr'] * 14
        plus_di = 100 * (plus_dm.rolling(14).mean() / tr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / tr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(14).mean()
        
        # Volume indicators
        df['volume_sma'] = df['tick_volume'].rolling(20).mean()
        df['volume_ratio'] = df['tick_volume'] / df['volume_sma']
        
        # Momentum
        df['momentum'] = df['close'].pct_change(periods=10)
        
        # Additional features
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_returns'].rolling(20).std()
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['tick_volume'].shift(lag)
        
        # Rolling statistics
        df['close_rolling_mean_10'] = df['close'].rolling(10).mean()
        df['close_rolling_std_10'] = df['close'].rolling(10).std()
        df['close_rolling_min_10'] = df['close'].rolling(10).min()
        df['close_rolling_max_10'] = df['close'].rolling(10).max()
        
        # Price position
        df['price_position'] = (df['close'] - df['close_rolling_min_10']) / (df['close_rolling_max_10'] - df['close_rolling_min_10'])
        
        # Trend indicators
        df['sma_20_50_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
        df['ema_sma_cross'] = (df['ema_20'] > df['sma_20']).astype(int)
        
        # Create labels (BUY/SELL based on future price)
        future_return = df['close'].shift(-24) / df['close'] - 1
        df['label'] = 0
        df.loc[future_return > 0.002, 'label'] = 1  # BUY
        df.loc[future_return < -0.002, 'label'] = -1  # SELL
        
        # Drop NaN
        df = df.dropna()
        
        return df
    
    def train_model(self, symbol, df):
        """Train ML model for a symbol"""
        logger.info(f"Training model for {symbol}...")
        
        # Prepare features
        exclude_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume', 'label']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove HOLD class (label = 0)
        df_binary = df[df['label'] != 0].copy()
        
        if len(df_binary) < 100:
            logger.error(f"Not enough data for {symbol}")
            return None, None, None
        
        # Convert labels: -1 (SELL) -> 0, 1 (BUY) -> 1
        df_binary['label'] = (df_binary['label'] == 1).astype(int)
        
        X = df_binary[feature_cols].values
        y = df_binary['label'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train_scaled, y_train)
        
        # Train Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=42
        )
        gb.fit(X_train_scaled, y_train)
        
        # Create ensemble
        class EnsembleModel:
            def __init__(self, rf, gb):
                self.rf = rf
                self.gb = gb
            
            def predict(self, X):
                rf_pred = self.rf.predict(X)
                gb_pred = self.gb.predict(X)
                # Weighted average (RF gets more weight)
                return ((rf_pred * 0.6 + gb_pred * 0.4) > 0.5).astype(int)
            
            def predict_proba(self, X):
                rf_proba = self.rf.predict_proba(X)
                gb_proba = self.gb.predict_proba(X)
                # Weighted average
                return rf_proba * 0.6 + gb_proba * 0.4
        
        ensemble = EnsembleModel(rf, gb)
        
        # Evaluate
        y_pred = ensemble.predict(X_test_scaled)
        y_pred_proba = ensemble.predict_proba(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        logger.info(f"{symbol} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        return ensemble, scaler, {'accuracy': accuracy, 'f1': f1, 'auc': auc}
    
    def save_model(self, symbol, model, scaler, metrics):
        """Save trained model"""
        models_dir = Path('ml_models_simple')
        models_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = models_dir / f"{symbol}_ensemble.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save scaler
        scaler_path = models_dir / f"{symbol}_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save metrics
        metrics_path = models_dir / f"{symbol}_metrics.pkl"
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)
        
        logger.info(f"Saved model for {symbol}")
    
    def retrain_all_models(self):
        """Retrain all models with fresh data"""
        logger.info("="*80)
        logger.info("STARTING AUTO-RETRAIN CYCLE")
        logger.info("="*80)
        logger.info(f"Time: {datetime.now()}")
        
        if not self.connect_mt5():
            logger.error("Failed to connect to MT5")
            return
        
        symbols = self.config['symbols']
        results = {}
        
        for symbol in symbols:
            try:
                # Collect fresh data
                df = self.collect_fresh_data(symbol, days=365)
                if df is None:
                    continue
                
                # Calculate indicators
                df = self.calculate_indicators(df)
                if len(df) < 100:
                    logger.warning(f"Not enough data for {symbol} after indicators")
                    continue
                
                # Train model
                model, scaler, metrics = self.train_model(symbol, df)
                if model is None:
                    continue
                
                # Save model
                self.save_model(symbol, model, scaler, metrics)
                
                results[symbol] = metrics
                
            except Exception as e:
                logger.error(f"Error retraining {symbol}: {e}", exc_info=True)
                continue
        
        mt5.shutdown()
        
        # Log summary
        logger.info("\n" + "="*80)
        logger.info("RETRAIN COMPLETE")
        logger.info("="*80)
        logger.info(f"Successfully retrained: {len(results)}/{len(symbols)} models")
        logger.info("\nPerformance Summary:")
        logger.info(f"{'Symbol':<10} {'Accuracy':<10} {'F1':<10} {'AUC':<10}")
        logger.info("-"*40)
        for symbol, metrics in results.items():
            logger.info(f"{symbol:<10} {metrics['accuracy']:<10.4f} {metrics['f1']:<10.4f} {metrics['auc']:<10.4f}")
        logger.info("="*80)
        
        self.last_retrain = datetime.now()
    
    def run(self):
        """Main loop - retrain every 12 hours"""
        logger.info("="*80)
        logger.info("AUTO-RETRAIN SYSTEM STARTED")
        logger.info("="*80)
        logger.info(f"Retrain interval: {self.retrain_interval / 3600} hours")
        logger.info("="*80)
        
        # Initial training
        self.retrain_all_models()
        
        # Continuous loop
        while self.running:
            try:
                # Calculate time until next retrain
                if self.last_retrain:
                    next_retrain = self.last_retrain + timedelta(seconds=self.retrain_interval)
                    time_until_next = (next_retrain - datetime.now()).total_seconds()
                    
                    if time_until_next > 0:
                        logger.info(f"\nNext retrain in {time_until_next / 3600:.1f} hours ({next_retrain})")
                        logger.info("Sleeping...")
                        time.sleep(min(time_until_next, 300))  # Check every 5 min
                    else:
                        # Time to retrain
                        self.retrain_all_models()
                else:
                    time.sleep(300)
                    
            except KeyboardInterrupt:
                logger.info("\nShutdown requested")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(300)
        
        logger.info("Auto-retrain system stopped")

def run_in_background():
    """Run auto-retrain system in background thread"""
    system = AutoRetrainSystem()
    thread = threading.Thread(target=system.run, daemon=True)
    thread.start()
    return thread

if __name__ == "__main__":
    system = AutoRetrainSystem()
    system.run()

