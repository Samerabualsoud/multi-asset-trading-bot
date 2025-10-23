#!/usr/bin/env python3
"""
ML + LLM Trading Bot - Optimized Version
- Faster LLM calls with timeout
- Larger lot sizes
- Graceful shutdown
- Better error handling
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import yaml
import logging
import time
import signal
import sys
from datetime import datetime
from pathlib import Path
import pickle
from openai import OpenAI
import concurrent.futures

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_llm_bot_optimized.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MLLLMTradingBot:
    def __init__(self, config_path='config.yaml'):
        self.config = self.load_config(config_path)
        self.models = {}
        self.scalers = {}
        self.running = True
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Initialize LLM client if enabled
        if self.config.get('llm_enabled', True):
            try:
                deepseek_key = self.config.get('deepseek_api_key')
                if deepseek_key and deepseek_key != "YOUR_DEEPSEEK_API_KEY_HERE":
                    self.llm_client = OpenAI(
                        api_key=deepseek_key,
                        base_url="https://api.deepseek.com"
                    )
                    logger.info("[OK] LLM client initialized (DeepSeek)")
                else:
                    self.llm_client = None
                    logger.warning("[WARN] DeepSeek API key not configured, LLM disabled")
            except Exception as e:
                self.llm_client = None
                logger.error(f"[ERROR] Failed to initialize LLM: {e}")
        else:
            self.llm_client = None
            logger.info("[INFO] LLM disabled in config")
        
        # Load ML models
        self.load_models()
        
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        logger.info("\n[SHUTDOWN] Received shutdown signal, stopping bot...")
        self.running = False
        mt5.shutdown()
        sys.exit(0)
    
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_models(self):
        """Load trained ML models"""
        models_dir = Path('ml_models_simple')
        
        for symbol in self.config['symbols']:
            try:
                # Load model
                model_path = models_dir / f"{symbol}_ensemble.pkl"
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        self.models[symbol] = pickle.load(f)
                    
                    # Load scaler
                    scaler_path = models_dir / f"{symbol}_scaler.pkl"
                    if scaler_path.exists():
                        with open(scaler_path, 'rb') as f:
                            self.scalers[symbol] = pickle.load(f)
                    
                    logger.info(f"[OK] Loaded model: {symbol}")
            except Exception as e:
                logger.error(f"[ERROR] Failed to load model for {symbol}: {e}")
        
        logger.info(f"\nLoaded {len(self.models)} models\n")
    
    def connect_mt5(self):
        """Connect to MT5"""
        if not mt5.initialize(
            path=self.config.get('mt5_path'),
            login=self.config['mt5_login'],
            password=self.config['mt5_password'],
            server=self.config['mt5_server']
        ):
            logger.error(f"[ERROR] MT5 initialization failed: {mt5.last_error()}")
            return False
        
        logger.info(f"[OK] Connected to MT5: {self.config['mt5_server']}")
        return True
    
    def get_market_data(self, symbol, timeframe=mt5.TIMEFRAME_M5, bars=2000):
        """Get recent market data"""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        
        if rates is None or len(rates) == 0:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    
    def calculate_indicators(self, df):
        """Calculate all technical indicators (same as training)"""
        # ... (keeping the full indicator calculation from the fixed version)
        # This is the same calculate_indicators method from ml_llm_trading_bot_fixed.py
        # Copying it here for completeness
        
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
        
        # Drop NaN
        df = df.dropna()
        
        return df
    
    def get_ml_prediction(self, symbol):
        """Get ML model prediction"""
        if symbol not in self.models:
            return None, 0.0
        
        # Get market data
        df = self.get_market_data(symbol)
        if df is None or len(df) < 100:
            return None, 0.0
        
        # Calculate indicators
        df_features = self.calculate_indicators(df)
        if len(df_features) == 0:
            return None, 0.0
        
        # Get latest features
        latest = df_features.iloc[-1]
        
        # Prepare features (exclude non-feature columns)
        exclude_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
        feature_cols = [col for col in df_features.columns if col not in exclude_cols]
        X = latest[feature_cols].values.reshape(1, -1)
        
        # Scale features
        if symbol in self.scalers:
            X = self.scalers[symbol].transform(X)
        
        # Get prediction
        model = self.models[symbol]
        pred_proba = model.predict_proba(X)[0]
        prediction = model.predict(X)[0]
        
        # Convert to BUY/SELL
        if prediction == 1:
            signal = "BUY"
            confidence = pred_proba[1]
        else:
            signal = "SELL"
            confidence = pred_proba[0]
        
        return signal, confidence
    
    def get_llm_analysis_with_timeout(self, symbol, ml_signal, ml_confidence, df_features, timeout=5):
        """Get LLM analysis with timeout"""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self.get_llm_analysis, symbol, ml_signal, ml_confidence, df_features)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                logger.warning(f"[TIMEOUT] LLM analysis for {symbol} took >{timeout}s, using ML only")
                return ml_signal, ml_confidence, "LLM timeout - using ML prediction"
            except Exception as e:
                logger.error(f"[ERROR] LLM analysis failed: {e}")
                return ml_signal, ml_confidence, f"LLM error: {str(e)}"
    
    def get_llm_analysis(self, symbol, ml_signal, ml_confidence, df_features):
        """Get LLM trading analysis"""
        if self.llm_client is None:
            return ml_signal, ml_confidence, "LLM not available"
        
        try:
            # Get latest indicators
            latest = df_features.iloc[-1]
            
            system_prompt = """You are an expert forex trading analyst. Analyze the technical indicators and ML prediction to make a final trading decision.

Provide your analysis in this EXACT format:
SIGNAL: [BUY/SELL/SKIP]
CONFIDENCE: [number between 0-100]
RISK: [LOW/MEDIUM/HIGH]
REASONING: [brief explanation]"""
            
            user_prompt = f"""Symbol: {symbol}
ML Prediction: {ml_signal} ({ml_confidence:.1%} confidence)

Technical Indicators:
- RSI: {latest.get('rsi', 0):.1f}
- MACD: {latest.get('macd', 0):.4f}
- Stochastic: {latest.get('stoch_k', 0):.1f}
- ADX: {latest.get('adx', 0):.1f}
- ATR: {latest.get('atr', 0):.2f}
- Price Change: {latest.get('price_change', 0):.2%}

Should we trade? Provide your analysis."""
            
            response = self.llm_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            analysis = response.choices[0].message.content
            
            # Parse response
            lines = analysis.strip().split('\n')
            signal = ml_signal
            confidence = ml_confidence
            reasoning = "LLM analysis"
            
            for line in lines:
                if line.startswith('SIGNAL:'):
                    signal = line.split(':')[1].strip()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        conf_str = line.split(':')[1].strip().replace('%', '')
                        confidence = float(conf_str) / 100
                    except:
                        pass
                elif line.startswith('REASONING:'):
                    reasoning = line.split(':', 1)[1].strip()
            
            return signal, confidence, reasoning
            
        except Exception as e:
            logger.error(f"LLM analysis error: {e}")
            return ml_signal, ml_confidence, f"LLM error: {str(e)}"
    
    def analyze_symbol(self, symbol):
        """Analyze a symbol and generate trading signal"""
        try:
            # Get ML prediction
            ml_signal, ml_confidence = self.get_ml_prediction(symbol)
            
            if ml_signal is None:
                return None
            
            logger.info(f"[ML] Prediction: {ml_signal} (confidence: {ml_confidence:.1%})")
            
            # Get market data for LLM
            df = self.get_market_data(symbol)
            df_features = self.calculate_indicators(df)
            
            # Get LLM analysis with timeout
            if self.llm_client and len(df_features) >= 2:
                final_signal, final_confidence, reasoning = self.get_llm_analysis_with_timeout(
                    symbol, ml_signal, ml_confidence, df_features, timeout=5
                )
            else:
                final_signal = ml_signal
                final_confidence = ml_confidence
                reasoning = "ML prediction only"
            
            result = {
                'symbol': symbol,
                'ml_signal': ml_signal,
                'ml_confidence': ml_confidence,
                'llm_signal': final_signal,
                'llm_confidence': final_confidence,
                'reasoning': reasoning
            }
            
            logger.info(f"\n[SIGNAL] {symbol}")
            logger.info(f"   Signal: {final_signal}")
            logger.info(f"   Confidence: {final_confidence:.1%}")
            logger.info(f"   Reasoning: {reasoning}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}", exc_info=True)
            return None
    
    def get_open_positions(self):
        """Get currently open positions"""
        positions = mt5.positions_get()
        if positions is None:
            return []
        return [pos.symbol for pos in positions]
    
    def calculate_lot_size(self, symbol, risk_percent=0.02):
        """Calculate lot size - LARGER LOTS"""
        account_info = mt5.account_info()
        if account_info is None:
            return 0.10
        
        balance = account_info.balance
        equity = account_info.equity
        
        # Use equity for calculation
        risk_amount = equity * risk_percent
        
        # More aggressive lot sizing
        base_lot = risk_amount / 100
        lot = base_lot * 5.0  # 5x multiplier
        
        # Enforce minimums based on balance
        if balance < 1000:
            lot = max(lot, 0.10)
        elif balance < 5000:
            lot = max(lot, 0.50)
        elif balance < 10000:
            lot = max(lot, 1.00)
        else:
            lot = max(lot, 2.00)
        
        # Round and ensure minimum
        lot = round(lot, 2)
        lot = max(lot, 0.10)
        
        return lot
    
    def place_trade(self, signal_data):
        """Place a trade"""
        symbol = signal_data['symbol']
        signal = signal_data['llm_signal']
        confidence = signal_data['llm_confidence']
        
        # Check existing position
        open_positions = self.get_open_positions()
        if symbol in open_positions:
            logger.info(f"[SKIP] Already have position for {symbol}")
            return False
        
        # Check max positions
        max_pos = self.config.get('max_positions', 5)
        if len(open_positions) >= max_pos:
            logger.info(f"[SKIP] Max positions reached ({len(open_positions)}/{max_pos})")
            return False
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"[ERROR] Symbol not found: {symbol}")
            return False
        
        if not symbol_info.visible:
            mt5.symbol_select(symbol, True)
        
        # Get price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"[ERROR] No tick data for {symbol}")
            return False
        
        # Determine order
        if signal == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
            tp = price * 1.02  # 2% TP
            sl = 0.0
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
            tp = price * 0.98
            sl = 0.0
        
        # Calculate lot
        risk_percent = self.config.get('risk_per_trade', 0.02)
        lot = self.calculate_lot_size(symbol, risk_percent)
        
        # Place order
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 234000,
            "comment": f"ML+LLM {confidence:.0%}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result is None:
            logger.error(f"[ERROR] Order failed: {mt5.last_error()}")
            return False
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"[ERROR] Order failed: {result.retcode} - {result.comment}")
            return False
        
        logger.info(f"\n[SUCCESS] Trade placed!")
        logger.info(f"   Symbol: {symbol}")
        logger.info(f"   Type: {signal}")
        logger.info(f"   Lot: {lot}")
        logger.info(f"   Price: {price:.5f}")
        logger.info(f"   TP: {tp:.5f}")
        logger.info(f"   SL: None")
        logger.info(f"   Ticket: {result.order}")
        logger.info(f"   Confidence: {confidence:.1%}")
        
        return True
    
    def run(self):
        """Main trading loop"""
        if not self.connect_mt5():
            return
        
        logger.info("="*80)
        logger.info("ML + LLM TRADING BOT STARTED (OPTIMIZED)")
        logger.info("="*80)
        logger.info(f"Models loaded: {len(self.models)}")
        logger.info(f"LLM enabled: {self.llm_client is not None}")
        logger.info(f"Scan interval: 15 seconds")
        logger.info("="*80)
        
        while self.running:
            try:
                logger.info(f"\n{'='*80}")
                logger.info(f"SCAN CYCLE: {datetime.now()}")
                logger.info(f"{'='*80}\n")
                
                symbols = self.config['symbols']
                
                for symbol in symbols:
                    if not self.running:
                        break
                    
                    try:
                        logger.info(f"{'='*80}")
                        logger.info(f"ANALYZING {symbol}")
                        logger.info(f"{'='*80}")
                        
                        result = self.analyze_symbol(symbol)
                        
                        if result and result['llm_signal'] != 'SKIP':
                            # Check confidence threshold
                            min_conf = self.config.get('min_confidence', 0.70)
                            if result['llm_confidence'] >= min_conf:
                                logger.info(f"\n[TRADE] Confidence {result['llm_confidence']:.1%} >= {min_conf:.0%}, placing trade...")
                                self.place_trade(result)
                            else:
                                logger.info(f"\n[SKIP] Confidence {result['llm_confidence']:.1%} < {min_conf:.0%}, not trading")
                        
                    except Exception as e:
                        logger.error(f"Error analyzing {symbol}: {e}", exc_info=True)
                        continue
                
                if self.running:
                    logger.info(f"\n[WAIT] Next scan in 15 seconds...")
                    time.sleep(15)
                    
            except KeyboardInterrupt:
                logger.info("\n[SHUTDOWN] Ctrl+C detected, stopping...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(15)
        
        mt5.shutdown()
        logger.info("[SHUTDOWN] Bot stopped")

if __name__ == "__main__":
    bot = MLLLMTradingBot()
    bot.run()

