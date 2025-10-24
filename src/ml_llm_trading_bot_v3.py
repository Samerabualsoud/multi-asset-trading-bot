#!/usr/bin/env python3
"""
ML + LLM Trading Bot V3 - With Auto-Discovery
- Supports auto-discovered symbols from MT5
- Faster LLM calls with timeout
- Larger lot sizes
- Graceful shutdown
- Better error handling
- Dynamic symbol loading
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

# Setup basic logging (will be reconfigured after loading config)
logger = logging.getLogger(__name__)

class MLLLMTradingBotV3:
    def __init__(self, config_path='config.yaml'):
        self.config = self.load_config(config_path)
        
        # Configure logging based on config
        log_level = getattr(logging, self.config.get('log_level', 'INFO').upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.get('log_file', 'ml_llm_bot_v3.log')),
                logging.StreamHandler()
            ],
            force=True  # Override any existing config
        )
        self.models = {}
        self.scalers = {}
        self.running = True
        self.scan_interval = 15  # seconds
        
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
        config_file = Path(config_path)
        if not config_file.exists():
            config_file = Path('config') / config_path
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def load_models(self):
        """Load trained ML models"""
        models_dir = Path('ml_models_simple')
        
        if not models_dir.exists():
            logger.error(f"[ERROR] Models directory not found: {models_dir}")
            logger.error("Please run auto_retrain_system_v2.py first to train models")
            return
        
        # Get list of symbols from config
        symbols = self.config.get('symbols', [])
        
        if not symbols:
            logger.error("[ERROR] No symbols in config.yaml")
            logger.error("Run symbol_discovery_enhanced.py to discover symbols")
            return
        
        logger.info(f"Loading models for {len(symbols)} symbols...")
        
        for symbol in symbols:
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
                else:
                    logger.warning(f"[SKIP] No model found for {symbol}")
            except Exception as e:
                logger.error(f"[ERROR] Failed to load model for {symbol}: {e}")
        
        logger.info(f"\n[SUMMARY] Loaded {len(self.models)} models out of {len(symbols)} symbols\n")
        
        if len(self.models) == 0:
            logger.error("[ERROR] No models loaded! Please run auto_retrain_system_v2.py first")
    
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
    
    def get_market_data(self, symbol, timeframe=mt5.TIMEFRAME_M30, bars=2000):
        """Get recent market data"""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        
        if rates is None or len(rates) == 0:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    
    def calculate_indicators(self, df):
        """Calculate all technical indicators (same as training)"""
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
        
        # Drop NaN
        df = df.dropna()
        
        return df
    
    def get_ml_prediction(self, symbol):
        """Get ML model prediction"""
        if symbol not in self.models:
            logger.debug(f"[ANALYSIS] {symbol}: Model not found")
            return None, 0.0
        
        # Get market data
        df = self.get_market_data(symbol)
        if df is None or len(df) < 100:
            logger.debug(f"[ANALYSIS] {symbol}: Insufficient data (got {len(df) if df is not None else 0} bars, need 100+)")
            return None, 0.0
        
        logger.debug(f"[ANALYSIS] {symbol}: Retrieved {len(df)} bars of market data")
        
        # Calculate indicators
        df_features = self.calculate_indicators(df)
        if len(df_features) == 0:
            logger.debug(f"[ANALYSIS] {symbol}: Feature calculation failed (0 features)")
            return None, 0.0
        
        logger.debug(f"[ANALYSIS] {symbol}: Calculated {len(df_features.columns)} features from {len(df_features)} bars")
        
        # Get latest features
        latest = df_features.iloc[-1]
        
        # Get feature columns from model
        model_data = self.models[symbol]
        if isinstance(model_data, dict) and 'feature_columns' in model_data:
            feature_cols = model_data['feature_columns']
            rf_model = model_data.get('rf')
            xgb_model = model_data.get('xgb')  # XGBoost (new)
            gb_model = model_data.get('gb')    # Gradient Boosting (legacy)
        else:
            # Fallback: exclude non-feature columns
            exclude_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
            feature_cols = [col for col in df_features.columns if col not in exclude_cols]
            rf_model = model_data
            gb_model = None
        
        # Prepare features
        X = latest[feature_cols].values.reshape(1, -1)
        
        # Scale features
        if symbol in self.scalers:
            X = self.scalers[symbol].transform(X)
        
        # Get ensemble prediction
        rf_pred_proba = rf_model.predict_proba(X)[0]
        rf_pred = rf_model.predict(X)[0]
        
        # Use XGBoost if available (preferred), otherwise Gradient Boosting (legacy)
        if xgb_model:
            xgb_pred_proba = xgb_model.predict_proba(X)[0]
            xgb_pred = xgb_model.predict(X)[0]
            # Average RF and XGBoost predictions
            pred_proba = (rf_pred_proba + xgb_pred_proba) / 2
            prediction = rf_pred if rf_pred == xgb_pred else rf_pred  # Use RF if disagreement
        elif gb_model:
            gb_pred_proba = gb_model.predict_proba(X)[0]
            gb_pred = gb_model.predict(X)[0]
            # Average RF and GB predictions
            pred_proba = (rf_pred_proba + gb_pred_proba) / 2
            prediction = rf_pred if rf_pred == gb_pred else rf_pred  # Use RF if disagreement
        else:
            # Use RF only
            pred_proba = rf_pred_proba
            prediction = rf_pred
        
        # Convert to BUY/SELL
        if prediction == 1:
            signal = "BUY"
            confidence = pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]
            logger.debug(f"[ANALYSIS] {symbol}: ML prediction = BUY (confidence: {confidence:.1%}, proba: {pred_proba})")
        elif prediction == -1:
            signal = "SELL"
            confidence = pred_proba[0] if len(pred_proba) > 1 else pred_proba[0]
            logger.debug(f"[ANALYSIS] {symbol}: ML prediction = SELL (confidence: {confidence:.1%}, proba: {pred_proba})")
        else:
            signal = "SKIP"
            confidence = 0.0
            logger.debug(f"[ANALYSIS] {symbol}: ML prediction = HOLD (no trade signal, proba: {pred_proba})")
        
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
- RSI_14: {latest.get('rsi_14', 0):.1f}
- MACD: {latest.get('macd', 0):.4f}
- Stochastic_14: {latest.get('stoch_14', 0):.1f}
- ADX: {latest.get('adx', 0):.1f}
- ATR_14: {latest.get('atr_14', 0):.4f}
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
            
            if ml_signal == "SKIP":
                # Log HOLD signals so user knows all symbols are being analyzed
                logger.debug(f"[HOLD] {symbol}: Model predicts HOLD (no trade)")
                return None
            
            # Get market data for LLM
            df = self.get_market_data(symbol)
            df_features = self.calculate_indicators(df)
            
            # Get LLM analysis with timeout
            if self.llm_client and len(df_features) > 0:
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
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def get_open_positions(self):
        """Get currently open positions"""
        positions = mt5.positions_get()
        if positions is None:
            return []
        return [pos.symbol for pos in positions]
    
    def calculate_lot_size(self, symbol, risk_percent=0.02):
        """Calculate lot size - LARGER LOTS (5x multiplier)"""
        account_info = mt5.account_info()
        if account_info is None:
            return 0.01
        
        balance = account_info.balance
        
        # Base lot size on balance with 5x multiplier
        if balance < 1000:
            base_lot = 0.01
        elif balance < 5000:
            base_lot = 0.05
        elif balance < 10000:
            base_lot = 0.10
        elif balance < 50000:
            base_lot = 0.50
        else:
            base_lot = 1.00
        
        # Apply 5x multiplier
        lot_size = base_lot * 5.0
        
        # Get symbol info for limits
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return lot_size
        
        # Respect min/max limits
        lot_size = max(symbol_info.volume_min, min(lot_size, symbol_info.volume_max))
        
        # Round to volume step
        lot_size = round(lot_size / symbol_info.volume_step) * symbol_info.volume_step
        
        return lot_size
    
    def place_trade(self, symbol, signal, confidence):
        """Place a trade"""
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"[ERROR] Symbol info not available for {symbol}")
                return False
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error(f"[ERROR] Tick data not available for {symbol}")
                return False
            
            # Calculate lot size
            lot_size = self.calculate_lot_size(symbol)
            
            # Determine order type and price
            if signal == "BUY":
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
                tp_price = price * 1.02  # 2% take profit
            else:  # SELL
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
                tp_price = price * 0.98  # 2% take profit
            
            # Prepare request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": order_type,
                "price": price,
                "tp": tp_price,
                "sl": 0,  # NO STOP LOSS as per user requirement
                "deviation": 20,
                "magic": 234000,
                "comment": f"ML+LLM Bot V3 ({confidence:.0%})",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"[ERROR] Order failed: {result.comment} (code: {result.retcode})")
                return False
            
            logger.info(f"\n[TRADE EXECUTED]")
            logger.info(f"   Symbol: {symbol}")
            logger.info(f"   Type: {signal}")
            logger.info(f"   Lot Size: {lot_size}")
            logger.info(f"   Entry Price: {price}")
            logger.info(f"   Take Profit: {tp_price}")
            logger.info(f"   Confidence: {confidence:.1%}")
            logger.info(f"   Order ID: {result.order}\n")
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to place trade: {e}")
            return False
    
    def print_analysis_table(self, results):
        """Print analysis results in a clean table format"""
        if not results:
            return
        
        logger.info("\n" + "="*100)
        logger.info("ANALYSIS SUMMARY")
        logger.info("="*100)
        
        # Table header
        header = f"{'Symbol':<10} | {'Signal':<6} | {'Confidence':<12} | {'Status':<10} | {'Action'}"
        logger.info(header)
        logger.info("-" * 100)
        
        # Sort by confidence (highest first)
        sorted_results = sorted(results, key=lambda x: x['confidence'], reverse=True)
        
        # Count by status
        hold_count = sum(1 for r in results if r['status'] == 'HOLD')
        signal_count = sum(1 for r in results if r['status'] == 'SIGNAL')
        low_conf_count = sum(1 for r in results if r['status'] == 'LOW_CONF')
        error_count = sum(1 for r in results if r['status'] == 'ERROR')
        open_pos_count = sum(1 for r in results if r['status'] == 'OPEN_POS')
        
        # Print each symbol
        for result in sorted_results:
            symbol = result['symbol']
            signal = result['signal']
            confidence = result['confidence']
            status = result['status']
            
            # Determine action
            if status == 'SIGNAL':
                action = f"✅ TRADE {signal}"
            elif status == 'LOW_CONF':
                action = f"⚠️  SKIP (conf < 70%)"
            elif status == 'HOLD':
                action = "⏸️  HOLD (no setup)"
            elif status == 'OPEN_POS':
                action = "📊 POSITION OPEN"
            else:
                action = "❌ ERROR"
            
            # Format confidence
            conf_str = f"{confidence:.1%}" if confidence > 0 else "N/A"
            
            row = f"{symbol:<10} | {signal:<6} | {conf_str:<12} | {status:<10} | {action}"
            logger.info(row)
        
        # Summary footer
        logger.info("-" * 100)
        logger.info(f"SUMMARY: {len(results)} symbols | "
                   f"✅ {signal_count} tradeable | "
                   f"⚠️  {low_conf_count} low confidence | "
                   f"⏸️  {hold_count} hold | "
                   f"📊 {open_pos_count} open positions | "
                   f"❌ {error_count} errors")
        logger.info("="*100 + "\n")
    
    def run(self):
        """Main trading loop"""
        logger.info("="*80)
        logger.info("ML + LLM TRADING BOT V3 STARTED")
        logger.info("="*80)
        logger.info(f"Symbols: {len(self.models)} models loaded")
        logger.info(f"LLM: {'Enabled' if self.llm_client else 'Disabled'}")
        logger.info(f"Scan interval: {self.scan_interval} seconds")
        logger.info(f"Max positions: {self.config.get('max_positions', 5)}")
        logger.info(f"Min confidence: {self.config.get('min_confidence', 0.70):.0%}")
        logger.info("="*80 + "\n")
        
        if not self.connect_mt5():
            logger.error("Failed to connect to MT5")
            return
        
        # Check AutoTrading
        terminal_info = mt5.terminal_info()
        if terminal_info and not terminal_info.trade_allowed:
            logger.error("="*80)
            logger.error("AUTOTRADING IS DISABLED IN MT5!")
            logger.error("Please enable it: Tools -> Options -> Expert Advisors")
            logger.error("Check 'Allow algorithmic trading'")
            logger.error("="*80)
            return
        
        while self.running:
            try:
                # Get open positions
                open_symbols = self.get_open_positions()
                max_positions = self.config.get('max_positions', 5)
                
                logger.info(f"\n[SCAN] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"Open positions: {len(open_symbols)}/{max_positions}")
                
                if len(open_symbols) >= max_positions:
                    logger.info("[SKIP] Maximum positions reached")
                    time.sleep(self.scan_interval)
                    continue
                
                # Collect analysis results for table
                analysis_results = []
                min_confidence = self.config.get('min_confidence', 0.70)
                
                # Scan ALL symbols (including those with positions for visibility)
                for symbol in self.models.keys():
                    if not self.running:
                        break
                    
                    # Check if already have position
                    if symbol in open_symbols:
                        analysis_results.append({
                            'symbol': symbol,
                            'signal': 'N/A',
                            'confidence': 0.0,
                            'status': 'OPEN_POS'
                        })
                        continue
                    
                    # Analyze symbol
                    result = self.analyze_symbol(symbol)
                    
                    # Store result for table (even if None)
                    if result is None:
                        # Get ML prediction directly for HOLD signals
                        ml_signal, ml_confidence = self.get_ml_prediction(symbol)
                        analysis_results.append({
                            'symbol': symbol,
                            'signal': ml_signal if ml_signal else 'ERROR',
                            'confidence': ml_confidence if ml_confidence else 0.0,
                            'status': 'HOLD' if ml_signal == 'SKIP' else 'ERROR'
                        })
                        continue
                    
                    # Determine status
                    if result['llm_confidence'] >= min_confidence:
                        status = 'SIGNAL'
                    else:
                        status = 'LOW_CONF'
                    
                    # Store result
                    analysis_results.append({
                        'symbol': symbol,
                        'signal': result['ml_signal'],
                        'confidence': result['ml_confidence'],
                        'status': status
                    })
                    
                    # Check confidence threshold
                    if result['llm_confidence'] < min_confidence:
                        logger.debug(f"[SKIP] {symbol}: Low confidence ({result['llm_confidence']:.1%})")
                        continue
                    
                    # Check if signal is actionable
                    if result['llm_signal'] not in ['BUY', 'SELL']:
                        logger.debug(f"[SKIP] {symbol}: Signal is {result['llm_signal']}")
                        continue
                    
                    # Place trade
                    logger.info(f"\n[SIGNAL] {symbol}")
                    logger.info(f"   ML: {result['ml_signal']} ({result['ml_confidence']:.1%})")
                    logger.info(f"   LLM: {result['llm_signal']} ({result['llm_confidence']:.1%})")
                    logger.info(f"   Reasoning: {result['reasoning']}")
                    
                    if self.place_trade(symbol, result['llm_signal'], result['llm_confidence']):
                        open_symbols.append(symbol)
                        
                        # Check if max positions reached
                        if len(open_symbols) >= max_positions:
                            logger.info("[INFO] Maximum positions reached")
                            break
                
                # Print summary table
                self.print_analysis_table(analysis_results)
                
                # Wait before next scan
                time.sleep(self.scan_interval)
                
            except KeyboardInterrupt:
                logger.info("\n[SHUTDOWN] Ctrl+C detected, stopping...")
                break
            except Exception as e:
                logger.error(f"[ERROR] Main loop error: {e}")
                time.sleep(self.scan_interval)
        
        mt5.shutdown()
        logger.info("\n[STOPPED] Bot stopped successfully")

def main():
    """Main function"""
    bot = MLLLMTradingBotV3()
    bot.run()

if __name__ == "__main__":
    main()

