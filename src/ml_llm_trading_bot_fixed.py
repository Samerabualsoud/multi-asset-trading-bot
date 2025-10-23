#!/usr/bin/env python3
"""
ML + LLM Trading Bot (Fixed)
- Complete feature calculation (matches training)
- 15 second scan interval
- No emoji characters (Windows compatible)
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import logging
import pickle
from pathlib import Path
import time
from openai import OpenAI
import json

logger = logging.getLogger(__name__)


class MLLLMTradingBot:
    """Trading bot that combines ML predictions with LLM analysis"""
    
    def __init__(self, config_file='config.yaml'):
        self.config = self.load_config(config_file)
        self.models = {}
        self.llm_client = None
        self.initialize_llm()
        self.load_all_models()
    
    def load_config(self, config_file):
        """Load configuration"""
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def initialize_llm(self):
        """Initialize LLM client (DeepSeek)"""
        api_key = self.config.get('deepseek_api_key', '')
        
        if not api_key or api_key == 'YOUR_DEEPSEEK_API_KEY_HERE':
            logger.warning("[WARNING] DeepSeek API key not configured!")
            logger.warning("LLM analyst will be disabled. Add API key to config.yaml")
            return
        
        try:
            self.llm_client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
            logger.info("[OK] LLM client initialized (DeepSeek)")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.llm_client = None
    
    def load_all_models(self):
        """Load all trained ML models"""
        model_dir = Path('ml_models_simple')
        
        if not model_dir.exists():
            logger.error(f"Model directory not found: {model_dir}")
            logger.error("Please train models first: python src/simple_trainer.py")
            return
        
        model_files = list(model_dir.glob('*_models.pkl'))
        
        for model_file in model_files:
            symbol = model_file.stem.replace('_models', '')
            
            try:
                with open(model_file, 'rb') as f:
                    self.models[symbol] = pickle.load(f)
                logger.info(f"[OK] Loaded model: {symbol}")
            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")
        
        logger.info(f"\nLoaded {len(self.models)} models")
    
    def initialize_mt5(self):
        """Initialize MT5 connection"""
        if not mt5.initialize(
            path=self.config['mt5_path'],
            login=self.config['mt5_login'],
            password=self.config['mt5_password'],
            server=self.config['mt5_server']
        ):
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        
        logger.info(f"[OK] Connected to MT5: {self.config['mt5_server']}")
        return True
    
    def get_market_data(self, symbol, timeframe=mt5.TIMEFRAME_H1, bars=500):
        """Get recent market data"""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        
        if rates is None or len(rates) == 0:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        return df
    
    def calculate_indicators(self, df):
        """Calculate ALL technical indicators (matches training exactly)"""
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
        
        # Fibonacci levels
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
        
        # Drop intermediate columns
        cols_to_drop = ['tr1', 'tr2', 'tr3', 'tr', 'plus_dm', 'minus_dm', 'dx',
                       'ema_12', 'ema_26', 'bb_std']
        data = data.drop(columns=[c for c in cols_to_drop if c in data.columns])
        
        return data.dropna()
    
    def get_ml_prediction(self, symbol):
        """Get ML model prediction"""
        if symbol not in self.models:
            logger.warning(f"No model found for {symbol}")
            return None, None
        
        # Get market data
        df = self.get_market_data(symbol)
        if df is None:
            return None, None
        
        # Calculate indicators
        df_features = self.calculate_indicators(df)
        
        if len(df_features) == 0:
            return None, None
        
        # Get latest row
        latest = df_features.iloc[-1]
        
        # Prepare features
        model_data = self.models[symbol]
        feature_cols = model_data['feature_cols']
        scaler = model_data['scaler']
        rf = model_data['random_forest']
        gb = model_data['gradient_boosting']
        weights = model_data['weights']
        
        # Extract features
        X = latest[feature_cols].values.reshape(1, -1)
        X_scaled = scaler.transform(X)
        
        # Get predictions
        rf_proba = rf.predict_proba(X_scaled)[0, 1]
        gb_proba = gb.predict_proba(X_scaled)[0, 1]
        
        # Ensemble
        ensemble_proba = weights['rf'] * rf_proba + weights['gb'] * gb_proba
        ensemble_pred = 1 if ensemble_proba > 0.5 else 0  # 1=BUY, 0=SELL
        
        signal = "BUY" if ensemble_pred == 1 else "SELL"
        confidence = ensemble_proba if ensemble_pred == 1 else (1 - ensemble_proba)
        
        return signal, confidence
    
    def get_llm_analysis(self, symbol, ml_signal, ml_confidence, market_data):
        """Get LLM trading analyst review"""
        if self.llm_client is None:
            return ml_signal, ml_confidence, "LLM not available"
        
        try:
            # Prepare market context
            latest = market_data.iloc[-1]
            prev = market_data.iloc[-2]
            
            context = f"""
You are an expert forex trading analyst. Review this ML prediction and provide your final decision.

Symbol: {symbol}
Current Price: {latest['close']:.5f}
Price Change: {((latest['close'] - prev['close']) / prev['close'] * 100):.2f}%

ML Model Prediction:
- Signal: {ml_signal}
- Confidence: {ml_confidence:.1%}

Technical Indicators:
- RSI: {latest['rsi']:.1f}
- MACD: {latest['macd']:.5f} (Signal: {latest['macd_signal']:.5f})
- ADX: {latest['adx']:.1f}
- Stochastic: {latest['stoch_k']:.1f}
- ATR: {latest['atr']:.5f}
- Trend (SMA20): {"Bullish" if latest['trend_20'] > 0 else "Bearish"}
- Trend (SMA50): {"Bullish" if latest['trend_50'] > 0 else "Bearish"}

Provide your analysis in JSON format:
{{
    "final_decision": "BUY" or "SELL" or "SKIP",
    "confidence": 0-100,
    "reasoning": "Brief explanation",
    "risk_level": "LOW", "MEDIUM", or "HIGH"
}}
"""
            
            response = self.llm_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are an expert forex trading analyst. Always respond with valid JSON only."},
                    {"role": "user", "content": context}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON
            if result_text.startswith('```json'):
                result_text = result_text.split('```json')[1].split('```')[0].strip()
            elif result_text.startswith('```'):
                result_text = result_text.split('```')[1].split('```')[0].strip()
            
            result = json.loads(result_text)
            
            final_signal = result['final_decision']
            final_confidence = result['confidence'] / 100
            reasoning = result['reasoning']
            risk = result['risk_level']
            
            logger.info(f"\n[LLM] Analysis for {symbol}:")
            logger.info(f"   ML: {ml_signal} ({ml_confidence:.1%})")
            logger.info(f"   LLM: {final_signal} ({final_confidence:.1%})")
            logger.info(f"   Risk: {risk}")
            logger.info(f"   Reasoning: {reasoning}")
            
            return final_signal, final_confidence, reasoning
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return ml_signal, ml_confidence, f"LLM error: {str(e)}"
    
    def analyze_symbol(self, symbol):
        """Complete analysis: ML + LLM"""
        logger.info(f"\n{'='*80}")
        logger.info(f"ANALYZING {symbol}")
        logger.info(f"{'='*80}")
        
        # Get ML prediction
        ml_signal, ml_confidence = self.get_ml_prediction(symbol)
        
        if ml_signal is None:
            logger.warning(f"ML prediction failed for {symbol}")
            return None
        
        logger.info(f"[ML] Prediction: {ml_signal} (confidence: {ml_confidence:.1%})")
        
        # Get market data for LLM
        df = self.get_market_data(symbol)
        df_features = self.calculate_indicators(df)
        
        # Check if we have enough data
        if len(df_features) < 2:
            logger.warning(f"Not enough data for LLM analysis ({len(df_features)} rows)")
            final_signal = ml_signal
            final_confidence = ml_confidence
            reasoning = "Insufficient data for LLM analysis"
        else:
            # Get LLM analysis
            final_signal, final_confidence, reasoning = self.get_llm_analysis(
                symbol, ml_signal, ml_confidence, df_features
            )
        
        result = {
            'symbol': symbol,
            'ml_signal': ml_signal,
            'ml_confidence': ml_confidence,
            'llm_signal': final_signal,
            'llm_confidence': final_confidence,
            'reasoning': reasoning,
            'timestamp': datetime.now()
        }
        
        return result
    
    def run(self):
        """Main trading loop"""
        logger.info("\n" + "="*80)
        logger.info("ML + LLM TRADING BOT STARTED")
        logger.info("="*80)
        logger.info(f"Models loaded: {len(self.models)}")
        logger.info(f"LLM enabled: {self.llm_client is not None}")
        logger.info(f"Scan interval: 15 seconds")
        logger.info("="*80)
        
        # Initialize MT5
        if not self.initialize_mt5():
            return
        
        symbols = list(self.models.keys())
        
        try:
            while True:
                logger.info(f"\n{'='*80}")
                logger.info(f"SCAN CYCLE: {datetime.now()}")
                logger.info(f"{'='*80}")
                
                for symbol in symbols:
                    try:
                        result = self.analyze_symbol(symbol)
                        
                        if result and result['llm_signal'] != 'SKIP':
                            logger.info(f"\n[SIGNAL] {symbol}")
                            logger.info(f"   Signal: {result['llm_signal']}")
                            logger.info(f"   Confidence: {result['llm_confidence']:.1%}")
                            logger.info(f"   Reasoning: {result['reasoning']}")
                            
                            # Here you would place the actual trade
                            # self.place_trade(result)
                        
                    except Exception as e:
                        logger.error(f"Error analyzing {symbol}: {e}", exc_info=True)
                        continue
                
                logger.info(f"\n[WAIT] Next scan in 15 seconds...")
                time.sleep(15)  # Wait 15 seconds
                
        except KeyboardInterrupt:
            logger.info("\n[STOP] Bot stopped by user")
        finally:
            mt5.shutdown()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ml_llm_bot.log'),
            logging.StreamHandler()
        ]
    )
    
    bot = MLLLMTradingBot()
    bot.run()

