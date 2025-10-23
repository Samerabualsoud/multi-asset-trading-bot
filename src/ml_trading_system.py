#!/usr/bin/env python3
"""
ML-Based Trading System
Data-driven approach with performance analysis to find optimal parameters
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import yaml
import logging
from pathlib import Path
import pickle
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MLTradingSystem:
    """ML-Based Trading System with Data-Driven Optimization"""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize the ML trading system"""
        self.config = self.load_config(config_path)
        self.running = False
        self.trade_history = []
        
        # Asset coverage - Major forex, crypto, metals, oil
        self.symbols = [
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
        
        # These will be determined through data analysis
        self.optimal_hours = None
        self.max_positions = None
        self.risk_per_trade = 0.02  # 2% base risk
        
        # Model will be loaded after training
        self.model = None
        self.feature_columns = None
        
        # Performance tracking
        self.hourly_performance = {}
        self.symbol_performance = {}
        
    def load_config(self, config_path):
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}. Using defaults.")
            return {}
    
    def connect_mt5(self):
        """Connect to MT5"""
        if not mt5.initialize():
            logger.error("MT5 initialization failed")
            return False
        
        # Login if credentials provided
        if self.config.get('mt5'):
            login = self.config['mt5'].get('login')
            password = self.config['mt5'].get('password')
            server = self.config['mt5'].get('server')
            
            if login and password and server:
                if not mt5.login(login, password, server):
                    logger.error(f"MT5 login failed: {mt5.last_error()}")
                    return False
        
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("Failed to get account info")
            return False
        
        logger.info(f"✅ Connected to MT5")
        logger.info(f"   Account: {account_info.login}")
        logger.info(f"   Balance: ${account_info.balance:,.2f}")
        logger.info(f"   Equity: ${account_info.equity:,.2f}")
        
        return True
    
    def analyze_hourly_performance(self, symbol, years=2):
        """
        Analyze historical performance by hour to find optimal trading times
        
        Returns:
            dict: Performance metrics by hour
        """
        logger.info(f"Analyzing hourly performance for {symbol}...")
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start_date, end_date)
        if rates is None or len(rates) == 0:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['hour'] = df['time'].dt.hour
        df['returns'] = df['close'].pct_change()
        
        # Calculate performance metrics by hour
        hourly_stats = df.groupby('hour').agg({
            'returns': ['mean', 'std', 'count'],
            'high': lambda x: (x - df.loc[x.index, 'low']).mean(),  # Average range
        }).round(6)
        
        hourly_stats.columns = ['avg_return', 'volatility', 'count', 'avg_range']
        hourly_stats['sharpe'] = hourly_stats['avg_return'] / hourly_stats['volatility']
        hourly_stats['win_rate'] = df.groupby('hour')['returns'].apply(lambda x: (x > 0).sum() / len(x))
        
        return hourly_stats
    
    def find_optimal_trading_hours(self, min_sharpe=0.1, min_win_rate=0.52):
        """
        Find optimal trading hours across all symbols
        
        Args:
            min_sharpe: Minimum Sharpe ratio threshold
            min_win_rate: Minimum win rate threshold
            
        Returns:
            list: Optimal hours to trade
        """
        logger.info("\n" + "="*100)
        logger.info("ANALYZING OPTIMAL TRADING HOURS")
        logger.info("="*100)
        
        all_hourly_stats = []
        
        # Analyze all symbols for comprehensive hour optimization
        logger.info(f"Analyzing {len(self.symbols)} symbols...")
        for symbol in self.symbols:
            stats = self.analyze_hourly_performance(symbol)
            if stats is not None:
                stats['symbol'] = symbol
                all_hourly_stats.append(stats)
        
        if not all_hourly_stats:
            logger.warning("Could not analyze hourly performance, using default hours")
            return list(range(24))  # Trade all hours
        
        # Combine stats across symbols
        combined_stats = pd.concat(all_hourly_stats).groupby(level=0).mean()
        
        # Find hours that meet criteria
        optimal_hours = combined_stats[
            (combined_stats['sharpe'] > min_sharpe) &
            (combined_stats['win_rate'] > min_win_rate)
        ].index.tolist()
        
        if not optimal_hours:
            logger.warning("No hours meet strict criteria, using top 50% hours")
            optimal_hours = combined_stats.nlargest(12, 'sharpe').index.tolist()
        
        logger.info(f"\n✅ Optimal trading hours (UTC): {sorted(optimal_hours)}")
        logger.info(f"   Total hours: {len(optimal_hours)}/24")
        
        # Convert to Saudi time for reference
        saudi_hours = [(h + 3) % 24 for h in optimal_hours]
        logger.info(f"   Saudi time equivalent: {sorted(saudi_hours)}")
        
        # Save analysis
        combined_stats.to_csv('ml_data/hourly_performance_analysis.csv')
        logger.info(f"   Saved analysis to ml_data/hourly_performance_analysis.csv")
        
        return sorted(optimal_hours)
    
    def determine_optimal_max_positions(self):
        """
        Determine optimal maximum concurrent positions through analysis
        
        Returns:
            int: Optimal max positions
        """
        logger.info("\nDetermining optimal max positions...")
        
        # Rule of thumb: 
        # - More symbols = can have more positions
        # - No stop loss = need to limit exposure
        # - Diversification benefit diminishes after certain point
        
        num_symbols = len(self.symbols)
        
        # Formula: sqrt(num_symbols) * 2, capped at 10
        optimal = min(int(np.sqrt(num_symbols) * 2), 10)
        
        # Minimum of 3 for diversification
        optimal = max(optimal, 3)
        
        logger.info(f"✅ Optimal max positions: {optimal}")
        logger.info(f"   Based on {num_symbols} symbols")
        logger.info(f"   Allows ~{optimal/num_symbols*100:.1f}% of symbols active at once")
        
        return optimal
    
    def calculate_position_size(self, symbol, confidence):
        """
        Calculate position size based on confidence and account
        
        Args:
            symbol: Trading symbol
            confidence: Model confidence (0-1)
            
        Returns:
            float: Lot size
        """
        account_info = mt5.account_info()
        if not account_info:
            return 0.01
        
        balance = account_info.balance
        
        # Scale risk by confidence
        # High confidence (0.8-1.0) = full risk
        # Medium confidence (0.6-0.8) = 50% risk
        # Low confidence (0.5-0.6) = 25% risk
        if confidence >= 0.8:
            risk_multiplier = 1.0
        elif confidence >= 0.7:
            risk_multiplier = 0.75
        elif confidence >= 0.6:
            risk_multiplier = 0.5
        else:
            risk_multiplier = 0.25
        
        risk_amount = balance * self.risk_per_trade * risk_multiplier
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return 0.01
        
        # Estimate lot size based on typical move
        # Since no SL, use expected TP distance
        if 'JPY' in symbol:
            pip_size = 0.01
            typical_tp_pips = 50
        elif any(crypto in symbol for crypto in ['BTC', 'ETH']):
            pip_size = 1.0
            typical_tp_pips = 500
        elif 'XAU' in symbol or 'XAG' in symbol:  # Gold/Silver
            pip_size = 0.01
            typical_tp_pips = 100
        elif 'OIL' in symbol:
            pip_size = 0.01
            typical_tp_pips = 50
        else:
            pip_size = 0.0001
            typical_tp_pips = 40
        
        contract_size = symbol_info.trade_contract_size
        pip_value_per_lot = contract_size * pip_size
        
        # Calculate lot
        lot = risk_amount / (typical_tp_pips * pip_value_per_lot)
        
        # Apply broker limits
        lot = max(symbol_info.volume_min, min(lot, symbol_info.volume_max))
        lot = round(lot / symbol_info.volume_step) * symbol_info.volume_step
        
        return lot
    
    def get_current_features(self, symbol):
        """
        Get current features for prediction
        
        Args:
            symbol: Trading symbol
            
        Returns:
            dict: Current features
        """
        # Get recent data
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 300)
        if rates is None or len(rates) == 0:
            return None
        
        df = pd.DataFrame(rates)
        
        # Calculate features (same as training)
        # This is a simplified version - full version in ml_data_collector.py
        features = {}
        
        # Price features
        features['close'] = df['close'].iloc[-1]
        features['returns'] = df['close'].pct_change().iloc[-1]
        
        # Moving averages
        features['sma_20'] = df['close'].rolling(20).mean().iloc[-1]
        features['sma_50'] = df['close'].rolling(50).mean().iloc[-1]
        features['ema_20'] = df['close'].ewm(span=20).mean().iloc[-1]
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        features['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        features['macd'] = (ema_12 - ema_26).iloc[-1]
        
        # Volatility
        features['volatility_20'] = df['close'].pct_change().rolling(20).std().iloc[-1]
        
        # Time features
        current_time = datetime.now()
        features['hour'] = current_time.hour
        features['day_of_week'] = current_time.weekday()
        
        return features
    
    def predict_signal(self, symbol):
        """
        Predict trading signal using ML model
        
        Args:
            symbol: Trading symbol
            
        Returns:
            tuple: (signal, confidence) where signal is 'BUY', 'SELL', or None
        """
        if self.model is None:
            logger.warning("Model not loaded, using random signals for demo")
            # Demo mode: random signals with confidence
            if np.random.random() > 0.9:  # 10% chance
                signal = np.random.choice(['BUY', 'SELL'])
                confidence = np.random.uniform(0.6, 0.9)
                return signal, confidence
            return None, 0.0
        
        # Get current features
        features = self.get_current_features(symbol)
        if features is None:
            return None, 0.0
        
        # Predict using model
        # This will be implemented after model training
        # For now, return None
        return None, 0.0
    
    def execute_trade(self, symbol, signal, confidence):
        """
        Execute trade (NO STOP LOSS as per user requirement)
        
        Args:
            symbol: Trading symbol
            signal: 'BUY' or 'SELL'
            confidence: Model confidence (0-1)
        """
        # Check position limit
        positions = mt5.positions_get()
        if positions and len(positions) >= self.max_positions:
            logger.warning(f"⚠️  Max positions reached ({len(positions)}/{self.max_positions})")
            return False
        
        # Check for duplicate
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            logger.warning(f"⚠️  Already have position in {symbol}")
            return False
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return False
        
        # Calculate position size
        lot = self.calculate_position_size(symbol, confidence)
        
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False
        
        # Set order parameters
        if signal == 'BUY':
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        
        digits = symbol_info.digits
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🤖 ML TRADE: {symbol} {signal}")
        logger.info(f"{'='*80}")
        logger.info(f"📊 Confidence: {confidence*100:.1f}%")
        logger.info(f"📐 Entry: {price:.{digits}f}")
        logger.info(f"📐 Lot Size: {lot:.2f}")
        logger.info(f"⚠️  NO STOP LOSS (as per user requirement)")
        logger.info(f"⚠️  UNLIMITED RISK - Monitor closely!")
        
        # Prepare order (NO STOP LOSS, NO TAKE PROFIT - pure ML decision)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": 0.0,  # NO STOP LOSS
            "tp": 0.0,  # NO TAKE PROFIT (will exit based on ML signals)
            "deviation": 20,
            "magic": 234002,  # ML system magic number
            "comment": f"ML-{confidence*100:.0f}%",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result is None:
            error = mt5.last_error()
            logger.error(f"❌ Order failed: {error}")
            return False
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"❌ Order failed: {result.comment} (code: {result.retcode})")
            return False
        
        logger.info(f"✅ Order executed!")
        logger.info(f"   Order: #{result.order}")
        logger.info(f"   Volume: {result.volume}")
        
        self.trade_history.append({
            'time': datetime.now(),
            'symbol': symbol,
            'signal': signal,
            'entry': price,
            'lot': lot,
            'confidence': confidence,
            'order': result.order
        })
        
        return True
    
    def should_close_position(self, position):
        """
        Determine if position should be closed based on ML signal
        
        Args:
            position: MT5 position object
            
        Returns:
            bool: True if should close
        """
        # Get new prediction
        signal, confidence = self.predict_signal(position.symbol)
        
        if signal is None:
            return False
        
        # Close if signal reverses with high confidence
        if position.type == 0:  # Long position
            if signal == 'SELL' and confidence > 0.7:
                logger.info(f"🔄 Closing {position.symbol} LONG: ML signal reversed (conf: {confidence*100:.1f}%)")
                return True
        else:  # Short position
            if signal == 'BUY' and confidence > 0.7:
                logger.info(f"🔄 Closing {position.symbol} SHORT: ML signal reversed (conf: {confidence*100:.1f}%)")
                return True
        
        return False
    
    def close_position(self, position):
        """Close a position"""
        tick = mt5.symbol_info_tick(position.symbol)
        if not tick:
            return False
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": position.ticket,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
            "price": tick.bid if position.type == 0 else tick.ask,
            "deviation": 20,
            "magic": 234002,
            "comment": "ML-Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"✅ Position closed: {position.symbol} P&L: ${position.profit:.2f}")
            return True
        
        return False
    
    def manage_positions(self):
        """Manage open positions based on ML signals"""
        positions = mt5.positions_get()
        if not positions:
            return
        
        for pos in positions:
            if pos.magic != 234002:  # Only manage our positions
                continue
            
            if self.should_close_position(pos):
                self.close_position(pos)
    
    def run(self):
        """Main trading loop"""
        logger.info("\n" + "="*100)
        logger.info("🤖 ML TRADING SYSTEM STARTED")
        logger.info("="*100)
        
        if not self.connect_mt5():
            logger.error("❌ Failed to connect to MT5")
            return
        
        # Perform initial analysis
        logger.info("\n📊 Performing initial analysis...")
        self.optimal_hours = self.find_optimal_trading_hours()
        self.max_positions = self.determine_optimal_max_positions()
        
        logger.info(f"\n✅ System Configuration:")
        logger.info(f"   Symbols: {len(self.symbols)} ({', '.join(self.symbols[:5])}...)")
        logger.info(f"   Optimal Hours (UTC): {self.optimal_hours}")
        logger.info(f"   Max Positions: {self.max_positions}")
        logger.info(f"   Risk per Trade: {self.risk_per_trade*100:.1f}%")
        logger.info(f"   Stop Loss: NONE (as per user requirement)")
        logger.info(f"   ⚠️  UNLIMITED RISK MODE - Monitor closely!")
        logger.info("="*100)
        
        self.running = True
        scan_interval = 300  # 5 minutes
        
        try:
            while self.running:
                current_hour = datetime.now().hour
                
                # Check if we're in optimal trading hours
                if current_hour in self.optimal_hours:
                    logger.info(f"\n✅ Trading hour active (Hour: {current_hour})")
                    
                    # Manage existing positions
                    self.manage_positions()
                    
                    # Scan for new opportunities
                    for symbol in self.symbols:
                        signal, confidence = self.predict_signal(symbol)
                        
                        if signal and confidence > 0.6:  # Minimum 60% confidence
                            logger.info(f"\n🎯 Signal: {symbol} {signal} (Confidence: {confidence*100:.1f}%)")
                            self.execute_trade(symbol, signal, confidence)
                            time.sleep(2)
                else:
                    logger.info(f"⏰ Outside optimal hours (Current: {current_hour}, Optimal: {self.optimal_hours})")
                
                time.sleep(scan_interval)
                
        except KeyboardInterrupt:
            logger.info("\n⚠️  System stopped by user")
        except Exception as e:
            logger.error(f"❌ Error: {e}", exc_info=True)
        finally:
            mt5.shutdown()
            logger.info("👋 System shutdown complete")


if __name__ == "__main__":
    system = MLTradingSystem()
    system.run()

