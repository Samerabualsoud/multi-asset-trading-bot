"""
Complete Multi-Asset Trading Bot
=================================
Production-ready bot with all strategies, risk management, and position monitoring
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
import time
import yaml
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'strategies'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'utils'))

# Import all modules
from indicators import EnhancedTechnicalIndicators
from market_analyzer import MarketAnalyzer
from risk_manager import EnhancedRiskManager
from position_monitor import PositionMonitor
from strategy_optimizer import StrategyOptimizer
from forex_strategies import ImprovedTradingStrategies
from crypto_strategies import CryptoTradingStrategies
from metals_strategies import MetalsTradingStrategies
from asset_detector import detect_asset_type
from config_validator import ConfigValidator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CompleteMultiAssetBot:
    """
    Complete trading bot with all strategies and risk management
    """
    
    def __init__(self, config_path='config/config.yaml'):
        """Initialize bot"""
        logger.info("=" * 60)
        logger.info("🤖 Initializing Complete Multi-Asset Trading Bot")
        logger.info("=" * 60)
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Validate configuration
        validator = ConfigValidator()
        is_valid, errors = validator.validate(self.config)
        if not is_valid:
            for error in errors:
                logger.error(f"[ERROR] Config error: {error}")
            raise ValueError("Invalid configuration")
        
        # Initialize components
        self.indicators = EnhancedTechnicalIndicators()
        self.market_analyzer = MarketAnalyzer()
        self.risk_manager = EnhancedRiskManager(self.config)
        self.position_monitor = PositionMonitor(self.config)
        self.strategy_optimizer = StrategyOptimizer(self.config)
        
        # Initialize strategy classes
        self.forex_strategies = ImprovedTradingStrategies(self.config)
        self.crypto_strategies = CryptoTradingStrategies(self.config)
        self.metals_strategies = MetalsTradingStrategies(self.config)
        
        self.running = False
        
        logger.info("[OK] Bot initialized successfully")
    
    def load_config(self, config_path):
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"[OK] Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"[ERROR] Config file not found: {config_path}")
            logger.info("💡 Create config/config.yaml from config/config.example.yaml")
            raise
        except Exception as e:
            logger.error(f"[ERROR] Error loading config: {e}")
            raise
    
    def connect_mt5(self):
        """Connect to MetaTrader 5"""
        try:
            # Initialize MT5
            if not mt5.initialize():
                logger.error(f"[ERROR] MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Login
            login = self.config.get('mt5_login')
            password = self.config.get('mt5_password')
            server = self.config.get('mt5_server')
            
            if not mt5.login(login, password, server):
                logger.error(f"[ERROR] MT5 login failed: {mt5.last_error()}")
                return False
            
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("[ERROR] Failed to get account info")
                return False
            
            logger.info("=" * 60)
            logger.info("[OK] Connected to MT5")
            logger.info(f"   Account: {account_info.login}")
            logger.info(f"   Server: {account_info.server}")
            logger.info(f"   Balance: ${account_info.balance:,.2f}")
            logger.info(f"   Equity: ${account_info.equity:,.2f}")
            logger.info(f"   Leverage: 1:{account_info.leverage}")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] MT5 connection error: {e}")
            return False
    
    def get_market_data(self, symbol: str, timeframe: int, bars: int = 500) -> Optional[pd.DataFrame]:
        """Get market data for a symbol"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"⚠️  No data for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            return df
            
        except Exception as e:
            logger.error(f"[ERROR] Error getting data for {symbol}: {e}")
            return None
    
    def analyze_symbol(self, symbol: str) -> List[Dict]:
        """Analyze a symbol and get trading opportunities"""
        try:
            # Detect asset type
            asset_type = detect_asset_type(symbol)
            
            logger.info(f"[ANALYZING] Analyzing {symbol} ({asset_type})...")
            
            # Get data for multiple timeframes
            df_m5 = self.get_market_data(symbol, mt5.TIMEFRAME_M5, 500)
            df_m15 = self.get_market_data(symbol, mt5.TIMEFRAME_M15, 500)
            df_h1 = self.get_market_data(symbol, mt5.TIMEFRAME_H1, 500)
            df_h4 = self.get_market_data(symbol, mt5.TIMEFRAME_H4, 500)
            
            if df_m15 is None or df_h1 is None:
                logger.warning(f"⚠️  Insufficient data for {symbol}")
                return []
            
            # Add indicators
            df_m15 = self.indicators.add_all_indicators(df_m15)
            df_h1 = self.indicators.add_all_indicators(df_h1)
            if df_h4 is not None:
                df_h4 = self.indicators.add_all_indicators(df_h4)
            
            # Get opportunities based on asset type
            opportunities = []
            
            if asset_type == 'forex':
                # Run all forex strategies
                strategies_to_run = [
                    ('trend_following', self.forex_strategies.strategy_1_trend_following),
                    ('fibonacci', self.forex_strategies.strategy_2_fibonacci_retracement),
                    ('mean_reversion', self.forex_strategies.strategy_3_mean_reversion),
                    ('breakout', self.forex_strategies.strategy_4_breakout),
                    ('momentum', self.forex_strategies.strategy_5_momentum),
                    ('multi_timeframe', self.forex_strategies.strategy_6_multi_timeframe_confluence),
                ]
                
                for strategy_name, strategy_func in strategies_to_run:
                    try:
                        # Call strategy based on its signature
                        if strategy_name in ['mean_reversion', 'momentum']:
                            signal, confidence, details = strategy_func(df_m15, symbol)
                        elif strategy_name == 'multi_timeframe':
                            if df_m5 is not None:
                                signal, confidence, details = strategy_func(df_m5, df_m15, df_h1, symbol)
                            else:
                                continue
                        else:
                            signal, confidence, details = strategy_func(df_m15, df_h1, symbol)
                        
                        if signal and confidence > 0:
                            # Apply strategy weight
                            weighted_confidence = self.strategy_optimizer.apply_weight(
                                symbol, strategy_name, confidence
                            )
                            
                            if weighted_confidence >= 65:
                                opportunities.append({
                                    'symbol': symbol,
                                    'signal': signal,
                                    'confidence': weighted_confidence,
                                    'strategy': strategy_name,
                                    'details': details,
                                    'asset_type': asset_type
                                })
                                logger.info(f"   [OK] {strategy_name}: {signal} ({weighted_confidence:.1f}%)")
                    except Exception as e:
                        logger.error(f"   [ERROR] Error in {strategy_name}: {e}")
            
            elif asset_type == 'crypto':
                # Run crypto strategies
                strategies_to_run = [
                    ('momentum_breakout', self.crypto_strategies.crypto_strategy_1_momentum_breakout),
                    ('support_resistance', self.crypto_strategies.crypto_strategy_2_support_resistance),
                    ('trend_following', self.crypto_strategies.crypto_strategy_3_trend_following),
                    ('volatility_breakout', self.crypto_strategies.crypto_strategy_4_volatility_breakout),
                ]
                
                for strategy_name, strategy_func in strategies_to_run:
                    try:
                        if strategy_name == 'trend_following' and df_h4 is not None:
                            signal, confidence, details = strategy_func(df_m15, df_h1, df_h4, symbol)
                        else:
                            signal, confidence, details = strategy_func(df_m15, df_h1, symbol)
                        
                        if signal and confidence > 0:
                            weighted_confidence = self.strategy_optimizer.apply_weight(
                                symbol, strategy_name, confidence
                            )
                            
                            if weighted_confidence >= 65:
                                opportunities.append({
                                    'symbol': symbol,
                                    'signal': signal,
                                    'confidence': weighted_confidence,
                                    'strategy': strategy_name,
                                    'details': details,
                                    'asset_type': asset_type
                                })
                                logger.info(f"   [OK] {strategy_name}: {signal} ({weighted_confidence:.1f}%)")
                    except Exception as e:
                        logger.error(f"   [ERROR] Error in {strategy_name}: {e}")
            
            elif asset_type == 'metal':
                # Run metals strategies
                strategies_to_run = [
                    ('safe_haven_flow', self.metals_strategies.metals_strategy_1_safe_haven_flow),
                    ('usd_correlation', self.metals_strategies.metals_strategy_2_usd_correlation),
                    ('technical_breakout', self.metals_strategies.metals_strategy_3_technical_breakout),
                ]
                
                for strategy_name, strategy_func in strategies_to_run:
                    try:
                        if strategy_name == 'usd_correlation':
                            # Need USD index data - skip for now or use proxy
                            continue
                        else:
                            if df_h4 is not None:
                                signal, confidence, details = strategy_func(df_h1, df_h4, symbol)
                            else:
                                continue
                        
                        if signal and confidence > 0:
                            weighted_confidence = self.strategy_optimizer.apply_weight(
                                symbol, strategy_name, confidence
                            )
                            
                            if weighted_confidence >= 65:
                                opportunities.append({
                                    'symbol': symbol,
                                    'signal': signal,
                                    'confidence': weighted_confidence,
                                    'strategy': strategy_name,
                                    'details': details,
                                    'asset_type': asset_type
                                })
                                logger.info(f"   [OK] {strategy_name}: {signal} ({weighted_confidence:.1f}%)")
                    except Exception as e:
                        logger.error(f"   [ERROR] Error in {strategy_name}: {e}")
            
            return opportunities
            
        except Exception as e:
            logger.error(f"[ERROR] Error analyzing {symbol}: {e}")
            return []
    
    def execute_trade(self, opportunity: Dict) -> bool:
        """Execute a trade based on opportunity"""
        try:
            symbol = opportunity['symbol']
            signal = opportunity['signal']
            confidence = opportunity['confidence']
            strategy = opportunity['strategy']
            details = opportunity['details']
            
            logger.info("=" * 60)
            logger.info(f"[TRADE] Executing Trade")
            logger.info(f"   Symbol: {symbol}")
            logger.info(f"   Signal: {signal}")
            logger.info(f"   Strategy: {strategy}")
            logger.info(f"   Confidence: {confidence:.1f}%")
            logger.info("=" * 60)
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error(f"[ERROR] Failed to get tick for {symbol}")
                return False
            
            price = tick.ask if signal == 'BUY' else tick.bid
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"[ERROR] Symbol info not available: {symbol}")
                return False
            
            # Extract SL/TP from strategy details
            sl_price = details.get('sl')
            tp_price = details.get('tp')
            
            if sl_price is None or tp_price is None:
                logger.error(f"[ERROR] Missing SL/TP in strategy details")
                return False
            
            # Calculate position size using risk manager
            account_info = mt5.account_info()
            if account_info is None:
                logger.error(f"[ERROR] Failed to get account info")
                return False
            
            # Calculate SL distance in pips
            point = symbol_info.point
            digits = symbol_info.digits
            
            if digits == 5 or digits == 3:
                pip_size = point * 10
            else:
                pip_size = point
            
            sl_distance_pips = abs(price - sl_price) / pip_size
            
            # Calculate lot size
            risk_percent = self.config.get('risk_management', {}).get('risk_per_trade', 0.005)
            risk_amount = account_info.balance * risk_percent
            
            contract_size = symbol_info.trade_contract_size
            if 'JPY' in symbol:
                pip_value_per_lot = contract_size * pip_size / price
            else:
                pip_value_per_lot = contract_size * pip_size
            
            lot = risk_amount / (sl_distance_pips * pip_value_per_lot)
            
            # Round and apply limits
            lot_step = symbol_info.volume_step
            lot = round(lot / lot_step) * lot_step
            lot = max(symbol_info.volume_min, min(lot, symbol_info.volume_max))
            
            logger.info(f"[INFO] Position Sizing:")
            logger.info(f"   Account: ${account_info.balance:,.2f}")
            logger.info(f"   Risk: {risk_percent*100:.2f}% = ${risk_amount:,.2f}")
            logger.info(f"   SL distance: {sl_distance_pips:.1f} pips")
            logger.info(f"   Lot size: {lot:.2f}")
            
            # Prepare order
            if signal == 'BUY':
                order_type = mt5.ORDER_TYPE_BUY
            else:
                order_type = mt5.ORDER_TYPE_SELL
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot,
                "type": order_type,
                "price": price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": 20,
                "magic": 234000,
                "comment": f"{strategy}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result is None:
                logger.error(f"[ERROR] Order failed: {mt5.last_error()}")
                return False
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"[ERROR] Order failed: {result.comment}")
                return False
            
            tp_distance_pips = abs(price - tp_price) / pip_size
            rr_ratio = tp_distance_pips / sl_distance_pips
            
            logger.info("[OK] Order Executed Successfully!")
            logger.info(f"   Entry: {price:.5f}")
            logger.info(f"   SL: {sl_price:.5f} ({sl_distance_pips:.1f} pips)")
            logger.info(f"   TP: {tp_price:.5f} ({tp_distance_pips:.1f} pips)")
            logger.info(f"   Risk:Reward: 1:{rr_ratio:.1f}")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Error executing trade: {e}")
            return False
    
    def run(self):
        """Main bot loop"""
        logger.info("🚀 Starting Complete Multi-Asset Trading Bot")
        
        # Connect to MT5
        if not self.connect_mt5():
            logger.error("[ERROR] Failed to connect to MT5. Exiting.")
            return
        
        # Get symbols
        symbols = self.config.get('symbols', [])
        if not symbols:
            logger.error("[ERROR] No symbols configured. Exiting.")
            return
        
        logger.info(f"[OK] Trading {len(symbols)} symbols: {', '.join(symbols)}")
        
        # Main loop
        self.running = True
        scan_interval = self.config.get('scan_interval_seconds', 300)  # 5 minutes default
        
        try:
            iteration = 0
            while self.running:
                iteration += 1
                logger.info("")
                logger.info("=" * 60)
                logger.info(f"[SCAN] Scan Iteration #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info("=" * 60)
                
                # Monitor existing positions
                positions = mt5.positions_get()
                if positions and len(positions) > 0:
                    logger.info(f"[INFO] Monitoring {len(positions)} open positions...")
                    self.position_monitor.monitor_positions()
                
                # Check if we can open new positions
                max_positions = self.config.get('risk_management', {}).get('max_positions', 5)
                current_positions = len(positions) if positions else 0
                
                if current_positions >= max_positions:
                    logger.info(f"⚠️  Max positions ({max_positions}) reached. Skipping new entries.")
                else:
                    # Scan all symbols
                    all_opportunities = []
                    
                    for symbol in symbols:
                        opportunities = self.analyze_symbol(symbol)
                        all_opportunities.extend(opportunities)
                    
                    if not all_opportunities:
                        logger.info("[INFO]  No trading opportunities found")
                    else:
                        logger.info(f"📈 Found {len(all_opportunities)} opportunities")
                        
                        # Rank opportunities by confidence
                        all_opportunities.sort(key=lambda x: x['confidence'], reverse=True)
                        
                        # Execute top opportunities
                        for opp in all_opportunities:
                            if current_positions >= max_positions:
                                break
                            
                            if self.execute_trade(opp):
                                current_positions += 1
                                time.sleep(2)  # Brief pause between trades
                
                # Wait for next scan
                logger.info(f"[WAIT] Next scan in {scan_interval} seconds...")
                time.sleep(scan_interval)
                
        except KeyboardInterrupt:
            logger.info("⚠️  Bot stopped by user")
        except Exception as e:
            logger.error(f"[ERROR] Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Shutdown bot"""
        logger.info("=" * 60)
        logger.info("🛑 Shutting down bot...")
        logger.info("=" * 60)
        mt5.shutdown()
        logger.info("[OK] Bot stopped successfully")


def main():
    """Main entry point"""
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║   Complete Multi-Asset Trading Bot v2.0              ║
    ║   Professional • Production-Ready • Profitable        ║
    ║                                                       ║
    ║   Assets: Forex • Crypto • Metals                    ║
    ║   Strategies: 13 Advanced Strategies                 ║
    ║   Features: Risk Management • Position Monitoring    ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    try:
        bot = CompleteMultiAssetBot()
        bot.run()
    except Exception as e:
        logger.error(f"[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

