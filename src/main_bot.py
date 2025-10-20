"""
Multi-Asset Trading Bot
=======================
Simple, working bot for Forex, Crypto, and Metals trading
"""

import MetaTrader5 as mt5
import logging
import time
import yaml
from datetime import datetime
from pathlib import Path

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


class MultiAssetTradingBot:
    """
    Simple multi-asset trading bot
    """
    
    def __init__(self, config_path='config/config.yaml'):
        """Initialize bot with configuration"""
        self.config = self.load_config(config_path)
        self.running = False
        
        logger.info("🤖 Multi-Asset Trading Bot initialized")
    
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"❌ Config file not found: {config_path}")
            logger.info("💡 Create config/config.yaml from config/config.example.yaml")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}")
            raise
    
    def connect_mt5(self):
        """Connect to MetaTrader 5"""
        try:
            # Initialize MT5
            if not mt5.initialize():
                logger.error(f"❌ MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Login
            login = self.config.get('mt5_login')
            password = self.config.get('mt5_password')
            server = self.config.get('mt5_server')
            
            if not mt5.login(login, password, server):
                logger.error(f"❌ MT5 login failed: {mt5.last_error()}")
                return False
            
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("❌ Failed to get account info")
                return False
            
            logger.info(f"✅ Connected to MT5")
            logger.info(f"   Account: {account_info.login}")
            logger.info(f"   Server: {account_info.server}")
            logger.info(f"   Balance: ${account_info.balance:.2f}")
            logger.info(f"   Equity: ${account_info.equity:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ MT5 connection error: {e}")
            return False
    
    def get_symbols(self):
        """Get list of symbols to trade"""
        symbols = self.config.get('symbols', [])
        if not symbols:
            logger.warning("⚠️  No symbols configured")
            return []
        
        # Verify symbols exist
        valid_symbols = []
        for symbol in symbols:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.warning(f"⚠️  Symbol not found: {symbol}")
            else:
                valid_symbols.append(symbol)
                logger.info(f"✅ Symbol added: {symbol}")
        
        return valid_symbols
    
    def scan_opportunities(self, symbols):
        """Scan for trading opportunities"""
        logger.info("🔍 Scanning for opportunities...")
        
        opportunities = []
        
        for symbol in symbols:
            # Get recent data
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"⚠️  No data for {symbol}")
                continue
            
            # Simple example: Check if price is above MA20
            closes = [r['close'] for r in rates]
            ma20 = sum(closes[-20:]) / 20
            current_price = closes[-1]
            
            if current_price > ma20:
                opportunities.append({
                    'symbol': symbol,
                    'signal': 'BUY',
                    'price': current_price,
                    'confidence': 65
                })
                logger.info(f"📈 BUY opportunity: {symbol} @ {current_price:.5f}")
            elif current_price < ma20:
                opportunities.append({
                    'symbol': symbol,
                    'signal': 'SELL',
                    'price': current_price,
                    'confidence': 65
                })
                logger.info(f"📉 SELL opportunity: {symbol} @ {current_price:.5f}")
        
        return opportunities
    
    def execute_trade(self, opportunity):
        """Execute a trade"""
        symbol = opportunity['symbol']
        signal = opportunity['signal']
        price = opportunity['price']
        
        logger.info(f"💼 Executing {signal} on {symbol}")
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"❌ Symbol info not available: {symbol}")
            return False
        
        # Get account info
        account_info = mt5.account_info()
        if account_info is None:
            logger.error(f"❌ Failed to get account info")
            return False
        
        account_balance = account_info.balance
        
        # Get risk percentage from config (default 0.5%)
        risk_percent = self.config.get('risk_management', {}).get('risk_per_trade', 0.005)
        
        # Calculate position size based on account balance and risk
        # Formula: Lot Size = (Account Balance × Risk %) / (SL in pips × Pip Value)
        
        # Get contract size (standard lot size)
        contract_size = symbol_info.trade_contract_size
        
        # Calculate pip value for 1 standard lot
        # For most pairs: pip_value = contract_size × pip_size
        # For JPY pairs and others, we need to adjust
        if 'JPY' in symbol:
            pip_value_per_lot = contract_size * pip_size / price
        else:
            pip_value_per_lot = contract_size * pip_size
        
        # Calculate SL/TP properly
        point = symbol_info.point
        digits = symbol_info.digits
        
        # Determine pip size based on digits
        if digits == 5 or digits == 3:
            # 5-digit broker (e.g., 1.23456) or 3-digit (JPY)
            pip_size = point * 10
        else:
            # 4-digit broker (e.g., 1.2345) or 2-digit (JPY)
            pip_size = point
        
        # Set SL/TP in pips (not points!)
        sl_pips = 30  # 30 pips stop loss
        tp_pips = 60  # 60 pips take profit (1:2 ratio)
        
        sl_distance = sl_pips * pip_size
        tp_distance = tp_pips * pip_size
        
        # Calculate lot size based on risk
        risk_amount = account_balance * risk_percent
        lot = risk_amount / (sl_pips * pip_value_per_lot)
        
        # Round to broker's lot step (usually 0.01)
        lot_step = symbol_info.volume_step
        lot = round(lot / lot_step) * lot_step
        
        # Apply min/max limits
        min_lot = symbol_info.volume_min
        max_lot = symbol_info.volume_max
        lot = max(min_lot, min(lot, max_lot))
        
        logger.info(f"📊 Position sizing:")
        logger.info(f"   Account balance: ${account_balance:,.2f}")
        logger.info(f"   Risk per trade: {risk_percent*100:.2f}% = ${risk_amount:,.2f}")
        logger.info(f"   SL: {sl_pips} pips")
        logger.info(f"   Calculated lot size: {lot:.2f}")
        
        if signal == 'BUY':
            order_type = mt5.ORDER_TYPE_BUY
            sl = price - sl_distance
            tp = price + tp_distance
        else:
            order_type = mt5.ORDER_TYPE_SELL
            sl = price + sl_distance
            tp = price - tp_distance
        
        # Prepare request
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
            "comment": "Multi-Asset Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result is None:
            logger.error(f"❌ Order failed: {mt5.last_error()}")
            return False
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"❌ Order failed: {result.comment}")
            return False
        
        logger.info(f"✅ Order executed: {signal} {lot} {symbol}")
        logger.info(f"   Entry: {price:.5f}")
        logger.info(f"   SL: {sl:.5f} ({sl_pips} pips)")
        logger.info(f"   TP: {tp:.5f} ({tp_pips} pips)")
        logger.info(f"   Risk:Reward = 1:{tp_pips/sl_pips:.1f}")
        
        return True
    
    def monitor_positions(self):
        """Monitor open positions"""
        positions = mt5.positions_get()
        
        if positions is None or len(positions) == 0:
            return
        
        logger.info(f"📊 Monitoring {len(positions)} positions...")
        
        for position in positions:
            profit = position.profit
            symbol = position.symbol
            
            if profit > 0:
                logger.info(f"   {symbol}: +${profit:.2f} ✅")
            else:
                logger.info(f"   {symbol}: ${profit:.2f} ❌")
    
    def run(self):
        """Main bot loop"""
        logger.info("🚀 Starting Multi-Asset Trading Bot...")
        
        # Connect to MT5
        if not self.connect_mt5():
            logger.error("❌ Failed to connect to MT5. Exiting.")
            return
        
        # Get symbols
        symbols = self.get_symbols()
        if not symbols:
            logger.error("❌ No valid symbols. Exiting.")
            return
        
        logger.info(f"✅ Trading {len(symbols)} symbols: {', '.join(symbols)}")
        
        # Main loop
        self.running = True
        scan_interval = self.config.get('scan_interval_seconds', 60)
        
        try:
            while self.running:
                # Scan for opportunities
                opportunities = self.scan_opportunities(symbols)
                
                # Execute trades
                max_positions = self.config.get('risk_management', {}).get('max_positions', 5)
                current_positions = len(mt5.positions_get() or [])
                
                for opp in opportunities:
                    if current_positions >= max_positions:
                        logger.info(f"⚠️  Max positions ({max_positions}) reached")
                        break
                    
                    if opp['confidence'] >= 65:
                        self.execute_trade(opp)
                        current_positions += 1
                
                # Monitor positions
                self.monitor_positions()
                
                # Wait
                logger.info(f"⏳ Waiting {scan_interval} seconds...")
                time.sleep(scan_interval)
                
        except KeyboardInterrupt:
            logger.info("⚠️  Bot stopped by user")
        except Exception as e:
            logger.error(f"❌ Error in main loop: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Shutdown bot"""
        logger.info("🛑 Shutting down bot...")
        mt5.shutdown()
        logger.info("✅ Bot stopped")


def main():
    """Main entry point"""
    print("""
    ╔═══════════════════════════════════════╗
    ║   Multi-Asset Trading Bot v2.0        ║
    ║   Forex • Crypto • Metals             ║
    ╚═══════════════════════════════════════╝
    """)
    
    try:
        bot = MultiAssetTradingBot()
        bot.run()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

