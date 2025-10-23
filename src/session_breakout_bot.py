#!/usr/bin/env python3
"""
Session Breakout Trading Bot
Professional strategy focusing on Asian range breakout at London open
Optimized for 8:30-11:30 AM Saudi time (good morning session)
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import yaml
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('session_breakout_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SessionBreakoutBot:
    """Professional Session Breakout Trading Bot"""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize the bot"""
        self.config = self.load_config(config_path)
        self.running = False
        self.trade_history = []
        self.daily_pnl = 0.0
        self.daily_trades = 0
        
        # Trading parameters
        self.symbols = ['EURUSD', 'GBPUSD', 'USDJPY']  # Major pairs only
        self.max_positions = 2  # Maximum 2 concurrent positions
        self.risk_per_trade = 0.01  # 1% risk per trade
        self.stop_loss_pips = 25  # 25 pip stop loss
        self.take_profit_pips = 50  # 50 pip take profit (2:1 R:R)
        self.trailing_start_pips = 30  # Start trailing after 30 pips profit
        self.trailing_distance_pips = 20  # Trail by 20 pips
        self.max_daily_loss_percent = 0.03  # 3% max daily loss
        
        # Session times (UTC)
        self.asian_start_hour = 0  # 00:00 UTC
        self.asian_end_hour = 8    # 08:00 UTC
        self.london_start_hour = 8  # 08:00 UTC
        self.london_end_hour = 12   # 12:00 UTC (11:30 AM Saudi = ~08:30 UTC)
        
        # Saudi time offset (UTC+3)
        self.saudi_offset = 3
        
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
    
    def is_good_morning_session(self):
        """Check if we're in the good morning session (8:30-11:30 AM Saudi time)"""
        now = datetime.utcnow()
        saudi_hour = (now.hour + self.saudi_offset) % 24
        saudi_minute = now.minute
        
        # 8:30 AM to 11:30 AM Saudi time
        if saudi_hour == 8 and saudi_minute >= 30:
            return True
        elif 9 <= saudi_hour < 11:
            return True
        elif saudi_hour == 11 and saudi_minute <= 30:
            return True
        
        return False
    
    def get_asian_range(self, symbol):
        """Get Asian session high and low"""
        now = datetime.utcnow()
        
        # Get data from Asian session (00:00 - 08:00 UTC)
        asian_start = now.replace(hour=self.asian_start_hour, minute=0, second=0, microsecond=0)
        asian_end = now.replace(hour=self.asian_end_hour, minute=0, second=0, microsecond=0)
        
        # If current time is before Asian end, use previous day
        if now.hour < self.asian_end_hour:
            asian_start -= timedelta(days=1)
            asian_end -= timedelta(days=1)
        
        # Get M15 data for Asian session
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, asian_start, asian_end)
        
        if rates is None or len(rates) == 0:
            return None, None, None
        
        df = pd.DataFrame(rates)
        
        asian_high = df['high'].max()
        asian_low = df['low'].min()
        range_size = asian_high - asian_low
        
        return asian_high, asian_low, range_size
    
    def check_breakout(self, symbol, asian_high, asian_low):
        """Check if price has broken out of Asian range"""
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return None, None
        
        current_price = tick.last
        
        # Check for bullish breakout
        if current_price > asian_high:
            return 'BUY', current_price
        
        # Check for bearish breakout
        if current_price < asian_low:
            return 'SELL', current_price
        
        return None, current_price
    
    def calculate_lot_size(self, symbol, stop_loss_pips):
        """Calculate lot size based on risk"""
        account_info = mt5.account_info()
        if not account_info:
            return 0.01
        
        balance = account_info.balance
        risk_amount = balance * self.risk_per_trade
        
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return 0.01
        
        # Calculate pip value
        if 'JPY' in symbol:
            pip_size = 0.01
        else:
            pip_size = 0.0001
        
        contract_size = symbol_info.trade_contract_size
        pip_value_per_lot = contract_size * pip_size
        
        # Calculate lot size
        lot = risk_amount / (stop_loss_pips * pip_value_per_lot)
        
        # Apply broker limits
        lot = max(symbol_info.volume_min, min(lot, symbol_info.volume_max))
        lot = round(lot / symbol_info.volume_step) * symbol_info.volume_step
        
        return lot
    
    def execute_trade(self, symbol, signal, entry_price, asian_high, asian_low):
        """Execute breakout trade"""
        # Check position limit
        positions = mt5.positions_get()
        if positions and len(positions) >= self.max_positions:
            logger.warning(f"⚠️  Maximum positions reached ({len(positions)}/{self.max_positions})")
            return False
        
        # Check for duplicate position
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            logger.warning(f"⚠️  Already have position in {symbol}")
            return False
        
        # Check daily loss limit
        account_info = mt5.account_info()
        if not account_info:
            return False
        
        daily_loss_limit = account_info.balance * self.max_daily_loss_percent
        if self.daily_pnl < -daily_loss_limit:
            logger.error(f"❌ Daily loss limit reached (${self.daily_pnl:.2f})")
            return False
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return False
        
        # Calculate pip size
        if 'JPY' in symbol:
            pip_size = 0.01
        else:
            pip_size = 0.0001
        
        # Calculate lot size
        lot = self.calculate_lot_size(symbol, self.stop_loss_pips)
        
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False
        
        # Set order parameters
        if signal == 'BUY':
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
            # Stop loss just below Asian low
            sl_price = asian_low - (5 * pip_size)  # 5 pips buffer
            # Take profit
            tp_price = price + (self.take_profit_pips * pip_size)
        else:  # SELL
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
            # Stop loss just above Asian high
            sl_price = asian_high + (5 * pip_size)  # 5 pips buffer
            # Take profit
            tp_price = price - (self.take_profit_pips * pip_size)
        
        # Round prices
        digits = symbol_info.digits
        sl_price = round(sl_price, digits)
        tp_price = round(tp_price, digits)
        price = round(price, digits)
        
        # Calculate actual SL distance
        if signal == 'BUY':
            sl_pips = (price - sl_price) / pip_size
        else:
            sl_pips = (sl_price - price) / pip_size
        
        # Calculate risk
        risk_amount = account_info.balance * self.risk_per_trade
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 SESSION BREAKOUT TRADE: {symbol} {signal}")
        logger.info(f"{'='*80}")
        logger.info(f"📊 Asian Range: {asian_low:.{digits}f} - {asian_high:.{digits}f}")
        logger.info(f"📐 Entry: {price:.{digits}f}")
        logger.info(f"📐 Stop Loss: {sl_price:.{digits}f} ({sl_pips:.1f} pips)")
        logger.info(f"📐 Take Profit: {tp_price:.{digits}f} ({self.take_profit_pips} pips)")
        logger.info(f"📐 Lot Size: {lot:.2f}")
        logger.info(f"📐 Risk: ${risk_amount:.2f} ({self.risk_per_trade*100:.1f}%)")
        logger.info(f"📐 R:R Ratio: 1:{self.take_profit_pips/sl_pips:.1f}")
        
        # Prepare order
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 20,
            "magic": 234001,  # Different magic number
            "comment": "Session Breakout",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result is None:
            error = mt5.last_error()
            logger.error(f"❌ Order failed: MT5 error {error}")
            return False
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"❌ Order failed: {result.comment} (code: {result.retcode})")
            return False
        
        logger.info(f"✅ Order executed successfully!")
        logger.info(f"   Order: #{result.order}")
        logger.info(f"   Volume: {result.volume}")
        logger.info(f"   Price: {result.price:.{digits}f}")
        
        self.trade_history.append({
            'time': datetime.now(),
            'symbol': symbol,
            'signal': signal,
            'entry': price,
            'sl': sl_price,
            'tp': tp_price,
            'lot': lot,
            'status': '✅ EXECUTED',
            'order': result.order
        })
        
        self.daily_trades += 1
        
        return True
    
    def manage_trailing_stops(self):
        """Manage trailing stops for open positions"""
        positions = mt5.positions_get()
        if not positions:
            return
        
        for pos in positions:
            # Only manage our positions
            if pos.magic != 234001:
                continue
            
            symbol = pos.symbol
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                continue
            
            # Calculate pip size
            if 'JPY' in symbol:
                pip_size = 0.01
            else:
                pip_size = 0.0001
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                continue
            
            current_price = tick.bid if pos.type == 0 else tick.ask
            
            # Calculate profit in pips
            if pos.type == 0:  # BUY
                profit_pips = (current_price - pos.price_open) / pip_size
            else:  # SELL
                profit_pips = (pos.price_open - current_price) / pip_size
            
            # Check if we should start trailing
            if profit_pips >= self.trailing_start_pips:
                # Calculate new stop loss
                if pos.type == 0:  # BUY
                    new_sl = current_price - (self.trailing_distance_pips * pip_size)
                else:  # SELL
                    new_sl = current_price + (self.trailing_distance_pips * pip_size)
                
                new_sl = round(new_sl, symbol_info.digits)
                
                # Only move SL in profit direction
                if pos.type == 0:  # BUY
                    if new_sl > pos.sl:
                        self.modify_position(pos, new_sl, pos.tp)
                else:  # SELL
                    if new_sl < pos.sl:
                        self.modify_position(pos, new_sl, pos.tp)
    
    def modify_position(self, position, new_sl, new_tp):
        """Modify position stop loss and take profit"""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": position.ticket,
            "sl": new_sl,
            "tp": new_tp,
        }
        
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"✅ Trailing stop updated for {position.symbol}: SL={new_sl}")
        else:
            logger.warning(f"⚠️  Failed to update trailing stop for {position.symbol}")
    
    def update_daily_pnl(self):
        """Update daily P&L"""
        positions = mt5.positions_get()
        if not positions:
            self.daily_pnl = 0.0
            return
        
        total_profit = sum(pos.profit for pos in positions if pos.magic == 234001)
        self.daily_pnl = total_profit
    
    def scan_opportunities(self):
        """Scan for breakout opportunities"""
        # Check if we're in good morning session
        if not self.is_good_morning_session():
            return []
        
        opportunities = []
        
        for symbol in self.symbols:
            # Get Asian range
            asian_high, asian_low, range_size = self.get_asian_range(symbol)
            
            if asian_high is None or asian_low is None:
                continue
            
            # Check for breakout
            signal, current_price = self.check_breakout(symbol, asian_high, asian_low)
            
            if signal:
                opportunities.append({
                    'symbol': symbol,
                    'signal': signal,
                    'current_price': current_price,
                    'asian_high': asian_high,
                    'asian_low': asian_low,
                    'range_size': range_size
                })
        
        return opportunities
    
    def run(self):
        """Main trading loop"""
        logger.info("\n" + "="*100)
        logger.info("🚀 SESSION BREAKOUT BOT STARTED")
        logger.info("="*100)
        logger.info("📊 Strategy: Asian Range Breakout at London Open")
        logger.info("⏰ Trading Hours: 8:30-11:30 AM Saudi Time (Good Morning Session)")
        logger.info(f"📈 Pairs: {', '.join(self.symbols)}")
        logger.info(f"💰 Risk per Trade: {self.risk_per_trade*100:.1f}%")
        logger.info(f"🛑 Stop Loss: {self.stop_loss_pips} pips")
        logger.info(f"🎯 Take Profit: {self.take_profit_pips} pips (R:R 1:2)")
        logger.info(f"📊 Max Positions: {self.max_positions}")
        logger.info(f"⚠️  Daily Loss Limit: {self.max_daily_loss_percent*100:.1f}%")
        logger.info("="*100)
        
        if not self.connect_mt5():
            logger.error("❌ Failed to connect to MT5")
            return
        
        self.running = True
        scan_interval = 60  # Check every minute
        last_day = datetime.now().day
        
        try:
            while self.running:
                current_time = datetime.now()
                
                # Reset daily counters at midnight
                if current_time.day != last_day:
                    self.daily_pnl = 0.0
                    self.daily_trades = 0
                    last_day = current_time.day
                    logger.info("\n🌅 New trading day started")
                
                # Update daily P&L
                self.update_daily_pnl()
                
                # Manage trailing stops
                self.manage_trailing_stops()
                
                # Check if we're in trading session
                if self.is_good_morning_session():
                    logger.info(f"\n🌅 GOOD MORNING SESSION ACTIVE")
                    logger.info(f"💰 Daily P&L: ${self.daily_pnl:.2f} | Trades: {self.daily_trades}")
                    
                    # Scan for opportunities
                    opportunities = self.scan_opportunities()
                    
                    if opportunities:
                        for opp in opportunities:
                            logger.info(f"\n🎯 BREAKOUT DETECTED: {opp['symbol']} {opp['signal']}")
                            logger.info(f"   Asian Range: {opp['asian_low']:.5f} - {opp['asian_high']:.5f}")
                            logger.info(f"   Current Price: {opp['current_price']:.5f}")
                            
                            self.execute_trade(
                                opp['symbol'],
                                opp['signal'],
                                opp['current_price'],
                                opp['asian_high'],
                                opp['asian_low']
                            )
                            
                            time.sleep(2)
                else:
                    # Outside trading hours
                    saudi_time = datetime.utcnow() + timedelta(hours=self.saudi_offset)
                    logger.info(f"⏰ Outside trading hours (Saudi time: {saudi_time.strftime('%H:%M')})")
                    logger.info(f"   Waiting for 8:30-11:30 AM Saudi time...")
                
                time.sleep(scan_interval)
                
        except KeyboardInterrupt:
            logger.info("\n⚠️  Bot stopped by user")
        except Exception as e:
            logger.error(f"❌ Error: {e}", exc_info=True)
        finally:
            mt5.shutdown()
            logger.info("👋 Bot shutdown complete")


if __name__ == "__main__":
    bot = SessionBreakoutBot()
    bot.run()

