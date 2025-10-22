"""
Multi-Asset Trading Bot - ENHANCED VERSION
==========================================
With detailed table logging and ultra-precise entry calculations
"""

import MetaTrader5 as mt5
import logging
import time
import yaml
from datetime import datetime
from pathlib import Path
from tabulate import tabulate

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
    Enhanced multi-asset trading bot with detailed logging
    """
    
    def __init__(self, config_path='config/config.yaml'):
        """Initialize bot with configuration"""
        self.config = self.load_config(config_path)
        self.running = False
        
        logger.info("🤖 Multi-Asset Trading Bot initialized (ENHANCED)")
    
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
            logger.info(f"   Balance: ${account_info.balance:,.2f}")
            logger.info(f"   Equity: ${account_info.equity:,.2f}")
            logger.info(f"   Margin Level: {account_info.margin_level:.2f}%")
            
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
            elif not symbol_info.visible:
                # Try to make it visible
                if not mt5.symbol_select(symbol, True):
                    logger.warning(f"⚠️  Cannot enable symbol: {symbol}")
                else:
                    valid_symbols.append(symbol)
            else:
                valid_symbols.append(symbol)
        
        return valid_symbols
    
    def get_session_info(self):
        """Get current trading session and parameters"""
        hour = datetime.utcnow().hour
        
        # Session times (UTC):
        # Asian: 00:00-08:00
        # London: 08:00-16:00
        # Overlap: 13:00-16:00
        # New York: 13:00-21:00
        
        if 0 <= hour < 8:
            return 'asian', 0.7, 65
        elif 8 <= hour < 13:
            return 'london', 0.9, 75
        elif 13 <= hour < 16:
            return 'overlap', 1.0, 80
        elif 16 <= hour < 21:
            return 'newyork', 0.9, 75
        else:
            return 'asian', 0.7, 65
    
    def check_margin_level(self):
        """Check if margin level is acceptable"""
        account_info = mt5.account_info()
        if account_info is None:
            return False
        
        margin_level = account_info.margin_level
        min_margin = self.config.get('risk_management', {}).get('min_margin_level', 700)
        
        if margin_level < min_margin:
            logger.warning(f"⚠️  Margin level too low: {margin_level:.2f}% < {min_margin}%")
            logger.warning(f"⚠️  STOPPING TRADING - Margin protection activated!")
            return False
        
        return True
    
    def calculate_precise_entry(self, symbol, signal, rates):
        """
        Calculate ultra-precise entry point using multiple indicators
        Returns: (entry_price, confidence, reason)
        """
        closes = [r['close'] for r in rates]
        highs = [r['high'] for r in rates]
        lows = [r['low'] for r in rates]
        opens = [r['open'] for r in rates]
        
        current_price = closes[-1]
        
        # Calculate multiple indicators for precision
        ma20 = sum(closes[-20:]) / 20
        ma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else ma20
        
        # Calculate momentum
        momentum = (current_price - closes[-10]) / closes[-10] * 100 if len(closes) >= 10 else 0
        
        # Calculate volatility (ATR approximation)
        tr_list = []
        for i in range(1, min(14, len(rates))):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_list.append(tr)
        atr = sum(tr_list) / len(tr_list) if tr_list else 0
        
        # Calculate RSI (14-period)
        rsi = 50  # Default neutral
        if len(closes) >= 15:
            gains = []
            losses = []
            for i in range(len(closes)-14, len(closes)):
                change = closes[i] - closes[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            avg_gain = sum(gains) / 14
            avg_loss = sum(losses) / 14
            
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
        
        # Determine confidence based on multiple factors
        confidence = 50  # Base confidence
        reasons = []
        
        # MA20 alignment
        if signal == 'BUY':
            if current_price > ma20:
                confidence += 15
                reasons.append("Above MA20")
            else:
                confidence -= 10
                reasons.append("Below MA20")
            
            # MA50 alignment
            if current_price > ma50:
                confidence += 10
                reasons.append("Above MA50")
            
            # RSI check
            if 30 < rsi < 70:
                confidence += 10
                reasons.append(f"RSI neutral ({rsi:.1f})")
            elif rsi < 30:
                confidence += 15
                reasons.append(f"RSI oversold ({rsi:.1f})")
            
            # Momentum check
            if momentum > 0:
                confidence += 10
                reasons.append(f"Positive momentum ({momentum:.2f}%)")
        
        else:  # SELL
            if current_price < ma20:
                confidence += 15
                reasons.append("Below MA20")
            else:
                confidence -= 10
                reasons.append("Above MA20")
            
            # MA50 alignment
            if current_price < ma50:
                confidence += 10
                reasons.append("Below MA50")
            
            # RSI check
            if 30 < rsi < 70:
                confidence += 10
                reasons.append(f"RSI neutral ({rsi:.1f})")
            elif rsi > 70:
                confidence += 15
                reasons.append(f"RSI overbought ({rsi:.1f})")
            
            # Momentum check
            if momentum < 0:
                confidence += 10
                reasons.append(f"Negative momentum ({momentum:.2f}%)")
        
        # Adjust entry price for precision (use bid/ask instead of last close)
        symbol_info = mt5.symbol_info_tick(symbol)
        if symbol_info:
            if signal == 'BUY':
                entry_price = symbol_info.ask  # Buy at ask
            else:
                entry_price = symbol_info.bid  # Sell at bid
        else:
            entry_price = current_price
        
        reason = ", ".join(reasons) if reasons else "Basic signal"
        
        return entry_price, min(confidence, 100), reason, atr
    
    def scan_opportunities(self, symbols):
        """Scan for trading opportunities with detailed table output"""
        logger.info("\n" + "="*100)
        logger.info("🔍 SCANNING FOR OPPORTUNITIES")
        logger.info("="*100)
        
        # Get current session info
        session, vol_mult, min_confidence = self.get_session_info()
        logger.info(f"🌍 Session: {session.upper()} | Vol Multiplier: {vol_mult}x | Min Confidence: {min_confidence}%")
        
        # Check margin level
        if not self.check_margin_level():
            logger.info("\n⛔ TRADING HALTED - Margin level too low\n")
            return []
        
        all_opportunities = []
        
        for symbol in symbols:
            # Get recent data
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
            
            if rates is None or len(rates) == 0:
                all_opportunities.append({
                    'symbol': symbol,
                    'signal': 'N/A',
                    'entry': 0,
                    'confidence': 0,
                    'status': 'NO DATA',
                    'reason': 'Failed to fetch market data'
                })
                continue
            
            # Calculate precise entry for BUY
            closes = [r['close'] for r in rates]
            ma20 = sum(closes[-20:]) / 20
            current_price = closes[-1]
            
            # Check both BUY and SELL possibilities
            if current_price > ma20:
                entry, confidence, reason, atr = self.calculate_precise_entry(symbol, 'BUY', rates)
                
                # Determine status
                if confidence >= min_confidence:
                    status = '✅ TRADE'
                else:
                    status = f'❌ SKIP (Conf: {confidence}% < {min_confidence}%)'
                
                all_opportunities.append({
                    'symbol': symbol,
                    'signal': 'BUY',
                    'entry': entry,
                    'confidence': confidence,
                    'status': status,
                    'reason': reason,
                    'atr': atr
                })
            
            elif current_price < ma20:
                entry, confidence, reason, atr = self.calculate_precise_entry(symbol, 'SELL', rates)
                
                # Determine status
                if confidence >= min_confidence:
                    status = '✅ TRADE'
                else:
                    status = f'❌ SKIP (Conf: {confidence}% < {min_confidence}%)'
                
                all_opportunities.append({
                    'symbol': symbol,
                    'signal': 'SELL',
                    'entry': entry,
                    'confidence': confidence,
                    'status': status,
                    'reason': reason,
                    'atr': atr
                })
            else:
                all_opportunities.append({
                    'symbol': symbol,
                    'signal': 'NEUTRAL',
                    'entry': current_price,
                    'confidence': 0,
                    'status': '⏸️  NEUTRAL',
                    'reason': 'Price at MA20',
                    'atr': 0
                })
        
        # Create detailed table
        table_data = []
        for opp in all_opportunities:
            table_data.append([
                opp['symbol'],
                opp['signal'],
                f"{opp['entry']:.5f}" if opp['entry'] > 0 else 'N/A',
                f"{opp['confidence']}%",
                opp['status'],
                opp['reason'][:50]  # Truncate long reasons
            ])
        
        headers = ['Symbol', 'Signal', 'Entry Price', 'Confidence', 'Status', 'Reason']
        
        print("\n" + tabulate(table_data, headers=headers, tablefmt='grid'))
        print()
        
        # Summary
        trade_count = sum(1 for opp in all_opportunities if '✅ TRADE' in opp['status'])
        skip_count = sum(1 for opp in all_opportunities if '❌ SKIP' in opp['status'])
        neutral_count = sum(1 for opp in all_opportunities if '⏸️  NEUTRAL' in opp['status'])
        
        logger.info(f"📊 SUMMARY: {trade_count} to trade | {skip_count} skipped | {neutral_count} neutral")
        
        # Return only opportunities that meet criteria
        return [opp for opp in all_opportunities if '✅ TRADE' in opp['status']]
    
    def execute_trade(self, opportunity):
        """Execute a trade with ultra-precise calculations"""
        symbol = opportunity['symbol']
        signal = opportunity['signal']
        entry_price = opportunity['entry']
        
        logger.info(f"\n💼 EXECUTING TRADE: {signal} {symbol}")
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"❌ Symbol info not available for {symbol}")
            return False
        
        # Get account info
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("❌ Account info not available")
            return False
        
        account_balance = account_info.balance
        risk_percent = self.config.get('risk_management', {}).get('risk_per_trade', 0.02)
        
        # ULTRA-PRECISE pip size calculation
        point = symbol_info.point
        digits = symbol_info.digits
        
        if 'JPY' in symbol:
            pip_size = 0.01  # Always 0.01 for JPY pairs
        elif 'XAU' in symbol or 'GOLD' in symbol:
            pip_size = 0.1 if digits == 2 else 0.01
        elif 'XAG' in symbol or 'SILVER' in symbol:
            pip_size = 0.01
        elif 'OIL' in symbol or 'WTI' in symbol or 'BRENT' in symbol:
            pip_size = 0.01
        elif 'BTC' in symbol:
            pip_size = 1.0
        elif 'ETH' in symbol:
            pip_size = 0.1
        elif 'LTC' in symbol or 'XRP' in symbol:
            pip_size = 0.01
        else:
            # Standard forex
            if digits == 5 or digits == 3:
                pip_size = point * 10
            else:
                pip_size = point
        
        # Get contract size
        contract_size = symbol_info.trade_contract_size
        
        # Calculate pip value
        if 'JPY' in symbol:
            pip_value_per_lot = (contract_size * pip_size) / entry_price
        else:
            pip_value_per_lot = contract_size * pip_size
        
        # Get session and ATR for dynamic SL/TP
        session, vol_mult, min_confidence = self.get_session_info()
        atr = opportunity.get('atr', 0)
        
        # Base SL/TP
        base_sl_pips = 25
        base_tp_pips = 50
        
        # Apply ATR-based adjustment if available
        if atr > 0:
            atr_pips = atr / pip_size
            # Use ATR but cap it
            base_sl_pips = min(max(20, atr_pips * 1.5), 40)
            base_tp_pips = base_sl_pips * 2
        
        # Apply session multiplier
        sl_pips = base_sl_pips * vol_mult
        tp_pips = base_tp_pips * vol_mult
        
        sl_distance = sl_pips * pip_size
        tp_distance = tp_pips * pip_size
        
        # Calculate lot size
        risk_amount = account_balance * risk_percent
        lot = risk_amount / (sl_pips * pip_value_per_lot)
        
        # Round to lot step
        lot_step = symbol_info.volume_step
        lot = round(lot / lot_step) * lot_step
        
        # Apply limits
        min_lot = symbol_info.volume_min
        max_lot = symbol_info.volume_max
        lot = max(min_lot, min(lot, max_lot))
        
        # Calculate SL/TP prices
        if signal == 'BUY':
            order_type = mt5.ORDER_TYPE_BUY
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:
            order_type = mt5.ORDER_TYPE_SELL
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance
        
        # Log details
        logger.info(f"📐 Entry: {entry_price:.5f}")
        logger.info(f"📐 SL: {sl:.5f} ({sl_pips:.1f} pips)")
        logger.info(f"📐 TP: {tp:.5f} ({tp_pips:.1f} pips)")
        logger.info(f"📐 Lot Size: {lot:.2f}")
        logger.info(f"📐 Risk: ${risk_amount:,.2f} ({risk_percent*100:.1f}%)")
        logger.info(f"📐 R:R = 1:{tp_pips/sl_pips:.1f}")
        
        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": entry_price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 234000,
            "comment": "Enhanced Bot",
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
        
        logger.info(f"✅ ORDER EXECUTED SUCCESSFULLY!")
        logger.info(f"   Ticket: {result.order}")
        
        return True
    
    def monitor_positions(self):
        """Monitor open positions"""
        positions = mt5.positions_get()
        
        if positions is None or len(positions) == 0:
            return
        
        logger.info(f"\n📊 Monitoring {len(positions)} positions...")
        
        table_data = []
        for position in positions:
            profit = position.profit
            symbol = position.symbol
            volume = position.volume
            price_open = position.price_open
            price_current = position.price_current
            
            status = "✅ Profit" if profit > 0 else "❌ Loss"
            
            table_data.append([
                position.ticket,
                symbol,
                "BUY" if position.type == 0 else "SELL",
                volume,
                f"{price_open:.5f}",
                f"{price_current:.5f}",
                f"${profit:.2f}",
                status
            ])
        
        headers = ['Ticket', 'Symbol', 'Type', 'Lot', 'Open', 'Current', 'P/L', 'Status']
        print("\n" + tabulate(table_data, headers=headers, tablefmt='grid'))
        print()
    
    def run(self):
        """Main bot loop"""
        logger.info("🚀 Starting Enhanced Multi-Asset Trading Bot...")
        
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
                for opp in opportunities:
                    self.execute_trade(opp)
                    time.sleep(2)  # Small delay between trades
                
                # Monitor positions
                self.monitor_positions()
                
                # Wait for next scan
                logger.info(f"\n⏰ Next scan in {scan_interval} seconds...\n")
                time.sleep(scan_interval)
                
        except KeyboardInterrupt:
            logger.info("\n⏹️  Bot stopped by user")
        except Exception as e:
            logger.error(f"❌ Error in main loop: {e}")
        finally:
            mt5.shutdown()
            logger.info("👋 Bot shutdown complete")


if __name__ == "__main__":
    bot = MultiAssetTradingBot()
    bot.run()

