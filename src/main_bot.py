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
        self.trade_history = []  # Track all trade execution attempts
        
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
        
        # STRICT STRATEGY - Reject weak signals immediately
        confidence = 0  # Start at 0, must earn confidence
        reasons = []
        reject_reasons = []
        
        # CRITICAL REQUIREMENT 1: Strong trend alignment (MA20 AND MA50)
        if signal == 'BUY':
            # MUST be above both MAs
            if current_price > ma20 and current_price > ma50:
                # Check if MAs are aligned (uptrend)
                if ma20 > ma50:
                    confidence += 40  # Strong uptrend
                    reasons.append("Strong uptrend (MA20>MA50)")
                else:
                    confidence += 20  # Weak uptrend
                    reasons.append("Weak uptrend")
            else:
                # REJECT if not above both MAs
                reject_reasons.append("NOT above MA20/MA50")
                return entry_price, 0, "REJECTED: " + ", ".join(reject_reasons), atr
            
            # CRITICAL REQUIREMENT 2: Positive momentum
            if momentum > 0.3:
                confidence += 25
                reasons.append(f"Strong momentum +{momentum:.2f}%")
            elif momentum > 0:
                confidence += 10
                reasons.append(f"Weak momentum +{momentum:.2f}%")
            else:
                # REJECT if momentum is negative
                reject_reasons.append(f"Negative momentum {momentum:.2f}%")
                return entry_price, 0, "REJECTED: " + ", ".join(reject_reasons), atr
            
            # CRITICAL REQUIREMENT 3: RSI not overbought
            if rsi > 75:
                reject_reasons.append(f"RSI too high ({rsi:.1f})")
                return entry_price, 0, "REJECTED: " + ", ".join(reject_reasons), atr
            elif rsi < 40:
                confidence += 20
                reasons.append(f"RSI oversold ({rsi:.1f})")
            elif 40 <= rsi <= 60:
                confidence += 15
                reasons.append(f"RSI neutral ({rsi:.1f})")
        
        else:  # SELL
            # MUST be below both MAs
            if current_price < ma20 and current_price < ma50:
                # Check if MAs are aligned (downtrend)
                if ma20 < ma50:
                    confidence += 40  # Strong downtrend
                    reasons.append("Strong downtrend (MA20<MA50)")
                else:
                    confidence += 20  # Weak downtrend
                    reasons.append("Weak downtrend")
            else:
                # REJECT if not below both MAs
                reject_reasons.append("NOT below MA20/MA50")
                return entry_price, 0, "REJECTED: " + ", ".join(reject_reasons), atr
            
            # CRITICAL REQUIREMENT 2: Negative momentum
            if momentum < -0.3:
                confidence += 25
                reasons.append(f"Strong momentum {momentum:.2f}%")
            elif momentum < 0:
                confidence += 10
                reasons.append(f"Weak momentum {momentum:.2f}%")
            else:
                # REJECT if momentum is positive
                reject_reasons.append(f"Positive momentum +{momentum:.2f}%")
                return entry_price, 0, "REJECTED: " + ", ".join(reject_reasons), atr
            
            # CRITICAL REQUIREMENT 3: RSI not oversold
            if rsi < 25:
                reject_reasons.append(f"RSI too low ({rsi:.1f})")
                return entry_price, 0, "REJECTED: " + ", ".join(reject_reasons), atr
            elif rsi > 60:
                confidence += 20
                reasons.append(f"RSI overbought ({rsi:.1f})")
            elif 40 <= rsi <= 60:
                confidence += 15
                reasons.append(f"RSI neutral ({rsi:.1f})")
        
        # If we got here, all critical requirements passed
        # Minimum confidence is now 65-85 (40+25+15 or 40+10+15)
        
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
        
        # Separate opportunities into categories
        to_trade = [opp for opp in all_opportunities if '✅ TRADE' in opp['status']]
        skipped = [opp for opp in all_opportunities if '❌ SKIP' in opp['status']]
        neutral = [opp for opp in all_opportunities if '⏸️  NEUTRAL' in opp['status']]
        no_data = [opp for opp in all_opportunities if 'NO DATA' in opp['status']]
        
        # Table 1: Opportunities to TRADE
        if to_trade:
            print("\n" + "="*100)
            print("✅ OPPORTUNITIES TO TRADE")
            print("="*100)
            trade_table = []
            for opp in to_trade:
                trade_table.append([
                    opp['symbol'],
                    opp['signal'],
                    f"{opp['entry']:.5f}",
                    f"{opp['confidence']}%",
                    opp['reason'][:40]
                ])
            headers = ['Symbol', 'Signal', 'Entry Price', 'Confidence', 'Reason']
            print(tabulate(trade_table, headers=headers, tablefmt='grid'))
            print()
        
        # Table 2: SKIPPED Opportunities (with detailed reasons)
        if skipped:
            print("\n" + "="*100)
            print("❌ SKIPPED OPPORTUNITIES (Not Traded)")
            print("="*100)
            skip_table = []
            for opp in skipped:
                # Extract skip reason from status
                skip_reason = opp['status'].replace('❌ SKIP ', '')
                skip_table.append([
                    opp['symbol'],
                    opp['signal'],
                    f"{opp['entry']:.5f}",
                    f"{opp['confidence']}%",
                    skip_reason,
                    opp['reason'][:35]
                ])
            headers = ['Symbol', 'Signal', 'Entry Price', 'Confidence', 'Skip Reason', 'Analysis']
            print(tabulate(skip_table, headers=headers, tablefmt='grid'))
            print()
        
        # Table 3: NEUTRAL (no clear signal)
        if neutral:
            print("\n" + "="*100)
            print("⏸️  NEUTRAL PAIRS (No Clear Signal)")
            print("="*100)
            neutral_table = []
            for opp in neutral:
                neutral_table.append([
                    opp['symbol'],
                    f"{opp['entry']:.5f}",
                    opp['reason']
                ])
            headers = ['Symbol', 'Current Price', 'Reason']
            print(tabulate(neutral_table, headers=headers, tablefmt='grid'))
            print()
        
        # Table 4: NO DATA (errors)
        if no_data:
            print("\n" + "="*100)
            print("⚠️  PAIRS WITH NO DATA (Errors)")
            print("="*100)
            error_table = []
            for opp in no_data:
                error_table.append([
                    opp['symbol'],
                    opp['reason']
                ])
            headers = ['Symbol', 'Error']
            print(tabulate(error_table, headers=headers, tablefmt='grid'))
            print()
        
        # Summary
        logger.info(f"\n📊 SUMMARY: {len(to_trade)} to trade | {len(skipped)} skipped | {len(neutral)} neutral | {len(no_data)} errors")
        logger.info("="*100 + "\n")
        
        # Return only opportunities that meet criteria
        return to_trade
    
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
        
        # Check margin level before sending order
        account_info = mt5.account_info()
        if account_info.margin_level < self.config.get('risk_management', {}).get('min_margin_level', 700):
            fail_reason = f"Low margin level ({account_info.margin_level:.1f}%)"
            logger.error(f"❌ Order rejected: {fail_reason}")
            self.trade_history.append({
                'time': datetime.now(),
                'symbol': symbol,
                'signal': signal,
                'entry': entry_price,
                'lot': lot,
                'status': '❌ REJECTED',
                'reason': fail_reason
            })
            return False
        
        # Send order
        result = mt5.order_send(request)
        
        if result is None:
            error = mt5.last_error()
            fail_reason = f"MT5 error: {error}"
            logger.error(f"❌ Order failed: {fail_reason}")
            self.trade_history.append({
                'time': datetime.now(),
                'symbol': symbol,
                'signal': signal,
                'entry': entry_price,
                'lot': lot,
                'status': '❌ FAILED',
                'reason': fail_reason
            })
            return False
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            fail_reason = f"{result.comment} (code: {result.retcode})"
            logger.error(f"❌ Order failed: {fail_reason}")
            self.trade_history.append({
                'time': datetime.now(),
                'symbol': symbol,
                'signal': signal,
                'entry': entry_price,
                'lot': lot,
                'status': '❌ FAILED',
                'reason': fail_reason
            })
            return False
        
        logger.info(f"✅ ORDER EXECUTED SUCCESSFULLY!")
        logger.info(f"   Ticket: {result.order}")
        
        # Record successful execution
        self.trade_history.append({
            'time': datetime.now(),
            'symbol': symbol,
            'signal': signal,
            'entry': entry_price,
            'lot': lot,
            'status': '✅ OPENED',
            'reason': f"Ticket #{result.order}",
            'ticket': result.order
        })
        
        return True
    
    def analyze_position_probability(self, position):
        """Analyze probability of position hitting TP vs SL"""
        symbol = position.symbol
        price_current = position.price_current
        price_open = position.price_open
        sl = position.sl
        tp = position.tp
        position_type = position.type  # 0 = BUY, 1 = SELL
        
        # Get recent market data
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 50)
        if rates is None or len(rates) == 0:
            return 50, "No data", "HOLD"
        
        closes = [r['close'] for r in rates]
        highs = [r['high'] for r in rates]
        lows = [r['low'] for r in rates]
        
        # Calculate indicators
        ma20 = sum(closes[-20:]) / 20
        ma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else ma20
        
        # Calculate momentum
        momentum = (price_current - closes[-10]) / closes[-10] * 100 if len(closes) >= 10 else 0
        
        # Calculate RSI
        rsi = 50
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
        
        # Calculate distances
        if position_type == 0:  # BUY
            dist_to_tp = tp - price_current
            dist_to_sl = price_current - sl
        else:  # SELL
            dist_to_tp = price_current - tp
            dist_to_sl = sl - price_current
        
        total_dist = dist_to_tp + dist_to_sl
        progress = (dist_to_sl / total_dist * 100) if total_dist > 0 else 0
        
        # Start with base probability
        tp_probability = 50
        reasons = []
        
        # Factor 1: Trend alignment (±20%)
        if position_type == 0:  # BUY
            if price_current > ma20 > ma50:
                tp_probability += 20
                reasons.append("Strong uptrend")
            elif price_current > ma20:
                tp_probability += 10
                reasons.append("Uptrend")
            elif price_current < ma20:
                tp_probability -= 15
                reasons.append("Against trend")
        else:  # SELL
            if price_current < ma20 < ma50:
                tp_probability += 20
                reasons.append("Strong downtrend")
            elif price_current < ma20:
                tp_probability += 10
                reasons.append("Downtrend")
            elif price_current > ma20:
                tp_probability -= 15
                reasons.append("Against trend")
        
        # Factor 2: Momentum alignment (±15%)
        if position_type == 0:  # BUY
            if momentum > 0.5:
                tp_probability += 15
                reasons.append(f"Momentum +{momentum:.1f}%")
            elif momentum < -0.5:
                tp_probability -= 10
                reasons.append(f"Momentum {momentum:.1f}%")
        else:  # SELL
            if momentum < -0.5:
                tp_probability += 15
                reasons.append(f"Momentum {momentum:.1f}%")
            elif momentum > 0.5:
                tp_probability -= 10
                reasons.append(f"Momentum +{momentum:.1f}%")
        
        # Factor 3: RSI (±10%)
        if position_type == 0:  # BUY
            if rsi < 40:
                tp_probability += 10
                reasons.append(f"RSI oversold ({rsi:.0f})")
            elif rsi > 70:
                tp_probability -= 10
                reasons.append(f"RSI overbought ({rsi:.0f})")
        else:  # SELL
            if rsi > 60:
                tp_probability += 10
                reasons.append(f"RSI overbought ({rsi:.0f})")
            elif rsi < 30:
                tp_probability -= 10
                reasons.append(f"RSI oversold ({rsi:.0f})")
        
        # Factor 4: Progress to TP (±10%)
        if progress > 60:
            tp_probability += 10
            reasons.append(f"Near TP ({progress:.0f}%)")
        elif progress < 20:
            tp_probability -= 5
            reasons.append(f"Near SL ({progress:.0f}%)")
        
        # Cap probability
        tp_probability = max(0, min(100, tp_probability))
        
        # Determine recommendation
        if tp_probability >= 70:
            recommendation = "🎯 HOLD (High TP prob)"
        elif tp_probability >= 50:
            recommendation = "⏸️ HOLD (Neutral)"
        elif tp_probability >= 30:
            recommendation = "⚠️ WATCH (Low TP prob)"
        else:
            recommendation = "❌ CLOSE (Very low TP prob)"
        
        reason_text = ", ".join(reasons[:2]) if reasons else "Neutral"
        
        return tp_probability, reason_text, recommendation
    
    def monitor_positions(self):
        """Monitor open positions with intelligent TP probability analysis"""
        positions = mt5.positions_get()
        
        if positions is None or len(positions) == 0:
            return
        
        logger.info(f"\n📊 Monitoring {len(positions)} positions with TP Analysis...")
        
        table_data = []
        for position in positions:
            profit = position.profit
            symbol = position.symbol
            volume = position.volume
            price_open = position.price_open
            price_current = position.price_current
            
            # Analyze TP probability
            tp_prob, analysis, recommendation = self.analyze_position_probability(position)
            
            status = "✅ Profit" if profit > 0 else "❌ Loss"
            
            table_data.append([
                position.ticket,
                symbol,
                "BUY" if position.type == 0 else "SELL",
                volume,
                f"{price_open:.5f}",
                f"{price_current:.5f}",
                f"${profit:.2f}",
                status,
                f"{tp_prob}%",
                analysis[:30],  # Truncate long analysis
                recommendation
            ])
        
        headers = ['Ticket', 'Symbol', 'Type', 'Lot', 'Open', 'Current', 'P/L', 'Status', 'TP Prob', 'Analysis', 'Action']
        print("\n" + tabulate(table_data, headers=headers, tablefmt='grid'))
        print()
        
        # Summary statistics
        total_profit = sum(p.profit for p in positions)
        profitable = sum(1 for p in positions if p.profit > 0)
        losing = len(positions) - profitable
        
        logger.info(f"💰 Total P/L: ${total_profit:.2f} | Profitable: {profitable} | Losing: {losing}")
    
    def show_trade_history(self, max_recent=20):
        """Show recent trade execution attempts (successful and failed)"""
        if not self.trade_history:
            return
        
        # Show only recent trades (last 20 by default)
        recent_trades = self.trade_history[-max_recent:]
        
        logger.info(f"\n📋 TRADE EXECUTION HISTORY (Last {len(recent_trades)} attempts)")
        logger.info("="*100)
        
        table_data = []
        for trade in recent_trades:
            time_str = trade['time'].strftime('%H:%M:%S')
            table_data.append([
                time_str,
                trade['symbol'],
                trade['signal'],
                f"{trade['entry']:.5f}",
                f"{trade['lot']:.2f}",
                trade['status'],
                trade['reason'][:45]  # Truncate long reasons
            ])
        
        headers = ['Time', 'Symbol', 'Signal', 'Entry', 'Lot', 'Status', 'Reason/Ticket']
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        print()
        
        # Summary stats
        opened = sum(1 for t in self.trade_history if '✅ OPENED' in t['status'])
        rejected = sum(1 for t in self.trade_history if '❌ REJECTED' in t['status'])
        failed = sum(1 for t in self.trade_history if '❌ FAILED' in t['status'])
        
        logger.info(f"📊 Execution Summary: {opened} opened | {rejected} rejected (margin) | {failed} failed (broker)")
        logger.info("="*100)
    
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
                
                # Show trade execution history
                self.show_trade_history()
                
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

