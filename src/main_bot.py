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
import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'utils'))

from asset_detector import AssetDetector

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
        self.failed_symbols = {}  # Track symbols that fail repeatedly {symbol: fail_count}
        self.asset_detector = AssetDetector()
        
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
        
        # If margin_level is 0 or very high (>100000), it means no positions open
        # In this case, margin is fine - allow trading
        if margin_level == 0 or margin_level > 100000:
            logger.info(f"✅ No open positions - Margin OK (unlimited)")
            return True
        
        # Check if margin level is too low
        if margin_level < min_margin:
            logger.warning(f"⚠️  Margin level too low: {margin_level:.2f}% < {min_margin}%")
            logger.warning(f"⚠️  STOPPING TRADING - Margin protection activated!")
            return False
        
        logger.info(f"✅ Margin level OK: {margin_level:.2f}% > {min_margin}%")
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
        
        # ============================================
        # SUPER ACCURATE ENTRY PRICE CALCULATION
        # ============================================
        
        # 1. Get real-time bid/ask (actual execution prices)
        symbol_info = mt5.symbol_info_tick(symbol)
        if not symbol_info:
            entry_price = current_price
        else:
            bid = symbol_info.bid
            ask = symbol_info.ask
            spread = ask - bid
            mid_price = (bid + ask) / 2  # True market price
            
            # 2. Calculate Support/Resistance levels (last 50 candles)
            # Find swing highs and lows
            swing_highs = []
            swing_lows = []
            for i in range(2, min(50, len(highs)-2)):
                # Swing high: higher than 2 candles before and after
                if highs[-i] > highs[-i-1] and highs[-i] > highs[-i-2] and \
                   highs[-i] > highs[-i+1] and highs[-i] > highs[-i+2]:
                    swing_highs.append(highs[-i])
                
                # Swing low: lower than 2 candles before and after
                if lows[-i] < lows[-i-1] and lows[-i] < lows[-i-2] and \
                   lows[-i] < lows[-i+1] and lows[-i] < lows[-i+2]:
                    swing_lows.append(lows[-i])
            
            # Find nearest support/resistance
            nearest_resistance = min([h for h in swing_highs if h > mid_price], default=mid_price * 1.01)
            nearest_support = max([l for l in swing_lows if l < mid_price], default=mid_price * 0.99)
            
            # 3. Calculate Fibonacci retracement levels (last 50 candles)
            recent_high = max(highs[-50:])
            recent_low = min(lows[-50:])
            fib_range = recent_high - recent_low
            
            # Key Fibonacci levels
            fib_236 = recent_high - (fib_range * 0.236)
            fib_382 = recent_high - (fib_range * 0.382)
            fib_500 = recent_high - (fib_range * 0.500)
            fib_618 = recent_high - (fib_range * 0.618)
            
            # 4. Calculate VWAP (Volume-Weighted Average Price) approximation
            # Use (high + low + close) / 3 as typical price
            typical_prices = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(-20, 0)]
            vwap = sum(typical_prices) / len(typical_prices)
            
            # 5. DETERMINE OPTIMAL ENTRY PRICE
            if signal == 'BUY':
                # For BUY: Use ask price (what we pay)
                base_entry = ask
                
                # Adjust for support levels (better entry near support)
                distance_to_support = abs(mid_price - nearest_support)
                if distance_to_support < fib_range * 0.02:  # Within 2% of support
                    # Near support = good entry, use actual ask
                    entry_price = ask
                    confidence_boost = 5
                else:
                    # Away from support = wait for better price
                    entry_price = ask
                    confidence_boost = 0
                
                # Check if near Fibonacci level (optimal entry)
                fib_tolerance = fib_range * 0.01  # 1% tolerance
                near_fib = False
                fib_level_name = ""
                
                if abs(mid_price - fib_382) < fib_tolerance:
                    near_fib = True
                    fib_level_name = "Fib 38.2%"
                    confidence_boost += 10
                elif abs(mid_price - fib_500) < fib_tolerance:
                    near_fib = True
                    fib_level_name = "Fib 50%"
                    confidence_boost += 8
                elif abs(mid_price - fib_618) < fib_tolerance:
                    near_fib = True
                    fib_level_name = "Fib 61.8%"
                    confidence_boost += 12
                
            else:  # SELL
                # For SELL: Use bid price (what we receive)
                base_entry = bid
                
                # Adjust for resistance levels (better entry near resistance)
                distance_to_resistance = abs(mid_price - nearest_resistance)
                if distance_to_resistance < fib_range * 0.02:  # Within 2% of resistance
                    # Near resistance = good entry, use actual bid
                    entry_price = bid
                    confidence_boost = 5
                else:
                    # Away from resistance = wait for better price
                    entry_price = bid
                    confidence_boost = 0
                
                # Check if near Fibonacci level (optimal entry)
                fib_tolerance = fib_range * 0.01  # 1% tolerance
                near_fib = False
                fib_level_name = ""
                
                if abs(mid_price - fib_382) < fib_tolerance:
                    near_fib = True
                    fib_level_name = "Fib 38.2%"
                    confidence_boost += 10
                elif abs(mid_price - fib_500) < fib_tolerance:
                    near_fib = True
                    fib_level_name = "Fib 50%"
                    confidence_boost += 8
                elif abs(mid_price - fib_618) < fib_tolerance:
                    near_fib = True
                    fib_level_name = "Fib 61.8%"
                    confidence_boost += 12
            
            # 6. Account for spread (slippage)
            # Wide spread = reduce confidence
            avg_spread_pct = (spread / mid_price) * 100
            if avg_spread_pct > 0.05:  # Spread > 0.05%
                confidence_boost -= 5  # Penalize wide spreads
            
            # Store for later use
            entry_context = {
                'spread': spread,
                'spread_pct': avg_spread_pct,
                'near_support': distance_to_support < fib_range * 0.02 if signal == 'BUY' else False,
                'near_resistance': distance_to_resistance < fib_range * 0.02 if signal == 'SELL' else False,
                'near_fib': near_fib,
                'fib_level': fib_level_name,
                'confidence_boost': confidence_boost
            }
        
        # ============================================
        # Calculate multiple indicators for precision
        # ============================================
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
        
        # Add entry quality boost from support/resistance/Fibonacci analysis
        if 'entry_context' in locals():
            confidence += entry_context['confidence_boost']
            if entry_context['near_fib']:
                reasons.append(f"Near {entry_context['fib_level']}")
            if entry_context.get('near_support'):
                reasons.append("Near support")
            if entry_context.get('near_resistance'):
                reasons.append("Near resistance")
        
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
        # Entry price already calculated at the beginning
        
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
        
        # CHECK IF SYMBOL FAILED TOO MANY TIMES - Skip problematic symbols
        if symbol in self.failed_symbols and self.failed_symbols[symbol] >= 3:
            logger.warning(f"⚠️ {symbol} has failed {self.failed_symbols[symbol]} times - SKIPPING")
            return False
        
        # CHECK FOR EXISTING OPEN POSITIONS - Prevent duplicates!
        # This is the CRITICAL fix - check actual positions, not just pending orders
        open_positions = mt5.positions_get(symbol=symbol)
        if open_positions:
            for position in open_positions:
                # Check if we already have an open position for this symbol and direction
                position_type = "BUY" if position.type == mt5.POSITION_TYPE_BUY else "SELL"
                if position_type == signal:
                    logger.warning(f"[DUPLICATE PREVENTION] {signal} position already exists for {symbol} (Ticket #{position.ticket})")
                    logger.warning(f"   Current P/L: ${position.profit:.2f} | Skipping to avoid duplicate")
                    return False  # Don't place duplicate
        
        # Also check pending orders
        pending_orders = mt5.orders_get(symbol=symbol)
        if pending_orders:
            for order in pending_orders:
                order_type_name = "BUY" if order.type in [mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP] else "SELL"
                if order_type_name == signal:
                    logger.warning(f"[DUPLICATE PREVENTION] Pending {signal} order exists for {symbol} (Ticket #{order.ticket})")
                    return False
        
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
        
        # Base SL/TP (asset-specific)
        asset_type = self.asset_detector.get_asset_type(symbol)
        
        if asset_type == 'crypto':
            # Crypto needs MUCH wider stops due to high volatility and price
            base_sl_pips = 150  # $150 for BTC, $15 for ETH, etc.
            base_tp_pips = 300
        elif 'TRY' in symbol or 'MXN' in symbol or 'ZAR' in symbol:
            # Exotic pairs
            base_sl_pips = 50
            base_tp_pips = 100
        else:
            # Standard forex
            base_sl_pips = 25
            base_tp_pips = 50
        
        # Apply ATR-based adjustment if available
        if atr > 0:
            atr_pips = atr / pip_size
            # Use ATR but cap it (different caps for different assets)
            if asset_type == 'crypto':
                base_sl_pips = min(max(100, atr_pips * 1.5), 300)  # 100-300 pips for crypto
            elif 'TRY' in symbol or 'MXN' in symbol or 'ZAR' in symbol:
                base_sl_pips = min(max(40, atr_pips * 1.5), 100)  # 40-100 pips for exotics
            else:
                base_sl_pips = min(max(20, atr_pips * 1.5), 40)  # 20-40 pips for forex
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
        
        # INSTANT EXECUTION: Get real-time market price for immediate execution
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            logger.error(f"❌ Failed to get tick data for {symbol}")
            return False
        
        current_bid = tick.bid
        current_ask = tick.ask
        mid_price = (current_bid + current_ask) / 2
        spread = current_ask - current_bid
        
        # ULTRA-PRECISE ENTRY PRICE CALCULATION
        # Use actual execution price (bid for SELL, ask for BUY)
        if signal == 'BUY':
            # BUY at ASK price (what we actually pay)
            execution_price = current_ask
            order_type = mt5.ORDER_TYPE_BUY
            
            # ULTRA-PRECISE SL/TP calculation
            # SL below entry, TP above entry
            sl_price = execution_price - sl_distance
            tp_price = execution_price + tp_distance
            
        else:  # SELL
            # SELL at BID price (what we actually receive)
            execution_price = current_bid
            order_type = mt5.ORDER_TYPE_SELL
            
            # ULTRA-PRECISE SL/TP calculation
            # SL above entry, TP below entry
            sl_price = execution_price + sl_distance
            tp_price = execution_price - tp_distance
        
        # Round SL/TP to proper decimal places (broker requirement)
        digits = symbol_info.digits
        sl_price = round(sl_price, digits)
        tp_price = round(tp_price, digits)
        execution_price = round(execution_price, digits)
        
        # Verify SL/TP distances meet broker minimum requirements
        stops_level = symbol_info.trade_stops_level * symbol_info.point
        
        if signal == 'BUY':
            # Check minimum distance for BUY
            actual_sl_distance = execution_price - sl_price
            actual_tp_distance = tp_price - execution_price
            
            if actual_sl_distance < stops_level:
                sl_price = execution_price - stops_level
                logger.warning(f"⚠️ SL adjusted to meet broker minimum distance")
            
            if actual_tp_distance < stops_level:
                tp_price = execution_price + stops_level
                logger.warning(f"⚠️ TP adjusted to meet broker minimum distance")
        else:
            # Check minimum distance for SELL
            actual_sl_distance = sl_price - execution_price
            actual_tp_distance = execution_price - tp_price
            
            if actual_sl_distance < stops_level:
                sl_price = execution_price + stops_level
                logger.warning(f"⚠️ SL adjusted to meet broker minimum distance")
            
            if actual_tp_distance < stops_level:
                tp_price = execution_price - stops_level
                logger.warning(f"⚠️ TP adjusted to meet broker minimum distance")
        
        # Log details
        logger.info(f"📐 INSTANT EXECUTION (Market Order)")
        logger.info(f"📐 Current: BID={current_bid:.{digits}f}, ASK={current_ask:.{digits}f}, Spread={spread:.{digits}f}")
        logger.info(f"📐 Execution Price: {execution_price:.{digits}f} (immediate fill)")
        logger.info(f"📐 SL: {sl_price:.{digits}f} ({sl_pips:.1f} pips)")
        logger.info(f"📐 TP: {tp_price:.{digits}f} ({tp_pips:.1f} pips)")
        logger.info(f"📐 Lot Size: {lot:.2f}")
        logger.info(f"📐 Risk: ${risk_amount:,.2f} ({risk_percent*100:.1f}%)")
        logger.info(f"📐 R:R = 1:{tp_pips/sl_pips:.1f}")
        
        # Store for request
        sl = sl_price
        tp = tp_price
        
        # Check margin level before sending order (FIX: handle 0 margin)
        account_info = mt5.account_info()
        margin_level = account_info.margin_level
        min_margin = self.config.get('risk_management', {}).get('min_margin_level', 700)
        
        # FIX: If margin_level is 0 or very high, it means no positions - allow trading
        if margin_level > 0 and margin_level < 100000:
            # We have positions, check margin level
            if margin_level < min_margin:
                fail_reason = f"Low margin level ({margin_level:.1f}%)"
                logger.error(f"❌ Order rejected: {fail_reason}")
                self.trade_history.append({
                    'time': datetime.now(),
                    'symbol': symbol,
                    'signal': signal,
                    'entry': execution_price,
                    'lot': lot,
                    'status': '❌ REJECTED',
                    'reason': fail_reason
                })
                return False
        
        # Prepare INSTANT EXECUTION request (Market Order)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": execution_price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 234000,
            "comment": "Enhanced Bot - Instant",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
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
                'entry': execution_price,
                'lot': lot,
                'status': '❌ FAILED',
                'reason': fail_reason
            })
            return False
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            fail_reason = f"{result.comment} (code: {result.retcode})"
            logger.error(f"❌ Order failed: {fail_reason}")
            
            # Track failed symbols to avoid repeated failures
            if symbol not in self.failed_symbols:
                self.failed_symbols[symbol] = 0
            self.failed_symbols[symbol] += 1
            
            if self.failed_symbols[symbol] >= 3:
                logger.warning(f"⚠️ {symbol} has failed {self.failed_symbols[symbol]} times - will be SKIPPED in future scans")
            
            self.trade_history.append({
                'time': datetime.now(),
                'symbol': symbol,
                'signal': signal,
                'entry': execution_price,
                'lot': lot,
                'status': '❌ FAILED',
                'reason': fail_reason
            })
            return False
        
        logger.info(f"✅ ORDER EXECUTED SUCCESSFULLY!")
        logger.info(f"   Ticket: {result.order}")
        logger.info(f"   Executed at: {execution_price:.{digits}f}")
        logger.info(f"   Actual fill: {result.price:.{digits}f}")
        
        # Record successful execution
        self.trade_history.append({
            'time': datetime.now(),
            'symbol': symbol,
            'signal': signal,
            'entry': execution_price,
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
        scan_interval = self.config.get('scan_interval_seconds', 15)  # Scan every 15 seconds
        
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

