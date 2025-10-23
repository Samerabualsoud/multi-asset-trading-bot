"""
Multi-Asset Trading Bot - ENHANCED VERSION V2
==============================================
With advanced indicators, better entry precision, and NO STOP LOSS
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


class AdvancedIndicators:
    """Advanced technical indicators for enhanced trading strategy"""
    
    @staticmethod
    def calculate_bollinger_bands(closes, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        if len(closes) < period:
            return None, None, None
        
        sma = sum(closes[-period:]) / period
        variance = sum([(x - sma) ** 2 for x in closes[-period:]]) / period
        std = variance ** 0.5
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def calculate_stochastic(highs, lows, closes, period=14):
        """Calculate Stochastic Oscillator"""
        if len(closes) < period:
            return 50, 50
        
        highest_high = max(highs[-period:])
        lowest_low = min(lows[-period:])
        current_close = closes[-1]
        
        if highest_high == lowest_low:
            return 50, 50
        
        k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        
        # Calculate %D (3-period SMA of %K)
        # For simplicity, we'll use current %K as %D
        d = k
        
        return k, d
    
    @staticmethod
    def calculate_adx(highs, lows, closes, period=14):
        """Calculate Average Directional Index (ADX) for trend strength"""
        if len(closes) < period + 1:
            return 25  # Neutral
        
        # Calculate True Range
        tr_list = []
        for i in range(1, period + 1):
            tr = max(
                highs[-i] - lows[-i],
                abs(highs[-i] - closes[-i-1]),
                abs(lows[-i] - closes[-i-1])
            )
            tr_list.append(tr)
        
        # Calculate +DM and -DM
        plus_dm_list = []
        minus_dm_list = []
        for i in range(1, period + 1):
            high_diff = highs[-i] - highs[-i-1]
            low_diff = lows[-i-1] - lows[-i]
            
            plus_dm = high_diff if high_diff > low_diff and high_diff > 0 else 0
            minus_dm = low_diff if low_diff > high_diff and low_diff > 0 else 0
            
            plus_dm_list.append(plus_dm)
            minus_dm_list.append(minus_dm)
        
        # Calculate smoothed averages
        atr = sum(tr_list) / len(tr_list)
        plus_di = (sum(plus_dm_list) / len(plus_dm_list) / atr * 100) if atr > 0 else 0
        minus_di = (sum(minus_dm_list) / len(minus_dm_list) / atr * 100) if atr > 0 else 0
        
        # Calculate ADX
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0
        
        return dx
    
    @staticmethod
    def detect_candlestick_pattern(opens, highs, lows, closes):
        """Detect bullish/bearish candlestick patterns"""
        if len(closes) < 3:
            return None, 0
        
        # Get last 3 candles
        o1, h1, l1, c1 = opens[-3], highs[-3], lows[-3], closes[-3]
        o2, h2, l2, c2 = opens[-2], highs[-2], lows[-2], closes[-2]
        o3, h3, l3, c3 = opens[-1], highs[-1], lows[-1], closes[-1]
        
        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        body3 = abs(c3 - o3)
        
        # Bullish Engulfing
        if c1 < o1 and c2 > o2 and o2 < c1 and c2 > o1:
            return "Bullish Engulfing", 15
        
        # Bearish Engulfing
        if c1 > o1 and c2 < o2 and o2 > c1 and c2 < o1:
            return "Bearish Engulfing", 15
        
        # Morning Star (bullish)
        if c1 < o1 and body2 < body1 * 0.3 and c3 > o3 and c3 > (o1 + c1) / 2:
            return "Morning Star", 20
        
        # Evening Star (bearish)
        if c1 > o1 and body2 < body1 * 0.3 and c3 < o3 and c3 < (o1 + c1) / 2:
            return "Evening Star", 20
        
        # Hammer (bullish)
        lower_shadow = min(o3, c3) - l3
        upper_shadow = h3 - max(o3, c3)
        if lower_shadow > body3 * 2 and upper_shadow < body3 * 0.5:
            return "Hammer", 10
        
        # Shooting Star (bearish)
        if upper_shadow > body3 * 2 and lower_shadow < body3 * 0.5:
            return "Shooting Star", 10
        
        return None, 0


class MultiAssetTradingBot:
    """
    Enhanced multi-asset trading bot with advanced indicators
    """
    
    def __init__(self, config_path='config/config.yaml'):
        """Initialize bot with configuration"""
        self.config = self.load_config(config_path)
        self.running = False
        self.trade_history = []
        self.failed_symbols = {}
        self.asset_detector = AssetDetector()
        self.indicators = AdvancedIndicators()
        
        logger.info("🤖 Multi-Asset Trading Bot V2 initialized (ENHANCED)")
    
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
            if not mt5.initialize():
                logger.error(f"❌ MT5 initialization failed: {mt5.last_error()}")
                return False
            
            login = self.config.get('mt5_login')
            password = self.config.get('mt5_password')
            server = self.config.get('mt5_server')
            
            if not mt5.login(login, password, server):
                logger.error(f"❌ MT5 login failed: {mt5.last_error()}")
                return False
            
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
        
        valid_symbols = []
        for symbol in symbols:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.warning(f"⚠️  Symbol not found: {symbol}")
            elif not symbol_info.visible:
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
        
        # Convert to Saudi time (UTC+3)
        saudi_hour = (hour + 3) % 24
        
        # ENHANCED: Boost confidence during good morning session (8:30-11:30 Saudi time)
        is_good_morning = (8 <= saudi_hour < 12)
        
        # Session times (UTC):
        if 0 <= hour < 8:
            session = 'asian'
            vol_mult = 0.7
            min_confidence = 70 if is_good_morning else 65
        elif 8 <= hour < 13:
            session = 'london'
            vol_mult = 0.9
            min_confidence = 70 if is_good_morning else 75
        elif 13 <= hour < 16:
            session = 'overlap'
            vol_mult = 1.0
            min_confidence = 75 if is_good_morning else 80
        elif 16 <= hour < 21:
            session = 'newyork'
            vol_mult = 0.9
            min_confidence = 75
        else:
            session = 'asian'
            vol_mult = 0.7
            min_confidence = 65
        
        return session, vol_mult, min_confidence, is_good_morning
    
    def check_margin_level(self):
        """Check if margin level is acceptable"""
        account_info = mt5.account_info()
        if account_info is None:
            return False
        
        margin_level = account_info.margin_level
        min_margin = self.config.get('risk_management', {}).get('min_margin_level', 700)
        
        if margin_level == 0 or margin_level > 100000:
            logger.info(f"✅ No open positions - Margin OK (unlimited)")
            return True
        
        if margin_level < min_margin:
            logger.warning(f"⚠️  Margin level too low: {margin_level:.2f}% < {min_margin}%")
            logger.warning(f"⚠️  STOPPING TRADING - Margin protection activated!")
            return False
        
        logger.info(f"✅ Margin level OK: {margin_level:.2f}% > {min_margin}%")
        return True
    
    def calculate_precise_entry(self, symbol, signal, rates):
        """
        Enhanced entry calculation with advanced indicators
        Returns: (entry_price, confidence, reason, atr)
        """
        closes = [r['close'] for r in rates]
        highs = [r['high'] for r in rates]
        lows = [r['low'] for r in rates]
        opens = [r['open'] for r in rates]
        
        current_price = closes[-1]
        
        # ============================================
        # 1. ENTRY PRICE CALCULATION
        # ============================================
        symbol_info = mt5.symbol_info_tick(symbol)
        if not symbol_info:
            entry_price = current_price
            spread = 0
            confidence_boost = 0
        else:
            bid = symbol_info.bid
            ask = symbol_info.ask
            spread = ask - bid
            mid_price = (bid + ask) / 2
            
            # Support/Resistance
            swing_highs = []
            swing_lows = []
            for i in range(2, min(50, len(highs)-2)):
                if highs[-i] > highs[-i-1] and highs[-i] > highs[-i-2] and \
                   highs[-i] > highs[-i+1] and highs[-i] > highs[-i+2]:
                    swing_highs.append(highs[-i])
                
                if lows[-i] < lows[-i-1] and lows[-i] < lows[-i-2] and \
                   lows[-i] < lows[-i+1] and lows[-i] < lows[-i+2]:
                    swing_lows.append(lows[-i])
            
            nearest_resistance = min([h for h in swing_highs if h > mid_price], default=mid_price * 1.01)
            nearest_support = max([l for l in swing_lows if l < mid_price], default=mid_price * 0.99)
            
            # Fibonacci levels
            recent_high = max(highs[-50:])
            recent_low = min(lows[-50:])
            fib_range = recent_high - recent_low
            
            fib_382 = recent_high - (fib_range * 0.382)
            fib_500 = recent_high - (fib_range * 0.500)
            fib_618 = recent_high - (fib_range * 0.618)
            
            confidence_boost = 0
            near_fib = False
            fib_level_name = ""
            
            if signal == 'BUY':
                entry_price = ask
                distance_to_support = abs(mid_price - nearest_support)
                
                if distance_to_support < fib_range * 0.02:
                    confidence_boost += 5
                
                fib_tolerance = fib_range * 0.01
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
                entry_price = bid
                distance_to_resistance = abs(mid_price - nearest_resistance)
                
                if distance_to_resistance < fib_range * 0.02:
                    confidence_boost += 5
                
                fib_tolerance = fib_range * 0.01
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
            
            avg_spread_pct = (spread / mid_price) * 100
            if avg_spread_pct > 0.05:
                confidence_boost -= 5
        
        # ============================================
        # 2. CALCULATE ADVANCED INDICATORS
        # ============================================
        
        # Basic indicators
        ma20 = sum(closes[-20:]) / 20
        ma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else ma20
        
        # EMA
        ema_multiplier = 2 / 21
        ema20 = closes[-1]
        for i in range(len(closes)-20, len(closes)):
            ema20 = (closes[i] - ema20) * ema_multiplier + ema20
        
        # MACD
        ema12 = closes[-1]
        ema26 = closes[-1]
        for i in range(len(closes)-26, len(closes)):
            ema12 = (closes[i] - ema12) * (2/13) + ema12
            ema26 = (closes[i] - ema26) * (2/27) + ema26
        macd = ema12 - ema26
        
        # Momentum
        momentum = (current_price - closes[-10]) / closes[-10] * 100 if len(closes) >= 10 else 0
        
        # ATR
        tr_list = []
        for i in range(1, min(14, len(rates))):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_list.append(tr)
        atr = sum(tr_list) / len(tr_list) if tr_list else 0
        
        # RSI
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
        
        # NEW: Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.indicators.calculate_bollinger_bands(closes)
        
        # NEW: Stochastic
        stoch_k, stoch_d = self.indicators.calculate_stochastic(highs, lows, closes)
        
        # NEW: ADX (trend strength)
        adx = self.indicators.calculate_adx(highs, lows, closes)
        
        # NEW: Candlestick patterns
        pattern_name, pattern_boost = self.indicators.detect_candlestick_pattern(opens, highs, lows, closes)
        
        # ============================================
        # 3. ENHANCED STRATEGY WITH NEW INDICATORS
        # ============================================
        
        confidence = 0
        reasons = []
        reject_reasons = []
        
        # Add initial boosts
        confidence += confidence_boost
        if near_fib:
            reasons.append(f"Near {fib_level_name}")
        
        if signal == 'BUY':
            # REQUIREMENT 1: Strong uptrend
            if current_price > ma20 and current_price > ma50 and current_price > ema20:
                if ma20 > ma50 and ema20 > ma20:
                    confidence += 50
                    reasons.append("Very strong uptrend")
                elif ma20 > ma50:
                    confidence += 35
                    reasons.append("Strong uptrend")
                else:
                    confidence += 15
                    reasons.append("Weak uptrend")
            else:
                reject_reasons.append("NOT above MAs")
                return entry_price, 0, "REJECTED: " + ", ".join(reject_reasons), atr
            
            # REQUIREMENT 2: Positive momentum
            if momentum > 0.3:
                confidence += 25
                reasons.append(f"Strong momentum +{momentum:.2f}%")
            elif momentum > 0:
                confidence += 10
                reasons.append(f"Weak momentum +{momentum:.2f}%")
            else:
                reject_reasons.append(f"Negative momentum")
                return entry_price, 0, "REJECTED: " + ", ".join(reject_reasons), atr
            
            # REQUIREMENT 3: MACD bullish
            if macd > 0:
                confidence += 10
                reasons.append("MACD bullish")
            
            # REQUIREMENT 4: RSI check
            if rsi > 75:
                reject_reasons.append(f"RSI overbought ({rsi:.1f})")
                return entry_price, 0, "REJECTED: " + ", ".join(reject_reasons), atr
            elif rsi < 40:
                confidence += 20
                reasons.append(f"RSI oversold ({rsi:.1f})")
            elif 40 <= rsi <= 60:
                confidence += 15
                reasons.append(f"RSI neutral ({rsi:.1f})")
            
            # NEW: Bollinger Bands confirmation
            if bb_lower and current_price <= bb_lower:
                confidence += 15
                reasons.append("Price at BB lower (oversold)")
            elif bb_middle and current_price > bb_middle:
                confidence += 5
                reasons.append("Price above BB middle")
            
            # NEW: Stochastic confirmation
            if stoch_k < 20:
                confidence += 15
                reasons.append(f"Stochastic oversold ({stoch_k:.1f})")
            elif stoch_k < 50:
                confidence += 5
                reasons.append("Stochastic bullish zone")
            
            # NEW: ADX trend strength
            if adx > 25:
                confidence += 15
                reasons.append(f"Strong trend (ADX {adx:.1f})")
            elif adx > 20:
                confidence += 8
                reasons.append(f"Moderate trend (ADX {adx:.1f})")
            
            # NEW: Candlestick pattern
            if pattern_name and "Bullish" in pattern_name or pattern_name in ["Morning Star", "Hammer"]:
                confidence += pattern_boost
                reasons.append(f"Pattern: {pattern_name}")
        
        else:  # SELL
            # REQUIREMENT 1: Strong downtrend
            if current_price < ma20 and current_price < ma50 and current_price < ema20:
                if ma20 < ma50 and ema20 < ma20:
                    confidence += 50
                    reasons.append("Very strong downtrend")
                elif ma20 < ma50:
                    confidence += 35
                    reasons.append("Strong downtrend")
                else:
                    confidence += 15
                    reasons.append("Weak downtrend")
            else:
                reject_reasons.append("NOT below MAs")
                return entry_price, 0, "REJECTED: " + ", ".join(reject_reasons), atr
            
            # REQUIREMENT 2: Negative momentum
            if momentum < -0.3:
                confidence += 25
                reasons.append(f"Strong momentum {momentum:.2f}%")
            elif momentum < 0:
                confidence += 10
                reasons.append(f"Weak momentum {momentum:.2f}%")
            else:
                reject_reasons.append(f"Positive momentum")
                return entry_price, 0, "REJECTED: " + ", ".join(reject_reasons), atr
            
            # REQUIREMENT 3: MACD bearish
            if macd < 0:
                confidence += 10
                reasons.append("MACD bearish")
            
            # REQUIREMENT 4: RSI check
            if rsi < 25:
                reject_reasons.append(f"RSI oversold ({rsi:.1f})")
                return entry_price, 0, "REJECTED: " + ", ".join(reject_reasons), atr
            elif rsi > 60:
                confidence += 20
                reasons.append(f"RSI overbought ({rsi:.1f})")
            elif 40 <= rsi <= 60:
                confidence += 15
                reasons.append(f"RSI neutral ({rsi:.1f})")
            
            # NEW: Bollinger Bands confirmation
            if bb_upper and current_price >= bb_upper:
                confidence += 15
                reasons.append("Price at BB upper (overbought)")
            elif bb_middle and current_price < bb_middle:
                confidence += 5
                reasons.append("Price below BB middle")
            
            # NEW: Stochastic confirmation
            if stoch_k > 80:
                confidence += 15
                reasons.append(f"Stochastic overbought ({stoch_k:.1f})")
            elif stoch_k > 50:
                confidence += 5
                reasons.append("Stochastic bearish zone")
            
            # NEW: ADX trend strength
            if adx > 25:
                confidence += 15
                reasons.append(f"Strong trend (ADX {adx:.1f})")
            elif adx > 20:
                confidence += 8
                reasons.append(f"Moderate trend (ADX {adx:.1f})")
            
            # NEW: Candlestick pattern
            if pattern_name and ("Bearish" in pattern_name or pattern_name in ["Evening Star", "Shooting Star"]):
                confidence += pattern_boost
                reasons.append(f"Pattern: {pattern_name}")
        
        reason = ", ".join(reasons) if reasons else "Basic signal"
        
        return entry_price, min(confidence, 100), reason, atr
    
    def scan_opportunities(self, symbols):
        """Scan for trading opportunities with enhanced filtering"""
        logger.info("\n" + "="*100)
        logger.info("🔍 SCANNING FOR OPPORTUNITIES (ENHANCED V2)")
        logger.info("="*100)
        
        session, vol_mult, min_confidence, is_good_morning = self.get_session_info()
        
        if is_good_morning:
            logger.info(f"🌅 GOOD MORNING SESSION ACTIVE (8:30-11:30 Saudi Time)")
            logger.info(f"🌍 Session: {session.upper()} | Vol Multiplier: {vol_mult}x | Min Confidence: {min_confidence}%")
        else:
            logger.info(f"🌍 Session: {session.upper()} | Vol Multiplier: {vol_mult}x | Min Confidence: {min_confidence}%")
        
        if not self.check_margin_level():
            logger.info("\n⛔ TRADING HALTED - Margin level too low\n")
            return []
        
        all_opportunities = []
        
        for symbol in symbols:
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
            
            closes = [r['close'] for r in rates]
            ma20 = sum(closes[-20:]) / 20
            current_price = closes[-1]
            
            if current_price > ma20:
                entry, confidence, reason, atr = self.calculate_precise_entry(symbol, 'BUY', rates)
                
                # Boost confidence during good morning session
                if is_good_morning and confidence >= 60:
                    confidence = min(confidence + 5, 100)
                
                status = '✅ TRADE' if confidence >= min_confidence else f'❌ SKIP (Conf: {confidence}% < {min_confidence}%)'
                
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
                
                # Boost confidence during good morning session
                if is_good_morning and confidence >= 60:
                    confidence = min(confidence + 5, 100)
                
                status = '✅ TRADE' if confidence >= min_confidence else f'❌ SKIP (Conf: {confidence}% < {min_confidence}%)'
                
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
        
        # Display results in tables
        to_trade = [opp for opp in all_opportunities if '✅ TRADE' in opp['status']]
        skipped = [opp for opp in all_opportunities if '❌ SKIP' in opp['status']]
        neutral = [opp for opp in all_opportunities if '⏸️  NEUTRAL' in opp['status']]
        no_data = [opp for opp in all_opportunities if 'NO DATA' in opp['status']]
        
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
                    opp['reason'][:50]
                ])
            headers = ['Symbol', 'Signal', 'Entry Price', 'Confidence', 'Reason']
            print(tabulate(trade_table, headers=headers, tablefmt='grid'))
            print()
        
        if skipped:
            print("\n" + "="*100)
            print("❌ SKIPPED OPPORTUNITIES")
            print("="*100)
            skip_table = []
            for opp in skipped[:10]:
                skip_table.append([
                    opp['symbol'],
                    opp['signal'],
                    f"{opp['confidence']}%",
                    opp['reason'][:50]
                ])
            headers = ['Symbol', 'Signal', 'Confidence', 'Reason']
            print(tabulate(skip_table, headers=headers, tablefmt='grid'))
            print()
        
        return to_trade
    
    def execute_trade(self, symbol, signal, opportunity):
        """Execute trade with NO STOP LOSS"""
        # Check for duplicate positions
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            for pos in positions:
                pos_type = "BUY" if pos.type == 0 else "SELL"
                if pos_type == signal:
                    logger.warning(f"⚠️  Duplicate position detected for {symbol} {signal} - SKIPPING")
                    return False
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"❌ Symbol info not available for {symbol}")
            return False
        
        # Calculate position size
        account_info = mt5.account_info()
        balance = account_info.balance
        equity = account_info.equity
        
        # INCREASED LOT SIZE: Since NO SL, we can use larger positions
        # Default risk increased from 2% to 5% per trade
        risk_percent = self.config.get('risk_management', {}).get('risk_per_trade', 0.05)
        risk_amount = equity * risk_percent  # Use equity instead of balance
        
        # Get pip size
        asset_type = self.asset_detector.get_asset_type(symbol)
        if 'JPY' in symbol:
            pip_size = 0.01
        elif asset_type == 'crypto':
            if 'BTC' in symbol or 'ETH' in symbol:
                pip_size = 1.0
            elif 'LTC' in symbol:
                pip_size = 0.1
            else:
                pip_size = 0.01
        else:
            pip_size = 0.0001
        
        # Calculate lot size
        contract_size = symbol_info.trade_contract_size
        pip_value_per_lot = contract_size * pip_size
        
        session, vol_mult, min_confidence, is_good_morning = self.get_session_info()
        atr = opportunity.get('atr', 0)
        
        # TP calculation (NO SL)
        if asset_type == 'crypto':
            tp_pips = 300
        elif 'JPY' in symbol:
            tp_pips = 40 * vol_mult
        else:
            tp_pips = 30 * vol_mult
        
        tp_distance = tp_pips * pip_size
        
        # ENHANCED LOT CALCULATION: Much larger positions
        # Calculate base lot from risk
        base_lot = risk_amount / (tp_pips * pip_value_per_lot)
        
        # MULTIPLY by 3x for larger positions (since no SL, we want meaningful size)
        lot = base_lot * 3.0
        
        # For small accounts, ensure minimum meaningful lot size
        if balance < 1000:
            # Very small account: use at least 0.1 lot
            lot = max(lot, 0.10)
        elif balance < 5000:
            # Small account: use at least 0.2 lot
            lot = max(lot, 0.20)
        elif balance < 10000:
            # Medium account: use at least 0.5 lot
            lot = max(lot, 0.50)
        else:
            # Large account: use at least 1.0 lot
            lot = max(lot, 1.00)
        
        # Apply broker limits
        lot = max(symbol_info.volume_min, min(lot, symbol_info.volume_max))
        lot = round(lot / symbol_info.volume_step) * symbol_info.volume_step
        
        # Get execution price
        current_tick = mt5.symbol_info_tick(symbol)
        if not current_tick:
            logger.error(f"❌ Cannot get tick data for {symbol}")
            return False
        
        current_bid = current_tick.bid
        current_ask = current_tick.ask
        
        if signal == 'BUY':
            order_type = mt5.ORDER_TYPE_BUY
            execution_price = current_ask
            tp_price = execution_price + tp_distance
        else:
            order_type = mt5.ORDER_TYPE_SELL
            execution_price = current_bid
            tp_price = execution_price - tp_distance
        
        # Round prices
        digits = symbol_info.digits
        tp_price = round(tp_price, digits)
        execution_price = round(execution_price, digits)
        
        # Verify TP distance
        stops_level = symbol_info.trade_stops_level * symbol_info.point
        
        if signal == 'BUY':
            actual_tp_distance = tp_price - execution_price
            if actual_tp_distance < stops_level:
                tp_price = execution_price + stops_level
                logger.warning(f"⚠️ TP adjusted to meet broker minimum distance")
        else:
            actual_tp_distance = execution_price - tp_price
            if actual_tp_distance < stops_level:
                tp_price = execution_price - stops_level
                logger.warning(f"⚠️ TP adjusted to meet broker minimum distance")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 EXECUTING TRADE: {symbol} {signal}")
        logger.info(f"{'='*80}")
        logger.info(f"📐 Execution Price: {execution_price:.{digits}f}")
        logger.info(f"📐 TP: {tp_price:.{digits}f} ({tp_pips:.1f} pips)")
        logger.info(f"📐 Lot Size: {lot:.2f} (3x multiplier applied)")
        logger.info(f"📐 Risk: ${risk_amount:,.2f} ({risk_percent*100:.1f}% of equity)")
        logger.info(f"📐 Balance: ${balance:,.2f} | Equity: ${equity:,.2f}")
        logger.info(f"⚠️  NO STOP LOSS - Unlimited Risk - LARGE POSITIONS")
        
        # Check margin
        account_info = mt5.account_info()
        margin_level = account_info.margin_level
        min_margin = self.config.get('risk_management', {}).get('min_margin_level', 700)
        
        if margin_level > 0 and margin_level < 100000:
            if margin_level < min_margin:
                logger.error(f"❌ Order rejected: Low margin level ({margin_level:.1f}%)")
                return False
        
        # Prepare order - NO STOP LOSS
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": execution_price,
            "sl": 0.0,  # NO STOP LOSS
            "tp": tp_price,
            "deviation": 20,
            "magic": 234000,
            "comment": "Enhanced V2 - No SL",
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
        logger.info(f"   Deal: #{result.deal}")
        logger.info(f"   Volume: {result.volume}")
        logger.info(f"   Price: {result.price:.{digits}f}")
        
        self.trade_history.append({
            'time': datetime.now(),
            'symbol': symbol,
            'signal': signal,
            'entry': execution_price,
            'tp': tp_price,
            'lot': lot,
            'status': '✅ EXECUTED',
            'order': result.order,
            'deal': result.deal
        })
        
        return True
    
    def run(self):
        """Main trading loop"""
        logger.info("\n" + "="*100)
        logger.info("🚀 STARTING ENHANCED TRADING BOT V2")
        logger.info("="*100)
        
        if not self.connect_mt5():
            logger.error("❌ Failed to connect to MT5")
            return
        
        symbols = self.get_symbols()
        if not symbols:
            logger.error("❌ No valid symbols to trade")
            return
        
        logger.info(f"✅ Trading {len(symbols)} symbols")
        logger.info(f"⚠️  WARNING: NO STOP LOSS - Unlimited Risk Mode")
        
        self.running = True
        scan_interval = 15  # seconds
        
        try:
            while self.running:
                opportunities = self.scan_opportunities(symbols)
                
                for opp in opportunities:
                    if '✅ TRADE' in opp['status']:
                        logger.info(f"\n🎯 Trading opportunity: {opp['symbol']} {opp['signal']} (Conf: {opp['confidence']}%)")
                        self.execute_trade(opp['symbol'], opp['signal'], opp)
                        time.sleep(2)
                
                logger.info(f"\n⏳ Waiting {scan_interval} seconds before next scan...")
                time.sleep(scan_interval)
                
        except KeyboardInterrupt:
            logger.info("\n⚠️  Bot stopped by user")
        finally:
            mt5.shutdown()
            logger.info("✅ MT5 connection closed")


if __name__ == "__main__":
    bot = MultiAssetTradingBot()
    bot.run()

