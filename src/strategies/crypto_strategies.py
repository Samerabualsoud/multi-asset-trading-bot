"""
Cryptocurrency Trading Strategies
Optimized for high volatility and 24/7 trading
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from indicators import EnhancedTechnicalIndicators as TI
from market_analyzer import MarketAnalyzer


class CryptoTradingStrategies:
    """
    Cryptocurrency-specific trading strategies
    Designed for high volatility, 24/7 markets
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.ti = TI()
        self.market_analyzer = MarketAnalyzer()
    
    def crypto_strategy_1_momentum_breakout(self, df_m15: pd.DataFrame, df_h1: pd.DataFrame, 
                                           symbol: str) -> Tuple[Optional[str], float, Dict]:
        """
        Crypto Strategy 1: Momentum Breakout
        
        Cryptos have strong momentum - ride the wave
        
        Entry:
        - Price breaks above/below 20-period high/low
        - Volume confirms (2x average)
        - RSI shows momentum (>60 for buy, <40 for sell)
        
        Exit:
        - Wider SL/TP for crypto volatility
        - Trailing stop (cryptos can run far)
        """
        try:
            if len(df_m15) < 100 or len(df_h1) < 50:
                return None, 0, {}
            
            # M15 indicators
            ema20_m15 = self.ti.ema(df_m15['close'], 20)
            rsi_m15 = self.ti.rsi(df_m15['close'], 14)
            atr_m15 = self.ti.atr(df_m15, 14)
            
            # Volume indicators
            volume_ma = df_m15['tick_volume'].rolling(20).mean()
            volume_ratio = df_m15['tick_volume'] / volume_ma
            
            # 20-period high/low
            high_20 = df_m15['high'].rolling(20).max()
            low_20 = df_m15['low'].rolling(20).min()
            
            # H1 trend
            ema50_h1 = self.ti.ema(df_h1['close'], 50)
            
            # Current values
            curr_close = df_m15['close'].iloc[-1]
            prev_high_20 = high_20.iloc[-2]
            prev_low_20 = low_20.iloc[-2]
            curr_rsi = rsi_m15.iloc[-1]
            curr_volume_ratio = volume_ratio.iloc[-1]
            curr_atr = atr_m15.iloc[-1]
            curr_price = curr_close
            
            # H1 trend
            h1_bullish = df_h1['close'].iloc[-1] > ema50_h1.iloc[-1]
            
            action = None
            confidence = 0
            
            # Bullish breakout (RELAXED: 1.5x volume instead of 2x, RSI > 55 instead of 60)
            if curr_close > prev_high_20 and curr_volume_ratio > 1.5 and curr_rsi > 55 and h1_bullish:
                action = 'BUY'
                confidence = 70
                
                if curr_rsi > 70:
                    confidence += 10
                if curr_volume_ratio > 2.5:
                    confidence += 5
            
            # Bearish breakout (RELAXED: 1.5x volume instead of 2x, RSI < 45 instead of 40)
            elif curr_close < prev_low_20 and curr_volume_ratio > 1.5 and curr_rsi < 45 and not h1_bullish:
                action = 'SELL'
                confidence = 70
                
                if curr_rsi < 30:
                    confidence += 10
                if curr_volume_ratio > 2.5:
                    confidence += 5
            
            if action:
                pip_size = self.market_analyzer.get_symbol_pip_size(symbol, curr_price)
                
                # Crypto-specific: Wider SL/TP (3x forex)
                sl_pips, tp_pips = self.market_analyzer.calculate_structure_based_sl_tp(
                    df_m15, action, curr_price, curr_atr, pip_size, 'breakout'
                )
                
                # Apply crypto multiplier
                sl_pips = sl_pips * 3.0
                tp_pips = tp_pips * 3.0
                
                # Use trailing stops for crypto
                use_trailing = True
                trailing_distance = sl_pips * 0.5
                
                return action, confidence, {
                    'strategy': 'CRYPTO_MOMENTUM_BREAKOUT',
                    'sl_pips': sl_pips,
                    'tp_pips': tp_pips,
                    'use_trailing_stop': use_trailing,
                    'trailing_distance_pips': trailing_distance,
                    'reason': f'Momentum breakout {action.lower()} (Volume: {curr_volume_ratio:.1f}x, RSI: {curr_rsi:.0f})',
                    'market_structure': self.market_analyzer.analyze_market_structure(df_h1)
                }
            
            return None, 0, {'strategy': 'CRYPTO_MOMENTUM_BREAKOUT', 'reason': 'No breakout'}
        
        except Exception as e:
            return None, 0, {'strategy': 'CRYPTO_MOMENTUM_BREAKOUT', 'error': str(e)}
    
    def crypto_strategy_2_support_resistance(self, df_h1: pd.DataFrame, df_h4: pd.DataFrame,
                                            symbol: str) -> Tuple[Optional[str], float, Dict]:
        """
        Crypto Strategy 2: Support/Resistance Bounce
        
        Cryptos respect key levels and round numbers
        
        Entry:
        - Bounce off major support/resistance
        - Round number psychology (BTC: 40k, 45k, 50k)
        - Volume spike on bounce
        
        Exit:
        - Target next key level
        - Wider stops for crypto volatility
        """
        try:
            if len(df_h1) < 100 or len(df_h4) < 50:
                return None, 0, {}
            
            # H1 indicators
            ema20_h1 = self.ti.ema(df_h1['close'], 20)
            rsi_h1 = self.ti.rsi(df_h1['close'], 14)
            atr_h1 = self.ti.atr(df_h1, 14)
            
            # Volume
            volume_ma = df_h1['tick_volume'].rolling(20).mean()
            volume_spike = df_h1['tick_volume'].iloc[-1] > volume_ma.iloc[-1] * 1.5
            
            # Support/Resistance levels (last 100 bars)
            recent_highs = df_h1['high'].rolling(20).max().iloc[-5:]
            recent_lows = df_h1['low'].rolling(20).min().iloc[-5:]
            
            resistance = recent_highs.max()
            support = recent_lows.min()
            
            # Current values
            curr_close = df_h1['close'].iloc[-1]
            curr_rsi = rsi_h1.iloc[-1]
            curr_atr = atr_h1.iloc[-1]
            curr_price = curr_close
            
            # Distance to levels
            dist_to_support = (curr_close - support) / support
            dist_to_resistance = (resistance - curr_close) / curr_close
            
            action = None
            confidence = 0
            
            # Bounce off support
            if dist_to_support < 0.02 and curr_rsi < 40 and volume_spike:
                action = 'BUY'
                confidence = 75
                
                if curr_rsi < 30:
                    confidence += 10
                if dist_to_support < 0.01:
                    confidence += 5
            
            # Bounce off resistance
            elif dist_to_resistance < 0.02 and curr_rsi > 60 and volume_spike:
                action = 'SELL'
                confidence = 75
                
                if curr_rsi > 70:
                    confidence += 10
                if dist_to_resistance < 0.01:
                    confidence += 5
            
            if action:
                pip_size = self.market_analyzer.get_symbol_pip_size(symbol, curr_price)
                
                # Calculate SL/TP
                sl_pips, tp_pips = self.market_analyzer.calculate_structure_based_sl_tp(
                    df_h1, action, curr_price, curr_atr, pip_size, 'support_resistance'
                )
                
                # Crypto multiplier
                sl_pips = sl_pips * 2.5
                tp_pips = tp_pips * 2.5
                
                return action, confidence, {
                    'strategy': 'CRYPTO_SUPPORT_RESISTANCE',
                    'sl_pips': sl_pips,
                    'tp_pips': tp_pips,
                    'use_trailing_stop': False,
                    'trailing_distance_pips': None,
                    'reason': f'Bounce off {"support" if action=="BUY" else "resistance"} (RSI: {curr_rsi:.0f})',
                    'market_structure': self.market_analyzer.analyze_market_structure(df_h1)
                }
            
            return None, 0, {'strategy': 'CRYPTO_SUPPORT_RESISTANCE', 'reason': 'No level bounce'}
        
        except Exception as e:
            return None, 0, {'strategy': 'CRYPTO_SUPPORT_RESISTANCE', 'error': str(e)}
    
    def crypto_strategy_3_trend_following(self, df_m15: pd.DataFrame, df_h1: pd.DataFrame,
                                         df_h4: pd.DataFrame, symbol: str) -> Tuple[Optional[str], float, Dict]:
        """
        Crypto Strategy 3: Multi-Timeframe Trend Following
        
        Cryptos trend strongly - ride the wave
        
        Entry:
        - H4 trend established
        - H1 pullback complete
        - M15 reversal signal
        
        Exit:
        - Trailing stop (let winners run)
        - Wide TP for big moves
        """
        try:
            if len(df_m15) < 50 or len(df_h1) < 100 or len(df_h4) < 50:
                return None, 0, {}
            
            # H4 trend
            ema20_h4 = self.ti.ema(df_h4['close'], 20)
            ema50_h4 = self.ti.ema(df_h4['close'], 50)
            
            # H1 pullback
            ema20_h1 = self.ti.ema(df_h1['close'], 20)
            rsi_h1 = self.ti.rsi(df_h1['close'], 14)
            atr_h1 = self.ti.atr(df_h1, 14)
            
            # M15 reversal
            ema10_m15 = self.ti.ema(df_m15['close'], 10)
            
            # Current values
            curr_price = df_m15['close'].iloc[-1]
            curr_atr = atr_h1.iloc[-1]
            curr_rsi = rsi_h1.iloc[-1]
            
            # Trend direction
            h4_bullish = ema20_h4.iloc[-1] > ema50_h4.iloc[-1]
            h1_above_ema = df_h1['close'].iloc[-1] > ema20_h1.iloc[-1]
            m15_above_ema = df_m15['close'].iloc[-1] > ema10_m15.iloc[-1]
            
            action = None
            confidence = 0
            
            # Bullish trend
            if h4_bullish and curr_rsi < 50 and m15_above_ema:
                action = 'BUY'
                confidence = 72
                
                if h1_above_ema:
                    confidence += 8
                if curr_rsi < 40:
                    confidence += 5
            
            # Bearish trend
            elif not h4_bullish and curr_rsi > 50 and not m15_above_ema:
                action = 'SELL'
                confidence = 72
                
                if not h1_above_ema:
                    confidence += 8
                if curr_rsi > 60:
                    confidence += 5
            
            if action:
                pip_size = self.market_analyzer.get_symbol_pip_size(symbol, curr_price)
                
                # Trend following: Wider TP
                sl_pips, tp_pips = self.market_analyzer.calculate_structure_based_sl_tp(
                    df_h1, action, curr_price, curr_atr, pip_size, 'trend'
                )
                
                # Crypto: Even wider for big moves
                sl_pips = sl_pips * 3.5
                tp_pips = tp_pips * 4.0  # Let winners run!
                
                # Definitely use trailing
                use_trailing = True
                trailing_distance = sl_pips * 0.6
                
                return action, confidence, {
                    'strategy': 'CRYPTO_TREND_FOLLOWING',
                    'sl_pips': sl_pips,
                    'tp_pips': tp_pips,
                    'use_trailing_stop': use_trailing,
                    'trailing_distance_pips': trailing_distance,
                    'reason': f'Multi-TF trend {action.lower()} (RSI: {curr_rsi:.0f})',
                    'market_structure': self.market_analyzer.analyze_market_structure(df_h1)
                }
            
            return None, 0, {'strategy': 'CRYPTO_TREND_FOLLOWING', 'reason': 'No trend alignment'}
        
        except Exception as e:
            return None, 0, {'strategy': 'CRYPTO_TREND_FOLLOWING', 'error': str(e)}
    
    def crypto_strategy_4_volatility_breakout(self, df_h1: pd.DataFrame, symbol: str) -> Tuple[Optional[str], float, Dict]:
        """
        Crypto Strategy 4: Volatility Breakout (Bollinger Bands)
        
        Cryptos consolidate then explode
        
        Entry:
        - BB squeeze (low volatility)
        - Price breaks out of bands
        - Volume confirms
        
        Exit:
        - Ride the volatility expansion
        - Trailing stop essential
        """
        try:
            if len(df_h1) < 100:
                return None, 0, {}
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.ti.bollinger_bands(df_h1['close'], 20, 2.0)
            atr_h1 = self.ti.atr(df_h1, 14)
            rsi_h1 = self.ti.rsi(df_h1['close'], 14)
            
            # BB width (squeeze detection)
            bb_width = (bb_upper - bb_lower) / bb_middle
            bb_width_ma = bb_width.rolling(20).mean()
            
            # Volume
            volume_ma = df_h1['tick_volume'].rolling(20).mean()
            volume_spike = df_h1['tick_volume'].iloc[-1] > volume_ma.iloc[-1] * 1.5
            
            # Current values
            curr_close = df_h1['close'].iloc[-1]
            curr_bb_upper = bb_upper.iloc[-1]
            curr_bb_lower = bb_lower.iloc[-1]
            curr_bb_width = bb_width.iloc[-1]
            avg_bb_width = bb_width_ma.iloc[-1]
            curr_atr = atr_h1.iloc[-1]
            curr_rsi = rsi_h1.iloc[-1]
            curr_price = curr_close
            
            # Squeeze: BB width below average
            squeeze = curr_bb_width < avg_bb_width * 0.8
            
            action = None
            confidence = 0
            
            # Bullish breakout
            if squeeze and curr_close > curr_bb_upper and volume_spike and curr_rsi > 55:
                action = 'BUY'
                confidence = 73
                
                if curr_rsi > 65:
                    confidence += 7
                if curr_bb_width < avg_bb_width * 0.6:
                    confidence += 5
            
            # Bearish breakout
            elif squeeze and curr_close < curr_bb_lower and volume_spike and curr_rsi < 45:
                action = 'SELL'
                confidence = 73
                
                if curr_rsi < 35:
                    confidence += 7
                if curr_bb_width < avg_bb_width * 0.6:
                    confidence += 5
            
            if action:
                pip_size = self.market_analyzer.get_symbol_pip_size(symbol, curr_price)
                
                # Volatility breakout: Expect big move
                sl_pips, tp_pips = self.market_analyzer.calculate_structure_based_sl_tp(
                    df_h1, action, curr_price, curr_atr, pip_size, 'breakout'
                )
                
                # Crypto: Wide targets for volatility expansion
                sl_pips = sl_pips * 3.0
                tp_pips = tp_pips * 4.5  # Expect explosive move
                
                # Must use trailing
                use_trailing = True
                trailing_distance = sl_pips * 0.5
                
                return action, confidence, {
                    'strategy': 'CRYPTO_VOLATILITY_BREAKOUT',
                    'sl_pips': sl_pips,
                    'tp_pips': tp_pips,
                    'use_trailing_stop': use_trailing,
                    'trailing_distance_pips': trailing_distance,
                    'reason': f'BB squeeze breakout {action.lower()} (Width: {curr_bb_width:.4f})',
                    'market_structure': self.market_analyzer.analyze_market_structure(df_h1)
                }
            
            return None, 0, {'strategy': 'CRYPTO_VOLATILITY_BREAKOUT', 'reason': 'No squeeze breakout'}
        
        except Exception as e:
            return None, 0, {'strategy': 'CRYPTO_VOLATILITY_BREAKOUT', 'error': str(e)}

