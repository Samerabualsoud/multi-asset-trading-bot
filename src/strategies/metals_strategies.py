"""
Metals Trading Strategies (Gold, Silver)
=========================================
Specialized strategies for precious metals trading:
- Safe-haven flows (risk-on/risk-off)
- USD inverse correlation
- Inflation hedging
- Technical patterns
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class MetalsTradingStrategies:
    """
    Precious metals trading strategies
    
    Key characteristics:
    - Inverse correlation with USD
    - Safe-haven asset (rises during uncertainty)
    - Inflation hedge
    - Lower leverage recommended
    - Best liquidity during US/London sessions
    """
    
    def __init__(self):
        from ..core.indicators import TechnicalIndicators
        from ..core.market_analyzer import MarketAnalyzer
        
        self.ti = TechnicalIndicators()
        self.market_analyzer = MarketAnalyzer()
    
    def metals_strategy_1_safe_haven_flow(self, df_h1: pd.DataFrame, df_h4: pd.DataFrame,
                                          symbol: str, usd_index_df: pd.DataFrame = None) -> Tuple[Optional[str], float, Dict]:
        """
        Strategy 1: Safe-Haven Flow Trading
        
        Gold/Silver rise during:
        - Market uncertainty (VIX up)
        - USD weakness
        - Risk-off sentiment
        
        Entry:
        - USD index falling
        - Gold breaking resistance
        - Risk-off indicators
        
        Exit:
        - Structure-based SL/TP
        - Trailing stop for trends
        """
        try:
            if len(df_h1) < 100 or len(df_h4) < 50:
                return None, 0, {}
            
            # Calculate indicators
            df_h1 = self.ti.calculate_ema(df_h1, 20)
            df_h1 = self.ti.calculate_ema(df_h1, 50)
            df_h1 = self.ti.calculate_rsi(df_h1, 14)
            df_h1 = self.ti.calculate_atr(df_h1, 14)
            
            # H4 trend
            df_h4 = self.ti.calculate_ema(df_h4, 20)
            df_h4 = self.ti.calculate_ema(df_h4, 50)
            
            close = df_h1['close'].iloc[-1]
            ema_20 = df_h1['ema_20'].iloc[-1]
            ema_50 = df_h1['ema_50'].iloc[-1]
            rsi = df_h1['rsi'].iloc[-1]
            atr = df_h1['atr'].iloc[-1]
            
            # H4 trend
            h4_trend = 'bullish' if df_h4['ema_20'].iloc[-1] > df_h4['ema_50'].iloc[-1] else 'bearish'
            
            # USD index analysis (if available)
            usd_trend = self._analyze_usd_trend(usd_index_df) if usd_index_df is not None else 'neutral'
            
            action = None
            confidence = 0
            
            # Bullish setup (risk-off, USD weakness)
            if h4_trend == 'bullish' and close > ema_20 > ema_50:
                action = 'BUY'
                confidence = 70
                
                # USD weakness confirms
                if usd_trend == 'bearish':
                    confidence += 10
                
                # RSI not overbought
                if 40 < rsi < 70:
                    confidence += 5
            
            # Bearish setup (risk-on, USD strength)
            elif h4_trend == 'bearish' and close < ema_20 < ema_50:
                action = 'SELL'
                confidence = 70
                
                # USD strength confirms
                if usd_trend == 'bullish':
                    confidence += 10
                
                # RSI not oversold
                if 30 < rsi < 60:
                    confidence += 5
            
            if not action:
                return None, 0, {}
            
            # Dynamic SL/TP
            market_analysis = self.market_analyzer.analyze_market_structure(df_h1, symbol)
            
            # Metals-specific adjustments (lower volatility than crypto, higher than forex)
            volatility_mult = 1.5
            
            sl_pips = market_analysis['recommended_sl'] * volatility_mult
            tp_pips = market_analysis['recommended_tp'] * volatility_mult
            
            # Minimum distances
            sl_pips = max(sl_pips, atr * 1.5)
            tp_pips = max(tp_pips, atr * 3.0)
            
            logger.info(f"Metals Safe-Haven: {symbol} {action} @ {close:.2f}")
            logger.info(f"  H4 Trend: {h4_trend}, USD: {usd_trend}, Confidence: {confidence}%")
            logger.info(f"  SL: {sl_pips:.1f}, TP: {tp_pips:.1f}")
            
            return action, confidence, {
                'sl_pips': sl_pips,
                'tp_pips': tp_pips,
                'strategy': 'Metals_Safe_Haven',
                'use_trailing_stop': True,
                'trailing_activation': 0.5,
                'entry_reason': f'{h4_trend.capitalize()} trend, USD {usd_trend}'
            }
        
        except Exception as e:
            logger.error(f"Error in metals_strategy_1: {e}")
            return None, 0, {}
    
    def metals_strategy_2_usd_correlation(self, df_h1: pd.DataFrame, symbol: str,
                                         usd_index_df: pd.DataFrame) -> Tuple[Optional[str], float, Dict]:
        """
        Strategy 2: USD Inverse Correlation Trading
        
        Gold/Silver typically move inverse to USD:
        - USD up → Gold down
        - USD down → Gold up
        
        Entry:
        - Strong USD move in one direction
        - Gold hasn't moved yet (lag)
        - Enter opposite direction
        
        Exit:
        - Quick scalp (correlation catch-up)
        - Tight stops
        """
        try:
            if len(df_h1) < 50 or usd_index_df is None or len(usd_index_df) < 50:
                return None, 0, {}
            
            # Calculate indicators
            df_h1 = self.ti.calculate_ema(df_h1, 20)
            df_h1 = self.ti.calculate_rsi(df_h1, 14)
            df_h1 = self.ti.calculate_atr(df_h1, 14)
            
            # USD index indicators
            usd_index_df = self.ti.calculate_ema(usd_index_df, 20)
            usd_index_df = self.ti.calculate_rsi(usd_index_df, 14)
            
            # Current values
            gold_close = df_h1['close'].iloc[-1]
            gold_rsi = df_h1['rsi'].iloc[-1]
            gold_atr = df_h1['atr'].iloc[-1]
            
            usd_close = usd_index_df['close'].iloc[-1]
            usd_ema = usd_index_df['ema_20'].iloc[-1]
            usd_rsi = usd_index_df['rsi'].iloc[-1]
            
            # Calculate recent moves
            gold_change = (gold_close - df_h1['close'].iloc[-5]) / df_h1['close'].iloc[-5]
            usd_change = (usd_close - usd_index_df['close'].iloc[-5]) / usd_index_df['close'].iloc[-5]
            
            action = None
            confidence = 0
            
            # USD strong up, Gold hasn't moved down yet
            if usd_change > 0.005 and abs(gold_change) < 0.003:  # USD +0.5%, Gold flat
                if usd_close > usd_ema and usd_rsi > 60:
                    action = 'SELL'  # Gold should fall
                    confidence = 65
                    
                    # Strong USD momentum
                    if usd_rsi > 70:
                        confidence += 10
            
            # USD strong down, Gold hasn't moved up yet
            elif usd_change < -0.005 and abs(gold_change) < 0.003:  # USD -0.5%, Gold flat
                if usd_close < usd_ema and usd_rsi < 40:
                    action = 'BUY'  # Gold should rise
                    confidence = 65
                    
                    # Strong USD weakness
                    if usd_rsi < 30:
                        confidence += 10
            
            if not action:
                return None, 0, {}
            
            # Tight stops for correlation scalp
            sl_pips = gold_atr * 1.2
            tp_pips = gold_atr * 2.5
            
            logger.info(f"Metals USD Correlation: {symbol} {action} @ {gold_close:.2f}")
            logger.info(f"  USD change: {usd_change*100:.2f}%, Gold change: {gold_change*100:.2f}%")
            logger.info(f"  Confidence: {confidence}%, SL: {sl_pips:.1f}, TP: {tp_pips:.1f}")
            
            return action, confidence, {
                'sl_pips': sl_pips,
                'tp_pips': tp_pips,
                'strategy': 'Metals_USD_Correlation',
                'use_trailing_stop': False,  # Quick scalp
                'entry_reason': f'USD {usd_change*100:+.2f}%, Gold lag'
            }
        
        except Exception as e:
            logger.error(f"Error in metals_strategy_2: {e}")
            return None, 0, {}
    
    def metals_strategy_3_technical_breakout(self, df_h1: pd.DataFrame, df_h4: pd.DataFrame,
                                            symbol: str) -> Tuple[Optional[str], float, Dict]:
        """
        Strategy 3: Technical Breakout Trading
        
        Classic technical analysis for metals:
        - Support/resistance breakouts
        - Triangle patterns
        - Channel breakouts
        
        Entry:
        - Price breaks key level
        - Volume confirms
        - Momentum supports
        
        Exit:
        - Target next key level
        - Trailing stop
        """
        try:
            if len(df_h1) < 100 or len(df_h4) < 100:
                return None, 0, {}
            
            # Calculate indicators
            df_h4 = self.ti.calculate_ema(df_h4, 20)
            df_h4 = self.ti.calculate_rsi(df_h4, 14)
            df_h4 = self.ti.calculate_atr(df_h4, 14)
            
            # Find key levels
            levels = self._find_key_levels(df_h4)
            
            if not levels:
                return None, 0, {}
            
            close = df_h4['close'].iloc[-1]
            rsi = df_h4['rsi'].iloc[-1]
            atr = df_h4['atr'].iloc[-1]
            
            # Volume
            df_h4['volume_ma'] = df_h4['tick_volume'].rolling(20).mean()
            volume_ratio = df_h4['tick_volume'].iloc[-1] / df_h4['volume_ma'].iloc[-1]
            
            action = None
            confidence = 0
            nearest_level = None
            
            # Check for resistance breakout
            for level in levels['resistance']:
                if abs(close - level) / level < 0.01:  # Within 1% of level
                    # Breakout above
                    if close > level and df_h4['low'].iloc[-1] < level:
                        action = 'BUY'
                        confidence = 70
                        nearest_level = level
                        
                        # Volume confirms
                        if volume_ratio > 1.5:
                            confidence += 10
                        
                        # RSI momentum
                        if rsi > 55:
                            confidence += 5
                        
                        break
            
            # Check for support breakdown
            if not action:
                for level in levels['support']:
                    if abs(close - level) / level < 0.01:
                        # Breakdown below
                        if close < level and df_h4['high'].iloc[-1] > level:
                            action = 'SELL'
                            confidence = 70
                            nearest_level = level
                            
                            # Volume confirms
                            if volume_ratio > 1.5:
                                confidence += 10
                            
                            # RSI momentum
                            if rsi < 45:
                                confidence += 5
                            
                            break
            
            if not action:
                return None, 0, {}
            
            # Calculate SL/TP
            if action == 'BUY':
                sl_pips = (close - nearest_level) * 1.3
                next_resistance = min([r for r in levels['resistance'] if r > close], default=close + atr * 4)
                tp_pips = next_resistance - close
            else:
                sl_pips = (nearest_level - close) * 1.3
                next_support = max([s for s in levels['support'] if s < close], default=close - atr * 4)
                tp_pips = close - next_support
            
            # Minimum distances
            sl_pips = max(sl_pips, atr * 1.5)
            tp_pips = max(tp_pips, atr * 3.0)
            
            logger.info(f"Metals Technical Breakout: {symbol} {action} @ {close:.2f}")
            logger.info(f"  Level: {nearest_level:.2f}, Volume: {volume_ratio:.1f}x")
            logger.info(f"  Confidence: {confidence}%, SL: {sl_pips:.1f}, TP: {tp_pips:.1f}")
            
            return action, confidence, {
                'sl_pips': sl_pips,
                'tp_pips': tp_pips,
                'strategy': 'Metals_Technical_Breakout',
                'use_trailing_stop': True,
                'trailing_activation': 0.5,
                'entry_reason': f'Breakout {nearest_level:.0f}, {volume_ratio:.1f}x volume'
            }
        
        except Exception as e:
            logger.error(f"Error in metals_strategy_3: {e}")
            return None, 0, {}
    
    def _analyze_usd_trend(self, usd_index_df: pd.DataFrame) -> str:
        """
        Analyze USD index trend
        
        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        if usd_index_df is None or len(usd_index_df) < 50:
            return 'neutral'
        
        try:
            usd_index_df = self.ti.calculate_ema(usd_index_df, 20)
            usd_index_df = self.ti.calculate_ema(usd_index_df, 50)
            
            close = usd_index_df['close'].iloc[-1]
            ema_20 = usd_index_df['ema_20'].iloc[-1]
            ema_50 = usd_index_df['ema_50'].iloc[-1]
            
            if close > ema_20 > ema_50:
                return 'bullish'
            elif close < ema_20 < ema_50:
                return 'bearish'
            else:
                return 'neutral'
        
        except Exception as e:
            logger.error(f"Error analyzing USD trend: {e}")
            return 'neutral'
    
    def _find_key_levels(self, df: pd.DataFrame, lookback: int = 100) -> Dict:
        """Find key support and resistance levels"""
        if len(df) < lookback:
            return {'support': [], 'resistance': []}
        
        df_recent = df.iloc[-lookback:]
        
        highs = []
        lows = []
        
        for i in range(2, len(df_recent) - 2):
            # Swing high
            if (df_recent['high'].iloc[i] > df_recent['high'].iloc[i-1] and
                df_recent['high'].iloc[i] > df_recent['high'].iloc[i-2] and
                df_recent['high'].iloc[i] > df_recent['high'].iloc[i+1] and
                df_recent['high'].iloc[i] > df_recent['high'].iloc[i+2]):
                highs.append(df_recent['high'].iloc[i])
            
            # Swing low
            if (df_recent['low'].iloc[i] < df_recent['low'].iloc[i-1] and
                df_recent['low'].iloc[i] < df_recent['low'].iloc[i-2] and
                df_recent['low'].iloc[i] < df_recent['low'].iloc[i+1] and
                df_recent['low'].iloc[i] < df_recent['low'].iloc[i+2]):
                lows.append(df_recent['low'].iloc[i])
        
        resistance = self._cluster_levels(highs)
        support = self._cluster_levels(lows)
        
        return {
            'support': sorted(support),
            'resistance': sorted(resistance, reverse=True)
        }
    
    def _cluster_levels(self, levels: list, threshold: float = 0.01) -> list:
        """Cluster levels that are close to each other"""
        if not levels:
            return []
        
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] < threshold:
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        clusters.append(np.mean(current_cluster))
        
        return clusters

