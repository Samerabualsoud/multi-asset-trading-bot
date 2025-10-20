"""
Strategy Optimizer - Intelligent Strategy-to-Pair Weighting
===========================================================
Optimizes strategy selection based on pair characteristics
"""

import logging
from typing import Dict, List, Tuple
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """
    Intelligent strategy weighting system
    
    Assigns confidence multipliers to strategies based on:
    - Pair characteristics (volatility, trend strength, etc.)
    - Historical performance
    - Market conditions
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        
        # Load default weights
        self.default_weights = self._get_default_weights()
        
        # Load custom weights from config
        self.custom_weights = self.config.get('strategy_weights', {})
        
        # Merge custom with defaults
        self.weights = self._merge_weights(self.default_weights, self.custom_weights)
        
        logger.info(f"✅ Strategy Optimizer initialized with {len(self.weights)} pair configurations")
    
    def get_weight(self, symbol: str, strategy_name: str) -> float:
        """
        Get confidence multiplier for strategy-pair combination
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            strategy_name: Strategy name (e.g., 'trend_following')
        
        Returns:
            Confidence multiplier (0.5 to 1.5)
        """
        # Check if symbol has custom weights
        if symbol in self.weights:
            return self.weights[symbol].get(strategy_name, 1.0)
        
        # Check if symbol matches a pattern (e.g., all JPY pairs)
        for pattern, weights in self.weights.items():
            if pattern in symbol or symbol.endswith(pattern):
                return weights.get(strategy_name, 1.0)
        
        # Default: neutral weight
        return 1.0
    
    def apply_weight(self, symbol: str, strategy_name: str, confidence: float) -> float:
        """
        Apply weight to confidence score
        
        Args:
            symbol: Trading symbol
            strategy_name: Strategy name
            confidence: Original confidence (0-100)
        
        Returns:
            Weighted confidence (0-100)
        """
        weight = self.get_weight(symbol, strategy_name)
        weighted_confidence = confidence * weight
        
        # Clamp to valid range
        weighted_confidence = max(0, min(100, weighted_confidence))
        
        if weight != 1.0:
            logger.debug(f"{symbol} {strategy_name}: {confidence:.1f}% × {weight:.2f} = {weighted_confidence:.1f}%")
        
        return weighted_confidence
    
    def _get_default_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Get default strategy weights based on pair characteristics
        
        Research-based weights optimized for each pair type
        """
        return {
            # === MAJOR FOREX PAIRS ===
            
            # EURUSD: Low volatility, strong trends, high liquidity
            'EURUSD': {
                'trend_following': 1.25,        # Excellent for trends
                'fibonacci': 1.15,              # Good retracements
                'mean_reversion': 0.75,         # Poor (trends too strong)
                'breakout': 0.90,               # Moderate
                'momentum': 1.10,               # Good
                'multi_timeframe': 1.20,        # Excellent
            },
            
            # GBPUSD: Medium volatility, good trends
            'GBPUSD': {
                'trend_following': 1.20,
                'fibonacci': 1.10,
                'mean_reversion': 0.80,
                'breakout': 1.00,
                'momentum': 1.15,
                'multi_timeframe': 1.15,
            },
            
            # USDJPY: Low volatility, respects technicals
            'USDJPY': {
                'trend_following': 1.15,
                'fibonacci': 1.20,              # Excellent fib respect
                'mean_reversion': 0.85,
                'breakout': 0.95,
                'momentum': 1.05,
                'multi_timeframe': 1.10,
            },
            
            # AUDUSD: Commodity currency, trend follower
            'AUDUSD': {
                'trend_following': 1.20,
                'fibonacci': 1.05,
                'mean_reversion': 0.80,
                'breakout': 1.00,
                'momentum': 1.15,
                'multi_timeframe': 1.15,
            },
            
            # === CROSS PAIRS ===
            
            # GBPJPY: High volatility, choppy, mean reversion works
            'GBPJPY': {
                'trend_following': 0.75,        # Poor (too choppy)
                'fibonacci': 0.85,
                'mean_reversion': 1.35,         # Excellent
                'breakout': 1.25,               # Good volatility
                'momentum': 1.20,               # Good for swings
                'multi_timeframe': 0.80,
            },
            
            # EURJPY: Medium volatility, good for all strategies
            'EURJPY': {
                'trend_following': 1.10,
                'fibonacci': 1.10,
                'mean_reversion': 1.10,
                'breakout': 1.10,
                'momentum': 1.10,
                'multi_timeframe': 1.10,
            },
            
            # EURGBP: Low volatility, range-bound
            'EURGBP': {
                'trend_following': 0.70,        # Poor (ranges)
                'fibonacci': 0.90,
                'mean_reversion': 1.30,         # Excellent for ranges
                'breakout': 1.15,               # Good when breaks
                'momentum': 0.80,
                'multi_timeframe': 0.85,
            },
            
            # === PATTERN MATCHING (for similar pairs) ===
            
            # All JPY pairs (if not specifically defined)
            'JPY': {
                'trend_following': 1.05,
                'fibonacci': 1.15,
                'mean_reversion': 1.10,
                'breakout': 1.05,
                'momentum': 1.10,
                'multi_timeframe': 1.05,
            },
            
            # All GBP pairs (volatile)
            'GBP': {
                'trend_following': 1.00,
                'fibonacci': 1.00,
                'mean_reversion': 1.15,
                'breakout': 1.15,
                'momentum': 1.15,
                'multi_timeframe': 1.00,
            },
            
            # === CRYPTOCURRENCIES ===
            
            # BTCUSD: High volatility, strong trends
            'BTCUSD': {
                'momentum_breakout': 1.30,      # Excellent
                'support_resistance': 1.20,     # Good (round numbers)
                'trend_following': 1.25,        # Excellent
                'volatility_breakout': 1.15,    # Good
            },
            
            # ETHUSD: Follows BTC but more volatile
            'ETHUSD': {
                'momentum_breakout': 1.25,
                'support_resistance': 1.15,
                'trend_following': 1.20,
                'volatility_breakout': 1.20,
            },
            
            # Generic crypto (for others)
            'BTC': {
                'momentum_breakout': 1.20,
                'support_resistance': 1.15,
                'trend_following': 1.15,
                'volatility_breakout': 1.15,
            },
            
            'ETH': {
                'momentum_breakout': 1.15,
                'support_resistance': 1.10,
                'trend_following': 1.15,
                'volatility_breakout': 1.15,
            },
            
            # === METALS ===
            
            # XAUUSD (Gold): Safe-haven, USD inverse
            'XAUUSD': {
                'safe_haven_flow': 1.30,        # Excellent
                'usd_correlation': 1.25,        # Excellent
                'technical_breakout': 1.15,     # Good
            },
            
            # XAGUSD (Silver): More volatile than gold
            'XAGUSD': {
                'safe_haven_flow': 1.20,
                'usd_correlation': 1.20,
                'technical_breakout': 1.25,     # Better for breakouts
            },
        }
    
    def _merge_weights(self, default: Dict, custom: Dict) -> Dict:
        """Merge custom weights with defaults"""
        merged = default.copy()
        
        for symbol, weights in custom.items():
            if symbol in merged:
                # Update existing
                merged[symbol].update(weights)
            else:
                # Add new
                merged[symbol] = weights
        
        return merged
    
    def get_best_strategies(self, symbol: str, top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Get top N strategies for a symbol
        
        Args:
            symbol: Trading symbol
            top_n: Number of top strategies to return
        
        Returns:
            List of (strategy_name, weight) tuples, sorted by weight
        """
        if symbol not in self.weights:
            return []
        
        strategies = self.weights[symbol]
        sorted_strategies = sorted(strategies.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_strategies[:top_n]
    
    def print_weights(self, symbol: str = None):
        """Print weights for debugging"""
        if symbol:
            if symbol in self.weights:
                logger.info(f"\nStrategy weights for {symbol}:")
                for strategy, weight in sorted(self.weights[symbol].items(), key=lambda x: x[1], reverse=True):
                    logger.info(f"  {strategy:25s}: {weight:.2f}")
            else:
                logger.info(f"No specific weights for {symbol}, using defaults")
        else:
            logger.info("\nAll strategy weights:")
            for sym, weights in self.weights.items():
                logger.info(f"\n{sym}:")
                for strategy, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                    logger.info(f"  {strategy:25s}: {weight:.2f}")
    
    def update_weights_from_performance(self, performance_data: Dict):
        """
        Update weights based on actual performance
        
        Args:
            performance_data: {
                'EURUSD': {
                    'trend_following': {'win_rate': 0.75, 'profit': 1250},
                    'mean_reversion': {'win_rate': 0.45, 'profit': -350},
                    ...
                }
            }
        """
        logger.info("Updating strategy weights based on performance...")
        
        for symbol, strategies in performance_data.items():
            if symbol not in self.weights:
                self.weights[symbol] = {}
            
            for strategy, metrics in strategies.items():
                win_rate = metrics.get('win_rate', 0.5)
                profit = metrics.get('profit', 0)
                
                # Calculate new weight
                # Base: win_rate (0.5 = 1.0, 0.7 = 1.4, 0.3 = 0.6)
                new_weight = 0.5 + win_rate
                
                # Adjust for profitability
                if profit > 0:
                    new_weight *= 1.1
                elif profit < 0:
                    new_weight *= 0.9
                
                # Clamp to reasonable range
                new_weight = max(0.5, min(1.5, new_weight))
                
                # Update
                old_weight = self.weights[symbol].get(strategy, 1.0)
                self.weights[symbol][strategy] = new_weight
                
                logger.info(f"  {symbol} {strategy}: {old_weight:.2f} → {new_weight:.2f} "
                          f"(WR: {win_rate:.1%}, P/L: {profit:+.0f})")
        
        logger.info("✅ Weights updated")
    
    def save_weights(self, filepath: str = 'config/strategy_weights.yaml'):
        """Save current weights to file"""
        try:
            with open(filepath, 'w') as f:
                yaml.dump(self.weights, f, default_flow_style=False)
            logger.info(f"✅ Weights saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save weights: {e}")
    
    def load_weights(self, filepath: str = 'config/strategy_weights.yaml'):
        """Load weights from file"""
        try:
            with open(filepath, 'r') as f:
                loaded_weights = yaml.safe_load(f)
            
            self.custom_weights = loaded_weights
            self.weights = self._merge_weights(self.default_weights, self.custom_weights)
            
            logger.info(f"✅ Weights loaded from {filepath}")
        except FileNotFoundError:
            logger.warning(f"Weights file not found: {filepath}, using defaults")
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")


# Example usage in bot:
"""
from core.strategy_optimizer import StrategyOptimizer

class TradingBot:
    def __init__(self, config):
        self.optimizer = StrategyOptimizer(config)
    
    def analyze_opportunity(self, symbol, strategy_name, base_confidence):
        # Apply weight
        weighted_confidence = self.optimizer.apply_weight(
            symbol, strategy_name, base_confidence
        )
        
        # Only take if above threshold
        if weighted_confidence >= 65:
            return True, weighted_confidence
        else:
            return False, weighted_confidence
"""

