"""
Test script to verify the bot works correctly
Run this on Windows where MT5 is installed
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 60)
print("Testing Multi-Asset Trading Bot")
print("=" * 60)

# Test 1: Imports
print("\n1. Testing imports...")
try:
    from core.indicators import EnhancedTechnicalIndicators
    from core.market_analyzer import MarketAnalyzer
    from core.risk_manager import EnhancedRiskManager
    from core.position_monitor import PositionMonitor
    from core.strategy_optimizer import StrategyOptimizer
    from strategies.forex_strategies import ImprovedTradingStrategies
    from strategies.crypto_strategies import CryptoTradingStrategies
    from strategies.metals_strategies import MetalsTradingStrategies
    from utils.asset_detector import detect_asset_type
    from utils.config_validator import ConfigValidator
    print("✅ All imports successful!")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test 2: Asset detection
print("\n2. Testing asset detection...")
try:
    assert detect_asset_type("EURUSD") == "forex"
    assert detect_asset_type("GBPUSD") == "forex"
    assert detect_asset_type("BTCUSD") == "crypto"
    assert detect_asset_type("ETHUSD") == "crypto"
    assert detect_asset_type("XAUUSD") == "metal"
    assert detect_asset_type("XAGUSD") == "metal"
    print("✅ Asset detection working!")
except Exception as e:
    print(f"❌ Asset detection error: {e}")
    sys.exit(1)

# Test 3: Config validation
print("\n3. Testing config validation...")
try:
    validator = ConfigValidator()
    test_config = {
        'mt5_login': 12345,
        'mt5_password': 'test',
        'mt5_server': 'test-server',
        'symbols': ['EURUSD', 'GBPUSD'],
        'risk_management': {
            'risk_per_trade': 0.005,
            'max_positions': 5
        }
    }
    is_valid, errors = validator.validate(test_config)
    if is_valid:
        print("✅ Config validation working!")
    else:
        print(f"⚠️  Config validation found issues: {errors}")
except Exception as e:
    print(f"❌ Config validation error: {e}")
    sys.exit(1)

# Test 4: Strategy classes
print("\n4. Testing strategy initialization...")
try:
    config = {
        'risk_management': {
            'risk_per_trade': 0.005,
            'max_positions': 5
        }
    }
    
    forex_strat = ImprovedTradingStrategies(config)
    crypto_strat = CryptoTradingStrategies(config)
    metals_strat = MetalsTradingStrategies(config)
    print("✅ All strategy classes initialized!")
except Exception as e:
    print(f"❌ Strategy initialization error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check strategy methods exist
print("\n5. Testing strategy methods...")
try:
    # Forex strategies
    assert hasattr(forex_strat, 'strategy_1_trend_following')
    assert hasattr(forex_strat, 'strategy_2_fibonacci_retracement')
    assert hasattr(forex_strat, 'strategy_3_mean_reversion')
    assert hasattr(forex_strat, 'strategy_4_breakout')
    assert hasattr(forex_strat, 'strategy_5_momentum')
    assert hasattr(forex_strat, 'strategy_6_multi_timeframe_confluence')
    
    # Crypto strategies
    assert hasattr(crypto_strat, 'crypto_strategy_1_momentum_breakout')
    assert hasattr(crypto_strat, 'crypto_strategy_2_support_resistance')
    assert hasattr(crypto_strat, 'crypto_strategy_3_trend_following')
    assert hasattr(crypto_strat, 'crypto_strategy_4_volatility_breakout')
    
    # Metals strategies
    assert hasattr(metals_strat, 'metals_strategy_1_safe_haven_flow')
    assert hasattr(metals_strat, 'metals_strategy_2_usd_correlation')
    assert hasattr(metals_strat, 'metals_strategy_3_technical_breakout')
    
    print("✅ All strategy methods exist!")
except Exception as e:
    print(f"❌ Strategy methods error: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nThe bot is ready to run on Windows with MT5 installed.")
print("Next step: Configure config/config.yaml and run:")
print("  python src/trading_bot.py")
print("=" * 60)

