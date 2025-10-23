#!/usr/bin/env python3
"""
Symbol Discovery Module
Automatically discovers all tradeable symbols in MT5 account
"""

import MetaTrader5 as mt5
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SymbolDiscovery:
    def __init__(self, config_path='config.yaml'):
        self.config = self.load_config(config_path)
        
    def load_config(self, config_path):
        """Load configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def connect_mt5(self):
        """Connect to MT5"""
        if not mt5.initialize(
            path=self.config.get('mt5_path'),
            login=self.config['mt5_login'],
            password=self.config['mt5_password'],
            server=self.config['mt5_server']
        ):
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        return True
    
    def discover_all_symbols(self):
        """Discover all available symbols in MT5"""
        if not self.connect_mt5():
            return []
        
        # Get all symbols
        all_symbols = mt5.symbols_get()
        
        if all_symbols is None:
            logger.error("Failed to get symbols")
            mt5.shutdown()
            return []
        
        logger.info(f"Found {len(all_symbols)} total symbols in MT5")
        
        tradeable_symbols = []
        
        for symbol in all_symbols:
            # Check if symbol is tradeable
            if not symbol.visible:
                continue
            
            # Check if trading is allowed
            if symbol.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
                continue
            
            # Check if we can get data
            rates = mt5.copy_rates_from_pos(symbol.name, mt5.TIMEFRAME_H1, 0, 100)
            if rates is None or len(rates) < 50:
                continue
            
            # Get symbol info
            info = {
                'name': symbol.name,
                'description': symbol.description,
                'path': symbol.path,
                'category': self.categorize_symbol(symbol.name, symbol.path),
                'spread': symbol.spread,
                'digits': symbol.digits,
                'trade_contract_size': symbol.trade_contract_size,
                'volume_min': symbol.volume_min,
                'volume_max': symbol.volume_max,
                'volume_step': symbol.volume_step
            }
            
            tradeable_symbols.append(info)
        
        mt5.shutdown()
        
        logger.info(f"Found {len(tradeable_symbols)} tradeable symbols")
        
        return tradeable_symbols
    
    def categorize_symbol(self, name, path):
        """Categorize symbol by type"""
        name_upper = name.upper()
        path_lower = path.lower() if path else ""
        
        # Forex pairs
        forex_majors = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']
        forex_crosses = ['EURJPY', 'GBPJPY', 'EURGBP', 'EURAUD', 'EURCHF', 'AUDJPY', 'GBPAUD', 'GBPCAD']
        
        if name in forex_majors:
            return 'forex_major'
        elif name in forex_crosses:
            return 'forex_cross'
        elif any(x in name_upper for x in ['EUR', 'USD', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']):
            return 'forex_exotic'
        
        # Crypto
        if any(x in name_upper for x in ['BTC', 'ETH', 'LTC', 'XRP', 'ADA', 'DOT', 'DOGE']):
            return 'crypto'
        
        # Metals
        if any(x in name_upper for x in ['XAU', 'GOLD', 'XAG', 'SILVER', 'XPT', 'PLATINUM', 'XPD', 'PALLADIUM']):
            return 'metal'
        
        # Indices
        if any(x in name_upper for x in ['US30', 'US500', 'NAS100', 'UK100', 'GER40', 'JPN225', 'AUS200', 'SPX', 'NDX', 'DJI']):
            return 'index'
        
        # Commodities
        if any(x in name_upper for x in ['OIL', 'WTI', 'BRENT', 'NGAS', 'COPPER', 'WHEAT', 'CORN', 'SOYBEAN']):
            return 'commodity'
        
        # Stocks
        if 'stock' in path_lower or 'equity' in path_lower:
            return 'stock'
        
        return 'other'
    
    def filter_symbols(self, symbols, categories=None, max_spread=None, min_volume=None):
        """Filter symbols by criteria"""
        filtered = symbols
        
        # Filter by category
        if categories:
            filtered = [s for s in filtered if s['category'] in categories]
        
        # Filter by spread
        if max_spread:
            filtered = [s for s in filtered if s['spread'] <= max_spread]
        
        # Filter by minimum volume
        if min_volume:
            filtered = [s for s in filtered if s['volume_min'] <= min_volume]
        
        return filtered
    
    def get_recommended_symbols(self):
        """Get recommended symbols for trading"""
        all_symbols = self.discover_all_symbols()
        
        # Recommended categories
        recommended_categories = [
            'forex_major',
            'forex_cross',
            'crypto',
            'metal',
            'index',
            'commodity'
        ]
        
        # Filter
        recommended = self.filter_symbols(
            all_symbols,
            categories=recommended_categories,
            max_spread=50,  # Max 50 points spread
            min_volume=0.01  # Can trade micro lots
        )
        
        # Sort by category and name
        recommended.sort(key=lambda x: (x['category'], x['name']))
        
        return recommended
    
    def save_to_config(self, symbols, config_path='config.yaml'):
        """Save discovered symbols to config file"""
        # Load existing config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update symbols
        config['symbols'] = [s['name'] for s in symbols]
        config['auto_discovered'] = True
        config['discovery_date'] = str(mt5.symbol_info_tick('EURUSD').time if mt5.initialize() else "")
        
        # Save
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Saved {len(symbols)} symbols to {config_path}")
    
    def print_summary(self, symbols):
        """Print summary of discovered symbols"""
        # Group by category
        by_category = {}
        for s in symbols:
            cat = s['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(s)
        
        print("\n" + "="*80)
        print("SYMBOL DISCOVERY SUMMARY")
        print("="*80)
        print(f"Total symbols found: {len(symbols)}")
        print("\nBy Category:")
        print("-"*80)
        
        for category, syms in sorted(by_category.items()):
            print(f"\n{category.upper().replace('_', ' ')} ({len(syms)} symbols):")
            for s in syms[:10]:  # Show first 10
                print(f"  - {s['name']:<15} {s['description'][:50]:<50} Spread: {s['spread']}")
            if len(syms) > 10:
                print(f"  ... and {len(syms) - 10} more")
        
        print("\n" + "="*80)
        print(f"Symbol names (for config.yaml):")
        print("-"*80)
        symbol_names = [s['name'] for s in symbols]
        for i in range(0, len(symbol_names), 5):
            print("  " + ", ".join(symbol_names[i:i+5]))
        print("="*80)

def main():
    """Main function"""
    discovery = SymbolDiscovery()
    
    print("Discovering symbols in MT5...")
    symbols = discovery.get_recommended_symbols()
    
    if not symbols:
        print("No symbols found!")
        return
    
    # Print summary
    discovery.print_summary(symbols)
    
    # Ask user if they want to save
    print("\nDo you want to save these symbols to config.yaml?")
    print("WARNING: This will replace your current symbol list!")
    response = input("Type 'yes' to confirm: ")
    
    if response.lower() == 'yes':
        discovery.save_to_config(symbols)
        print("\nSymbols saved to config.yaml!")
        print("You can now run the auto-retrain system to train models for all symbols.")
    else:
        print("\nNot saved. You can manually copy the symbol names above to config.yaml")

if __name__ == "__main__":
    main()

