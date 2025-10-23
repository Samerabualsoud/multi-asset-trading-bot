#!/usr/bin/env python3
"""
Enhanced Symbol Discovery Module
Automatically discovers all tradeable symbols in MT5 account with advanced filtering
"""

import MetaTrader5 as mt5
import yaml
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedSymbolDiscovery:
    def __init__(self, config_path='config.yaml'):
        self.config = self.load_config(config_path)
        
    def load_config(self, config_path):
        """Load configuration"""
        config_file = Path(config_path)
        if not config_file.exists():
            config_file = Path('config') / config_path
        with open(config_file, 'r') as f:
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
    
    def check_liquidity(self, symbol, min_bars=100):
        """Check if symbol has sufficient liquidity and data"""
        try:
            # Check H1 data availability
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, min_bars)
            if rates is None or len(rates) < min_bars:
                return False, "Insufficient H1 data"
            
            # Check recent activity (last 24 hours)
            recent_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 24)
            if recent_rates is None or len(recent_rates) < 20:
                return False, "Low recent activity"
            
            # Check tick data availability
            ticks = mt5.copy_ticks_from(symbol, datetime.now() - timedelta(hours=1), 10, mt5.COPY_TICKS_ALL)
            if ticks is None or len(ticks) < 5:
                return False, "No recent ticks"
            
            return True, "OK"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def get_symbol_stats(self, symbol):
        """Get trading statistics for symbol"""
        try:
            # Get 7 days of data
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 168)
            if rates is None or len(rates) < 100:
                return None
            
            df = pd.DataFrame(rates)
            
            # Calculate statistics
            stats = {
                'avg_volume': df['tick_volume'].mean(),
                'volatility': df['close'].pct_change().std() * 100,  # Percentage
                'avg_spread': df['spread'].mean() if 'spread' in df else 0,
                'price_range': (df['high'].max() - df['low'].min()) / df['close'].mean() * 100
            }
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get stats for {symbol}: {e}")
            return None
    
    def discover_all_symbols(self, enable_liquidity_check=True):
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
        logger.info("Filtering tradeable symbols (this may take a few minutes)...")
        
        tradeable_symbols = []
        processed = 0
        
        for symbol in all_symbols:
            processed += 1
            if processed % 50 == 0:
                logger.info(f"Processed {processed}/{len(all_symbols)} symbols...")
            
            # Basic checks
            if not symbol.visible:
                continue
            
            # Check if trading is allowed
            if symbol.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
                continue
            
            # Check if symbol is currently tradeable
            if not symbol.select:
                # Try to enable it
                if not mt5.symbol_select(symbol.name, True):
                    continue
            
            # Liquidity check
            if enable_liquidity_check:
                is_liquid, reason = self.check_liquidity(symbol.name)
                if not is_liquid:
                    continue
            
            # Get trading statistics
            stats = self.get_symbol_stats(symbol.name)
            if stats is None:
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
                'volume_step': symbol.volume_step,
                'point': symbol.point,
                'currency_base': symbol.currency_base,
                'currency_profit': symbol.currency_profit,
                'avg_volume': stats['avg_volume'],
                'volatility': stats['volatility'],
                'avg_spread': stats['avg_spread'],
                'price_range': stats['price_range']
            }
            
            tradeable_symbols.append(info)
        
        mt5.shutdown()
        
        logger.info(f"Found {len(tradeable_symbols)} tradeable symbols with sufficient liquidity")
        
        return tradeable_symbols
    
    def categorize_symbol(self, name, path):
        """Categorize symbol by type"""
        name_upper = name.upper()
        path_lower = path.lower() if path else ""
        
        # Forex pairs
        forex_majors = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']
        forex_crosses = ['EURJPY', 'GBPJPY', 'EURGBP', 'EURAUD', 'EURCHF', 'AUDJPY', 'GBPAUD', 'GBPCAD', 'CADJPY', 'NZDJPY']
        
        if name in forex_majors:
            return 'forex_major'
        elif name in forex_crosses:
            return 'forex_cross'
        elif any(x in name_upper for x in ['EUR', 'USD', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD', 'TRY', 'MXN', 'ZAR', 'SEK', 'NOK', 'DKK']):
            return 'forex_exotic'
        
        # Crypto
        if any(x in name_upper for x in ['BTC', 'ETH', 'LTC', 'XRP', 'ADA', 'DOT', 'DOGE', 'SOL', 'AVAX', 'MATIC']):
            return 'crypto'
        
        # Metals
        if any(x in name_upper for x in ['XAU', 'GOLD', 'XAG', 'SILVER', 'XPT', 'PLATINUM', 'XPD', 'PALLADIUM']):
            return 'metal'
        
        # Indices
        if any(x in name_upper for x in ['US30', 'US500', 'NAS100', 'UK100', 'GER40', 'JPN225', 'AUS200', 'SPX', 'NDX', 'DJI', 'DAX', 'FTSE']):
            return 'index'
        
        # Commodities
        if any(x in name_upper for x in ['OIL', 'WTI', 'BRENT', 'NGAS', 'COPPER', 'WHEAT', 'CORN', 'SOYBEAN', 'COFFEE', 'SUGAR']):
            return 'commodity'
        
        # Stocks
        if 'stock' in path_lower or 'equity' in path_lower or 'share' in path_lower:
            return 'stock'
        
        return 'other'
    
    def filter_symbols(self, symbols, categories=None, max_spread=None, min_volume=None, 
                      min_volatility=None, max_volatility=None):
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
            filtered = [s for s in filtered if s['avg_volume'] >= min_volume]
        
        # Filter by volatility range
        if min_volatility:
            filtered = [s for s in filtered if s['volatility'] >= min_volatility]
        if max_volatility:
            filtered = [s for s in filtered if s['volatility'] <= max_volatility]
        
        return filtered
    
    def get_recommended_symbols(self, mode='balanced'):
        """Get recommended symbols for trading
        
        Modes:
        - 'conservative': Low spread, high liquidity, major pairs only
        - 'balanced': Mix of majors, crosses, crypto, metals (default)
        - 'aggressive': Include exotics, commodities, indices
        - 'all': All tradeable symbols
        """
        logger.info(f"Discovering symbols in {mode} mode...")
        all_symbols = self.discover_all_symbols()
        
        if mode == 'conservative':
            recommended_categories = ['forex_major', 'metal']
            recommended = self.filter_symbols(
                all_symbols,
                categories=recommended_categories,
                max_spread=30,
                min_volume=1000
            )
        
        elif mode == 'balanced':
            recommended_categories = ['forex_major', 'forex_cross', 'crypto', 'metal', 'index']
            recommended = self.filter_symbols(
                all_symbols,
                categories=recommended_categories,
                max_spread=50,
                min_volume=500
            )
        
        elif mode == 'aggressive':
            recommended_categories = ['forex_major', 'forex_cross', 'forex_exotic', 'crypto', 
                                     'metal', 'index', 'commodity']
            recommended = self.filter_symbols(
                all_symbols,
                categories=recommended_categories,
                max_spread=100,
                min_volume=100
            )
        
        else:  # 'all'
            recommended = all_symbols
        
        # Sort by category and name
        recommended.sort(key=lambda x: (x['category'], x['name']))
        
        return recommended
    
    def save_to_config(self, symbols, config_path='config.yaml'):
        """Save discovered symbols to config file"""
        config_file = Path(config_path)
        if not config_file.exists():
            config_file = Path('config') / config_path
        
        # Load existing config
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update symbols
        config['symbols'] = [s['name'] for s in symbols]
        config['auto_discovered'] = True
        config['discovery_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        config['total_symbols'] = len(symbols)
        
        # Save
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved {len(symbols)} symbols to {config_file}")
    
    def save_detailed_report(self, symbols, output_path='symbol_discovery_report.csv'):
        """Save detailed report as CSV"""
        df = pd.DataFrame(symbols)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved detailed report to {output_path}")
    
    def print_summary(self, symbols):
        """Print summary of discovered symbols"""
        # Group by category
        by_category = {}
        for s in symbols:
            cat = s['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(s)
        
        print("\n" + "="*100)
        print("SYMBOL DISCOVERY SUMMARY")
        print("="*100)
        print(f"Total symbols found: {len(symbols)}")
        print("\nBy Category:")
        print("-"*100)
        
        for category, syms in sorted(by_category.items()):
            print(f"\n{category.upper().replace('_', ' ')} ({len(syms)} symbols):")
            print(f"{'Symbol':<15} {'Description':<40} {'Spread':<8} {'Volatility':<10} {'Volume':<10}")
            print("-"*100)
            for s in syms[:15]:  # Show first 15
                desc = s['description'][:38] if len(s['description']) > 38 else s['description']
                print(f"{s['name']:<15} {desc:<40} {s['spread']:<8} {s['volatility']:<10.2f} {s['avg_volume']:<10.0f}")
            if len(syms) > 15:
                print(f"  ... and {len(syms) - 15} more")
        
        print("\n" + "="*100)
        print(f"Symbol names (for config.yaml):")
        print("-"*100)
        symbol_names = [s['name'] for s in symbols]
        for i in range(0, len(symbol_names), 6):
            print("  " + ", ".join(symbol_names[i:i+6]))
        print("="*100)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Discover tradeable symbols in MT5')
    parser.add_argument('--mode', choices=['conservative', 'balanced', 'aggressive', 'all'], 
                       default='balanced', help='Discovery mode (default: balanced)')
    parser.add_argument('--save', action='store_true', help='Save to config.yaml')
    parser.add_argument('--report', action='store_true', help='Save detailed CSV report')
    args = parser.parse_args()
    
    discovery = EnhancedSymbolDiscovery()
    
    print(f"\nDiscovering symbols in '{args.mode}' mode...")
    print("This may take a few minutes depending on your broker...\n")
    
    symbols = discovery.get_recommended_symbols(mode=args.mode)
    
    if not symbols:
        print("No symbols found!")
        return
    
    # Print summary
    discovery.print_summary(symbols)
    
    # Save detailed report if requested
    if args.report:
        discovery.save_detailed_report(symbols)
        print(f"\nDetailed report saved to symbol_discovery_report.csv")
    
    # Ask user if they want to save
    if args.save:
        print("\nSaving symbols to config.yaml...")
        discovery.save_to_config(symbols)
        print("\nSymbols saved to config.yaml!")
        print("You can now run the auto-retrain system to train models for all symbols.")
    else:
        print("\n" + "="*100)
        print("To save these symbols to config.yaml, run:")
        print(f"  python src/symbol_discovery_enhanced.py --mode {args.mode} --save")
        print("\nTo generate a detailed CSV report, add --report flag")
        print("="*100)

if __name__ == "__main__":
    main()

