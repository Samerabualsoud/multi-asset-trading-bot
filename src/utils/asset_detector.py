"""
Asset type detection utilities
"""


def detect_asset_type(symbol: str) -> str:
    """
    Detect asset type from symbol
    
    Args:
        symbol: Trading symbol (e.g., 'EURUSD', 'BTCUSD', 'XAUUSD')
    
    Returns:
        Asset type: 'forex', 'crypto', 'metal', or 'unknown'
    """
    symbol_upper = symbol.upper()
    
    # Metals
    if symbol_upper.startswith('XAU') or symbol_upper.startswith('XAG') or 'GOLD' in symbol_upper or 'SILVER' in symbol_upper:
        return 'metal'
    
    # Oil/Energy (FIXED: Added oil support)
    if ('OIL' in symbol_upper or 'WTI' in symbol_upper or 'BRENT' in symbol_upper or 
        symbol_upper.startswith('USO') or symbol_upper.startswith('UKO') or
        'CRUDE' in symbol_upper or symbol_upper.startswith('XTI') or symbol_upper.startswith('XBR')):
        return 'oil'
    
    # Cryptocurrencies
    crypto_prefixes = ['BTC', 'ETH', 'LTC', 'XRP', 'BCH', 'EOS', 'ADA', 'DOT', 'LINK', 'UNI']
    for prefix in crypto_prefixes:
        if symbol_upper.startswith(prefix):
            return 'crypto'
    
    # Forex (default)
    forex_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF']
    for currency in forex_currencies:
        if currency in symbol_upper:
            return 'forex'
    
    return 'unknown'


def is_forex(symbol: str) -> bool:
    """Check if symbol is forex"""
    return detect_asset_type(symbol) == 'forex'


def is_crypto(symbol: str) -> bool:
    """Check if symbol is cryptocurrency"""
    return detect_asset_type(symbol) == 'crypto'


def is_metal(symbol: str) -> bool:
    """Check if symbol is metal"""
    return detect_asset_type(symbol) == 'metal'


def is_oil(symbol: str) -> bool:
    """Check if symbol is oil/energy"""
    return detect_asset_type(symbol) == 'oil'

