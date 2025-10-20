"""
Configuration validator
"""

from typing import Tuple, List


class ConfigValidator:
    """Validates trading bot configuration"""
    
    def validate(self, config: dict) -> Tuple[bool, List[str]]:
        """
        Validate configuration
        
        Args:
            config: Configuration dictionary
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check MT5 connection
        if 'mt5_login' not in config:
            errors.append("Missing 'mt5_login'")
        if 'mt5_password' not in config:
            errors.append("Missing 'mt5_password'")
        if 'mt5_server' not in config:
            errors.append("Missing 'mt5_server'")
        
        # Check symbols
        if 'symbols' not in config or not config['symbols']:
            errors.append("No trading symbols specified")
        
        # Check risk management
        if 'risk_management' in config:
            rm = config['risk_management']
            
            if 'risk_per_trade' in rm:
                if not 0 < rm['risk_per_trade'] < 0.1:
                    errors.append("risk_per_trade should be between 0 and 0.1 (0-10%)")
            
            if 'max_positions' in rm:
                if rm['max_positions'] < 1:
                    errors.append("max_positions must be at least 1")
        
        return (len(errors) == 0, errors)

