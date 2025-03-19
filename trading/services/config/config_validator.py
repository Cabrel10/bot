"""
Configuration validator service.
"""

from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Validation result data structure."""
    is_valid: bool
    errors: List[str]
    warnings: List[str] = None

class ConfigValidator:
    """Validates trading system configuration."""
    
    def __init__(self):
        """Initialize config validator."""
        self._valid_exchanges = ['binance', 'bitget', 'kucoin']
        self._valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        self._valid_position_sizing = ['fixed', 'risk_based', 'kelly']
        self._valid_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT']
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration."""
        errors = []
        warnings = []
        
        # Validate exchange configuration
        if 'exchange' not in config:
            errors.append("Missing exchange configuration")
        else:
            exchange_errors = self._validate_exchange_config(config['exchange'])
            errors.extend(exchange_errors)
        
        # Validate trading configuration
        if 'trading' not in config:
            errors.append("Missing trading configuration")
        else:
            trading_errors = self._validate_trading_config(config['trading'])
            errors.extend(trading_errors)
        
        # Validate risk configuration
        if 'risk' not in config:
            errors.append("Missing risk configuration")
        else:
            risk_errors = self._validate_risk_config(config['risk'])
            errors.extend(risk_errors)
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _validate_exchange_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate exchange configuration."""
        errors = []
        
        # Validate exchange name
        if 'name' not in config:
            errors.append("Missing exchange name")
        elif config['name'] not in self._valid_exchanges:
            errors.append(f"Invalid exchange name: {config['name']}")
        
        # Validate API credentials
        if 'api_key' not in config:
            errors.append("Missing API key")
        if 'api_secret' not in config:
            errors.append("Missing API secret")
        
        # Validate timeframes
        if 'timeframes' not in config:
            errors.append("Missing timeframes")
        else:
            for timeframe in config['timeframes']:
                if timeframe not in self._valid_timeframes:
                    errors.append(f"Invalid timeframe: {timeframe}")
        
        # Validate symbols
        if 'symbols' not in config:
            errors.append("Missing symbols")
        else:
            for symbol in config['symbols']:
                if symbol not in self._valid_symbols:
                    errors.append(f"Invalid symbol: {symbol}")
        
        return errors
    
    def _validate_trading_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate trading configuration."""
        errors = []
        
        # Validate initial capital
        if 'initial_capital' not in config:
            errors.append("Missing initial capital")
        elif not isinstance(config['initial_capital'], (int, float)) or config['initial_capital'] <= 0:
            errors.append("Invalid initial capital")
        
        # Validate position size
        if 'max_position_size' not in config:
            errors.append("Missing max position size")
        elif not isinstance(config['max_position_size'], (int, float)) or config['max_position_size'] <= 0:
            errors.append("Invalid max position size")
        
        # Validate stop loss
        if 'stop_loss_pct' not in config:
            errors.append("Missing stop loss percentage")
        elif not isinstance(config['stop_loss_pct'], (int, float)) or config['stop_loss_pct'] <= 0 or config['stop_loss_pct'] > 0.1:
            errors.append("Invalid stop loss percentage")
        
        # Validate take profit
        if 'take_profit_pct' not in config:
            errors.append("Missing take profit percentage")
        elif not isinstance(config['take_profit_pct'], (int, float)) or config['take_profit_pct'] <= 0:
            errors.append("Invalid take profit percentage")
        
        # Validate stop loss vs take profit
        if 'stop_loss_pct' in config and 'take_profit_pct' in config:
            if config['take_profit_pct'] <= config['stop_loss_pct']:
                errors.append("Take profit must be greater than stop loss")
        
        return errors
    
    def _validate_risk_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate risk configuration."""
        errors = []
        
        # Validate max drawdown
        if 'max_drawdown' not in config:
            errors.append("Missing max drawdown")
        elif not isinstance(config['max_drawdown'], (int, float)) or config['max_drawdown'] <= 0 or config['max_drawdown'] > 1:
            errors.append("Invalid max drawdown")
        
        # Validate max leverage
        if 'max_leverage' not in config:
            errors.append("Missing max leverage")
        elif not isinstance(config['max_leverage'], (int, float)) or config['max_leverage'] < 1:
            errors.append("Invalid max leverage")
        
        # Validate position sizing method
        if 'position_sizing' not in config:
            errors.append("Missing position sizing method")
        elif config['position_sizing'] not in self._valid_position_sizing:
            errors.append(f"Invalid position sizing method: {config['position_sizing']}")
        
        return errors 