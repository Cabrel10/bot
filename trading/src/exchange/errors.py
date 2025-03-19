"""
Module de gestion des erreurs pour les exchanges.

Ce module définit les différentes classes d'erreurs qui peuvent
survenir lors de l'interaction avec les exchanges.
"""

class ExchangeError(Exception):
    """Erreur de base pour les exchanges."""
    pass

class AuthenticationError(ExchangeError):
    """Erreur d'authentification."""
    pass

class InsufficientFundsError(ExchangeError):
    """Erreur de fonds insuffisants."""
    pass

class InvalidOrderError(ExchangeError):
    """Erreur de paramètres d'ordre invalides."""
    pass

class OrderNotFoundError(ExchangeError):
    """Erreur d'ordre non trouvé."""
    pass

class RateLimitError(ExchangeError):
    """Erreur de limite de taux atteinte."""
    pass

class NetworkError(ExchangeError):
    """Erreur de réseau."""
    pass

class WebSocketError(ExchangeError):
    """Erreur de WebSocket."""
    pass

class MarketError(ExchangeError):
    """Erreur liée au marché."""
    pass

class PositionError(ExchangeError):
    """Erreur liée aux positions."""
    pass

class ConfigurationError(ExchangeError):
    """Erreur de configuration."""
    pass

class MaintenanceError(ExchangeError):
    """Erreur de maintenance de l'exchange."""
    pass

class PermissionError(ExchangeError):
    """Erreur de permission."""
    pass

class BinanceError(ExchangeError):
    """Erreur spécifique à Binance."""
    
    def __init__(self, code: int, message: str):
        """
        Initialise l'erreur Binance.
        
        Args:
            code: Code d'erreur Binance
            message: Message d'erreur
        """
        self.code = code
        super().__init__(f"Binance error {code}: {message}")

class BitgetError(ExchangeError):
    """Erreur spécifique à Bitget."""
    
    def __init__(self, code: str, message: str):
        """
        Initialise l'erreur Bitget.
        
        Args:
            code: Code d'erreur Bitget
            message: Message d'erreur
        """
        self.code = code
        super().__init__(f"Bitget error {code}: {message}")

def handle_binance_error(code: int, message: str) -> ExchangeError:
    """
    Convertit une erreur Binance en erreur appropriée.
    
    Args:
        code: Code d'erreur Binance
        message: Message d'erreur
        
    Returns:
        Exception appropriée
    """
    error_map = {
        -1002: RateLimitError,
        -1003: RateLimitError,
        -1010: InsufficientFundsError,
        -1021: NetworkError,
        -1022: AuthenticationError,
        -2010: InvalidOrderError,
        -2011: OrderNotFoundError,
        -2013: OrderNotFoundError,
        -2014: InvalidOrderError,
        -2015: RateLimitError
    }
    
    error_class = error_map.get(code, BinanceError)
    if error_class == BinanceError:
        return BinanceError(code, message)
    return error_class(message)

def handle_bitget_error(code: str, message: str) -> ExchangeError:
    """
    Convertit une erreur Bitget en erreur appropriée.
    
    Args:
        code: Code d'erreur Bitget
        message: Message d'erreur
        
    Returns:
        Exception appropriée
    """
    error_map = {
        '40001': AuthenticationError,
        '40002': PermissionError,
        '40003': RateLimitError,
        '40004': InsufficientFundsError,
        '40005': InvalidOrderError,
        '40006': OrderNotFoundError,
        '40007': MarketError,
        '40008': PositionError,
        '40009': ConfigurationError,
        '40010': MaintenanceError
    }
    
    error_class = error_map.get(code, BitgetError)
    if error_class == BitgetError:
        return BitgetError(code, message)
    return error_class(message) 