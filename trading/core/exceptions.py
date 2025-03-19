"""
Module définissant les exceptions personnalisées pour le système de trading.
"""

class TradingException(Exception):
    """Exception de base pour toutes les erreurs du système de trading."""
    pass

class ValidationError(TradingException):
    """Exception levée lorsque la validation des données échoue."""
    pass

class CalculationError(TradingException):
    """Exception levée lorsqu'une erreur de calcul se produit."""
    pass

class ConfigurationError(TradingException):
    """Exception levée lorsqu'une erreur de configuration se produit."""
    pass

class DataError(TradingException):
    """Exception levée lorsqu'une erreur liée aux données se produit."""
    pass

class InvalidDataError(DataError):
    """Exception levée lorsque les données ne respectent pas le format attendu."""
    pass

class InsufficientDataError(DataError):
    """Exception levée lorsque les données sont insuffisantes pour un traitement."""
    pass

class APIError(TradingException):
    """Exception levée lorsqu'une erreur d'API se produit."""
    pass

class ExchangeError(TradingException):
    """Exception levée lorsqu'une erreur d'échange se produit."""
    pass

class OrderError(TradingException):
    """Exception levée lorsqu'une erreur d'ordre se produit."""
    pass

class PositionError(TradingException):
    """Exception levée lorsqu'une erreur de position se produit."""
    pass

class StrategyError(TradingException):
    """Exception levée lorsqu'une erreur de stratégie se produit."""
    pass

class ModelError(TradingException):
    """Exception levée lorsqu'une erreur de modèle se produit."""
    pass

class OptimizationError(TradingException):
    """Exception levée lorsqu'une erreur d'optimisation se produit."""
    pass

class BacktestError(TradingException):
    """Exception levée lorsqu'une erreur de backtest se produit."""
    pass

class ExecutionError(TradingException):
    """Exception levée lorsqu'une erreur d'exécution se produit."""
    pass

class RiskError(TradingException):
    """Exception levée lorsqu'une erreur de gestion des risques se produit."""
    pass
