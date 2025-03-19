"""
Module de logging pour le système de trading.
"""

import logging
from typing import Optional
from datetime import datetime

class TradingLogger:
    """Logger personnalisé pour le système de trading."""
    
    def __init__(self, name: str = "trading", level: int = logging.INFO):
        """
        Initialise le logger.
        
        Args:
            name: Nom du logger
            level: Niveau de logging
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Ajout d'un handler console si aucun n'existe
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, message: str, **kwargs):
        """Log un message de niveau INFO."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log un message de niveau WARNING."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log un message de niveau ERROR."""
        self.logger.error(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log un message de niveau DEBUG."""
        self.logger.debug(message, **kwargs)
