"""
Module d'erreurs pour le client Bitget.

Ce module définit les exceptions spécifiques au client Bitget
pour une meilleure gestion des erreurs.
"""

from typing import Optional, Dict, Any

class BitgetError(Exception):
    """Classe de base pour les erreurs Bitget."""
    
    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        request_info: Optional[Dict[str, Any]] = None
    ):
        """
        Initialise une erreur Bitget.
        
        Args:
            message: Message d'erreur
            code: Code d'erreur (optionnel)
            request_info: Informations sur la requête (optionnel)
        """
        self.code = code
        self.message = message
        self.request_info = request_info or {}
        super().__init__(self.message)
        
    def __str__(self) -> str:
        """
        Représentation en chaîne de l'erreur.
        
        Returns:
            Message d'erreur formaté
        """
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message

class BitgetAPIError(BitgetError):
    """Erreur retournée par l'API REST."""
    pass

class BitgetRequestError(BitgetError):
    """Erreur lors de l'envoi d'une requête."""
    pass

class BitgetParamsError(BitgetError):
    """Erreur de paramètres invalides."""
    pass

class BitgetTimeoutError(BitgetError):
    """Erreur de timeout."""
    pass

class BitgetWSError(BitgetError):
    """Erreur WebSocket."""
    pass

class BitgetAuthenticationError(BitgetError):
    """Erreur d'authentification."""
    pass

class BitgetOrderError(BitgetError):
    """Erreur liée aux ordres."""
    pass

class BitgetPositionError(BitgetError):
    """Erreur liée aux positions."""
    pass

class BitgetBalanceError(BitgetError):
    """Erreur liée aux soldes."""
    pass

class BitgetRateLimitError(BitgetError):
    """Erreur de limite de taux atteinte."""
    pass

# Mapping des codes d'erreur vers les exceptions
ERROR_CODES = {
    # Erreurs d'authentification
    '40001': BitgetAuthenticationError,
    '40002': BitgetAuthenticationError,
    '40003': BitgetAuthenticationError,
    '40004': BitgetAuthenticationError,
    
    # Erreurs de paramètres
    '40101': BitgetParamsError,
    '40102': BitgetParamsError,
    '40103': BitgetParamsError,
    
    # Erreurs d'ordres
    '41001': BitgetOrderError,
    '41002': BitgetOrderError,
    '41003': BitgetOrderError,
    '41004': BitgetOrderError,
    
    # Erreurs de positions
    '42001': BitgetPositionError,
    '42002': BitgetPositionError,
    '42003': BitgetPositionError,
    
    # Erreurs de solde
    '43001': BitgetBalanceError,
    '43002': BitgetBalanceError,
    
    # Erreurs de limite de taux
    '45001': BitgetRateLimitError,
    '45002': BitgetRateLimitError
}

def get_error_class(code: str) -> type:
    """
    Obtient la classe d'exception appropriée pour un code d'erreur.
    
    Args:
        code: Code d'erreur
        
    Returns:
        Classe d'exception
    """
    return ERROR_CODES.get(code, BitgetAPIError) 