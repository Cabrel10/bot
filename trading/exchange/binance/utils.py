"""
Module d'utilitaires pour le client Binance.

Ce module fournit des fonctions utilitaires pour le client Binance,
notamment pour la signature des requêtes, la conversion des données,
et la gestion des erreurs.
"""

import hmac
import hashlib
import time
from typing import Dict, Any, Optional, Union
from urllib.parse import urlencode
import json
from decimal import Decimal
from datetime import datetime, timezone

from ..errors import handle_binance_error
from .constants import (
    ORDER_TYPES,
    ORDER_SIDES,
    ORDER_STATUS,
    TIME_IN_FORCE,
    VALID_INTERVALS
)

def generate_signature(secret_key: str, query_string: str) -> str:
    """
    Génère une signature HMAC SHA256 pour l'authentification.
    
    Args:
        secret_key: Clé secrète API
        query_string: Chaîne de requête à signer
        
    Returns:
        Signature hexadécimale
    """
    return hmac.new(
        secret_key.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def get_timestamp() -> int:
    """
    Obtient le timestamp actuel en millisecondes.
    
    Returns:
        Timestamp en millisecondes
    """
    return int(time.time() * 1000)

def create_query_string(params: Dict[str, Any]) -> str:
    """
    Crée une chaîne de requête à partir des paramètres.
    
    Args:
        params: Paramètres de la requête
        
    Returns:
        Chaîne de requête encodée
    """
    return urlencode(sort_params(params))

def sort_params(params: Dict[str, Any]) -> Dict[str, str]:
    """
    Trie et convertit les paramètres pour la signature.
    
    Args:
        params: Paramètres à trier
        
    Returns:
        Paramètres triés et convertis
    """
    return {
        key: str(value)
        for key, value in sorted(params.items())
        if value is not None
    }

def validate_symbol(symbol: str) -> str:
    """
    Valide et normalise un symbole.
    
    Args:
        symbol: Symbole à valider
        
    Returns:
        Symbole normalisé
        
    Raises:
        ValueError: Si le symbole est invalide
    """
    if not isinstance(symbol, str):
        raise ValueError("Le symbole doit être une chaîne de caractères")
    
    symbol = symbol.upper().strip()
    if not symbol or '/' in symbol:
        raise ValueError("Format de symbole invalide")
    
    return symbol

def validate_interval(interval: str) -> str:
    """
    Valide un intervalle de temps.
    
    Args:
        interval: Intervalle à valider
        
    Returns:
        Intervalle validé
        
    Raises:
        ValueError: Si l'intervalle est invalide
    """
    if interval not in VALID_INTERVALS:
        raise ValueError(
            f"Intervalle invalide. Valeurs possibles: {', '.join(VALID_INTERVALS)}"
        )
    return interval

def validate_order_type(order_type: str) -> str:
    """
    Valide un type d'ordre.
    
    Args:
        order_type: Type d'ordre à valider
        
    Returns:
        Type d'ordre validé
        
    Raises:
        ValueError: Si le type d'ordre est invalide
    """
    order_type = order_type.upper()
    if order_type not in ORDER_TYPES:
        raise ValueError(
            f"Type d'ordre invalide. Valeurs possibles: {', '.join(ORDER_TYPES)}"
        )
    return ORDER_TYPES[order_type]

def validate_order_side(side: str) -> str:
    """
    Valide un côté d'ordre.
    
    Args:
        side: Côté d'ordre à valider
        
    Returns:
        Côté d'ordre validé
        
    Raises:
        ValueError: Si le côté d'ordre est invalide
    """
    side = side.upper()
    if side not in ORDER_SIDES:
        raise ValueError(
            f"Côté d'ordre invalide. Valeurs possibles: {', '.join(ORDER_SIDES)}"
        )
    return ORDER_SIDES[side]

def validate_time_in_force(time_in_force: str) -> str:
    """
    Valide un time in force.
    
    Args:
        time_in_force: Time in force à valider
        
    Returns:
        Time in force validé
        
    Raises:
        ValueError: Si le time in force est invalide
    """
    time_in_force = time_in_force.upper()
    if time_in_force not in TIME_IN_FORCE:
        raise ValueError(
            f"Time in force invalide. Valeurs possibles: {', '.join(TIME_IN_FORCE)}"
        )
    return TIME_IN_FORCE[time_in_force]

def parse_timestamp(timestamp: Union[int, float, str, datetime]) -> int:
    """
    Convertit un timestamp en millisecondes.
    
    Args:
        timestamp: Timestamp à convertir
        
    Returns:
        Timestamp en millisecondes
    """
    if isinstance(timestamp, datetime):
        return int(timestamp.timestamp() * 1000)
    elif isinstance(timestamp, (int, float)):
        return int(timestamp * 1000 if timestamp < 1e12 else timestamp)
    elif isinstance(timestamp, str):
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return int(dt.timestamp() * 1000)
        except ValueError:
            raise ValueError("Format de timestamp invalide")
    else:
        raise ValueError("Type de timestamp non supporté")

def format_number(
    number: Union[int, float, str, Decimal],
    precision: Optional[int] = None
) -> str:
    """
    Formate un nombre pour l'API Binance.
    
    Args:
        number: Nombre à formater
        precision: Précision décimale
        
    Returns:
        Nombre formaté
    """
    if isinstance(number, str):
        number = Decimal(number)
    elif isinstance(number, (int, float)):
        number = Decimal(str(number))
    
    if precision is not None:
        number = round(number, precision)
    
    return format(number, 'f')

def parse_order_status(status: str) -> str:
    """
    Parse le statut d'un ordre.
    
    Args:
        status: Statut à parser
        
    Returns:
        Statut parsé
        
    Raises:
        ValueError: Si le statut est invalide
    """
    status = status.upper()
    if status not in ORDER_STATUS:
        raise ValueError(
            f"Statut d'ordre invalide. Valeurs possibles: {', '.join(ORDER_STATUS)}"
        )
    return ORDER_STATUS[status]

def handle_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gère la réponse de l'API Binance.
    
    Args:
        response: Réponse à gérer
        
    Returns:
        Réponse traitée
        
    Raises:
        BinanceError: Si la réponse contient une erreur
    """
    if 'code' in response and 'msg' in response:
        raise handle_binance_error(response['code'], response['msg'])
    return response

def parse_ws_message(message: Union[str, bytes, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Parse un message WebSocket.
    
    Args:
        message: Message à parser
        
    Returns:
        Message parsé
    """
    if isinstance(message, bytes):
        message = message.decode('utf-8')
    if isinstance(message, str):
        message = json.loads(message)
    
    if 'error' in message:
        raise handle_binance_error(
            message['error'].get('code', -1),
            message['error'].get('msg', 'Unknown error')
        )
    
    return message 