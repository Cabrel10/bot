"""
Module d'utilitaires pour le client Bitget.

Ce module fournit des fonctions utilitaires pour le client Bitget,
notamment pour la validation des paramètres, le formatage des données,
et la signature des requêtes.
"""

import hmac
import hashlib
import base64
import json
import time
from typing import Dict, Any, Optional, Union
from datetime import datetime
from urllib.parse import urlencode

from .constants import (
    VALID_INTERVALS,
    ORDER_TYPES,
    ORDER_SIDES,
    ORDER_STATUS,
    POSITION_SIDE,
    MARGIN_TYPE
)

def generate_signature(
    timestamp: str,
    method: str,
    request_path: str,
    body: Optional[Union[Dict, str]] = None,
    secret_key: Optional[str] = None
) -> str:
    """
    Génère une signature HMAC SHA256 pour l'authentification.
    
    Args:
        timestamp: Timestamp en millisecondes
        method: Méthode HTTP (GET, POST, etc.)
        request_path: Chemin de la requête
        body: Corps de la requête (optionnel)
        secret_key: Clé secrète API
        
    Returns:
        Signature encodée en base64
    """
    if not secret_key:
        raise ValueError("Secret key is required for signature generation")
        
    if isinstance(body, dict):
        body = json.dumps(body)
    
    message = timestamp + method.upper() + request_path + (body or '')
    
    signature = hmac.new(
        secret_key.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).digest()
    
    return base64.b64encode(signature).decode('utf-8')

def get_timestamp() -> str:
    """
    Obtient le timestamp actuel en millisecondes.
    
    Returns:
        Timestamp en millisecondes sous forme de chaîne
    """
    return str(int(time.time() * 1000))

def create_query_string(params: Dict[str, Any]) -> str:
    """
    Crée une chaîne de requête à partir des paramètres.
    
    Args:
        params: Dictionnaire des paramètres
        
    Returns:
        Chaîne de requête encodée
    """
    return urlencode(sorted(params.items()))

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
    if not symbol:
        raise ValueError("Le symbole ne peut pas être vide")
        
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
    interval = interval.lower()
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
            f"Type d'ordre invalide. Valeurs possibles: {', '.join(ORDER_TYPES.keys())}"
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
            f"Côté d'ordre invalide. Valeurs possibles: {', '.join(ORDER_SIDES.keys())}"
        )
    return ORDER_SIDES[side]

def validate_position_side(side: str) -> str:
    """
    Valide un côté de position.
    
    Args:
        side: Côté de position à valider
        
    Returns:
        Côté de position validé
        
    Raises:
        ValueError: Si le côté de position est invalide
    """
    side = side.upper()
    if side not in POSITION_SIDE:
        raise ValueError(
            f"Côté de position invalide. Valeurs possibles: {', '.join(POSITION_SIDE.keys())}"
        )
    return POSITION_SIDE[side]

def validate_margin_type(margin_type: str) -> str:
    """
    Valide un type de marge.
    
    Args:
        margin_type: Type de marge à valider
        
    Returns:
        Type de marge validé
        
    Raises:
        ValueError: Si le type de marge est invalide
    """
    margin_type = margin_type.upper()
    if margin_type not in MARGIN_TYPE:
        raise ValueError(
            f"Type de marge invalide. Valeurs possibles: {', '.join(MARGIN_TYPE.keys())}"
        )
    return MARGIN_TYPE[margin_type]

def parse_timestamp(timestamp: Union[int, float, str, datetime]) -> int:
    """
    Convertit différents formats de timestamp en millisecondes.
    
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
            return int(float(timestamp) * 1000)
        except ValueError:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return int(dt.timestamp() * 1000)
    else:
        raise ValueError("Format de timestamp non supporté")

def format_number(number: Union[int, float, str]) -> str:
    """
    Formate un nombre pour l'API.
    
    Args:
        number: Nombre à formater
        
    Returns:
        Nombre formaté
    """
    if isinstance(number, str):
        number = float(number)
    return f"{number:f}".rstrip('0').rstrip('.')

def parse_order_status(status: str) -> str:
    """
    Parse et valide un statut d'ordre.
    
    Args:
        status: Statut à parser
        
    Returns:
        Statut validé
        
    Raises:
        ValueError: Si le statut est invalide
    """
    status = status.upper()
    if status not in ORDER_STATUS:
        raise ValueError(
            f"Statut d'ordre invalide. Valeurs possibles: {', '.join(ORDER_STATUS.keys())}"
        )
    return ORDER_STATUS[status]

def handle_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gère la réponse de l'API et vérifie les erreurs.
    
    Args:
        response: Réponse de l'API
        
    Returns:
        Données de la réponse
        
    Raises:
        BitgetAPIError: Si la réponse contient une erreur
    """
    if not isinstance(response, dict):
        raise ValueError("La réponse doit être un dictionnaire")
        
    if response.get('code') != '00000':
        raise BitgetAPIError(
            code=response.get('code'),
            message=response.get('msg', 'Unknown error')
        )
        
    return response.get('data', response)

def parse_ws_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse un message WebSocket.
    
    Args:
        message: Message WebSocket
        
    Returns:
        Message parsé
        
    Raises:
        BitgetWSError: Si le message contient une erreur
    """
    if not isinstance(message, dict):
        raise ValueError("Le message doit être un dictionnaire")
        
    if message.get('code') != '00000':
        raise BitgetWSError(
            code=message.get('code'),
            message=message.get('msg', 'Unknown error')
        )
        
    return message.get('data', message) 