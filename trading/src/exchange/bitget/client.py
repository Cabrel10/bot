"""
Module client de base pour Bitget.

Ce module implémente la classe de base pour les clients Bitget,
fournissant les fonctionnalités communes aux clients spot et futures.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, Union, List, Callable
import aiohttp
from datetime import datetime

from ..base import ExchangeClient
from .constants import (
    DEFAULT_HEADERS,
    HTTP_TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY,
    RETRY_MULTIPLIER,
    MAX_RETRY_DELAY,
    WS_CONNECTION_TIMEOUT,
    WS_PING_INTERVAL,
    WS_PING_TIMEOUT,
    WS_CLOSE_TIMEOUT
)
from .utils import (
    generate_signature,
    get_timestamp,
    create_query_string,
    handle_response,
    parse_ws_message
)
from .errors import (
    BitgetError,
    BitgetAPIError,
    BitgetRequestError,
    BitgetTimeoutError,
    BitgetWSError,
    BitgetAuthenticationError,
    get_error_class
)

logger = logging.getLogger(__name__)

class BitgetClientBase(ExchangeClient):
    """Classe de base pour les clients Bitget."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        passphrase: Optional[str] = None,
        testnet: bool = False
    ):
        """
        Initialise le client Bitget.
        
        Args:
            api_key: Clé API (optionnelle)
            api_secret: Secret API (optionnel)
            passphrase: Phrase secrète API (optionnelle)
            testnet: Utiliser le testnet
        """
        super().__init__()
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.testnet = testnet
        
        self._session: Optional[aiohttp.ClientSession] = None
        self.ws_manager = None
        self._closed = False
        
    @property
    def base_url(self) -> str:
        """URL de base de l'API."""
        raise NotImplementedError("Doit être implémenté par les sous-classes")
        
    @property
    def ws_url(self) -> str:
        """URL de base du WebSocket."""
        raise NotImplementedError("Doit être implémenté par les sous-classes")
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Obtient une session HTTP.
        
        Returns:
            Session HTTP
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers=DEFAULT_HEADERS,
                timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
            )
        return self._session
        
    def _get_headers(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        signed: bool = False
    ) -> Dict[str, str]:
        """
        Génère les en-têtes pour une requête.
        
        Args:
            method: Méthode HTTP
            endpoint: Endpoint de l'API
            params: Paramètres de requête (optionnel)
            data: Corps de la requête (optionnel)
            signed: Requête authentifiée
            
        Returns:
            En-têtes HTTP
            
        Raises:
            BitgetAuthenticationError: Si les clés API sont manquantes
        """
        headers = DEFAULT_HEADERS.copy()
        
        if signed:
            if not all([self.api_key, self.api_secret, self.passphrase]):
                raise BitgetAuthenticationError(
                    "API key, secret and passphrase are required for authenticated endpoints"
                )
                
            timestamp = get_timestamp()
            
            # Construction du message à signer
            request_path = endpoint
            if params:
                request_path += '?' + create_query_string(params)
                
            signature = generate_signature(
                timestamp=timestamp,
                method=method,
                request_path=request_path,
                body=data,
                secret_key=self.api_secret
            )
            
            headers.update({
                'ACCESS-KEY': self.api_key,
                'ACCESS-SIGN': signature,
                'ACCESS-TIMESTAMP': timestamp,
                'ACCESS-PASSPHRASE': self.passphrase
            })
            
        return headers
        
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        signed: bool = False,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """
        Envoie une requête HTTP à l'API.
        
        Args:
            method: Méthode HTTP
            endpoint: Endpoint de l'API
            params: Paramètres de requête (optionnel)
            data: Corps de la requête (optionnel)
            signed: Requête authentifiée
            retry_count: Nombre de tentatives
            
        Returns:
            Réponse de l'API
            
        Raises:
            BitgetAPIError: Si l'API retourne une erreur
            BitgetRequestError: Si la requête échoue
            BitgetTimeoutError: Si la requête expire
        """
        session = await self._get_session()
        headers = self._get_headers(method, endpoint, params, data, signed)
        
        url = self.base_url + endpoint
        
        try:
            async with session.request(
                method,
                url,
                params=params,
                json=data,
                headers=headers
            ) as response:
                if response.status == 429:  # Rate limit
                    if retry_count < MAX_RETRIES:
                        delay = min(
                            RETRY_DELAY * (RETRY_MULTIPLIER ** retry_count),
                            MAX_RETRY_DELAY
                        )
                        await asyncio.sleep(delay)
                        return await self._request(
                            method,
                            endpoint,
                            params,
                            data,
                            signed,
                            retry_count + 1
                        )
                    raise BitgetAPIError(
                        message="Rate limit exceeded",
                        code="429",
                        request_info={'url': url, 'method': method}
                    )
                    
                response_data = await response.json()
                
                if response.status >= 400:
                    error_class = get_error_class(str(response.status))
                    raise error_class(
                        message=response_data.get('msg', 'Unknown error'),
                        code=response_data.get('code'),
                        request_info={'url': url, 'method': method}
                    )
                    
                return handle_response(response_data)
                
        except asyncio.TimeoutError:
            raise BitgetTimeoutError(
                message="Request timed out",
                request_info={'url': url, 'method': method}
            )
        except aiohttp.ClientError as e:
            raise BitgetRequestError(
                message=str(e),
                request_info={'url': url, 'method': method}
            )
            
    async def _create_ws_connection(
        self,
        stream: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Crée une connexion WebSocket.
        
        Args:
            stream: Flux à écouter
            callback: Fonction de rappel pour les messages
            
        Raises:
            BitgetWSError: Si la connexion échoue
        """
        try:
            session = await self._get_session()
            
            async with session.ws_connect(
                self.ws_url,
                timeout=WS_CONNECTION_TIMEOUT,
                heartbeat=WS_PING_INTERVAL,
                receive_timeout=WS_PING_TIMEOUT
            ) as ws:
                # Authentification si nécessaire
                if self.api_key and self.api_secret and self.passphrase:
                    timestamp = get_timestamp()
                    signature = generate_signature(
                        timestamp=timestamp,
                        method='GET',
                        request_path='/users/self/verify',
                        secret_key=self.api_secret
                    )
                    
                    await ws.send_json({
                        'op': 'login',
                        'args': [{
                            'apiKey': self.api_key,
                            'passphrase': self.passphrase,
                            'timestamp': timestamp,
                            'sign': signature
                        }]
                    })
                    
                    response = await ws.receive_json()
                    if not response.get('success'):
                        raise BitgetWSError(
                            message="WebSocket authentication failed",
                            code=response.get('code')
                        )
                    
                # Souscription au flux
                await ws.send_json({
                    'op': 'subscribe',
                    'args': [stream]
                })
                
                response = await ws.receive_json()
                if not response.get('success'):
                    raise BitgetWSError(
                        message="WebSocket subscription failed",
                        code=response.get('code')
                    )
                    
                # Boucle de réception des messages
                while not self._closed:
                    try:
                        msg = await ws.receive_json()
                        if msg.get('event') == 'error':
                            logger.error(f"WebSocket error: {msg}")
                            continue
                            
                        parsed_msg = parse_ws_message(msg)
                        callback(parsed_msg)
                        
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message: {e}")
                        
        except Exception as e:
            raise BitgetWSError(
                message=f"WebSocket connection failed: {str(e)}"
            )
            
    async def close(self) -> None:
        """Ferme les connexions."""
        self._closed = True
        
        if self.ws_manager:
            await self.ws_manager.close()
            
        if self._session and not self._session.closed:
            await self._session.close()
            
    async def __aenter__(self):
        """Support du context manager."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Support du context manager."""
        await self.close() 