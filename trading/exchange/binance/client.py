"""
Module client Binance.

Ce module implémente le client Binance avec support pour le spot
et les futures, incluant les fonctionnalités de trading et de
streaming de données en temps réel.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import json
from decimal import Decimal

from ..base import ExchangeClient, WebSocketManager
from ..errors import (
    AuthenticationError,
    NetworkError,
    RateLimitError,
    handle_binance_error
)
from .constants import (
    BINANCE_SPOT_MAINNET,
    BINANCE_SPOT_TESTNET,
    BINANCE_FUTURES_MAINNET,
    BINANCE_FUTURES_TESTNET,
    BINANCE_WS_SPOT_MAINNET,
    BINANCE_WS_SPOT_TESTNET,
    BINANCE_WS_FUTURES_MAINNET,
    BINANCE_WS_FUTURES_TESTNET,
    ENDPOINTS,
    DEFAULT_HEADERS,
    HTTP_TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY,
    RETRY_MULTIPLIER,
    MAX_RETRY_DELAY,
    DEFAULT_RECV_WINDOW
)
from .utils import (
    generate_signature,
    get_timestamp,
    create_query_string,
    validate_symbol,
    validate_interval,
    validate_order_type,
    validate_order_side,
    validate_time_in_force,
    parse_timestamp,
    format_number,
    parse_order_status,
    handle_response,
    parse_ws_message
)

logger = logging.getLogger(__name__)

class BinanceClientBase(ExchangeClient):
    """Classe de base pour les clients Binance."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True,
        recv_window: int = DEFAULT_RECV_WINDOW
    ):
        """
        Initialise le client Binance.
        
        Args:
            api_key: Clé API
            api_secret: Secret API
            testnet: Utilisation du testnet
            recv_window: Fenêtre de réception en millisecondes
        """
        super().__init__(api_key, api_secret, testnet)
        
        self.recv_window = recv_window
        self.session: Optional[aiohttp.ClientSession] = None
        self._initialized = False
        
    @property
    def base_url(self) -> str:
        """URL de base de l'API."""
        raise NotImplementedError
        
    @property
    def ws_url(self) -> str:
        """URL de base du WebSocket."""
        raise NotImplementedError
        
    async def initialize(self) -> None:
        """Initialise la connexion à l'exchange."""
        if not self._initialized:
            self.session = aiohttp.ClientSession(
                headers=DEFAULT_HEADERS,
                timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
            )
            self._initialized = True
            
    async def close(self) -> None:
        """Ferme la connexion à l'exchange."""
        if self.session:
            await self.session.close()
            self.session = None
        self._initialized = False
        
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
        api_version: str = 'v3'
    ) -> Dict[str, Any]:
        """
        Effectue une requête HTTP vers l'API Binance.
        
        Args:
            method: Méthode HTTP
            endpoint: Endpoint de l'API
            params: Paramètres de la requête
            signed: Requête signée ou non
            api_version: Version de l'API
            
        Returns:
            Réponse de l'API
            
        Raises:
            NetworkError: En cas d'erreur réseau
            AuthenticationError: En cas d'erreur d'authentification
            RateLimitError: En cas de dépassement de limite
        """
        if not self._initialized:
            await self.initialize()
            
        url = f"{self.base_url}{endpoint}"
        headers = {}
        params = params or {}
        
        if signed:
            if not self.api_key or not self.api_secret:
                raise AuthenticationError("API key and secret required")
                
            params['timestamp'] = get_timestamp()
            params['recvWindow'] = self.recv_window
            query_string = create_query_string(params)
            params['signature'] = generate_signature(self.api_secret, query_string)
            headers['X-MBX-APIKEY'] = self.api_key
            
        elif self.api_key:
            headers['X-MBX-APIKEY'] = self.api_key
            
        retry_count = 0
        retry_delay = RETRY_DELAY
        
        while retry_count < MAX_RETRIES:
            try:
                if method == 'GET':
                    async with self.session.get(
                        url,
                        params=params,
                        headers=headers
                    ) as response:
                        data = await response.json()
                        
                elif method == 'POST':
                    async with self.session.post(
                        url,
                        params=params,
                        headers=headers
                    ) as response:
                        data = await response.json()
                        
                elif method == 'DELETE':
                    async with self.session.delete(
                        url,
                        params=params,
                        headers=headers
                    ) as response:
                        data = await response.json()
                        
                else:
                    raise ValueError(f"Méthode HTTP non supportée: {method}")
                    
                return handle_response(data)
                
            except aiohttp.ClientError as e:
                retry_count += 1
                if retry_count >= MAX_RETRIES:
                    raise NetworkError(f"Erreur réseau: {str(e)}")
                    
                await asyncio.sleep(retry_delay)
                retry_delay = min(
                    retry_delay * RETRY_MULTIPLIER,
                    MAX_RETRY_DELAY
                )
                
            except Exception as e:
                if isinstance(e, RateLimitError):
                    retry_count += 1
                    if retry_count >= MAX_RETRIES:
                        raise
                    
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(
                        retry_delay * RETRY_MULTIPLIER,
                        MAX_RETRY_DELAY
                    )
                else:
                    raise
                    
    async def _create_ws_connection(
        self,
        streams: Union[str, List[str]],
        callback: callable
    ) -> WebSocketManager:
        """
        Crée une connexion WebSocket.
        
        Args:
            streams: Stream(s) à écouter
            callback: Fonction de callback
            
        Returns:
            Gestionnaire WebSocket
        """
        if isinstance(streams, str):
            streams = [streams]
            
        stream_path = '/'.join(streams)
        ws_url = f"{self.ws_url}/{stream_path}"
        
        ws_manager = WebSocketManager(
            url=ws_url,
            on_message=callback
        )
        
        await ws_manager.connect()
        return ws_manager
        
    def _handle_user_data_callback(self, msg: Dict[str, Any]) -> None:
        """
        Gère les messages de données utilisateur.
        
        Args:
            msg: Message WebSocket
        """
        raise NotImplementedError
        
    def _handle_market_data_callback(self, msg: Dict[str, Any]) -> None:
        """
        Gère les messages de données de marché.
        
        Args:
            msg: Message WebSocket
        """
        raise NotImplementedError 