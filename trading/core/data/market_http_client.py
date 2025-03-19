"""
Client HTTP personnalisé pour les interactions avec l'API de l'exchange.
Supporte les opérations spot et futures avec gestion avancée des erreurs.
"""

from typing import Dict, Optional, Any, Union
import aiohttp
import asyncio
import json
import time
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from trading.core.exceptions import (
    APIError, RateLimitError, ConnectionError,
    AuthenticationError, OrderError, MarketError
)
from trading.utils.logging import setup_logger

class MarketType(Enum):
    """Type de marché."""
    SPOT = 'spot'
    FUTURES = 'futures'
    MARGIN = 'margin'

@dataclass
class ClientConfig:
    """Configuration du client HTTP."""
    base_url: str
    api_key: str = ''
    api_secret: str = ''
    timeout: int = 30
    proxy: Optional[str] = None
    ssl_verify: bool = True
    market_type: MarketType = MarketType.SPOT
    rate_limit: Dict[str, int] = None
    retry: Dict[str, Union[int, float]] = None
    futures_config: Optional[Dict] = None

class MarketHTTPClient:
    """Client HTTP personnalisé pour les interactions avec l'API de l'exchange."""

    def __init__(self, config: Optional[Union[Dict, ClientConfig]] = None):
        """
        Initialise le client HTTP.
        
        Args:
            config: Configuration du client
        """
        self.logger = setup_logger(__name__)
        
        if isinstance(config, dict):
            self.config = self._create_config_from_dict(config)
        elif isinstance(config, ClientConfig):
            self.config = config
        else:
            self.config = self._default_config()
            
        self.session = None
        self.last_request_time = 0
        self._request_count = 0
        self._request_window_start = time.time()
        
    def _create_config_from_dict(self, config_dict: Dict) -> ClientConfig:
        """Crée une configuration à partir d'un dictionnaire."""
        return ClientConfig(
            base_url=config_dict.get('base_url', ''),
            api_key=config_dict.get('api_key', ''),
            api_secret=config_dict.get('api_secret', ''),
            timeout=config_dict.get('timeout', 30),
            proxy=config_dict.get('proxy'),
            ssl_verify=config_dict.get('ssl_verify', True),
            market_type=MarketType(config_dict.get('market_type', 'spot')),
            rate_limit=config_dict.get('rate_limit', {
                'requests_per_second': 10,
                'requests_per_minute': 500
            }),
            retry=config_dict.get('retry', {
                'max_attempts': 3,
                'delay': 1.0,
                'backoff_factor': 2.0
            }),
            futures_config=config_dict.get('futures_config')
        )

    def _default_config(self) -> ClientConfig:
        """Configuration par défaut."""
        return ClientConfig(
            base_url='',
            timeout=30,
            rate_limit={
                'requests_per_second': 10,
                'requests_per_minute': 500
            },
            retry={
                'max_attempts': 3,
                'delay': 1.0,
                'backoff_factor': 2.0
            }
        )

    async def _init_session(self) -> None:
        """Initialise la session HTTP avec la configuration spécifiée."""
        if not self.session or self.session.closed:
            connector = aiohttp.TCPConnector(
                ssl=self.config.ssl_verify,
                limit=100  # Limite de connexions simultanées
            )
            
            self.session = aiohttp.ClientSession(
                headers=self._get_default_headers(),
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )

    def _get_default_headers(self) -> Dict[str, str]:
        """Prépare les en-têtes HTTP par défaut avec authentification si nécessaire."""
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'TradingBot/1.0'
        }
        
        if self.config.api_key:
            headers['X-API-KEY'] = self.config.api_key
            
        return headers

    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict:
        """
        Traite la réponse HTTP et gère les erreurs spécifiques.
        
        Args:
            response: Réponse HTTP
            
        Returns:
            Données de la réponse
            
        Raises:
            RateLimitError: Limite de taux atteinte
            AuthenticationError: Erreur d'authentification
            OrderError: Erreur liée aux ordres
            MarketError: Erreur de marché
            APIError: Autre erreur API
        """
        try:
            if response.status == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                raise RateLimitError(f'Rate limit atteint, attente de {retry_after}s')
                
            elif response.status == 401:
                raise AuthenticationError('Authentification invalide')
                
            elif response.status == 400:
                data = await response.json()
                if 'order' in str(data).lower():
                    raise OrderError(f'Erreur d\'ordre: {data}')
                raise MarketError(f'Erreur de marché: {data}')
                
            elif response.status >= 500:
                raise APIError(f'Erreur serveur: {response.status}')
                
            try:
                return await response.json()
            except json.JSONDecodeError as e:
                raise APIError(f'Réponse JSON invalide: {str(e)}')
                
        except aiohttp.ClientError as e:
            raise ConnectionError(f'Erreur de connexion: {str(e)}')

    def _respect_rate_limit(self) -> None:
        """
        Gère les limites de taux d'appels API avec fenêtre glissante.
        
        Raises:
            RateLimitError: Si la limite est atteinte
        """
        current_time = time.time()
        
        # Réinitialisation du compteur après une minute
        if current_time - self._request_window_start >= 60:
            self._request_count = 0
            self._request_window_start = current_time
            
        # Vérification de la limite par minute
        if self._request_count >= self.config.rate_limit['requests_per_minute']:
            raise RateLimitError('Limite de requêtes par minute atteinte')
            
        # Respect de l'intervalle minimum entre requêtes
        elapsed = current_time - self.last_request_time
        min_interval = 1.0 / self.config.rate_limit['requests_per_second']
        
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
            
        self._request_count += 1
        self.last_request_time = time.time()

    async def request(self,
                     method: str,
                     endpoint: str,
                     params: Optional[Dict] = None,
                     data: Optional[Dict] = None,
                     futures: bool = False) -> Dict:
        """
        Effectue une requête HTTP vers l'API avec gestion avancée des erreurs.
        
        Args:
            method: Méthode HTTP
            endpoint: Point d'accès API
            params: Paramètres de requête
            data: Données de la requête
            futures: Si True, utilise l'API futures
            
        Returns:
            Réponse de l'API
            
        Raises:
            Various exceptions selon le type d'erreur
        """
        await self._init_session()
        
        # Ajustement de l'URL selon le type de marché
        base_url = self.config.base_url
        if futures and self.config.futures_config:
            base_url = self.config.futures_config.get('base_url', base_url)
            
        url = f"{base_url}{endpoint}"
        attempts = 0
        last_error = None
        
        while attempts < self.config.retry['max_attempts']:
            try:
                self._respect_rate_limit()
                
                # Configuration du proxy si spécifié
                proxy = self.config.proxy if self.config.proxy else None
                
                async with self.session.request(
                    method,
                    url,
                    params=params,
                    json=data,
                    proxy=proxy
                ) as response:
                    return await self._handle_response(response)
                    
            except (RateLimitError, AuthenticationError) as e:
                # Pas de retry pour ces erreurs
                raise
                
            except Exception as e:
                attempts += 1
                last_error = e
                
                if attempts == self.config.retry['max_attempts']:
                    self.logger.error(f'Toutes les tentatives ont échoué: {str(e)}')
                    raise
                    
                # Délai exponentiel entre les tentatives
                delay = self.config.retry['delay'] * (
                    self.config.retry['backoff_factor'] ** (attempts - 1)
                )
                self.logger.warning(
                    f'Tentative {attempts} échouée: {str(e)}. '
                    f'Nouvelle tentative dans {delay}s'
                )
                await asyncio.sleep(delay)

    async def close(self) -> None:
        """Ferme proprement la session HTTP."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.logger.info('Session HTTP fermée')
            
    async def __aenter__(self):
        """Support du context manager asynchrone."""
        await self._init_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Fermeture propre à la sortie du context manager."""
        await self.close()