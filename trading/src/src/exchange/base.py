"""
Module de base pour les clients d'échange.

Ce module définit l'interface commune et les fonctionnalités de base
pour tous les clients d'échange supportés par le système.
"""

from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from abc import ABC, abstractmethod
import asyncio
import json
from pathlib import Path
import websockets
from decimal import Decimal

logger = logging.getLogger(__name__)

@dataclass
class OrderRequest:
    """Requête de création d'ordre."""
    symbol: str
    side: str  # 'buy' ou 'sell'
    type: str  # 'market', 'limit', etc.
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = 'GTC'
    reduce_only: bool = False
    post_only: bool = False
    leverage: Optional[int] = None
    client_order_id: Optional[str] = None

@dataclass
class OrderResponse:
    """Réponse à une création d'ordre."""
    exchange_order_id: str
    client_order_id: Optional[str]
    symbol: str
    status: str
    side: str
    type: str
    quantity: float
    price: float
    filled_quantity: float
    remaining_quantity: float
    average_price: Optional[float]
    commission: Optional[float]
    commission_asset: Optional[str]
    created_at: datetime
    updated_at: datetime
    raw_response: Dict[str, Any]

@dataclass
class PositionInfo:
    """Information sur une position."""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    leverage: int
    unrealized_pnl: float
    margin_type: str
    liquidation_price: Optional[float]
    raw_info: Dict[str, Any]

@dataclass
class AccountBalance:
    """Information sur le solde du compte."""
    asset: str
    free: float
    locked: float
    total: float
    raw_balance: Dict[str, Any]

@dataclass
class MarketInfo:
    """Information sur un marché."""
    symbol: str
    base_asset: str
    quote_asset: str
    price_precision: int
    quantity_precision: int
    min_quantity: float
    max_quantity: float
    min_notional: float
    raw_info: Dict[str, Any]

class WebSocketManager:
    """Gestionnaire de connexions WebSocket."""
    
    def __init__(self, url: str, on_message, on_error=None):
        """
        Initialise le gestionnaire WebSocket.
        
        Args:
            url: URL du WebSocket
            on_message: Callback pour les messages
            on_error: Callback pour les erreurs
        """
        self.url = url
        self.on_message = on_message
        self.on_error = on_error or self._default_error_handler
        self.ws = None
        self.running = False
        self._reconnect_delay = 1
        self._max_reconnect_delay = 60
        
    async def connect(self):
        """Établit la connexion WebSocket."""
        while True:
            try:
                async with websockets.connect(self.url) as ws:
                    self.ws = ws
                    self.running = True
                    self._reconnect_delay = 1
                    logger.info(f"Connecté au WebSocket: {self.url}")
                    
                    while self.running:
                        try:
                            message = await ws.recv()
                            await self.on_message(json.loads(message))
                        except websockets.ConnectionClosed:
                            logger.warning("Connexion WebSocket fermée")
                            break
                            
            except Exception as e:
                if self.running:
                    await self.on_error(e)
                    await asyncio.sleep(self._reconnect_delay)
                    self._reconnect_delay = min(
                        self._reconnect_delay * 2,
                        self._max_reconnect_delay
                    )
                else:
                    break
                    
    async def disconnect(self):
        """Ferme la connexion WebSocket."""
        self.running = False
        if self.ws:
            await self.ws.close()
            
    @staticmethod
    async def _default_error_handler(error):
        """Gestionnaire d'erreur par défaut."""
        logger.error(f"Erreur WebSocket: {str(error)}")

class ExchangeClient(ABC):
    """Classe de base pour les clients d'échange."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True
    ):
        """
        Initialise le client d'échange.
        
        Args:
            api_key: Clé API
            api_secret: Secret API
            testnet: Utilisation du testnet
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.ws_manager = None
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialise la connexion à l'exchange."""
        pass
        
    @abstractmethod
    async def close(self) -> None:
        """Ferme la connexion à l'exchange."""
        pass
        
    @abstractmethod
    async def get_exchange_info(self) -> Dict[str, Any]:
        """
        Récupère les informations de l'exchange.
        
        Returns:
            Informations de l'exchange
        """
        pass
        
    @abstractmethod
    async def get_market_info(self, symbol: str) -> MarketInfo:
        """
        Récupère les informations d'un marché.
        
        Args:
            symbol: Symbole du marché
            
        Returns:
            Informations du marché
        """
        pass
        
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Récupère le ticker d'un symbole.
        
        Args:
            symbol: Symbole du marché
            
        Returns:
            Ticker du symbole
        """
        pass
        
    @abstractmethod
    async def get_orderbook(
        self,
        symbol: str,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Récupère le carnet d'ordres d'un symbole.
        
        Args:
            symbol: Symbole du marché
            limit: Nombre de niveaux à récupérer
            
        Returns:
            Carnet d'ordres
        """
        pass
        
    @abstractmethod
    async def get_recent_trades(
        self,
        symbol: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Récupère les trades récents d'un symbole.
        
        Args:
            symbol: Symbole du marché
            limit: Nombre de trades à récupérer
            
        Returns:
            Liste des trades
        """
        pass
        
    @abstractmethod
    async def get_historical_trades(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Récupère l'historique des trades d'un symbole.
        
        Args:
            symbol: Symbole du marché
            start_time: Début de la période
            end_time: Fin de la période
            limit: Nombre de trades à récupérer
            
        Returns:
            Liste des trades
        """
        pass
        
    @abstractmethod
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Récupère les chandeliers d'un symbole.
        
        Args:
            symbol: Symbole du marché
            interval: Intervalle des chandeliers
            start_time: Début de la période
            end_time: Fin de la période
            limit: Nombre de chandeliers à récupérer
            
        Returns:
            Liste des chandeliers
        """
        pass
        
    @abstractmethod
    async def create_order(self, order: OrderRequest) -> OrderResponse:
        """
        Crée un ordre.
        
        Args:
            order: Requête de création d'ordre
            
        Returns:
            Réponse de création d'ordre
        """
        pass
        
    @abstractmethod
    async def cancel_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Annule un ordre.
        
        Args:
            symbol: Symbole du marché
            order_id: ID de l'ordre
            client_order_id: ID client de l'ordre
            
        Returns:
            Réponse d'annulation
        """
        pass
        
    @abstractmethod
    async def get_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None
    ) -> OrderResponse:
        """
        Récupère un ordre.
        
        Args:
            symbol: Symbole du marché
            order_id: ID de l'ordre
            client_order_id: ID client de l'ordre
            
        Returns:
            Informations de l'ordre
        """
        pass
        
    @abstractmethod
    async def get_open_orders(
        self,
        symbol: Optional[str] = None
    ) -> List[OrderResponse]:
        """
        Récupère les ordres ouverts.
        
        Args:
            symbol: Symbole du marché (optionnel)
            
        Returns:
            Liste des ordres ouverts
        """
        pass
        
    @abstractmethod
    async def get_account_balance(self) -> List[AccountBalance]:
        """
        Récupère le solde du compte.
        
        Returns:
            Liste des soldes par actif
        """
        pass
        
    @abstractmethod
    async def get_positions(
        self,
        symbol: Optional[str] = None
    ) -> List[PositionInfo]:
        """
        Récupère les positions ouvertes.
        
        Args:
            symbol: Symbole du marché (optionnel)
            
        Returns:
            Liste des positions
        """
        pass
        
    @abstractmethod
    async def set_leverage(
        self,
        symbol: str,
        leverage: int
    ) -> Dict[str, Any]:
        """
        Définit l'effet de levier pour un symbole.
        
        Args:
            symbol: Symbole du marché
            leverage: Effet de levier
            
        Returns:
            Réponse de l'exchange
        """
        pass
        
    @abstractmethod
    async def set_margin_type(
        self,
        symbol: str,
        margin_type: str
    ) -> Dict[str, Any]:
        """
        Définit le type de marge pour un symbole.
        
        Args:
            symbol: Symbole du marché
            margin_type: Type de marge ('ISOLATED' ou 'CROSSED')
            
        Returns:
            Réponse de l'exchange
        """
        pass
        
    @abstractmethod
    async def subscribe_ticker(self, symbol: str) -> None:
        """
        Souscrit aux mises à jour du ticker.
        
        Args:
            symbol: Symbole du marché
        """
        pass
        
    @abstractmethod
    async def subscribe_orderbook(self, symbol: str) -> None:
        """
        Souscrit aux mises à jour du carnet d'ordres.
        
        Args:
            symbol: Symbole du marché
        """
        pass
        
    @abstractmethod
    async def subscribe_trades(self, symbol: str) -> None:
        """
        Souscrit aux mises à jour des trades.
        
        Args:
            symbol: Symbole du marché
        """
        pass
        
    @abstractmethod
    async def subscribe_klines(self, symbol: str, interval: str) -> None:
        """
        Souscrit aux mises à jour des chandeliers.
        
        Args:
            symbol: Symbole du marché
            interval: Intervalle des chandeliers
        """
        pass
        
    @abstractmethod
    async def subscribe_user_data(self) -> None:
        """Souscrit aux mises à jour des données utilisateur."""
        pass 