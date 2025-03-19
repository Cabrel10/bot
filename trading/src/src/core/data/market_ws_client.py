from typing import Dict, Optional, Callable, Any, List, Union, Set
import json
import asyncio
import websockets
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import logging
import aiohttp
from collections import defaultdict
from asyncio import Queue, Task
from contextlib import asynccontextmanager

from trading.core.exceptions import (
    WebSocketError, ConnectionError, SubscriptionError,
    MessageError, ConfigError
)

class MarketType(Enum):
    """Type de marché supporté."""
    SPOT = 'spot'
    FUTURES = 'futures'
    MARGIN = 'margin'

class MessageType(Enum):
    """Types de messages WebSocket."""
    TRADE = 'trade'
    KLINE = 'kline'
    TICKER = 'ticker'
    BOOK = 'book'
    BOOK_TICKER = 'bookTicker'
    FUNDING_RATE = 'funding_rate'
    MARK_PRICE = 'mark_price'
    LIQUIDATION = 'liquidation'

@dataclass
class WSConfig:
    """Configuration du client WebSocket."""
    exchange_id: str = 'binance'
    market_type: MarketType = MarketType.SPOT
    ping_interval: int = 30
    pong_timeout: int = 10
    reconnect_delay: int = 5
    max_reconnect_attempts: int = 5
    connection_timeout: int = 30
    message_timeout: int = 60
    buffer_size: int = 1000
    max_message_size: int = 10 * 1024 * 1024  # 10MB
    compression: Optional[str] = 'deflate'
    verify_ssl: bool = True
    proxy: Optional[str] = None
    futures_config: Dict[str, Any] = field(default_factory=lambda: {
        'mark_price_interval': 3000,  # ms
        'funding_rate_interval': 8 * 3600 * 1000,  # 8h en ms
        'liquidation_only': False
    })

class MessageHandler:
    """Gestionnaire de messages WebSocket."""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.message_queues: Dict[str, Queue] = defaultdict(lambda: Queue(maxsize=buffer_size))
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
    async def process_message(self, message: Dict) -> None:
        """Traite un message reçu."""
        try:
            # Identification du type de message
            msg_type = self._identify_message_type(message)
            
            # Mise en queue et notification des callbacks
            await self._queue_message(msg_type, message)
            await self._notify_callbacks(msg_type, message)
            
        except Exception as e:
            raise MessageError(f"Erreur traitement message: {str(e)}")
            
    def _identify_message_type(self, message: Dict) -> str:
        """Identifie le type de message."""
        if 'stream' in message:
            stream = message['stream']
            for msg_type in MessageType:
                if msg_type.value in stream:
                    return msg_type.value
        return 'unknown'
        
    async def _queue_message(self, msg_type: str, message: Dict) -> None:
        """Met le message en queue."""
        try:
            # Suppression du plus ancien message si la queue est pleine
            if self.message_queues[msg_type].full():
                await self.message_queues[msg_type].get()
                
            await self.message_queues[msg_type].put(message)
        except Exception as e:
            raise MessageError(f"Erreur queue message: {str(e)}")
            
    async def _notify_callbacks(self, msg_type: str, message: Dict) -> None:
        """Notifie les callbacks enregistrés."""
        for callback in self.callbacks[msg_type]:
            try:
                await callback(message)
            except Exception as e:
                logging.error(f"Erreur callback {msg_type}: {str(e)}")
                
    def register_callback(self, msg_type: str, callback: Callable) -> None:
        """Enregistre un callback pour un type de message."""
        if not callable(callback):
            raise ValueError("Le callback doit être callable")
        self.callbacks[msg_type].append(callback)
        
    def unregister_callback(self, msg_type: str, callback: Callable) -> None:
        """Désenregistre un callback."""
        if callback in self.callbacks[msg_type]:
            self.callbacks[msg_type].remove(callback)

class MarketWebSocketClient:
    """Client WebSocket avancé pour les données de marché en temps réel."""

    def __init__(self, config: Optional[Union[Dict, WSConfig]] = None):
        """
        Initialise le client WebSocket.
        
        Args:
            config: Configuration du client
        """
        self.config = (
            config if isinstance(config, WSConfig)
            else WSConfig(**(config or {}))
        )
        
        self.logger = self._setup_logger()
        self.handler = MessageHandler(buffer_size=self.config.buffer_size)
        
        self.ws = None
        self.session = None
        self.is_connected = False
        self.last_message_time = None
        self.ping_task: Optional[Task] = None
        self.listen_task: Optional[Task] = None
        self.subscriptions: Set[str] = set()
        self.reconnect_count = 0

    def _setup_logger(self) -> logging.Logger:
        """Configure le logger."""
        logger = logging.getLogger(f"{__name__}.{self.config.exchange_id}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    @asynccontextmanager
    async def connection(self, url: str):
        """Context manager pour la connexion WebSocket."""
        try:
            await self.connect(url)
            yield self
        finally:
            await self.close()

    async def connect(self, url: str) -> None:
        """
        Établit une connexion WebSocket avec l'exchange.
        
        Args:
            url: URL du WebSocket
        """
        try:
            # Configuration de la session
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            # Configuration des options WebSocket
            ws_options = {
                'timeout': self.config.connection_timeout,
                'max_size': self.config.max_message_size,
                'compress': self.config.compression is not None,
                'ssl': self.config.verify_ssl,
                'proxy': self.config.proxy
            }
            
            # Établissement de la connexion
            self.ws = await self.session.ws_connect(url, **ws_options)
            self.is_connected = True
            self.last_message_time = datetime.utcnow()
            self.reconnect_count = 0
            
            # Démarrage des tâches de maintenance
            self.ping_task = asyncio.create_task(self._start_ping_loop())
            self.listen_task = asyncio.create_task(self.listen())
            
            self.logger.info(f'WebSocket connecté à {url}')
            
        except Exception as e:
            self.logger.error(f'Erreur connexion WebSocket: {str(e)}')
            raise ConnectionError(f'Échec connexion WebSocket: {str(e)}')

    async def _start_ping_loop(self) -> None:
        """Maintient la connexion active avec des pings réguliers."""
        while self.is_connected:
            try:
                await asyncio.sleep(self.config.ping_interval)
                
                if not self.ws or self.ws.closed:
                    raise ConnectionError('WebSocket fermé')
                    
                # Vérification du dernier message reçu
                if self.last_message_time:
                    elapsed = (datetime.utcnow() - self.last_message_time).total_seconds()
                    if elapsed > self.config.message_timeout:
                        raise ConnectionError(f'Pas de message depuis {elapsed}s')
                        
                # Envoi du ping
                await self.ws.ping()
                
                # Attente du pong
                try:
                    async with asyncio.timeout(self.config.pong_timeout):
                        await self.ws.pong()
                except asyncio.TimeoutError:
                    raise ConnectionError('Timeout pong')
                    
            except Exception as e:
                self.logger.warning(f'Erreur ping: {str(e)}')
                await self.reconnect()

    async def subscribe(self,
                       channels: Union[str, List[str]],
                       callback: Optional[Callable] = None) -> None:
        """
        S'abonne à un ou plusieurs canaux de données.
        
        Args:
            channels: Canal(aux) à suivre
            callback: Fonction de callback optionnelle
        """
        if not self.is_connected:
            raise ConnectionError('WebSocket non connecté')
            
        if isinstance(channels, str):
            channels = [channels]
            
        try:
            # Préparation du message de souscription
            message = {
                'method': 'SUBSCRIBE',
                'params': channels,
                'id': len(self.subscriptions)
            }
            
            # Envoi de la demande
            await self.ws.send_json(message)
            
            # Attente de la confirmation
            response = await self.ws.receive_json()
            if not response.get('result'):
                raise SubscriptionError(f'Échec souscription: {response}')
                
            # Enregistrement des souscriptions
            for channel in channels:
                self.subscriptions.add(channel)
                if callback:
                    msg_type = self._get_message_type(channel)
                    self.handler.register_callback(msg_type, callback)
                    
            self.logger.info(f'Souscription réussie: {channels}')
            
        except Exception as e:
            self.logger.error(f'Erreur souscription: {str(e)}')
            raise SubscriptionError(f'Échec souscription: {str(e)}')

    def _get_message_type(self, channel: str) -> str:
        """Détermine le type de message d'un canal."""
        for msg_type in MessageType:
            if msg_type.value in channel:
                return msg_type.value
        return 'unknown'

    async def unsubscribe(self, channels: Union[str, List[str]]) -> None:
        """
        Se désabonne d'un ou plusieurs canaux.
        
        Args:
            channels: Canal(aux) à quitter
        """
        if isinstance(channels, str):
            channels = [channels]
            
        try:
            # Préparation du message de désabonnement
            message = {
                'method': 'UNSUBSCRIBE',
                'params': channels,
                'id': len(self.subscriptions)
            }
            
            # Envoi de la demande
            await self.ws.send_json(message)
            
            # Attente de la confirmation
            response = await self.ws.receive_json()
            if not response.get('result'):
                raise SubscriptionError(f'Échec désabonnement: {response}')
                
            # Mise à jour des souscriptions
            for channel in channels:
                self.subscriptions.discard(channel)
                
            self.logger.info(f'Désabonnement réussi: {channels}')
            
        except Exception as e:
            self.logger.error(f'Erreur désabonnement: {str(e)}')
            raise SubscriptionError(f'Échec désabonnement: {str(e)}')

    async def listen(self) -> None:
        """Écoute les messages entrants et les traite."""
        while self.is_connected:
            try:
                # Réception du message
                msg = await self.ws.receive()
                self.last_message_time = datetime.utcnow()
                
                # Traitement selon le type
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await self.handler.process_message(data)
                    
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    data = json.loads(msg.data.decode())
                    await self.handler.process_message(data)
                    
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    raise ConnectionError('WebSocket fermé par le serveur')
                    
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    raise WebSocketError(f'Erreur WebSocket: {self.ws.exception()}')
                    
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.error(f'Erreur décodage message: {str(e)}')
                
            except Exception as e:
                self.logger.error(f'Erreur traitement message: {str(e)}')
                await self.reconnect()

    async def reconnect(self) -> None:
        """Tente de rétablir la connexion WebSocket."""
        self.is_connected = False
        self.reconnect_count += 1
        
        if self.reconnect_count > self.config.max_reconnect_attempts:
            raise ConnectionError(
                f'Échec après {self.config.max_reconnect_attempts} tentatives'
            )
            
        try:
            # Nettoyage des ressources
            if self.ping_task:
                self.ping_task.cancel()
            if self.listen_task:
                self.listen_task.cancel()
            if self.ws:
                await self.ws.close()
                
            # Attente avant reconnexion
            delay = self.config.reconnect_delay * (2 ** (self.reconnect_count - 1))
            self.logger.info(f'Tentative reconnexion dans {delay}s...')
            await asyncio.sleep(delay)
            
            # Reconnexion
            await self.connect(self.ws.url.human_repr())
            
            # Réabonnement
            if self.subscriptions:
                await self.subscribe(list(self.subscriptions))
                
        except Exception as e:
            self.logger.error(f'Erreur reconnexion: {str(e)}')
            await self.reconnect()

    async def close(self) -> None:
        """Ferme proprement la connexion WebSocket."""
        self.is_connected = False
        
        try:
            # Annulation des tâches
            if self.ping_task:
                self.ping_task.cancel()
            if self.listen_task:
                self.listen_task.cancel()
                
            # Fermeture WebSocket
            if self.ws:
                await self.ws.close()
                
            # Fermeture session
            if self.session:
                await self.session.close()
                
            self.logger.info('WebSocket fermé proprement')
            
        except Exception as e:
            self.logger.error(f'Erreur fermeture WebSocket: {str(e)}')
            raise WebSocketError(f'Échec fermeture WebSocket: {str(e)}')