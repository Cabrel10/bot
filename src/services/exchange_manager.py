"""
Gestionnaire d'exchanges pour le système de trading.
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging
from dataclasses import dataclass
import asyncio
from abc import ABC, abstractmethod

@dataclass
class ExchangeConfig:
    """Configuration pour la connexion aux exchanges."""
    exchange_id: str
    api_key: str
    api_secret: str
    testnet: bool = False
    rate_limit: int = 100  # Limite de requêtes par minute
    timeout: int = 30000   # Timeout en millisecondes

class BaseExchange(ABC):
    """Classe de base pour les exchanges."""
    
    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Récupère les données historiques."""
        pass
    
    @abstractmethod
    def get_latest_data(
        self,
        symbol: str,
        timeframe: str,
        n_points: int = 100
    ) -> pd.DataFrame:
        """Récupère les dernières données."""
        pass
    
    @abstractmethod
    def place_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None
    ) -> Dict:
        """Place un ordre."""
        pass
    
    @abstractmethod
    def cancel_order(
        self,
        symbol: str,
        order_id: str
    ) -> bool:
        """Annule un ordre."""
        pass
    
    @abstractmethod
    def get_balance(self) -> Dict:
        """Récupère le solde."""
        pass

class ExchangeManager:
    """Gestionnaire d'exchanges pour le système de trading."""
    
    def __init__(self, config: ExchangeConfig):
        self.config = config
        self.exchange = None
        self._setup_logging()
        self._initialize_exchange()
        
    def _setup_logging(self):
        """Configure la journalisation."""
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _initialize_exchange(self):
        """Initialise la connexion à l'exchange."""
        try:
            self.exchange = getattr(ccxt, self.config.exchange_id)({
                'apiKey': self.config.api_key,
                'secret': self.config.api_secret,
                'enableRateLimit': True,
                'timeout': self.config.timeout
            })
            
            if self.config.testnet:
                self.exchange.set_sandbox_mode(True)
                
            self.logger.info(f"Exchange {self.config.exchange_id} initialisé avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation de l'exchange: {e}")
            raise
    
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Récupère les données historiques.
        
        Args:
            symbol: Paire de trading
            timeframe: Intervalle de temps
            start_date: Date de début
            end_date: Date de fin
            
        Returns:
            DataFrame avec les données OHLCV
        """
        try:
            since = int(start_date.timestamp() * 1000)
            end_timestamp = int(end_date.timestamp() * 1000)
            
            all_candles = []
            while since < end_timestamp:
                candles = await self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    since=since,
                    limit=1000
                )
                
                if not candles:
                    break
                    
                all_candles.extend(candles)
                since = candles[-1][0] + 1
                
                # Respect du rate limit
                await asyncio.sleep(self.exchange.rateLimit / 1000)
            
            if not all_candles:
                return pd.DataFrame()
            
            df = pd.DataFrame(
                all_candles,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des données: {e}")
            return pd.DataFrame()
    
    async def get_latest_data(
        self,
        symbol: str,
        timeframe: str,
        n_points: int = 100
    ) -> pd.DataFrame:
        """
        Récupère les dernières données.
        
        Args:
            symbol: Paire de trading
            timeframe: Intervalle de temps
            n_points: Nombre de points à récupérer
            
        Returns:
            DataFrame avec les dernières données
        """
        try:
            candles = await self.exchange.fetch_ohlcv(
                symbol,
                timeframe,
                limit=n_points
            )
            
            if not candles:
                return pd.DataFrame()
            
            df = pd.DataFrame(
                candles,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des dernières données: {e}")
            return pd.DataFrame()
    
    async def place_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None
    ) -> Dict:
        """
        Place un ordre.
        
        Args:
            symbol: Paire de trading
            order_type: Type d'ordre (market, limit)
            side: Côté (buy, sell)
            amount: Quantité
            price: Prix (optionnel)
            
        Returns:
            Détails de l'ordre
        """
        try:
            params = {}
            if price is not None:
                params['price'] = price
                
            order = await self.exchange.create_order(
                symbol,
                order_type,
                side,
                amount,
                **params
            )
            
            self.logger.info(f"Ordre placé avec succès: {order['id']}")
            return order
            
        except Exception as e:
            self.logger.error(f"Erreur lors du placement de l'ordre: {e}")
            raise
    
    async def cancel_order(
        self,
        symbol: str,
        order_id: str
    ) -> bool:
        """
        Annule un ordre.
        
        Args:
            symbol: Paire de trading
            order_id: ID de l'ordre
            
        Returns:
            True si l'annulation a réussi
        """
        try:
            await self.exchange.cancel_order(order_id, symbol)
            self.logger.info(f"Ordre {order_id} annulé avec succès")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'annulation de l'ordre: {e}")
            return False
    
    async def get_balance(self) -> Dict:
        """
        Récupère le solde.
        
        Returns:
            Dictionnaire avec les soldes
        """
        try:
            balance = await self.exchange.fetch_balance()
            return balance
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération du solde: {e}")
            raise 