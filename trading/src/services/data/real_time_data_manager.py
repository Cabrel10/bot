"""
Gestionnaire de données en temps réel pour les exchanges crypto.
"""
from typing import Dict, List, Optional, Union
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from trading.exchanges.bitget import BitgetClient
from trading.exchanges.binance import BinanceClient
from trading.core.data_types import MarketData, OHLCV
from trading.utils.logging import TradingLogger

@dataclass
class RealTimeConfig:
    """Configuration pour la collecte de données en temps réel."""
    exchanges: List[str]
    symbols: List[str]
    timeframes: List[str]
    update_interval: int = 1  # secondes
    max_retries: int = 3
    retry_delay: int = 5  # secondes
    buffer_size: int = 1000
    sync_historical: bool = True
    indicators: List[str] = None

class RealTimeDataManager:
    """Gestionnaire de données en temps réel multi-exchange."""
    
    def __init__(self, config: RealTimeConfig):
        self.config = config
        self.logger = TradingLogger()
        self._running = False
        self._tasks = []
        self._data_buffers: Dict[str, Dict] = {}
        self._last_update = {}
        self._exchange_clients = {}
        self._initialize_exchanges()
        self._setup_buffers()

    def _initialize_exchanges(self):
        """Initialise les connexions aux exchanges."""
        for exchange in self.config.exchanges:
            if exchange.lower() == 'bitget':
                self._exchange_clients[exchange] = BitgetClient()
            elif exchange.lower() == 'binance':
                self._exchange_clients[exchange] = BinanceClient()
            else:
                raise ValueError(f"Exchange non supporté: {exchange}")

    def _setup_buffers(self):
        """Initialise les buffers de données pour chaque paire/timeframe."""
        for exchange in self.config.exchanges:
            self._data_buffers[exchange] = {}
            for symbol in self.config.symbols:
                self._data_buffers[exchange][symbol] = {
                    tf: {
                        'ohlcv': pd.DataFrame(),
                        'volume': pd.DataFrame(),
                        'indicators': pd.DataFrame()
                    } for tf in self.config.timeframes
                }

    async def start(self):
        """Démarre la collecte de données en temps réel."""
        if self._running:
            self.logger.warning("Le gestionnaire de données est déjà en cours d'exécution")
            return

        self._running = True
        self.logger.info("Démarrage de la collecte de données en temps réel...")

        # Synchronisation initiale avec les données historiques si nécessaire
        if self.config.sync_historical:
            await self._sync_historical_data()

        # Démarrage des tâches de collecte pour chaque exchange/symbol/timeframe
        for exchange in self.config.exchanges:
            for symbol in self.config.symbols:
                for timeframe in self.config.timeframes:
                    task = asyncio.create_task(
                        self._collect_data_loop(exchange, symbol, timeframe)
                    )
                    self._tasks.append(task)

    async def stop(self):
        """Arrête la collecte de données en temps réel."""
        if not self._running:
            return

        self._running = False
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self.logger.info("Collecte de données arrêtée")

    async def _collect_data_loop(self, exchange: str, symbol: str, timeframe: str):
        """Boucle principale de collecte de données."""
        retry_count = 0
        
        while self._running:
            try:
                # Collecte des données OHLCV
                ohlcv_data = await self._exchange_clients[exchange].fetch_ohlcv(
                    symbol, timeframe
                )
                
                # Collecte des volumes
                volume_data = await self._exchange_clients[exchange].fetch_volume(
                    symbol, timeframe
                )
                
                # Mise à jour des buffers
                await self._update_buffers(exchange, symbol, timeframe, ohlcv_data, volume_data)
                
                # Calcul des indicateurs si configurés
                if self.config.indicators:
                    await self._calculate_indicators(exchange, symbol, timeframe)
                
                # Réinitialisation du compteur d'erreurs
                retry_count = 0
                
                # Attente avant la prochaine mise à jour
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                retry_count += 1
                self.logger.error(
                    f"Erreur lors de la collecte de données pour {exchange}/{symbol}/{timeframe}: {e}"
                )
                
                if retry_count >= self.config.max_retries:
                    self.logger.error(f"Nombre maximum de tentatives atteint pour {exchange}/{symbol}")
                    break
                    
                await asyncio.sleep(self.config.retry_delay)

    async def _sync_historical_data(self):
        """Synchronise avec les données historiques."""
        self.logger.info("Synchronisation avec les données historiques...")
        
        for exchange in self.config.exchanges:
            for symbol in self.config.symbols:
                for timeframe in self.config.timeframes:
                    try:
                        # Chargement des données historiques
                        historical_data = await self._exchange_clients[exchange].fetch_historical_data(
                            symbol, timeframe, limit=self.config.buffer_size
                        )
                        
                        # Mise à jour des buffers avec les données historiques
                        self._data_buffers[exchange][symbol][timeframe]['ohlcv'] = historical_data
                        
                        self.logger.info(
                            f"Données historiques synchronisées pour {exchange}/{symbol}/{timeframe}"
                        )
                        
                    except Exception as e:
                        self.logger.error(
                            f"Erreur lors de la synchronisation historique pour "
                            f"{exchange}/{symbol}/{timeframe}: {e}"
                        )

    async def _update_buffers(self, 
                            exchange: str, 
                            symbol: str, 
                            timeframe: str,
                            ohlcv_data: pd.DataFrame,
                            volume_data: pd.DataFrame):
        """Met à jour les buffers de données."""
        buffer = self._data_buffers[exchange][symbol][timeframe]
        
        # Mise à jour OHLCV
        buffer['ohlcv'] = pd.concat([buffer['ohlcv'], ohlcv_data]).tail(self.config.buffer_size)
        buffer['volume'] = pd.concat([buffer['volume'], volume_data]).tail(self.config.buffer_size)
        
        # Mise à jour du timestamp
        self._last_update[f"{exchange}_{symbol}_{timeframe}"] = datetime.now()

    async def _calculate_indicators(self, exchange: str, symbol: str, timeframe: str):
        """Calcule les indicateurs techniques."""
        buffer = self._data_buffers[exchange][symbol][timeframe]
        ohlcv_data = buffer['ohlcv']
        
        indicators_data = pd.DataFrame(index=ohlcv_data.index)
        
        for indicator in self.config.indicators:
            try:
                # Calcul de l'indicateur
                indicator_value = await self._calculate_indicator(indicator, ohlcv_data)
                indicators_data[indicator] = indicator_value
                
            except Exception as e:
                self.logger.error(f"Erreur lors du calcul de l'indicateur {indicator}: {e}")
        
        buffer['indicators'] = indicators_data

    async def get_latest_data(self, 
                            exchange: str, 
                            symbol: str, 
                            timeframe: str) -> MarketData:
        """
        Récupère les dernières données pour une paire/timeframe.
        
        Returns:
            MarketData: Dernières données de marché
        """
        buffer = self._data_buffers[exchange][symbol][timeframe]
        
        return MarketData(
            timestamp=buffer['ohlcv'].index,
            open=buffer['ohlcv']['open'],
            high=buffer['ohlcv']['high'],
            low=buffer['ohlcv']['low'],
            close=buffer['ohlcv']['close'],
            volume=buffer['volume']['volume'],
            indicators=buffer['indicators']
        )

    def get_buffer_status(self) -> Dict:
        """Retourne l'état des buffers de données."""
        status = {}
        
        for exchange in self.config.exchanges:
            status[exchange] = {}
            for symbol in self.config.symbols:
                status[exchange][symbol] = {}
                for timeframe in self.config.timeframes:
                    buffer = self._data_buffers[exchange][symbol][timeframe]
                    last_update = self._last_update.get(f"{exchange}_{symbol}_{timeframe}")
                    
                    status[exchange][symbol][timeframe] = {
                        'data_points': len(buffer['ohlcv']),
                        'last_update': last_update,
                        'indicators_available': list(buffer['indicators'].columns)
                    }
        
        return status 