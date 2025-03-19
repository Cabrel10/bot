"""
Module de collecte de données avec support pour différentes sources et types de données.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import asyncio
import aiohttp
import ccxt.async_support as ccxt
import logging
from pathlib import Path
import yaml
from tenacity import retry, stop_after_attempt, wait_exponential

from trading.core.exceptions import DataCollectionError
from trading.utils.logging import setup_logger
from .data_validation import DataValidator
from .data_types import MarketData, FuturesData, OrderBookData, TradeData

class DataCollector:
    """Collecteur de données avec support multi-sources et multi-assets."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialise le collecteur de données.
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.config = self._load_config(config_path)
        self.logger = setup_logger('data_collector')
        self.validator = DataValidator()
        
        # Initialisation des connexions aux exchanges
        self.exchanges = self._initialize_exchanges()
        
        # Cache pour les données
        self.data_cache = {}
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Charge la configuration depuis un fichier."""
        default_config = {
            'exchanges': {
                'binance': {
                    'api_key': '',
                    'api_secret': '',
                    'timeout': 30000,
                    'enableRateLimit': True
                }
            },
            'symbols': ['BTC/USDT', 'ETH/USDT'],
            'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
            'save_path': 'data/raw',
            'max_retries': 3,
            'retry_delay': 5,
            'batch_size': 1000,
            'use_async': True
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                default_config.update(user_config)
            except Exception as e:
                self.logger.error(f"Erreur lors du chargement de la configuration: {str(e)}")
                
        return default_config
        
    def _initialize_exchanges(self) -> Dict[str, ccxt.Exchange]:
        """Initialise les connexions aux exchanges."""
        exchanges = {}
        
        for exchange_id, params in self.config['exchanges'].items():
            try:
                exchange_class = getattr(ccxt, exchange_id)
                exchanges[exchange_id] = exchange_class(params)
                self.logger.info(f"Exchange {exchange_id} initialisé")
            except Exception as e:
                self.logger.error(f"Erreur lors de l'initialisation de {exchange_id}: {str(e)}")
                
        return exchanges
        
    async def fetch_ohlcv(self, exchange_id: str, symbol: str, 
                         timeframe: str, since: Optional[int] = None,
                         limit: Optional[int] = None) -> pd.DataFrame:
        """
        Récupère les données OHLCV depuis un exchange.
        
        Args:
            exchange_id: ID de l'exchange
            symbol: Symbole de trading
            timeframe: Intervalle temporel
            since: Timestamp de début
            limit: Nombre maximum de bougies
            
        Returns:
            DataFrame avec les données OHLCV
        """
        try:
            exchange = self.exchanges[exchange_id]
            
            @retry(stop=stop_after_attempt(self.config['max_retries']),
                   wait=wait_exponential(multiplier=self.config['retry_delay']))
            async def _fetch():
                data = await exchange.fetch_ohlcv(
                    symbol, timeframe, since, limit
                )
                return data
                
            # Récupération des données
            ohlcv_data = await _fetch()
            
            # Conversion en DataFrame
            df = pd.DataFrame(
                ohlcv_data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Conversion des timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            raise DataCollectionError(
                f"Erreur lors de la récupération des données OHLCV: {str(e)}"
            )
            
    async def collect_historical_data(self, exchange_id: str, symbol: str,
                                    timeframe: str, start_date: datetime,
                                    end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Collecte des données historiques.
        
        Args:
            exchange_id: ID de l'exchange
            symbol: Symbole de trading
            timeframe: Intervalle temporel
            start_date: Date de début
            end_date: Date de fin (optionnel)
            
        Returns:
            DataFrame avec les données historiques
        """
        try:
            exchange = self.exchanges[exchange_id]
            end_date = end_date or datetime.now()
            
            # Conversion des dates en timestamps
            since = int(start_date.timestamp() * 1000)
            until = int(end_date.timestamp() * 1000)
            
            # Calcul du nombre de bougies nécessaires
            timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
            total_candles = (until - since) // timeframe_ms
            
            # Collecte par lots
            all_data = []
            current_since = since
            
            while current_since < until:
                try:
                    batch = await self.fetch_ohlcv(
                        exchange_id, symbol, timeframe,
                        since=current_since,
                        limit=self.config['batch_size']
                    )
                    
                    if batch.empty:
                        break
                        
                    all_data.append(batch)
                    last_timestamp = batch.index[-1].timestamp() * 1000
                    current_since = int(last_timestamp + timeframe_ms)
                    
                    # Pause pour respecter les limites de l'API
                    await asyncio.sleep(exchange.rateLimit / 1000)
                    
                except Exception as e:
                    self.logger.error(f"Erreur lors de la collecte du lot: {str(e)}")
                    continue
                    
            if not all_data:
                raise DataCollectionError("Aucune donnée collectée")
                
            # Concaténation des données
            df = pd.concat(all_data)
            df = df[~df.index.duplicated(keep='first')]
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            raise DataCollectionError(
                f"Erreur lors de la collecte des données historiques: {str(e)}"
            )
            
    async def collect_orderbook(self, exchange_id: str, symbol: str,
                              depth: Optional[int] = None) -> OrderBookData:
        """
        Collecte les données du carnet d'ordres.
        
        Args:
            exchange_id: ID de l'exchange
            symbol: Symbole de trading
            depth: Profondeur du carnet
            
        Returns:
            OrderBookData
        """
        try:
            exchange = self.exchanges[exchange_id]
            
            @retry(stop=stop_after_attempt(self.config['max_retries']),
                   wait=wait_exponential(multiplier=self.config['retry_delay']))
            async def _fetch():
                orderbook = await exchange.fetch_order_book(symbol, depth)
                return orderbook
                
            # Récupération des données
            orderbook = await _fetch()
            
            return OrderBookData(
                timestamp=datetime.now(),
                symbol=symbol,
                exchange=exchange_id,
                bids=orderbook['bids'],
                asks=orderbook['asks'],
                last_update_id=orderbook.get('nonce')
            )
            
        except Exception as e:
            raise DataCollectionError(
                f"Erreur lors de la collecte du carnet d'ordres: {str(e)}"
            )
            
    async def collect_trades(self, exchange_id: str, symbol: str,
                           since: Optional[int] = None,
                           limit: Optional[int] = None) -> List[TradeData]:
        """
        Collecte les transactions récentes.
        
        Args:
            exchange_id: ID de l'exchange
            symbol: Symbole de trading
            since: Timestamp de début
            limit: Nombre maximum de transactions
            
        Returns:
            Liste de TradeData
        """
        try:
            exchange = self.exchanges[exchange_id]
            
            @retry(stop=stop_after_attempt(self.config['max_retries']),
                   wait=wait_exponential(multiplier=self.config['retry_delay']))
            async def _fetch():
                trades = await exchange.fetch_trades(symbol, since, limit)
                return trades
                
            # Récupération des données
            trades = await _fetch()
            
            return [
                TradeData(
                    timestamp=datetime.fromtimestamp(trade['timestamp'] / 1000),
                    symbol=symbol,
                    exchange=exchange_id,
                    price=trade['price'],
                    amount=trade['amount'],
                    side=trade['side'],
                    trade_id=trade['id'],
                    maker=trade.get('maker'),
                    taker=trade.get('taker'),
                    fee=trade.get('fee', {}).get('cost'),
                    fee_currency=trade.get('fee', {}).get('currency')
                )
                for trade in trades
            ]
            
        except Exception as e:
            raise DataCollectionError(
                f"Erreur lors de la collecte des transactions: {str(e)}"
            )
            
    async def collect_futures_data(self, exchange_id: str,
                                 symbol: str) -> FuturesData:
        """
        Collecte les données spécifiques aux contrats futures.
        
        Args:
            exchange_id: ID de l'exchange
            symbol: Symbole de trading
            
        Returns:
            FuturesData
        """
        try:
            exchange = self.exchanges[exchange_id]
            
            # Vérification que l'exchange supporte les futures
            if not hasattr(exchange, 'fapiPrivate'):
                raise DataCollectionError(f"{exchange_id} ne supporte pas les futures")
                
            @retry(stop=stop_after_attempt(self.config['max_retries']),
                   wait=wait_exponential(multiplier=self.config['retry_delay']))
            async def _fetch_ticker():
                return await exchange.fetch_ticker(symbol)
                
            @retry(stop=stop_after_attempt(self.config['max_retries']),
                   wait=wait_exponential(multiplier=self.config['retry_delay']))
            async def _fetch_funding_rate():
                return await exchange.fetch_funding_rate(symbol)
                
            # Récupération des données
            ticker = await _fetch_ticker()
            funding = await _fetch_funding_rate()
            
            return FuturesData(
                timestamp=datetime.fromtimestamp(ticker['timestamp'] / 1000),
                symbol=symbol,
                exchange=exchange_id,
                open=ticker['open'],
                high=ticker['high'],
                low=ticker['low'],
                close=ticker['close'],
                volume=ticker['baseVolume'],
                funding_rate=funding['fundingRate'],
                mark_price=ticker['mark'] if 'mark' in ticker else ticker['close'],
                index_price=ticker['index'] if 'index' in ticker else ticker['close'],
                open_interest=ticker['openInterest'] if 'openInterest' in ticker else 0,
                next_funding_time=datetime.fromtimestamp(
                    funding['nextFundingTime'] / 1000
                ) if 'nextFundingTime' in funding else datetime.now(),
                predicted_funding_rate=funding.get('predictedFundingRate')
            )
            
        except Exception as e:
            raise DataCollectionError(
                f"Erreur lors de la collecte des données futures: {str(e)}"
            )
            
    def save_data(self, data: Union[pd.DataFrame, List[Any]],
                  filename: str, format: str = 'parquet') -> None:
        """
        Sauvegarde les données collectées.
        
        Args:
            data: Données à sauvegarder
            filename: Nom du fichier
            format: Format de sauvegarde ('parquet', 'csv', 'json')
        """
        try:
            save_path = Path(self.config['save_path'])
            save_path.mkdir(parents=True, exist_ok=True)
            filepath = save_path / f"{filename}.{format}"
            
            if isinstance(data, pd.DataFrame):
                if format == 'parquet':
                    data.to_parquet(filepath)
                elif format == 'csv':
                    data.to_csv(filepath)
                elif format == 'json':
                    data.to_json(filepath)
                else:
                    raise ValueError(f"Format non supporté: {format}")
            else:
                if format == 'json':
                    import json
                    with open(filepath, 'w') as f:
                        json.dump([d.__dict__ for d in data], f)
                else:
                    raise ValueError(f"Format {format} non supporté pour ce type de données")
                    
            self.logger.info(f"Données sauvegardées: {filepath}")
            
        except Exception as e:
            raise DataCollectionError(f"Erreur lors de la sauvegarde des données: {str(e)}")
            
    async def close(self):
        """Ferme proprement les connexions aux exchanges."""
        for exchange in self.exchanges.values():
            try:
                await exchange.close()
            except Exception as e:
                self.logger.error(f"Erreur lors de la fermeture de {exchange.id}: {str(e)}")

# Exemple d'utilisation
async def main():
    # Création d'une instance
    collector = DataCollector()
    
    try:
        # Collecte de données historiques
        df = await collector.collect_historical_data(
            'binance',
            'BTC/USDT',
            '1h',
            start_date=datetime.now() - timedelta(days=30)
        )
        
        # Sauvegarde des données
        collector.save_data(df, 'BTCUSDT_1h')
        
        # Collecte de données futures
        futures_data = await collector.collect_futures_data(
            'binance',
            'BTC/USDT'
        )
        
        print("Collecte terminée avec succès")
        
    except Exception as e:
        print(f"Erreur: {str(e)}")
        
    finally:
        await collector.close()

if __name__ == "__main__":
    asyncio.run(main()) 