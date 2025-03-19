"""
Gestionnaire de données historiques avec support avancé pour spot et futures.
Inclut la gestion des données OHLCV, funding rates, et autres métriques de marché.
"""

import os
import logging
from typing import Optional, Dict, Any, List, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt
from pathlib import Path
import aiohttp
import asyncio
import json
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor

from trading.core.exceptions import (
    DataError, ValidationError, StorageError,
    ExchangeError, ConfigError
)

class MarketType(Enum):
    """Type de marché supporté."""
    SPOT = 'spot'
    FUTURES = 'futures'
    MARGIN = 'margin'

@dataclass
class DataConfig:
    """Configuration du gestionnaire de données."""
    exchange_id: str = 'binance'
    market_type: MarketType = MarketType.SPOT
    base_path: Union[str, Path] = field(default="data")
    cache_enabled: bool = True
    compression: str = 'snappy'
    max_retries: int = 3
    retry_delay: float = 5.0
    timeout: int = 30000
    rate_limit: int = 1200
    batch_size: int = 1000
    parallel_downloads: int = 3
    validate_data: bool = True
    outlier_threshold: float = 3.0
    futures_config: Dict[str, Any] = field(default_factory=lambda: {
        'funding_rate_history': True,
        'mark_price_history': True,
        'open_interest_history': True
    })

class DataValidation:
    """Validation des données historiques."""
    
    @staticmethod
    def validate_ohlcv(df: pd.DataFrame) -> Tuple[bool, str]:
        """Valide les données OHLCV."""
        if df.empty:
            return False, "DataFrame vide"
            
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            return False, f"Colonnes manquantes. Requises: {required_columns}"
            
        # Validation des valeurs
        if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
            return False, "Prix négatifs ou nuls détectés"
            
        if (df['high'] < df['low']).any():
            return False, "High inférieur au Low détecté"
            
        if (df['volume'] < 0).any():
            return False, "Volume négatif détecté"
            
        return True, "Validation OK"

    @staticmethod
    def detect_outliers(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """Détecte et traite les outliers."""
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                continue
                
            # Calcul des z-scores
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            
            # Marquage des outliers
            df[f'{col}_outlier'] = z_scores > threshold
            
            # Remplacement des outliers par la moyenne mobile
            mask = df[f'{col}_outlier']
            df.loc[mask, col] = df[col].rolling(window=5, center=True).mean()
            
        return df

class HistoricalDataManager:
    """Gestionnaire des données historiques avec support avancé."""
    
    def __init__(self, config: Optional[Union[Dict, DataConfig]] = None):
        """
        Initialise le gestionnaire de données historiques.
        
        Args:
            config: Configuration du gestionnaire
        """
        self.config = (
            config if isinstance(config, DataConfig)
            else DataConfig(**(config or {}))
        )
        
        self.base_path = Path(self.config.base_path)
        self._setup_directories()
        
        # Configuration de l'exchange
        exchange_config = {
            'enableRateLimit': True,
            'timeout': self.config.timeout,
            'rateLimit': self.config.rate_limit,
            'options': {
                'defaultType': self.config.market_type.value
            }
        }
        
        self.exchange = getattr(ccxt, self.config.exchange_id)(exchange_config)
        self.logger = self._setup_logger()
        self.executor = ThreadPoolExecutor(max_workers=self.config.parallel_downloads)
        self.validation = DataValidation()

    def _setup_directories(self) -> None:
        """Configure les répertoires de données."""
        directories = [
            self.base_path / 'raw',
            self.base_path / 'processed',
            self.base_path / 'cache'
        ]
        
        if self.config.market_type == MarketType.FUTURES:
            directories.extend([
                self.base_path / 'futures/funding_rates',
                self.base_path / 'futures/mark_prices',
                self.base_path / 'futures/open_interest'
            ])
            
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

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

    async def _fetch_with_retry(self, method, *args, **kwargs) -> Any:
        """Exécute une requête avec gestion des retries et backoff exponentiel."""
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                return await method(*args, **kwargs)
            except (ccxt.NetworkError, ccxt.ExchangeError, aiohttp.ClientError) as e:
                last_error = e
                if attempt == self.config.max_retries - 1:
                    raise ExchangeError(f"Échec après {self.config.max_retries} tentatives: {str(e)}")
                    
                delay = self.config.retry_delay * (2 ** attempt)
                self.logger.warning(
                    f"Tentative {attempt + 1} échouée. "
                    f"Nouvel essai dans {delay:.1f} secondes..."
                )
                await asyncio.sleep(delay)
                
        raise ExchangeError(f"Échec inattendu: {str(last_error)}")

    async def fetch_ohlcv(self,
                         symbol: str,
                         timeframe: str = '1h',
                         since: Optional[int] = None,
                         limit: Optional[int] = None) -> pd.DataFrame:
        """
        Récupère les données OHLCV.
        
        Args:
            symbol: Symbole de l'instrument
            timeframe: Intervalle temporel
            since: Timestamp de début
            limit: Nombre maximum de candles
            
        Returns:
            DataFrame avec les données OHLCV
        """
        try:
            # Adaptation du symbole selon l'exchange
            formatted_symbol = (
                symbol.replace('/', '')
                if self.config.exchange_id == 'binance'
                else symbol
            )
            
            # Récupération des données
            ohlcv = await self._fetch_with_retry(
                self.exchange.fetch_ohlcv,
                formatted_symbol,
                timeframe,
                since=since,
                limit=limit or self.config.batch_size
            )
            
            if not ohlcv:
                return pd.DataFrame()
            
            # Conversion en DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            
            # Validation si activée
            if self.config.validate_data:
                valid, message = self.validation.validate_ohlcv(df)
                if not valid:
                    raise ValidationError(f"Validation OHLCV échouée: {message}")
                    
                df = self.validation.detect_outliers(
                    df,
                    threshold=self.config.outlier_threshold
                )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération OHLCV pour {symbol}: {str(e)}")
            raise DataError(f"Échec récupération OHLCV: {str(e)}")

    async def fetch_funding_rates(self,
                                symbol: str,
                                since: Optional[int] = None,
                                limit: Optional[int] = None) -> pd.DataFrame:
        """
        Récupère l'historique des funding rates (futures uniquement).
        
        Args:
            symbol: Symbole du contrat futures
            since: Timestamp de début
            limit: Nombre maximum d'entrées
            
        Returns:
            DataFrame avec l'historique des funding rates
        """
        if self.config.market_type != MarketType.FUTURES:
            raise ConfigError("Funding rates disponibles uniquement en futures")
            
        try:
            rates = await self._fetch_with_retry(
                self.exchange.fetch_funding_rate_history,
                symbol,
                since=since,
                limit=limit or self.config.batch_size
            )
            
            if not rates:
                return pd.DataFrame()
                
            df = pd.DataFrame(rates)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des funding rates pour {symbol}: {str(e)}")
            raise DataError(f"Échec récupération funding rates: {str(e)}")

    async def update_historical_data(self,
                                   symbol: str,
                                   timeframe: str,
                                   start_date: datetime,
                                   end_date: Optional[datetime] = None,
                                   include_funding_rates: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Met à jour les données historiques.
        
        Args:
            symbol: Symbole de l'instrument
            timeframe: Intervalle temporel
            start_date: Date de début
            end_date: Date de fin (optionnelle)
            include_funding_rates: Inclure les funding rates (futures)
            
        Returns:
            Dictionnaire avec les DataFrames (OHLCV et funding rates)
        """
        try:
            since = int(start_date.timestamp() * 1000)
            end_timestamp = int((end_date or datetime.now()).timestamp() * 1000)
            
            # Récupération OHLCV
            ohlcv_data = []
            current_since = since
            
            while current_since < end_timestamp:
                df = await self.fetch_ohlcv(
                    symbol,
                    timeframe,
                    since=current_since
                )
                
                if df.empty:
                    break
                    
                ohlcv_data.append(df)
                current_since = int(df.index[-1].timestamp() * 1000) + 1
                
            result = {
                'ohlcv': pd.concat(ohlcv_data)[~pd.concat(ohlcv_data).index.duplicated(keep='first')]
                if ohlcv_data else pd.DataFrame()
            }
            
            # Récupération des funding rates si demandé et en futures
            if (include_funding_rates and
                self.config.market_type == MarketType.FUTURES and
                self.config.futures_config['funding_rate_history']):
                
                funding_data = []
                current_since = since
                
                while current_since < end_timestamp:
                    df = await self.fetch_funding_rates(
                        symbol,
                        since=current_since
                    )
                    
                    if df.empty:
                        break
                        
                    funding_data.append(df)
                    current_since = int(df.index[-1].timestamp() * 1000) + 1
                    
                result['funding_rates'] = pd.concat(funding_data)[~pd.concat(funding_data).index.duplicated(keep='first')]
                if funding_data else pd.DataFrame()
                
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la mise à jour des données pour {symbol}: {str(e)}")
            raise DataError(f"Échec mise à jour données: {str(e)}")

    async def save_data(self,
                       symbol: str,
                       timeframe: str,
                       data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> None:
        """
        Sauvegarde les données historiques.
        
        Args:
            symbol: Symbole de l'instrument
            timeframe: Intervalle temporel
            data: DataFrame ou dictionnaire de DataFrames à sauvegarder
        """
        try:
            if isinstance(data, pd.DataFrame):
                data = {'ohlcv': data}
                
            for data_type, df in data.items():
                if df.empty:
                    continue
                    
                file_path = self._get_file_path(symbol, timeframe, data_type)
                
                # Optimisation du stockage
                table = pa.Table.from_pandas(df)
                pq.write_table(
                    table,
                    file_path,
                    compression=self.config.compression
                )
                
                self.logger.info(f"Données {data_type} sauvegardées pour {symbol} {timeframe}")
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde des données: {str(e)}")
            raise StorageError(f"Échec sauvegarde données: {str(e)}")

    async def load_data(self,
                       symbol: str,
                       timeframe: str,
                       data_types: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Charge les données historiques.
        
        Args:
            symbol: Symbole de l'instrument
            timeframe: Intervalle temporel
            data_types: Types de données à charger (ex: ['ohlcv', 'funding_rates'])
            
        Returns:
            Dictionnaire des DataFrames chargés
        """
        try:
            if data_types is None:
                data_types = ['ohlcv']
                if self.config.market_type == MarketType.FUTURES:
                    data_types.extend(['funding_rates'])
                    
            result = {}
            
            for data_type in data_types:
                file_path = self._get_file_path(symbol, timeframe, data_type)
                if not file_path.exists():
                    self.logger.warning(f"Pas de données {data_type} pour {symbol} {timeframe}")
                    continue
                    
                # Lecture optimisée
                table = pq.read_table(file_path)
                df = table.to_pandas()
                
                if isinstance(df.index, pd.RangeIndex):
                    df.set_index('timestamp', inplace=True)
                    
                result[data_type] = df
                self.logger.info(f"Données {data_type} chargées pour {symbol} {timeframe}")
                
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement des données: {str(e)}")
            raise StorageError(f"Échec chargement données: {str(e)}")

    def _get_file_path(self,
                       symbol: str,
                       timeframe: str,
                       data_type: str = 'ohlcv') -> Path:
        """
        Construit le chemin du fichier de données.
        
        Args:
            symbol: Symbole de l'instrument
            timeframe: Intervalle temporel
            data_type: Type de données
            
        Returns:
            Chemin du fichier
        """
        # Sélection du sous-répertoire selon le type
        if data_type == 'ohlcv':
            subdir = 'raw'
        elif data_type == 'funding_rates':
            subdir = 'futures/funding_rates'
        elif data_type == 'mark_prices':
            subdir = 'futures/mark_prices'
        elif data_type == 'open_interest':
            subdir = 'futures/open_interest'
        else:
            subdir = 'processed'
            
        return self.base_path / subdir / f"{symbol}_{timeframe}_{data_type}.parquet"

    async def list_available_data(self) -> List[Dict[str, str]]:
        """
        Liste les données disponibles.
        
        Returns:
            Liste des données disponibles avec symbole, timeframe et type
        """
        try:
            available_data = []
            
            # Parcours récursif des répertoires
            for data_type in ['raw', 'futures/funding_rates', 'futures/mark_prices', 'futures/open_interest']:
                data_dir = self.base_path / data_type
                if not data_dir.exists():
                    continue
                    
                for file_path in data_dir.glob("*.parquet"):
                    parts = file_path.stem.split("_")
                    if len(parts) >= 3:
                        available_data.append({
                            "symbol": parts[0],
                            "timeframe": parts[1],
                            "type": parts[2],
                            "path": str(file_path)
                        })
                        
            return available_data
            
        except Exception as e:
            self.logger.error(f"Erreur lors du listage des données: {str(e)}")
            raise StorageError(f"Échec listage données: {str(e)}")

    async def delete_data(self,
                         symbol: str,
                         timeframe: str,
                         data_types: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Supprime les données historiques.
        
        Args:
            symbol: Symbole de l'instrument
            timeframe: Intervalle temporel
            data_types: Types de données à supprimer
            
        Returns:
            Dictionnaire indiquant le succès de la suppression pour chaque type
        """
        try:
            if data_types is None:
                data_types = ['ohlcv']
                if self.config.market_type == MarketType.FUTURES:
                    data_types.extend(['funding_rates', 'mark_prices', 'open_interest'])
                    
            result = {}
            
            for data_type in data_types:
                file_path = self._get_file_path(symbol, timeframe, data_type)
                if not file_path.exists():
                    self.logger.warning(f"Pas de données {data_type} à supprimer pour {symbol} {timeframe}")
                    result[data_type] = False
                    continue
                    
                file_path.unlink()
                self.logger.info(f"Données {data_type} supprimées pour {symbol} {timeframe}")
                result[data_type] = True
                
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la suppression des données: {str(e)}")
            raise StorageError(f"Échec suppression données: {str(e)}")

    async def __aenter__(self):
        """Support du context manager asynchrone."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Fermeture propre des ressources."""
        await self.exchange.close()
        self.executor.shutdown(wait=True)