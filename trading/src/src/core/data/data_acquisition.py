"""
Module d'acquisition de données avec support avancé pour les futures et la validation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from pathlib import Path
import yaml
import json
import time
from functools import wraps
import ccxt.async_support as ccxt
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from .historical_data import HistoricalDataManager
from trading.utils.logging.logger import TradingLogger
from trading.utils.helpers import TradingHelpers
from trading.core.exceptions import (
    DataAcquisitionError,
    NetworkError,
    ExchangeError,
    ValidationError,
    ConfigurationError,
    FuturesError,
    RateLimitError,
    AuthenticationError
)

def load_config(config_path: Optional[str] = None) -> Dict:
    """
    Charge la configuration depuis un fichier YAML ou JSON.
    
    Args:
        config_path: Chemin vers le fichier de configuration
        
    Returns:
        Dict: Configuration chargée
        
    Raises:
        ConfigurationError: Si le fichier est invalide ou inaccessible
    """
    if not config_path:
        return {}
        
    try:
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                return yaml.safe_load(f)
            elif config_path.endswith('.json'):
                return json.load(f)
            else:
                raise ConfigurationError("Format de fichier de configuration non supporté")
    except Exception as e:
        raise ConfigurationError(f"Erreur lors du chargement de la configuration: {str(e)}")

def retry_with_backoff(retries: int = 3, backoff_in_seconds: int = 1):
    """
    Décorateur pour réessayer une opération avec un backoff exponentiel.
    
    Args:
        retries: Nombre de tentatives
        backoff_in_seconds: Délai initial entre les tentatives
        
    Returns:
        Decorator: Décorateur de retry
    """
    def should_retry(exception):
        return isinstance(exception, (
            NetworkError,
            ExchangeError,
            RateLimitError,
            requests.exceptions.RequestException
        ))
    
    return retry(
        stop=stop_after_attempt(retries),
        wait=wait_exponential(multiplier=backoff_in_seconds, min=4, max=10),
        retry=retry_if_exception_type(should_retry),
        reraise=True
    )

class DataAcquisitionBase(ABC):
    """Classe abstraite de base pour l'acquisition de données."""
    
    def __init__(self, config: Optional[Union[Dict, str]] = None):
        """
        Initialise l'acquisition de données.
        
        Args:
            config: Configuration sous forme de dictionnaire ou chemin vers un fichier
                   de configuration (YAML ou JSON)
        """
        if isinstance(config, str):
            self.config = load_config(config)
        else:
            self.config = config or self._default_config()
            
        self.logger = TradingLogger()
        self.helpers = TradingHelpers()
        self.historical_manager = HistoricalDataManager(self.config)
        
        # Initialisation des scalers
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(feature_range=(-1, 1)),
            'robust': RobustScaler(),
            'decimal': lambda x: x / 100,  # Normalisation décimale simple
            'log': lambda x: np.log1p(x)   # Normalisation logarithmique
        }
        
        # Validation de la configuration
        self._validate_config()
        
        # Configuration des futures si activés
        if self.config.get('futures', {}).get('enabled'):
            self._setup_futures()

    def _validate_config(self) -> None:
        """
        Valide la configuration.
        
        Raises:
            ConfigurationError: Si la configuration est invalide
        """
        required_fields = ['default_exchange', 'validate_data']
        missing_fields = [field for field in required_fields if field not in self.config]
        if missing_fields:
            raise ConfigurationError(f"Champs de configuration manquants: {missing_fields}")
            
        # Validation des paramètres futures si activés
        if self.config.get('futures', {}).get('enabled'):
            futures_fields = ['margin_mode', 'leverage', 'hedge_mode']
            missing_futures = [f for f in futures_fields 
                             if f not in self.config.get('futures', {})]
            if missing_futures:
                raise ConfigurationError(
                    f"Configuration futures incomplète. Champs manquants: {missing_futures}"
                )

    def _setup_futures(self) -> None:
        """
        Configure les paramètres pour le trading de futures.
        
        Raises:
            FuturesError: Si la configuration des futures échoue
        """
        try:
            futures_config = self.config['futures']
            exchange = self.historical_manager.get_exchange(
                self.config['default_exchange']
            )
            
            # Vérification du support des futures
            if not exchange.has['future']:
                raise FuturesError("L'exchange ne supporte pas les futures")
                
            # Configuration du mode de marge
            if futures_config['margin_mode'] not in ['cross', 'isolated']:
                raise FuturesError("Mode de marge invalide")
                
            # Configuration du mode hedge si spécifié
            if futures_config.get('hedge_mode'):
                if not exchange.has['setPositionMode']:
                    raise FuturesError("L'exchange ne supporte pas le mode hedge")
                    
            self.logger.info("Configuration futures réussie", futures_config)
            
        except Exception as e:
            raise FuturesError(f"Erreur lors de la configuration des futures: {str(e)}")

    @abstractmethod
    def _default_config(self) -> Dict:
        """Configuration par défaut pour l'acquisition."""
        pass

    @abstractmethod
    async def fetch_data(self, **kwargs) -> pd.DataFrame:
        """Méthode abstraite pour récupérer les données."""
        pass

    @abstractmethod
    def validate_raw_data(self, data: Any) -> bool:
        """Méthode abstraite pour valider les données brutes."""
        pass

    def handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Gère les données manquantes selon la stratégie configurée.
        
        Args:
            data: DataFrame à traiter
            
        Returns:
            DataFrame: Données traitées
            
        Raises:
            ValidationError: Si les données sont invalides
        """
        if data.empty:
            raise ValidationError("Données vides")
            
        # Détection des valeurs manquantes
        missing_count = data.isnull().sum()
        if missing_count.any():
            self.logger.log_warning(
                "Valeurs manquantes détectées",
                {'missing_counts': missing_count.to_dict()}
            )
            
            strategy = self.config.get('missing_data_strategy', 'ffill')
            
            if strategy == 'drop':
                data = data.dropna()
            elif strategy == 'interpolate':
                data = data.interpolate(method='time')
            elif strategy == 'mean':
                data = data.fillna(data.mean())
            elif strategy == 'median':
                data = data.fillna(data.median())
            elif strategy == 'ffill':
                data = data.fillna(method='ffill').fillna(method='bfill')
            else:
                raise ValueError(f"Stratégie de gestion des données manquantes invalide: {strategy}")
                
        return data

    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise les données selon la méthode spécifiée.
        
        Args:
            data: DataFrame à normaliser
            
        Returns:
            DataFrame: Données normalisées
            
        Raises:
            DataAcquisitionError: Si la normalisation échoue
        """
        if not self.config.get('normalize_data'):
            return data
            
        method = self.config.get('normalization_method', 'minmax')
        
        try:
            if method not in self.scalers:
                raise ValueError(f"Méthode de normalisation non supportée: {method}")
                
            scaler = self.scalers[method]
            
            # Colonnes numériques uniquement
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if isinstance(scaler, (StandardScaler, MinMaxScaler, RobustScaler)):
                data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
            else:
                # Pour les méthodes personnalisées (decimal, log)
                data[numeric_cols] = data[numeric_cols].apply(scaler)
                
            return data
            
        except Exception as e:
            raise DataAcquisitionError(f"Erreur lors de la normalisation: {str(e)}")

class MarketDataAcquisition(DataAcquisitionBase):
    """Acquisition de données de marché avec support avancé pour les futures."""

    def _default_config(self) -> Dict:
        """
        Configuration par défaut pour l'acquisition de données de marché.
        
        Returns:
            Dict: Configuration par défaut
        """
        return {
            'default_exchange': 'bitget',
            'default_timeframe': '1h',
            'max_retries': 3,
            'retry_delay': 1.0,
            'validate_data': True,
            'cache_data': True,
            'symbols': ['BTC/USDT', 'ETH/USDT'],
            'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
            'data_types': ['ohlcv', 'trades', 'orderbook'],
            'missing_data_strategy': 'interpolate',
            'normalize_data': False,
            'normalization_method': 'minmax',
            'proxy': None,
            'futures': {
                'enabled': False,
                'margin_mode': 'cross',
                'leverage': 1,
                'hedge_mode': False,
                'position_mode': 'one_way',
                'risk_limits': {
                    'max_position': 100000,
                    'maintenance_margin': 0.01
                }
            },
            'rate_limits': {
                'max_requests_per_second': 10,
                'max_requests_per_minute': 500,
                'retry_after': 60
            },
            'validation': {
                'price_threshold': 0.1,  # 10% de variation max
                'volume_threshold': 1000000,  # Volume max
                'gap_threshold': 300,  # 5 minutes en secondes
                'check_sequence': True,
                'check_gaps': True
            }
        }

    @retry_with_backoff()
    async def fetch_data(self, 
                        symbol: str,
                        data_type: str = 'ohlcv',
                        timeframe: Optional[str] = None,
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None,
                        **kwargs) -> pd.DataFrame:
        """
        Récupère les données selon le type spécifié.
        
        Args:
            symbol: Symbole de trading
            data_type: Type de données ('ohlcv', 'trades', 'orderbook')
            timeframe: Intervalle temporel
            start_time: Date de début
            end_time: Date de fin
            **kwargs: Arguments additionnels
            
        Returns:
            pd.DataFrame: Données récupérées
            
        Raises:
            NetworkError: En cas d'erreur réseau
            ExchangeError: En cas d'erreur de l'exchange
            ValidationError: En cas de données invalides
        """
        try:
            # Paramètres par défaut
            timeframe = timeframe or self.config['default_timeframe']
            end_time = end_time or datetime.utcnow()
            start_time = start_time or (end_time - timedelta(days=1))

            # Configuration des futures si nécessaire
            if self.config['futures'].get('enabled'):
                await self._setup_futures_market(symbol, **kwargs)

            # Sélection de la méthode de récupération
            try:
                if data_type == 'ohlcv':
                    data = await self._fetch_ohlcv(
                        symbol, timeframe, start_time, end_time, **kwargs
                    )
                elif data_type == 'trades':
                    data = await self._fetch_trades(
                        symbol, start_time, end_time, **kwargs
                    )
                elif data_type == 'orderbook':
                    data = await self._fetch_orderbook(
                        symbol, **kwargs
                    )
                elif data_type == 'funding_rate':
                    if not self.config['futures'].get('enabled'):
                        raise FuturesError("Les funding rates sont uniquement disponibles en futures")
                    data = await self._fetch_funding_rate(
                        symbol, start_time, end_time, **kwargs
                    )
                else:
                    raise ValueError(f"Type de données non supporté: {data_type}")

            except ccxt.NetworkError as e:
                raise NetworkError(f"Erreur réseau lors de la récupération des données: {str(e)}")
            except ccxt.ExchangeError as e:
                raise ExchangeError(f"Erreur de l'exchange: {str(e)}")
            except ccxt.RateLimitExceeded as e:
                raise RateLimitError(f"Limite de requêtes atteinte: {str(e)}")
            except Exception as e:
                raise DataAcquisitionError(f"Erreur lors de la récupération des données: {str(e)}")

            # Validation et traitement des données
            if self.config['validate_data']:
                if not self.validate_raw_data(data):
                    raise ValidationError("Données invalides")

            # Gestion des données manquantes et normalisation
            if isinstance(data, pd.DataFrame):
                data = self.handle_missing_data(data)
                data = self.normalize_data(data)

            return data

        except Exception as e:
            self.logger.log_error(e, {
                'action': 'fetch_data',
                'symbol': symbol,
                'data_type': data_type
            })
            raise

    async def _setup_futures_market(self, symbol: str, **kwargs) -> None:
        """
        Configure le marché futures pour un symbole.
        
        Args:
            symbol: Symbole de trading
            **kwargs: Arguments additionnels
            
        Raises:
            FuturesError: Si la configuration échoue
        """
        try:
            exchange = self.historical_manager.get_exchange(
                self.config['default_exchange']
            )
            
            # Configuration du mode de marge
            margin_mode = self.config['futures']['margin_mode']
            await exchange.fapiPrivatePostMarginType({
                'symbol': symbol,
                'marginType': margin_mode.upper()
            })
            
            # Configuration du levier
            leverage = kwargs.get('leverage', self.config['futures']['leverage'])
            await exchange.fapiPrivatePostLeverage({
                'symbol': symbol,
                'leverage': leverage
            })
            
            # Configuration du mode de position si nécessaire
            if self.config['futures'].get('hedge_mode'):
                await exchange.fapiPrivatePostPositionSide({
                    'dualSidePosition': 'true'
                })
                
            self.logger.info("Configuration futures réussie", {
                'symbol': symbol,
                'margin_mode': margin_mode,
                'leverage': leverage
            })
            
        except Exception as e:
            raise FuturesError(f"Erreur lors de la configuration du marché futures: {str(e)}")

    async def _fetch_funding_rate(self, 
                                symbol: str,
                                start_time: datetime,
                                end_time: datetime,
                                **kwargs) -> pd.DataFrame:
        """
        Récupère les taux de funding pour un symbole.
        
        Args:
            symbol: Symbole de trading
            start_time: Date de début
            end_time: Date de fin
            **kwargs: Arguments additionnels
            
        Returns:
            pd.DataFrame: Données de funding rate
        """
        try:
            exchange = self.historical_manager.get_exchange(
                self.config['default_exchange']
            )
            
            funding_rates = await exchange.fetch_funding_rate_history(
                symbol,
                int(start_time.timestamp() * 1000),
                int(end_time.timestamp() * 1000)
            )
            
            df = pd.DataFrame(funding_rates)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            raise DataAcquisitionError(f"Erreur lors de la récupération des funding rates: {str(e)}")

    def validate_raw_data(self, data: Any) -> bool:
        """
        Valide les données brutes selon leur type.
        
        Args:
            data: Données à valider
            
        Returns:
            bool: True si les données sont valides
        """
        try:
            if isinstance(data, pd.DataFrame):
                if data.empty:
                    return False
                    
                # Vérification des colonnes requises selon le type de données
                required_columns = {
                    'ohlcv': ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
                    'trades': ['timestamp', 'price', 'amount', 'side'],
                    'orderbook': ['timestamp', 'bids', 'asks'],
                    'funding_rate': ['timestamp', 'fundingRate']
                }
                
                # Détermination du type de données
                data_type = self._determine_data_type(data.columns)
                if not data_type:
                    self.logger.warning("Type de données non reconnu")
                    return False
                    
                # Vérification des colonnes
                if not all(col in data.columns for col in required_columns[data_type]):
                    return False
                    
                # Validation spécifique selon le type
                if data_type == 'ohlcv':
                    return self._validate_ohlcv(data)
                elif data_type == 'trades':
                    return self._validate_trades(data)
                elif data_type == 'orderbook':
                    return self._validate_orderbook(data)
                elif data_type == 'funding_rate':
                    return self._validate_funding_rate(data)
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la validation: {str(e)}")
            return False

    def _determine_data_type(self, columns: pd.Index) -> Optional[str]:
        """
        Détermine le type de données basé sur les colonnes.
        
        Args:
            columns: Colonnes du DataFrame
            
        Returns:
            str: Type de données ou None si non reconnu
        """
        columns = set(columns)
        if {'open', 'high', 'low', 'close'}.issubset(columns):
            return 'ohlcv'
        elif {'price', 'amount', 'side'}.issubset(columns):
            return 'trades'
        elif {'bids', 'asks'}.issubset(columns):
            return 'orderbook'
        elif {'fundingRate'}.issubset(columns):
            return 'funding_rate'
        return None

    def _validate_ohlcv(self, data: pd.DataFrame) -> bool:
        """
        Valide les données OHLCV.
        
        Args:
            data: DataFrame OHLCV
            
        Returns:
            bool: True si les données sont valides
        """
        validation = self.config['validation']
        
        # Vérification des prix
        if not all(data['low'] <= data[col] <= data['high'] 
                  for col in ['open', 'close']):
            return False
            
        # Vérification des variations de prix
        price_changes = data['close'].pct_change().abs()
        if (price_changes > validation['price_threshold']).any():
            self.logger.warning("Variations de prix suspectes détectées")
            
        # Vérification du volume
        if (data['volume'] > validation['volume_threshold']).any():
            self.logger.warning("Volumes suspects détectés")
            
        # Vérification des gaps temporels
        if validation['check_gaps']:
            gaps = data.index.to_series().diff().dt.total_seconds()
            if (gaps > validation['gap_threshold']).any():
                self.logger.warning("Gaps temporels détectés")
                
        return True

    def _validate_trades(self, data: pd.DataFrame) -> bool:
        """
        Valide les données de trades.
        
        Args:
            data: DataFrame de trades
            
        Returns:
            bool: True si les données sont valides
        """
        # Vérification des prix et volumes positifs
        if (data['price'] <= 0).any() or (data['amount'] <= 0).any():
            return False
            
        # Vérification des sides valides
        if not data['side'].isin(['buy', 'sell']).all():
            return False
            
        return True

    def _validate_orderbook(self, data: pd.DataFrame) -> bool:
        """
        Valide les données du carnet d'ordres.
        
        Args:
            data: DataFrame du carnet d'ordres
            
        Returns:
            bool: True si les données sont valides
        """
        # Vérification de la structure des bids/asks
        if not (isinstance(data['bids'].iloc[0], list) and 
                isinstance(data['asks'].iloc[0], list)):
            return False
            
        # Vérification de l'ordre des prix
        for row in data.itertuples():
            bids = row.bids
            asks = row.asks
            if not (all(b[0] <= a[0] for b in bids for a in asks)):
                return False
                
        return True

    def _validate_funding_rate(self, data: pd.DataFrame) -> bool:
        """
        Valide les données de funding rate.
        
        Args:
            data: DataFrame de funding rate
            
        Returns:
            bool: True si les données sont valides
        """
        # Vérification des valeurs dans une plage raisonnable
        if (data['fundingRate'].abs() > 0.01).any():  # 1% max
            self.logger.warning("Funding rates suspects détectés")
            
        return True

class BatchDataAcquisition(DataAcquisitionBase):
    """Acquisition de données par lots avec support pour les futures."""

    def _default_config(self) -> Dict:
        """
        Configuration par défaut pour l'acquisition par lots.
        
        Returns:
            Dict: Configuration par défaut
        """
        return {
            'batch_size': 1000,
            'max_concurrent_requests': 5,
            'max_retries': 3,
            'retry_delay': 1.0,
            'validate_data': True,
            'cache_data': True,
            'save_format': 'parquet',
            'compression': 'snappy',
            'futures': {
                'enabled': False,
                'margin_mode': 'cross',
                'leverage': 1,
                'hedge_mode': False
            },
            'rate_limits': {
                'max_requests_per_second': 5,
                'max_requests_per_minute': 300,
                'retry_after': 60
            }
        }

    async def fetch_data(self, symbols: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Récupère les données pour plusieurs symboles en parallèle.
        
        Args:
            symbols: Liste des symboles
            **kwargs: Arguments additionnels
            
        Returns:
            Dict[str, pd.DataFrame]: Données par symbole
            
        Raises:
            DataAcquisitionError: En cas d'erreur lors de la récupération
        """
        try:
            # Création d'un sémaphore pour limiter les requêtes concurrentes
            semaphore = asyncio.Semaphore(self.config['max_concurrent_requests'])
            
            # Création des tâches pour chaque symbole
            tasks = []
            for symbol in symbols:
                task = self._fetch_with_semaphore(semaphore, symbol, **kwargs)
                tasks.append(task)
                
            # Exécution des tâches en parallèle
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Traitement des résultats
            data = {}
            for symbol, result in zip(symbols, results):
                if isinstance(result, Exception):
                    self.logger.error(f"Erreur pour {symbol}: {str(result)}")
                    continue
                if result is not None:
                    data[symbol] = result
                    
            if not data:
                raise DataAcquisitionError("Aucune donnée récupérée")
                
            return data
            
        except Exception as e:
            raise DataAcquisitionError(f"Erreur lors de la récupération des données: {str(e)}")

    async def _fetch_with_semaphore(self, 
                                  semaphore: asyncio.Semaphore,
                                  symbol: str,
                                  **kwargs) -> Optional[pd.DataFrame]:
        """
        Récupère les données pour un symbole avec un sémaphore.
        
        Args:
            semaphore: Sémaphore pour limiter les requêtes
            symbol: Symbole de trading
            **kwargs: Arguments additionnels
            
        Returns:
            Optional[pd.DataFrame]: Données récupérées ou None en cas d'erreur
        """
        async with semaphore:
            try:
                # Configuration des futures si nécessaire
                if self.config['futures'].get('enabled'):
                    await self._setup_futures_market(symbol, **kwargs)
                    
                # Récupération des données
                data = await self.historical_manager.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=kwargs.get('timeframe', '1h'),
                    start_time=kwargs.get('start_time'),
                    end_time=kwargs.get('end_time')
                )
                
                # Validation et traitement
                if self.config['validate_data']:
                    if not self.validate_raw_data({symbol: data}):
                        self.logger.warning(f"Données invalides pour {symbol}")
                        return None
                        
                return data
                
            except Exception as e:
                self.logger.error(f"Erreur pour {symbol}: {str(e)}")
                return None

    def validate_raw_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """
        Valide les données pour tous les symboles.
        
        Args:
            data: Dictionnaire des données par symbole
            
        Returns:
            bool: True si toutes les données sont valides
        """
        try:
            if not data:
                return False
                
            # Validation de chaque symbole
            for symbol, df in data.items():
                if df is None or df.empty:
                    self.logger.warning(f"Données manquantes pour {symbol}")
                    return False
                    
                # Vérification des colonnes OHLCV
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    self.logger.warning(f"Colonnes manquantes pour {symbol}")
                    return False
                    
                # Vérification des valeurs numériques
                if not all(df[col].dtype.kind in 'fc' for col in required_cols):
                    self.logger.warning(f"Types de données invalides pour {symbol}")
                    return False
                    
                # Vérification de la cohérence des prix
                if not (
                    (df['low'] <= df['high']).all() and
                    (df['low'] <= df['close']).all() and
                    (df['low'] <= df['open']).all() and
                    (df['high'] >= df['close']).all() and
                    (df['high'] >= df['open']).all()
                ):
                    self.logger.warning(f"Prix incohérents pour {symbol}")
                    return False
                    
                # Vérification des volumes
                if (df['volume'] < 0).any():
                    self.logger.warning(f"Volumes négatifs pour {symbol}")
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la validation: {str(e)}")
            return False

    def save_batch(self, data: Dict[str, pd.DataFrame], batch_name: str) -> None:
        """
        Sauvegarde les données par lots.
        
        Args:
            data: Dictionnaire des données par symbole
            batch_name: Nom du lot
            
        Raises:
            DataAcquisitionError: En cas d'erreur lors de la sauvegarde
        """
        try:
            save_dir = Path(self.config.get('save_path', 'data/batches'))
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Création du répertoire pour le lot
            batch_dir = save_dir / batch_name
            batch_dir.mkdir(exist_ok=True)
            
            # Sauvegarde des données pour chaque symbole
            for symbol, df in data.items():
                # Nettoyage du nom de fichier
                clean_symbol = symbol.replace('/', '_')
                file_path = batch_dir / f"{clean_symbol}.{self.config['save_format']}"
                
                # Sauvegarde selon le format configuré
                if self.config['save_format'] == 'parquet':
                    df.to_parquet(
                        file_path,
                        compression=self.config['compression']
                    )
                elif self.config['save_format'] == 'csv':
                    df.to_csv(file_path)
                elif self.config['save_format'] == 'feather':
                    df.to_feather(file_path)
                else:
                    raise ValueError(f"Format non supporté: {self.config['save_format']}")
                    
            # Sauvegarde des métadonnées
            metadata = {
                'batch_name': batch_name,
                'symbols': list(data.keys()),
                'timestamp': datetime.now().isoformat(),
                'config': self.config
            }
            
            with open(batch_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
                
            self.logger.info(f"Lot {batch_name} sauvegardé avec succès")
            
        except Exception as e:
            raise DataAcquisitionError(f"Erreur lors de la sauvegarde du lot: {str(e)}")

    async def _setup_futures_market(self, symbol: str, **kwargs) -> None:
        """
        Configure le marché futures pour un symbole.
        
        Args:
            symbol: Symbole de trading
            **kwargs: Arguments additionnels
            
        Raises:
            FuturesError: Si la configuration échoue
        """
        try:
            exchange = self.historical_manager.get_exchange(
                self.config['default_exchange']
            )
            
            # Configuration du mode de marge
            margin_mode = self.config['futures']['margin_mode']
            await exchange.fapiPrivatePostMarginType({
                'symbol': symbol,
                'marginType': margin_mode.upper()
            })
            
            # Configuration du levier
            leverage = kwargs.get('leverage', self.config['futures']['leverage'])
            await exchange.fapiPrivatePostLeverage({
                'symbol': symbol,
                'leverage': leverage
            })
            
            # Configuration du mode de position si nécessaire
            if self.config['futures'].get('hedge_mode'):
                await exchange.fapiPrivatePostPositionSide({
                    'dualSidePosition': 'true'
                })
                
            self.logger.info("Configuration futures réussie", {
                'symbol': symbol,
                'margin_mode': margin_mode,
                'leverage': leverage
            })
            
        except Exception as e:
            raise FuturesError(f"Erreur lors de la configuration du marché futures: {str(e)}")

# Exemple d'utilisation
async def main():
    """Exemple d'utilisation des classes d'acquisition de données."""
    try:
        # Configuration
        config = {
            'default_exchange': 'binance',
            'validate_data': True,
            'futures': {
                'enabled': True,
                'margin_mode': 'cross',
                'leverage': 2
            }
        }
        
        # Création des instances
        market_data = MarketDataAcquisition(config)
        batch_data = BatchDataAcquisition(config)
        
        # Récupération des données spot
        spot_data = await market_data.fetch_data(
            symbol='BTC/USDT',
            timeframe='1h',
            start_time=datetime.now() - timedelta(days=1)
        )
        
        # Récupération des données futures
        futures_data = await market_data.fetch_data(
            symbol='BTC/USDT',
            timeframe='1h',
            start_time=datetime.now() - timedelta(days=1),
            data_type='funding_rate'
        )
        
        # Récupération par lots
        symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT']
        batch_results = await batch_data.fetch_data(
            symbols=symbols,
            timeframe='1h',
            start_time=datetime.now() - timedelta(days=1)
        )
        
        # Sauvegarde des résultats
        batch_data.save_batch(batch_results, 'daily_update')
        
        print("Acquisition des données terminée avec succès")
        
    except Exception as e:
        print(f"Erreur: {str(e)}")
        
    finally:
        # Fermeture des connexions
        await market_data.historical_manager.close()
        await batch_data.historical_manager.close()

if __name__ == "__main__":
    asyncio.run(main())