from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Union, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from .data_types import OHLCVData, TechnicalIndicators, ProcessedData
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
# Remplacer talib par notre module de remplacement
from ...utils.talib_mock import SMA, EMA, RSI, MACD, BBANDS, ATR, STOCH
import json
from ...utils.logging.logger import TradingLogger
import ta
import joblib

from .data_acquisition import MarketDataAcquisition, BatchDataAcquisition
from ...utils.data_preprocessing import DataPreprocessor
from ...utils.helpers import TradingHelpers
from ..types import PreprocessingResult, ProcessingConfig
from trading.core.exceptions import DataPreparationError

@dataclass
class PreparedData:
    """Structure de données pour les données préparées."""
    raw_data: pd.DataFrame
    processed_data: pd.DataFrame
    feature_data: pd.DataFrame
    normalized_data: pd.DataFrame
    metadata: Dict

@dataclass
class CacheMetadata:
    """Métadonnées pour le cache des données."""
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    features: List[str]
    last_update: datetime
    data_points: int

@dataclass
class PreparationConfig:
    """Configuration pour la préparation des données."""
    sequence_length: int = 60
    target_column: str = 'close'
    features: List[str] = None
    technical_indicators: Dict[str, Dict] = None
    scaling_method: str = 'standard'
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    random_state: int = 42
    outlier_threshold: float = 3.0
    missing_threshold: float = 0.1
    min_periods: int = 20
    futures_enabled: bool = False
    cache_dir: Optional[Path] = None

class DataPreparation:
    """Préparation sophistiquée des données pour le trading."""
    
    def __init__(self, config: Optional[Union[Dict, PreparationConfig]] = None):
        """
        Initialise le préparateur de données.
        
        Args:
            config: Configuration de préparation
        """
        self.logger = TradingLogger()
        
        if isinstance(config, dict):
            self.config = self._create_config_from_dict(config)
        elif isinstance(config, PreparationConfig):
            self.config = config
        else:
            self.config = PreparationConfig()
            
        self._initialize_scalers()
        self._setup_technical_indicators()
        
        if self.config.cache_dir:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)
            
    def _create_config_from_dict(self, config_dict: Dict) -> PreparationConfig:
        """Crée une configuration à partir d'un dictionnaire."""
        return PreparationConfig(
            sequence_length=config_dict.get('sequence_length', 60),
            target_column=config_dict.get('target_column', 'close'),
            features=config_dict.get('features'),
            technical_indicators=config_dict.get('technical_indicators'),
            scaling_method=config_dict.get('scaling_method', 'standard'),
            train_split=config_dict.get('train_split', 0.7),
            val_split=config_dict.get('val_split', 0.15),
            test_split=config_dict.get('test_split', 0.15),
            random_state=config_dict.get('random_state', 42),
            outlier_threshold=config_dict.get('outlier_threshold', 3.0),
            missing_threshold=config_dict.get('missing_threshold', 0.1),
            min_periods=config_dict.get('min_periods', 20),
            futures_enabled=config_dict.get('futures_enabled', False),
            cache_dir=Path(config_dict['cache_dir']) if 'cache_dir' in config_dict else None
        )
        
    def _initialize_scalers(self) -> None:
        """Initialise les scalers disponibles."""
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(feature_range=(-1, 1)),
            'robust': RobustScaler(),
            'log': lambda x: np.log1p(x),
            'decimal': lambda x: x / 100
        }
        
        self.fitted_scalers = {}
        
    def _setup_technical_indicators(self) -> None:
        """Configure les indicateurs techniques disponibles."""
        self.default_indicators = {
            'sma': {'window': 20},
            'ema': {'window': 20},
            'macd': {'window_slow': 26, 'window_fast': 12, 'window_sign': 9},
            'rsi': {'window': 14},
            'bollinger': {'window': 20, 'window_dev': 2},
            'atr': {'window': 14},
            'stoch': {'window': 14, 'smooth_window': 3},
            'obv': {},
            'vwap': {'window': 14}
        }
        
        if not self.config.technical_indicators:
            self.config.technical_indicators = self.default_indicators
            
    def prepare_data(self, 
                    data: pd.DataFrame,
                    add_technical_indicators: bool = True,
                    handle_outliers: bool = True) -> pd.DataFrame:
        """
        Prépare les données pour l'entraînement.
        
        Args:
            data: DataFrame source
            add_technical_indicators: Si True, ajoute les indicateurs techniques
            handle_outliers: Si True, gère les outliers
            
        Returns:
            DataFrame préparé
        """
        try:
            # Copie pour éviter la modification en place
            prepared_data = data.copy()
            
            # Vérification des données manquantes
            self._check_missing_data(prepared_data)
            
            # Ajout des indicateurs techniques
            if add_technical_indicators:
                prepared_data = self.add_technical_indicators(prepared_data)
                
            # Gestion des outliers
            if handle_outliers:
                prepared_data = self.handle_outliers(prepared_data)
                
            # Ajout de features temporelles
            prepared_data = self.add_temporal_features(prepared_data)
            
            # Features spécifiques aux futures si activé
            if self.config.futures_enabled:
                prepared_data = self.add_futures_features(prepared_data)
                
            # Normalisation des données
            prepared_data = self.normalize_features(prepared_data)
            
            return prepared_data
            
        except Exception as e:
            raise DataPreparationError(f"Erreur lors de la préparation des données: {str(e)}")
            
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les indicateurs techniques configurés."""
        try:
            result = data.copy()
            
            for indicator, params in self.config.technical_indicators.items():
                if indicator == 'sma':
                    sma = SMAIndicator(
                        close=data['close'],
                        window=params['window']
                    )
                    result[f'sma_{params["window"]}'] = sma.sma_indicator()
                    
                elif indicator == 'ema':
                    ema = EMAIndicator(
                        close=data['close'],
                        window=params['window']
                    )
                    result[f'ema_{params["window"]}'] = ema.ema_indicator()
                    
                elif indicator == 'macd':
                    macd = MACD(
                        close=data['close'],
                        window_slow=params['window_slow'],
                        window_fast=params['window_fast'],
                        window_sign=params['window_sign']
                    )
                    result['macd'] = macd.macd()
                    result['macd_signal'] = macd.macd_signal()
                    result['macd_diff'] = macd.macd_diff()
                    
                elif indicator == 'rsi':
                    rsi = RSIIndicator(
                        close=data['close'],
                        window=params['window']
                    )
                    result[f'rsi_{params["window"]}'] = rsi.rsi()
                    
                elif indicator == 'bollinger':
                    bollinger = BollingerBands(
                        close=data['close'],
                        window=params['window'],
                        window_dev=params['window_dev']
                    )
                    result['bb_high'] = bollinger.bollinger_hband()
                    result['bb_mid'] = bollinger.bollinger_mavg()
                    result['bb_low'] = bollinger.bollinger_lband()
                    
                elif indicator == 'atr':
                    atr = AverageTrueRange(
                        high=data['high'],
                        low=data['low'],
                        close=data['close'],
                        window=params['window']
                    )
                    result[f'atr_{params["window"]}'] = atr.average_true_range()
                    
                elif indicator == 'stoch':
                    stoch = StochasticOscillator(
                        high=data['high'],
                        low=data['low'],
                        close=data['close'],
                        window=params['window'],
                        smooth_window=params['smooth_window']
                    )
                    result['stoch_k'] = stoch.stoch()
                    result['stoch_d'] = stoch.stoch_signal()
                    
                elif indicator == 'obv':
                    obv = OnBalanceVolumeIndicator(
                        close=data['close'],
                        volume=data['volume']
                    )
                    result['obv'] = obv.on_balance_volume()
                    
                elif indicator == 'vwap':
                    vwap = VolumeWeightedAveragePrice(
                        high=data['high'],
                        low=data['low'],
                        close=data['close'],
                        volume=data['volume'],
                        window=params['window']
                    )
                    result['vwap'] = vwap.volume_weighted_average_price()
                    
            return result
            
        except Exception as e:
            raise DataPreparationError(f"Erreur lors de l'ajout des indicateurs techniques: {str(e)}")
            
    def handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Gère les outliers dans les données.
        
        Args:
            data: DataFrame à traiter
            
        Returns:
            DataFrame sans outliers
        """
        try:
            result = data.copy()
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                # Calcul des limites basées sur l'IQR
                Q1 = result[col].quantile(0.25)
                Q3 = result[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.config.outlier_threshold * IQR
                upper_bound = Q3 + self.config.outlier_threshold * IQR
                
                # Identification des outliers
                outliers = (result[col] < lower_bound) | (result[col] > upper_bound)
                
                if outliers.any():
                    self.logger.warning(f"Outliers détectés dans {col}: {outliers.sum()} points")
                    
                    # Remplacement des outliers par les bornes
                    result.loc[result[col] < lower_bound, col] = lower_bound
                    result.loc[result[col] > upper_bound, col] = upper_bound
                    
            return result
            
        except Exception as e:
            raise DataPreparationError(f"Erreur lors de la gestion des outliers: {str(e)}")
            
    def add_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute des features temporelles au DataFrame.
        
        Args:
            data: DataFrame source
            
        Returns:
            DataFrame avec features temporelles
        """
        try:
            result = data.copy()
            
            # Conversion de l'index en datetime si nécessaire
            if not isinstance(result.index, pd.DatetimeIndex):
                result.index = pd.to_datetime(result.index)
                
            # Features cycliques pour l'heure
            result['hour_sin'] = np.sin(2 * np.pi * result.index.hour / 24)
            result['hour_cos'] = np.cos(2 * np.pi * result.index.hour / 24)
            
            # Features cycliques pour le jour de la semaine
            result['day_sin'] = np.sin(2 * np.pi * result.index.dayofweek / 7)
            result['day_cos'] = np.cos(2 * np.pi * result.index.dayofweek / 7)
            
            # Features cycliques pour le mois
            result['month_sin'] = np.sin(2 * np.pi * result.index.month / 12)
            result['month_cos'] = np.cos(2 * np.pi * result.index.month / 12)
            
            # Indicateur de jour ouvré
            result['is_business_day'] = result.index.dayofweek < 5
            
            # Heure de la journée normalisée
            result['day_progress'] = (result.index.hour * 3600 + 
                                    result.index.minute * 60 + 
                                    result.index.second) / 86400
                                    
            return result
            
        except Exception as e:
            raise DataPreparationError(f"Erreur lors de l'ajout des features temporelles: {str(e)}")
            
    def add_futures_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute des features spécifiques aux futures.
        
        Args:
            data: DataFrame source
            
        Returns:
            DataFrame avec features futures
        """
        try:
            if not self.config.futures_enabled:
                return data
                
            result = data.copy()
            
            # Calcul du basis (différence entre prix spot et futures)
            if 'index_price' in result.columns:
                result['basis'] = result['close'] - result['index_price']
                result['basis_pct'] = result['basis'] / result['index_price'] * 100
                
            # Calcul des variations de funding rate
            if 'funding_rate' in result.columns:
                result['funding_rate_change'] = result['funding_rate'].diff()
                result['funding_rate_ma'] = result['funding_rate'].rolling(
                    window=self.config.min_periods
                ).mean()
                
            # Calcul de l'open interest normalisé
            if 'open_interest' in result.columns:
                result['open_interest_change'] = result['open_interest'].diff()
                result['open_interest_ma'] = result['open_interest'].rolling(
                    window=self.config.min_periods
                ).mean()
                
            # Calcul de la volatilité implicite
            if all(col in result.columns for col in ['high', 'low', 'close']):
                atr = AverageTrueRange(
                    high=result['high'],
                    low=result['low'],
                    close=result['close'],
                    window=self.config.min_periods
                )
                result['implied_volatility'] = atr.average_true_range()
                
            return result
            
        except Exception as e:
            raise DataPreparationError(f"Erreur lors de l'ajout des features futures: {str(e)}")
            
    def normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise les features selon la méthode spécifiée.
        
        Args:
            data: DataFrame à normaliser
            
        Returns:
            DataFrame normalisé
        """
        try:
            result = data.copy()
            
            # Sélection des colonnes numériques
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            
            # Application du scaler approprié
            scaler = self.scalers.get(self.config.scaling_method)
            if scaler is None:
                raise ValueError(f"Méthode de scaling invalide: {self.config.scaling_method}")
                
            if isinstance(scaler, (StandardScaler, MinMaxScaler, RobustScaler)):
                # Fit et transform pour les scalers sklearn
                scaled_data = scaler.fit_transform(result[numeric_cols])
                result[numeric_cols] = scaled_data
                self.fitted_scalers[self.config.scaling_method] = scaler
            else:
                # Application directe pour les fonctions de transformation
                result[numeric_cols] = result[numeric_cols].apply(scaler)
                
            return result
            
        except Exception as e:
            raise DataPreparationError(f"Erreur lors de la normalisation: {str(e)}")
            
    def create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crée des séquences pour l'apprentissage.
        
        Args:
            data: DataFrame source
            
        Returns:
            Tuple de (X, y) pour l'entraînement
        """
        try:
            # Sélection des features si spécifiées
            if self.config.features:
                features_data = data[self.config.features]
            else:
                features_data = data
                
            # Création des séquences
            sequences = []
            targets = []
            
            for i in range(len(data) - self.config.sequence_length):
                # Séquence de features
                seq = features_data.iloc[i:i + self.config.sequence_length].values
                sequences.append(seq)
                
                # Valeur cible
                target = data[self.config.target_column].iloc[i + self.config.sequence_length]
                targets.append(target)
                
            return np.array(sequences), np.array(targets)
            
        except Exception as e:
            raise DataPreparationError(f"Erreur lors de la création des séquences: {str(e)}")
            
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Divise les données en ensembles d'entraînement, validation et test.
        
        Args:
            X: Features
            y: Cibles
            
        Returns:
            Dictionnaire contenant les ensembles divisés
        """
        try:
            # Vérification des proportions
            total_split = self.config.train_split + self.config.val_split + self.config.test_split
            if not np.isclose(total_split, 1.0):
                raise ValueError("La somme des proportions doit être égale à 1")
                
            # Calcul des indices de division
            n_samples = len(X)
            indices = np.arange(n_samples)
            
            if self.config.random_state is not None:
                np.random.seed(self.config.random_state)
                np.random.shuffle(indices)
                
            # Points de division
            train_end = int(n_samples * self.config.train_split)
            val_end = train_end + int(n_samples * self.config.val_split)
            
            # Division des données
            train_indices = indices[:train_end]
            val_indices = indices[train_end:val_end]
            test_indices = indices[val_end:]
            
            return {
                'X_train': X[train_indices],
                'y_train': y[train_indices],
                'X_val': X[val_indices],
                'y_val': y[val_indices],
                'X_test': X[test_indices],
                'y_test': y[test_indices]
            }
            
        except Exception as e:
            raise DataPreparationError(f"Erreur lors de la division des données: {str(e)}")
            
    def save_scalers(self, filepath: Union[str, Path]) -> None:
        """
        Sauvegarde les scalers entraînés.
        
        Args:
            filepath: Chemin de sauvegarde
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(self.fitted_scalers, filepath)
            self.logger.info(f"Scalers sauvegardés dans {filepath}")
            
        except Exception as e:
            raise DataPreparationError(f"Erreur lors de la sauvegarde des scalers: {str(e)}")
            
    def load_scalers(self, filepath: Union[str, Path]) -> None:
        """
        Charge les scalers sauvegardés.
        
        Args:
            filepath: Chemin vers les scalers
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileNotFoundError(f"Fichier non trouvé: {filepath}")
                
            self.fitted_scalers = joblib.load(filepath)
            self.logger.info(f"Scalers chargés depuis {filepath}")
            
        except Exception as e:
            raise DataPreparationError(f"Erreur lors du chargement des scalers: {str(e)}")
            
    def _check_missing_data(self, data: pd.DataFrame) -> None:
        """Vérifie les données manquantes."""
        missing_ratio = data.isnull().sum() / len(data)
        problematic_cols = missing_ratio[missing_ratio > self.config.missing_threshold]
        
        if not problematic_cols.empty:
            self.logger.warning(
                "Colonnes avec trop de valeurs manquantes: "
                f"{problematic_cols.to_dict()}"
            )
            raise DataPreparationError(
                f"Colonnes avec plus de {self.config.missing_threshold*100}% "
                "de valeurs manquantes"
        )

class DataCache:
    """Gère le stockage local des données historiques."""
    def __init__(self, cache_dir: str = "./data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def save_to_cache(self, data: ProcessedData, symbol: str, timeframe: str) -> None:
        """Sauvegarde les données dans un fichier Parquet avec métadonnées."""
        filename = self.cache_dir / f"{symbol}_{timeframe}.parquet"
        metadata_file = self.cache_dir / f"{symbol}_{timeframe}_metadata.parquet"
        
        # Sauvegarde des données
        pd.DataFrame(data['raw_data']).to_parquet(filename)
        
        # Sauvegarde des métadonnées
        metadata = CacheMetadata(
            symbol=symbol,
            timeframe=timeframe,
            last_update=datetime.now(),
            data_points=len(data['raw_data']['timestamp'])
        )
        pd.DataFrame([metadata]).to_parquet(metadata_file)

    def load_from_cache(self, symbol: str, timeframe: str) -> tuple[Optional[ProcessedData], Optional[CacheMetadata]]:
        """Charge les données depuis le cache avec leurs métadonnées."""
        filename = self.cache_dir / f"{symbol}_{timeframe}.parquet"
        metadata_file = self.cache_dir / f"{symbol}_{timeframe}_metadata.parquet"
        
        if not (filename.exists() and metadata_file.exists()):
            return None, None
            
        data = pd.read_parquet(filename)
        metadata = pd.read_parquet(metadata_file).iloc[0].to_dict()
        
        return ProcessedData(
            raw_data=OHLCVData(**data.to_dict('list')),
            indicators={},  # Les indicateurs seront recalculés si nécessaire
            normalized_data=pd.DataFrame()
        ), CacheMetadata(**metadata)

class DataPreparationPipeline:
    """Pipeline de préparation des données pour le trading."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialise le pipeline de préparation.
        
        Args:
            config: Configuration du pipeline
        """
        self.logger = TradingLogger()
        self.helpers = TradingHelpers()
        self.config = config or self._default_config()
        self.preprocessor = DataPreprocessor(self.config.get('preprocessing'))
        self.market_data = MarketDataAcquisition(self.config.get('market_data'))
        self.cache_dir = Path(self.config.get('cache_dir', 'data/prepared'))
        self.processing_config = ProcessingConfig(
            feature_engineering=self.config.get('feature_engineering', True),
            normalization=self.config.get('normalization', True),
            handle_missing=self.config.get('handle_missing', True),
            handle_outliers=self.config.get('handle_outliers', True),
            technical_indicators=self.config.get('technical_indicators', True)
        )

    def _default_config(self) -> Dict:
        """Configuration par défaut du pipeline."""
        return {
            'cache_dir': 'data/prepared',
            'cache_enabled': True,
            'feature_engineering': {
                'technical_indicators': True,
                'price_derivatives': True,
                'volume_analysis': True,
                'market_indicators': True
            },
            'preprocessing': {
                'normalization': 'min_max',
                'sequence_length': 60,
                'target_type': 'returns'
            },
            'validation': {
                'min_data_points': 1000,
                'max_missing_values': 0.01,
                'correlation_threshold': 0.95
            }
        }

    async def prepare_data(self,
                          symbol: str,
                          timeframe: str,
                          start_time: datetime,
                          end_time: datetime,
                          use_cache: bool = True) -> PreparedData:
        """Prépare les données pour l'entraînement ou le trading.
        
        Args:
            symbol: Symbole de trading
            timeframe: Intervalle temporel
            start_time: Date de début
            end_time: Date de fin
            use_cache: Utiliser le cache si disponible
            
        Returns:
            PreparedData: Données préparées et métadonnées
        """
        try:
            # Vérification du cache
            if use_cache and self.config['cache_enabled']:
                cached_data = self._load_from_cache(symbol, timeframe, start_time, end_time)
                if cached_data is not None:
                    return cached_data

            # Acquisition des données brutes
            raw_data = await self.market_data.fetch_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )

            # Préparation des données
            prepared_data = self._prepare_single_symbol(
                raw_data,
                symbol,
                timeframe
            )

            # Sauvegarde dans le cache
            if self.config['cache_enabled']:
                self._save_to_cache(prepared_data, symbol, timeframe)

            return prepared_data

        except Exception as e:
            self.logger.log_error(e, {
                'action': 'prepare_data',
                'symbol': symbol,
                'timeframe': timeframe
            })
            raise

    def _prepare_single_symbol(self,
                             data: pd.DataFrame,
                             symbol: str,
                             timeframe: str) -> PreparedData:
        """Prépare les données pour un symbole."""
        try:
            # Nettoyage initial
            processed_data = self.preprocessor.process_data(data)

            # Création des features
            feature_data = self._engineer_features(processed_data)

            # Normalisation
            normalized_data = self._normalize_data(feature_data)

            # Validation
            if not self._validate_prepared_data(normalized_data):
                raise ValueError("Les données préparées ne passent pas la validation")

            # Métadonnées
            metadata = {
                'symbol': symbol,
                'timeframe': timeframe,
                'start_date': data.index.min(),
                'end_date': data.index.max(),
                'features': list(feature_data.columns),
                'data_points': len(data),
                'last_update': datetime.now()
            }

            return PreparedData(
                raw_data=data,
                processed_data=processed_data,
                feature_data=feature_data,
                normalized_data=normalized_data,
                metadata=metadata
            )

        except Exception as e:
            self.logger.log_error(e, {
                'action': '_prepare_single_symbol',
                'symbol': symbol
            })
            raise

    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crée des features additionnelles."""
        feature_data = data.copy()
        
        if self.config['feature_engineering']['technical_indicators']:
            # Indicateurs techniques déjà calculés par le preprocessor
            pass

        if self.config['feature_engineering']['price_derivatives']:
            # Dérivées des prix
            feature_data['price_velocity'] = feature_data['close'].diff()
            feature_data['price_acceleration'] = feature_data['price_velocity'].diff()

        if self.config['feature_engineering']['volume_analysis']:
            # Analyse du volume
            feature_data['volume_ma'] = feature_data['volume'].rolling(20).mean()
            feature_data['volume_std'] = feature_data['volume'].rolling(20).std()
            feature_data['volume_zscore'] = (feature_data['volume'] - feature_data['volume_ma']) / feature_data['volume_std']

        if self.config['feature_engineering']['market_indicators']:
            # Indicateurs de marché
            feature_data['daily_range'] = feature_data['high'] - feature_data['low']
            feature_data['daily_range_ma'] = feature_data['daily_range'].rolling(20).mean()
            feature_data['range_expansion'] = feature_data['daily_range'] / feature_data['daily_range_ma']

        return feature_data.dropna()

    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalise les données selon la configuration."""
        return self.preprocessor._normalize_features(data)

    def _validate_prepared_data(self, data: pd.DataFrame) -> bool:
        """Valide les données préparées."""
        try:
            # Vérification du nombre minimum de points
            if len(data) < self.config['validation']['min_data_points']:
                return False

            # Vérification des valeurs manquantes
            missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
            if missing_ratio > self.config['validation']['max_missing_values']:
                return False

            # Vérification des corrélations
            correlation_matrix = data.corr().abs()
            mask = np.triu(np.ones_like(correlation_matrix), k=1)
            high_corr = (correlation_matrix * mask > self.config['validation']['correlation_threshold']).any().any()
            if high_corr:
                self.logger.logger.warning("Haute corrélation détectée dans les features")

            return True

        except Exception as e:
            self.logger.log_error(e, {'action': '_validate_prepared_data'})
            return False

    def _save_to_cache(self,
                      prepared_data: PreparedData,
                      symbol: str,
                      timeframe: str) -> None:
        """Sauvegarde les données préparées dans le cache."""
        try:
            # Création du répertoire de cache
            cache_dir = self.cache_dir / symbol / timeframe
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Nom de fichier basé sur la période
            start_date = prepared_data.metadata['start_date'].strftime('%Y%m%d')
            end_date = prepared_data.metadata['end_date'].strftime('%Y%m%d')
            base_name = f"{start_date}_{end_date}"

            # Sauvegarde des données
            prepared_data.normalized_data.to_parquet(cache_dir / f"{base_name}_normalized.parquet")
            prepared_data.feature_data.to_parquet(cache_dir / f"{base_name}_features.parquet")
            
            # Sauvegarde des métadonnées
            with open(cache_dir / f"{base_name}_metadata.json", 'w') as f:
                json.dump(prepared_data.metadata, f, default=str)

        except Exception as e:
            self.logger.log_error(e, {
                'action': '_save_to_cache',
                'symbol': symbol,
                'timeframe': timeframe
            })

    def _load_from_cache(self,
                        symbol: str,
                        timeframe: str,
                        start_time: datetime,
                        end_time: datetime) -> Optional[PreparedData]:
        """Charge les données depuis le cache."""
        try:
            cache_dir = self.cache_dir / symbol / timeframe
            if not cache_dir.exists():
                return None

            # Recherche des fichiers de cache appropriés
            for metadata_file in cache_dir.glob('*_metadata.json'):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                cache_start = datetime.fromisoformat(metadata['start_date'])
                cache_end = datetime.fromisoformat(metadata['end_date'])
                
                if cache_start <= start_time and cache_end >= end_time:
                    base_name = metadata_file.stem.replace('_metadata', '')
                    
                    # Chargement des données
                    normalized_data = pd.read_parquet(cache_dir / f"{base_name}_normalized.parquet")
                    feature_data = pd.read_parquet(cache_dir / f"{base_name}_features.parquet")
                    
                    # Filtrage de la période demandée
                    mask = (normalized_data.index >= start_time) & (normalized_data.index <= end_time)
                    
                    return PreparedData(
                        raw_data=feature_data[mask],  # On utilise feature_data comme raw_data
                        processed_data=feature_data[mask],
                        feature_data=feature_data[mask],
                        normalized_data=normalized_data[mask],
                        metadata=metadata
                    )

            return None

        except Exception as e:
            self.logger.log_error(e, {
                'action': '_load_from_cache',
                'symbol': symbol,
                'timeframe': timeframe
            })
            return None

# Exemple d'utilisation
if __name__ == "__main__":
    async def main():
        # Configuration
        config = {
            'cache_enabled': True,
            'feature_engineering': {
                'technical_indicators': True,
                'price_derivatives': True,
                'volume_analysis': True
            }
        }

        # Création du pipeline
        pipeline = DataPreparationPipeline(config)
        
        try:
            # Préparation des données
            start_time = datetime(2023, 1, 1)
            end_time = datetime(2023, 12, 31)
            
            prepared_data = await pipeline.prepare_data(
                symbol='BTC/USDT',
                timeframe='1h',
                start_time=start_time,
                end_time=end_time
            )
            
            print("Données préparées avec succès")
            print(f"Nombre de features: {prepared_data.feature_data.shape[1]}")
            print(f"Période: {prepared_data.metadata['start_date']} - {prepared_data.metadata['end_date']}")
            
        except Exception as e:
            print(f"Erreur: {e}")

    # Exécution
    import asyncio
    asyncio.run(main())