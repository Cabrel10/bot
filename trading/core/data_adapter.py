"""Module d'adaptation et de préparation des données.

Ce module fournit des classes et fonctions pour :
- La validation et le nettoyage des données
- La transformation et la normalisation
- La gestion des séries temporelles
- L'optimisation de la mémoire
- La conversion entre différents formats
- Le monitoring des performances
- La gestion du cache
"""

from typing import Tuple, Dict, Optional, List, Union, Any, Protocol, Callable
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from pathlib import Path
import logging
from functools import lru_cache
import numba
from concurrent.futures import ThreadPoolExecutor

from ..core.data_types import MarketData
from ..utils.logger import TradingLogger
from ..utils.memory_manager import MemoryManager
from ..utils.validation import validate_market_data, validate_positive
from ..utils.metrics import MetricsCollector

class DataTransformer(Protocol):
    """Protocol pour les transformateurs de données."""
    def fit(self, X: np.ndarray) -> 'DataTransformer':
        """Entraîne le transformateur."""
        ...
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Applique la transformation."""
        ...
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Entraîne et applique la transformation."""
        ...

class OutlierDetector(BaseEstimator, TransformerMixin):
    """Détecteur d'outliers configurable."""
    
    def __init__(self, method: str = 'zscore', threshold: float = 3.0):
        """
        Initialise le détecteur.
        
        Args:
            method: Méthode de détection ('zscore', 'iqr', 'isolation_forest')
            threshold: Seuil de détection
        """
        self.method = method
        self.threshold = threshold
        self._stats = {}
        
    def fit(self, X: np.ndarray) -> 'OutlierDetector':
        """Calcule les statistiques."""
        if self.method == 'zscore':
            self._stats['mean'] = np.mean(X, axis=0)
            self._stats['std'] = np.std(X, axis=0)
        elif self.method == 'iqr':
            q1 = np.percentile(X, 25, axis=0)
            q3 = np.percentile(X, 75, axis=0)
            self._stats['iqr'] = q3 - q1
            self._stats['q1'] = q1
            self._stats['q3'] = q3
        elif self.method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            self.model = IsolationForest(contamination=0.1)
            self.model.fit(X)
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Détecte et remplace les outliers."""
        if self.method == 'zscore':
            z_scores = np.abs((X - self._stats['mean']) / self._stats['std'])
            outliers = z_scores > self.threshold
        elif self.method == 'iqr':
            iqr = self._stats['iqr']
            lower = self._stats['q1'] - self.threshold * iqr
            upper = self._stats['q3'] + self.threshold * iqr
            outliers = (X < lower) | (X > upper)
        else:
            outliers = self.model.predict(X) == -1
            
        # Remplacement des outliers par la moyenne mobile
        X_clean = X.copy()
        if outliers.any():
            window = 5
            rolling_mean = pd.DataFrame(X).rolling(
                window=window, min_periods=1
            ).mean().values
            X_clean[outliers] = rolling_mean[outliers]
            
        return X_clean

@dataclass
class DataTransformConfig:
    """Configuration des transformations de données."""
    
    # Paramètres de base
    normalization_method: str = 'standard'
    window_size: int = 20
    target_column: str = 'close'
    feature_columns: List[str] = field(default_factory=list)
    batch_size: int = 32
    validation_split: float = 0.2
    shuffle: bool = True
    
    # Gestion des données manquantes
    fill_method: str = 'ffill'
    interpolation_method: str = 'time'
    max_missing_ratio: float = 0.1
    
    # Détection des outliers
    outlier_method: str = 'zscore'
    outlier_threshold: float = 3.0
    
    # Optimisation
    use_cache: bool = True
    cache_size: int = 1000
    n_jobs: int = -1
    
    # Monitoring
    collect_metrics: bool = True
    metrics_interval: int = 60
    
    def __post_init__(self):
        """Valide la configuration."""
        self._validate_normalization()
        self._validate_window_params()
        self._validate_missing_params()
        self._validate_outlier_params()
        self._validate_optimization_params()
        
    def _validate_normalization(self):
        """Valide les paramètres de normalisation."""
        valid_methods = ['standard', 'minmax', 'robust']
        if self.normalization_method not in valid_methods:
            raise ValueError(
                f"Méthode de normalisation invalide: {self.normalization_method}"
                f"\nMéthodes valides: {valid_methods}"
            )
            
    def _validate_window_params(self):
        """Valide les paramètres de fenêtre."""
        validate_positive(self.window_size, 'window_size')
        validate_positive(self.batch_size, 'batch_size')
        
        if not 0 <= self.validation_split <= 1:
            raise ValueError("validation_split doit être entre 0 et 1")
            
    def _validate_missing_params(self):
        """Valide les paramètres de gestion des valeurs manquantes."""
        valid_methods = ['ffill', 'bfill', 'interpolate']
        if self.fill_method not in valid_methods:
            raise ValueError(
                f"Méthode de remplissage invalide: {self.fill_method}"
                f"\nMéthodes valides: {valid_methods}"
            )
            
        if not 0 <= self.max_missing_ratio <= 1:
            raise ValueError("max_missing_ratio doit être entre 0 et 1")
            
    def _validate_outlier_params(self):
        """Valide les paramètres de détection des outliers."""
        valid_methods = ['zscore', 'iqr', 'isolation_forest']
        if self.outlier_method not in valid_methods:
            raise ValueError(
                f"Méthode de détection des outliers invalide: {self.outlier_method}"
                f"\nMéthodes valides: {valid_methods}"
            )
            
        if self.outlier_threshold <= 0:
            raise ValueError("outlier_threshold doit être positif")
            
    def _validate_optimization_params(self):
        """Valide les paramètres d'optimisation."""
        if self.cache_size <= 0:
            raise ValueError("cache_size doit être positif")
            
        if self.n_jobs == 0:
            raise ValueError("n_jobs ne peut pas être 0")

class DataAdapter:
    """Adaptateur pour la conversion et la gestion des données."""
    
    def __init__(
        self,
        config: DataTransformConfig,
        memory_manager: Optional[MemoryManager] = None
    ):
        """
        Initialise l'adaptateur de données.
        
        Args:
            config: Configuration des transformations
            memory_manager: Gestionnaire de mémoire optionnel
        """
        self.config = config
        self.logger = TradingLogger()
        self.memory_manager = memory_manager or MemoryManager()
        
        # Métriques
        if self.config.collect_metrics:
            self.metrics = MetricsCollector(
                interval=self.config.metrics_interval
            )
        
        # Pool de threads
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.n_jobs if self.config.n_jobs > 0
            else None
        )
        
        # Initialisation des transformateurs
        self.transformers = {}
        self._initialize_transformers()
        
        # Cache LRU
        if self.config.use_cache:
            self._cache = {}
            self.get_cached_data = lru_cache(maxsize=self.config.cache_size)(
                self._get_data
            )
        
    def _initialize_transformers(self) -> None:
        """Initialise les pipelines de transformation."""
        # Sélection du scaler
        if self.config.normalization_method == 'standard':
            scaler = StandardScaler()
        elif self.config.normalization_method == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()
            
        # Détecteur d'outliers
        outlier_detector = OutlierDetector(
            method=self.config.outlier_method,
            threshold=self.config.outlier_threshold
        )
        
        # Pipeline principal
        self.transformers['main'] = Pipeline([
            ('outliers', outlier_detector),
            ('scaler', scaler)
        ])
        
    @numba.jit(nopython=True)
    def _fast_rolling_window(self, data: np.ndarray, window: int) -> np.ndarray:
        """Version optimisée de la création de fenêtres glissantes."""
        shape = (data.shape[0] - window + 1, window, data.shape[1])
        strides = (data.strides[0], data.strides[0], data.strides[1])
        return np.lib.stride_tricks.as_strided(
            data,
            shape=shape,
            strides=strides,
            writeable=False
        )
        
    async def prepare_training_data(
        self,
        market_data: MarketData,
        validation: bool = True
    ) -> Union[
        Tuple[tf.data.Dataset, tf.data.Dataset],
        tf.data.Dataset
    ]:
        """
        Prépare les données pour l'entraînement.
        
        Args:
            market_data: Données de marché
            validation: Inclure un set de validation
            
        Returns:
            Dataset(s) pour l'entraînement (et validation)
            
        Raises:
            ValueError: Si les données sont invalides
        """
        try:
            # Validation des données
            self._validate_input_data(market_data)
            
            # Utilisation du cache si activé
            if self.config.use_cache:
                data = self.get_cached_data(market_data)
            else:
                data = await self._preprocess_data(market_data)
            
            # Vérification de la mémoire
            if not self._check_memory_requirements(data):
                self.logger.warning("Mémoire limitée, utilisation du mode streaming")
                return await self._prepare_streaming_data(data, validation)
                
            # Préparation standard
            return await self._prepare_batch_data(data, validation)
            
        except Exception as e:
            self.logger.error(f"Erreur de préparation: {str(e)}")
            if self.config.collect_metrics:
                self.metrics.record_error('data_preparation', str(e))
            raise
            
    def _validate_input_data(self, market_data: MarketData) -> None:
        """Valide les données d'entrée."""
        # Validation de base
        validate_market_data(market_data)
        
        # Vérification des colonnes requises
        missing_cols = [
            col for col in self.config.feature_columns
            if col not in market_data.columns
        ]
        if missing_cols:
            raise ValueError(f"Colonnes manquantes: {missing_cols}")
            
        # Vérification des valeurs manquantes
        missing_ratio = market_data.isnull().mean()
        if (missing_ratio > self.config.max_missing_ratio).any():
            problematic_cols = missing_ratio[
                missing_ratio > self.config.max_missing_ratio
            ].index.tolist()
            raise ValueError(
                f"Trop de valeurs manquantes dans: {problematic_cols}"
            )
            
    def _get_data(self, market_data: MarketData) -> pd.DataFrame:
        """Récupère les données (utilisé avec le cache)."""
        return self._preprocess_data(market_data)
        
    async def _preprocess_data(self, market_data: MarketData) -> pd.DataFrame:
        """
        Prétraite les données.
        
        Args:
            market_data: Données brutes
            
        Returns:
            DataFrame prétraité
        """
        start_time = datetime.now()
        
        try:
            # Sélection des features
            data = market_data.get_features(self.config.feature_columns)
            
            # Gestion des valeurs manquantes
            data = await self._handle_missing_values(data)
            
            # Normalisation et gestion des outliers via pipeline
            data = self.transformers['main'].fit_transform(data)
            
            # Conversion en DataFrame
            data = pd.DataFrame(
                data,
                index=market_data.index,
                columns=self.config.feature_columns
            )
            
            # Enregistrement des métriques
            if self.config.collect_metrics:
                duration = (datetime.now() - start_time).total_seconds()
                self.metrics.record_metric(
                    'preprocessing_time',
                    duration
                )
                self.metrics.record_metric(
                    'data_size',
                    len(data)
                )
                
            return data
            
        except Exception as e:
            self.logger.error(f"Erreur de prétraitement: {str(e)}")
            if self.config.collect_metrics:
                self.metrics.record_error('preprocessing', str(e))
            raise
            
    async def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Gère les valeurs manquantes."""
        if data.isnull().any().any():
            self.logger.warning("Valeurs manquantes détectées")
            
            if self.config.fill_method == 'interpolate':
                return data.interpolate(method=self.config.interpolation_method)
            return getattr(data, self.config.fill_method)()
            
        return data
        
    def _check_memory_requirements(self, data: pd.DataFrame) -> bool:
        """Vérifie si les données peuvent tenir en mémoire."""
        required_memory = (
            len(data) *
            len(data.columns) *
            data.dtypes.iloc[0].itemsize *
            2  # Pour les copies temporaires
        )
        
        return self.memory_manager.check_memory_available(required_memory)
        
    async def _prepare_streaming_data(
        self,
        data: pd.DataFrame,
        validation: bool
    ) -> Union[
        Tuple[tf.data.Dataset, tf.data.Dataset],
        tf.data.Dataset
    ]:
        """Prépare les données en mode streaming."""
        dataset = TimeSeriesDataset(
            data,
            self.config.window_size,
            self.config.target_column
        )
        
        if not validation:
            return tf.data.Dataset.from_tensor_slices(dataset)
        
        # Split train/validation
        train_size = int(len(dataset) * (1 - self.config.validation_split))
        train_dataset, val_dataset = tf.data.Dataset.from_tensor_slices(
            dataset[:train_size]
        ), tf.data.Dataset.from_tensor_slices(
            dataset[train_size:]
        )
        
        return train_dataset, val_dataset
        
    async def _prepare_batch_data(
        self,
        data: pd.DataFrame,
        validation: bool
    ) -> Union[
        Tuple[tf.data.Dataset, tf.data.Dataset],
        tf.data.Dataset
    ]:
        """Prépare les données en mode batch."""
        X, y = [], []
        
        for i in range(len(data) - self.config.window_size):
            X.append(data.iloc[i:i+self.config.window_size].values)
            y.append(data[self.config.target_column].iloc[i+self.config.window_size])
            
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        
        if not validation:
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
            return dataset.batch(self.config.batch_size)
            
        # Split train/validation
        train_size = int(len(X) * (1 - self.config.validation_split))
        
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        
        return train_dataset.batch(self.config.batch_size), val_dataset.batch(self.config.batch_size)
        
    def save_transformers(self, path: Union[str, Path]) -> None:
        """Sauvegarde les transformateurs."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.transformers, path)
        
    def load_transformers(self, path: Union[str, Path]) -> None:
        """Charge les transformateurs."""
        self.transformers = joblib.load(path)
        
    async def cleanup(self) -> None:
        """Nettoie les ressources."""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
            
        if hasattr(self, 'metrics'):
            await self.metrics.cleanup()

class TimeSeriesDataset:
    """Dataset pour les séries temporelles."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int,
        target_column: str
    ):
        """
        Initialise le dataset.
        
        Args:
            data: Données temporelles
            window_size: Taille de la fenêtre
            target_column: Colonne cible
        """
        self.data = data
        self.window_size = window_size
        self.target_column = target_column
        
        # Création des fenêtres glissantes
        self.features = np.array([
            col for col in data.columns
            if col != target_column
        ])
        self.targets = data[target_column].values
        self.windows = self._create_windows()
        
    def _create_windows(self) -> np.ndarray:
        """Crée les fenêtres glissantes de manière optimisée."""
        return self._fast_rolling_window(
            self.data.values,
            self.window_size
        )
        
    def __len__(self) -> int:
        """Retourne la taille du dataset."""
        return len(self.windows)
        
    def __getitem__(self, idx: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Retourne un élément du dataset."""
        window = self.windows[idx]
        target_idx = idx + self.window_size - 1
        
        return (
            tf.convert_to_tensor(window, dtype=tf.float32),
            tf.convert_to_tensor([self.targets[target_idx]], dtype=tf.float32)
        )
        
    @staticmethod
    def market_data_to_tensor(
        market_data: MarketData,
        features: Optional[List[str]] = None
    ) -> tf.Tensor:
        """
        Convertit MarketData en tenseur TensorFlow.
        
        Args:
            market_data: Données de marché
            features: Liste des features à inclure
            
        Returns:
            tf.Tensor: Données au format tenseur
            
        Raises:
            ValueError: Si la conversion échoue
        """
        try:
            if features is None:
                features = market_data.get_feature_names()
                
            data = market_data.get_features(features)
            return tf.convert_to_tensor(data.values, dtype=tf.float32)
            
        except Exception as e:
            raise ValueError(f"Erreur de conversion: {str(e)}")
            
    @staticmethod
    def tensor_to_market_data(
        tensor: tf.Tensor,
        timestamps: pd.DatetimeIndex,
        feature_names: List[str]
    ) -> MarketData:
        """
        Convertit un tenseur en MarketData.
        
        Args:
            tensor: Données au format tenseur
            timestamps: Index temporel
            feature_names: Noms des features
            
        Returns:
            MarketData: Données converties
            
        Raises:
            ValueError: Si la conversion échoue
        """
        try:
            data = pd.DataFrame(
                tensor.numpy(),
                index=timestamps,
                columns=feature_names
            )
            return MarketData(data)
            
        except Exception as e:
            raise ValueError(f"Erreur de conversion inverse: {str(e)}") 