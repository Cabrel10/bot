from typing import (
    List, Optional, Union, Dict, Any, Tuple, Protocol,
    TypeVar, Generic, Callable, Iterator, Type
)
from dataclasses import dataclass, asdict, field
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging
from abc import ABC, abstractmethod
import pyarrow as pa
import pyarrow.parquet as pq
from functools import lru_cache
import warnings
from enum import Enum
import hashlib
import zlib
from concurrent.futures import ThreadPoolExecutor

# Type générique pour les données
T = TypeVar('T')

class DataFormat(Enum):
    """Formats de sérialisation supportés."""
    JSON = 'json'
    PARQUET = 'parquet'
    PICKLE = 'pickle'
    CSV = 'csv'

class ValidationLevel(Enum):
    """Niveaux de validation des données."""
    NONE = 0
    BASIC = 1
    STRICT = 2

class Validatable(Protocol):
    """Protocol pour les classes validables."""
    def validate(self, level: ValidationLevel = ValidationLevel.BASIC) -> Tuple[bool, List[str]]:
        """
        Valide l'objet et retourne (is_valid, errors).
        
        Args:
            level: Niveau de validation
        """
        ...
        
    def is_valid(self) -> bool:
        """Vérifie rapidement si l'objet est valide."""
        ...

class Serializable(Protocol):
    """Protocol pour les classes sérialisables."""
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'objet en dictionnaire."""
        ...
        
    def to_bytes(self, format: DataFormat = DataFormat.JSON) -> bytes:
        """Convertit l'objet en bytes."""
        ...
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Serializable':
        """Crée une instance depuis un dictionnaire."""
        ...
        
    @classmethod
    def from_bytes(cls, data: bytes, format: DataFormat = DataFormat.JSON) -> 'Serializable':
        """Crée une instance depuis des bytes."""
        ...

class TimeSeriesData(Generic[T]):
    """Interface pour les données temporelles."""
    
    @abstractmethod
    def resample(self, timeframe: str) -> 'TimeSeriesData[T]':
        """
        Rééchantillonne les données.
        
        Args:
            timeframe: Période de rééchantillonnage (e.g. '1H', '1D')
        """
        ...
        
    @abstractmethod
    def slice_time(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> 'TimeSeriesData[T]':
        """
        Découpe les données selon une période.
        
        Args:
            start: Date de début (None = depuis le début)
            end: Date de fin (None = jusqu'à la fin)
        """
        ...
        
    @abstractmethod
    def rolling_window(
        self,
        window: int,
        min_periods: Optional[int] = None
    ) -> Iterator['TimeSeriesData[T]']:
        """
        Crée un itérateur de fenêtres glissantes.
        
        Args:
            window: Taille de la fenêtre
            min_periods: Nombre minimum de périodes
        """
        ...
        
    @abstractmethod
    def aggregate(
        self,
        freq: str,
        func: Union[str, Callable] = 'mean'
    ) -> 'TimeSeriesData[T]':
        """
        Agrège les données selon une fréquence.
        
        Args:
            freq: Fréquence d'agrégation
            func: Fonction d'agrégation
        """
        ...

@dataclass
class BaseData:
    """Classe de base pour les données."""
    
    # Métadonnées - utilisation de KW_ONLY pour éviter les conflits d'ordre
    metadata: Dict[str, Any] = field(default_factory=dict, kw_only=True)
    
    # Cache pour les calculs (optionnel)
    _validation_cache: Dict[str, Tuple[bool, List[str]]] = field(
        default_factory=dict,
        init=False,
        repr=False
    )
    _hash_cache: Optional[str] = field(
        default=None,
        init=False,
        repr=False
    )
    
    def add_metadata(self, key: str, value: Any) -> None:
        """
        Ajoute une métadonnée.
        
        Args:
            key: Clé de la métadonnée
            value: Valeur à ajouter
        """
        self.metadata[key] = value
        self._invalidate_cache()
        
    def get_metadata(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Récupère une métadonnée.
        
        Args:
            key: Clé de la métadonnée
            default: Valeur par défaut
        """
        return self.metadata.get(key, default)
        
    def update_metadata(self, updates: Dict[str, Any]) -> None:
        """
        Met à jour plusieurs métadonnées.
        
        Args:
            updates: Dictionnaire de mises à jour
        """
        self.metadata.update(updates)
        self._invalidate_cache()
        
    def save(
        self,
        path: Union[str, Path],
        format: DataFormat = DataFormat.JSON,
        compress: bool = True
    ) -> None:
        """
        Sauvegarde les données.
        
        Args:
            path: Chemin de sauvegarde
            format: Format de sérialisation
            compress: Compression des données
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Conversion en bytes
        data = self.to_bytes(format)
        
        # Compression si demandée
        if compress:
            data = zlib.compress(data)
        
        # Sauvegarde
        with open(path, 'wb') as f:
            f.write(data)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        format: DataFormat = DataFormat.JSON,
        compressed: bool = True
    ) -> 'BaseData':
        """
        Charge les données.
        
        Args:
            path: Chemin du fichier
            format: Format de sérialisation
            compressed: Données compressées
        """
        path = Path(path)
        
        # Lecture des données
        with open(path, 'rb') as f:
            data = f.read()
            
        # Décompression si nécessaire
        if compressed:
            data = zlib.decompress(data)
            
        return cls.from_bytes(data, format)
        
    def to_bytes(self, format: DataFormat = DataFormat.JSON) -> bytes:
        """
        Convertit en bytes.
        
        Args:
            format: Format de sérialisation
        """
        if format == DataFormat.JSON:
            return json.dumps(
                self.to_dict(),
                default=self._json_serialize
            ).encode()
        elif format == DataFormat.PARQUET:
            table = pa.Table.from_pydict(self.to_dict())
            return table.serialize().to_pybytes()
        elif format == DataFormat.PICKLE:
            import pickle
            return pickle.dumps(self)
        else:
            raise ValueError(f"Format non supporté: {format}")

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        format: DataFormat = DataFormat.JSON
    ) -> 'BaseData':
        """
        Crée une instance depuis des bytes.
        
        Args:
            data: Données sérialisées
            format: Format de sérialisation
        """
        if format == DataFormat.JSON:
            return cls.from_dict(json.loads(data))
        elif format == DataFormat.PARQUET:
            table = pa.deserialize(data)
            return cls.from_dict(table.to_pydict())
        elif format == DataFormat.PICKLE:
            import pickle
            return pickle.loads(data)
        else:
            raise ValueError(f"Format non supporté: {format}")
            
    def _json_serialize(self, obj: Any) -> Any:
        """Sérialise les types spéciaux pour JSON."""
        if isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, pd.DatetimeIndex):
            return [d.isoformat() for d in obj]
        if isinstance(obj, Enum):
            return obj.value
        return str(obj)
        
    def _invalidate_cache(self) -> None:
        """Invalide les caches."""
        self._validation_cache.clear()
        self._hash_cache = None
        
    def __hash__(self) -> int:
        """Calcule le hash de l'objet."""
        if self._hash_cache is None:
            # Création d'un hash stable
            hasher = hashlib.sha256()
            hasher.update(str(self.to_dict()).encode())
            self._hash_cache = hasher.hexdigest()
        return int(self._hash_cache, 16)

@dataclass
class MarketData(BaseData):
    """Données de marché."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str

@dataclass
class OHLCV(BaseData):
    """Données OHLCV (Open, High, Low, Close, Volume)."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class OrderData(BaseData):
    """Données d'un ordre."""
    order_id: str
    symbol: str
    order_type: str
    side: str
    quantity: float
    timestamp: Optional[datetime] = None
    price: Optional[float] = None
    stop_price: Optional[float] = None

@dataclass
class TradeData(BaseData):
    """Données d'un trade."""
    trade_id: str
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    fee: float = 0.0

@dataclass
class PositionData(BaseData):
    """Données d'une position."""
    symbol: str
    quantity: float
    average_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime

@dataclass
class StrategyParameters(BaseData):
    """Paramètres d'une stratégie."""
    name: str
    timeframe: str
    symbols: List[str]
    parameters: Dict[str, Union[float, int, str, bool]]

@dataclass
class BacktestResult(BaseData):
    """Résultats d'un backtest."""
    trades: List[TradeData]
    positions: List[PositionData]
    equity_curve: pd.Series
    metrics: Dict[str, float]
    parameters: StrategyParameters

@dataclass
class ProcessedData(BaseData, Validatable, Serializable):
    """Données prétraitées pour l'entraînement ou la prédiction."""
    # Champs obligatoires
    X: np.ndarray
    feature_names: List[str]
    
    # Champs optionnels
    y: Optional[np.ndarray] = None
    timestamps: Optional[Union[pd.DatetimeIndex, List[datetime]]] = None
    target_names: Optional[List[str]] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    
    def __post_init__(self):
        """Validation après initialisation."""
        if self.timestamps is not None:
            self.timestamps = pd.DatetimeIndex(self.timestamps)
            
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(self.X.shape[1])]
            
        if self.y is not None and self.target_names is None:
            self.target_names = [f"target_{i}" for i in range(self.y.shape[1])]

    def validate(self) -> Tuple[bool, List[str]]:
        """Valide l'intégrité des données."""
        errors = []
        
        try:
            # Vérification des dimensions
            if self.y is not None and len(self.X) != len(self.y):
                errors.append(
                    f"Dimensions X/y incohérentes: {self.X.shape}/{self.y.shape}"
                )
                
            if (self.timestamps is not None and
                len(self.timestamps) != len(self.X)):
                errors.append(
                    "Nombre de timestamps incorrect"
                )
                
            # Vérification des noms de features
            if len(self.feature_names) != self.X.shape[1]:
                errors.append(
                    "Nombre de noms de features incorrect"
                )
                
            # Vérification des valeurs manquantes
            if np.isnan(self.X).any():
                errors.append("Valeurs manquantes dans X")
                
            if self.y is not None and np.isnan(self.y).any():
                errors.append("Valeurs manquantes dans y")
                
        except Exception as e:
            errors.append(f"Erreur de validation: {str(e)}")
            
        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        data = {
            'X': self.X.tolist(),
            'feature_names': self.feature_names,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'metadata': self.metadata
        }
        
        if self.y is not None:
            data['y'] = self.y.tolist()
        if self.timestamps is not None:
            data['timestamps'] = [t.isoformat() for t in self.timestamps]
        if self.target_names is not None:
            data['target_names'] = self.target_names
            
        return data
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessedData':
        """Crée une instance depuis un dictionnaire."""
        # Conversion des données
        X = np.array(data['X'])
        y = np.array(data['y']) if 'y' in data else None
        
        # Conversion des timestamps
        timestamps = None
        if 'timestamps' in data:
            timestamps = pd.DatetimeIndex([
                pd.to_datetime(t) for t in data['timestamps']
            ])
            
        return cls(
            X=X,
            y=y,
            timestamps=timestamps,
            feature_names=data.get('feature_names'),
            target_names=data.get('target_names'),
            symbol=data.get('symbol'),
            timeframe=data.get('timeframe'),
            metadata=data.get('metadata', {})
        )

@dataclass
class TrainingData:
    """Données pour l'entraînement des modèles."""
    # Champs obligatoires
    X_train: np.ndarray
    y_train: np.ndarray
    
    # Champs optionnels
    X_val: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None
    X_test: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    target_names: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ModelPrediction:
    """Prédiction individuelle du modèle."""
    # Champs obligatoires
    timestamp: datetime
    symbol: str
    side: str
    price: float
    amount: float
    cost: float
    fee: float
    
    # Champs optionnels
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class PredictionResult:
    """Résultat complet des prédictions."""
    # Champs obligatoires
    predictions: List[ModelPrediction]
    
    # Champs optionnels
    metadata: Optional[Dict[str, Any]] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convertit les prédictions en DataFrame."""
        data = []
        for pred in self.predictions:
            data.append({
                'timestamp': pred.timestamp,
                'symbol': pred.symbol,
                'side': pred.side,
                'price': pred.price,
                'amount': pred.amount,
                'cost': pred.cost,
                'fee': pred.fee,
                'realized_pnl': pred.realized_pnl,
                'unrealized_pnl': pred.unrealized_pnl
            })
        return pd.DataFrame(data)

@dataclass
class ValidationResult:
    """Résultat de la validation des données."""
    # Champs obligatoires
    is_valid: bool
    missing_values: Dict[str, int]
    outliers: Dict[str, List[int]]
    data_quality_score: float
    
    # Champs optionnels
    errors: List[str] = field(default_factory=list)

@dataclass
class ModelMetrics:
    """Métriques d'évaluation du modèle."""
    # Champs obligatoires
    model_name: str
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    profitable_trades: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Champs optionnels
    additional_metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.additional_metrics is None:
            self.additional_metrics = {}

@dataclass
class TrainingConfig:
    """Configuration pour l'entraînement."""
    # Champs obligatoires
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    shuffle: bool = True
    random_state: int = 42
    use_gpu: bool = True
    
    # Champs optionnels
    save_checkpoints: bool = True
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'

@dataclass
class TimeSeriesMarketData(BaseData, Validatable, Serializable, TimeSeriesData['TimeSeriesMarketData']):
    """Données de marché complètes avec fonctionnalités de séries temporelles."""
    # Champs obligatoires
    timestamp: Union[pd.DatetimeIndex, List[datetime]]
    open: Union[np.ndarray, List[float]]
    high: Union[np.ndarray, List[float]]
    low: Union[np.ndarray, List[float]]
    close: Union[np.ndarray, List[float]]
    volume: Union[np.ndarray, List[float]]
    
    # Champs optionnels
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    indicators: Optional[pd.DataFrame] = None

    def __post_init__(self):
        """Validation après initialisation."""
        self.timestamp = pd.DatetimeIndex(self.timestamp)
        self.open = np.array(self.open, dtype=float)
        self.high = np.array(self.high, dtype=float)
        self.low = np.array(self.low, dtype=float)
        self.close = np.array(self.close, dtype=float)
        self.volume = np.array(self.volume, dtype=float)
        
        if self.indicators is not None:
            self.indicators.index = self.timestamp

    def validate(self) -> Tuple[bool, List[str]]:
        """Valide l'intégrité des données."""
        errors = []
        
        try:
            # Vérification des dimensions
            lengths = {
                'timestamp': len(self.timestamp),
                'open': len(self.open),
                'high': len(self.high),
                'low': len(self.low),
                'close': len(self.close),
                'volume': len(self.volume)
            }
            
            if len(set(lengths.values())) != 1:
                errors.append(
                    f"Dimensions incohérentes: {lengths}"
                )
            
            # Vérification des valeurs
            if np.any(self.low > self.high):
                errors.append("Low > High détecté")
            if np.any(self.open > self.high):
                errors.append("Open > High détecté")
            if np.any(self.close > self.high):
                errors.append("Close > High détecté")
            if np.any(self.open < self.low):
                errors.append("Open < Low détecté")
            if np.any(self.close < self.low):
                errors.append("Close < Low détecté")
            if np.any(self.volume < 0):
                errors.append("Volume négatif détecté")
            
            # Vérification des timestamps
            if not self.timestamp.is_monotonic_increasing:
                errors.append("Timestamps non monotones")
                
            # Vérification des indicateurs
            if (self.indicators is not None and
                len(self.indicators) != len(self.timestamp)):
                errors.append("Dimensions des indicateurs incohérentes")
                
        except Exception as e:
            errors.append(f"Erreur de validation: {str(e)}")
            
        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        data = {
            'timestamp': [t.isoformat() for t in self.timestamp],
            'open': self.open.tolist(),
            'high': self.high.tolist(),
            'low': self.low.tolist(),
            'close': self.close.tolist(),
            'volume': self.volume.tolist(),
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'metadata': self.metadata
        }
        
        if self.indicators is not None:
            data['indicators'] = self.indicators.to_dict('records')
            
        return data
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimeSeriesMarketData':
        """Crée une instance depuis un dictionnaire."""
        # Conversion des timestamps
        timestamps = pd.DatetimeIndex([
            pd.to_datetime(t) for t in data['timestamp']
        ])
        
        # Conversion des indicateurs
        indicators = None
        if 'indicators' in data:
            indicators = pd.DataFrame(data['indicators'])
            indicators.index = timestamps
            
        return cls(
            timestamp=timestamps,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            volume=data['volume'],
            indicators=indicators,
            symbol=data.get('symbol'),
            timeframe=data.get('timeframe'),
            metadata=data.get('metadata', {})
        )

    def resample(self, timeframe: str) -> 'TimeSeriesMarketData':
        """Rééchantillonne les données."""
        df = self.to_dataframe()
        
        # Règles d'agrégation
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Ajout des règles pour les indicateurs
        if self.indicators is not None:
            for col in self.indicators.columns:
                agg_rules[col] = 'last'
        
        resampled = df.resample(timeframe).agg(agg_rules)
        
        # Mise à jour du timeframe
        result = self.from_dataframe(resampled)
        result.timeframe = timeframe
        return result
        
    def slice_time(
        self,
        start: datetime,
        end: datetime
    ) -> 'TimeSeriesMarketData':
        """Découpe les données selon une période."""
        mask = (self.timestamp >= start) & (self.timestamp <= end)
        
        indicators = None
        if self.indicators is not None:
            indicators = self.indicators.loc[mask]
            
        return TimeSeriesMarketData(
            timestamp=self.timestamp[mask],
            open=self.open[mask],
            high=self.high[mask],
            low=self.low[mask],
            close=self.close[mask],
            volume=self.volume[mask],
            indicators=indicators,
            symbol=self.symbol,
            timeframe=self.timeframe,
            metadata=self.metadata.copy()
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convertit les données en DataFrame."""
        df = pd.DataFrame({
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }, index=self.timestamp)
        
        if self.indicators is not None:
            df = pd.concat([df, self.indicators], axis=1)
        
        return df

@dataclass
class OrderBook:
    """Carnet d'ordres."""
    # Champs obligatoires
    timestamp: datetime
    symbol: str
    bids: List[List[float]]
    asks: List[List[float]]
    
    def get_mid_price(self) -> float:
        """Calcule le prix moyen entre le meilleur bid et le meilleur ask."""
        if not self.bids or not self.asks:
            return 0.0
        return (self.bids[0][0] + self.asks[0][0]) / 2
    
    def get_spread(self) -> float:
        """Calcule le spread entre le meilleur bid et le meilleur ask."""
        if not self.bids or not self.asks:
            return 0.0
        return self.asks[0][0] - self.bids[0][0]

@dataclass
class PerformanceData:
    """Données de performance du modèle."""
    # Champs obligatoires
    equity_curve: pd.Series
    trades: List[TradeData]
    metrics: ModelMetrics
    
    # Champs optionnels
    drawdown_series: Optional[pd.Series] = None
    daily_returns: Optional[pd.Series] = None
    metadata: Optional[Dict[str, Any]] = None

    def calculate_statistics(self) -> Dict[str, float]:
        """Calcule les statistiques de performance supplémentaires."""
        stats = {}
        
        # Calcul des rendements cumulés
        if self.equity_curve is not None:
            stats['total_return'] = (
                self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1
            )
            
        # Calcul du drawdown maximum si non fourni
        if self.drawdown_series is None and self.equity_curve is not None:
            rolling_max = self.equity_curve.expanding().max()
            drawdown = (self.equity_curve - rolling_max) / rolling_max
            stats['max_drawdown'] = abs(drawdown.min())
            
        # Calcul des rendements quotidiens si non fournis
        if self.daily_returns is None and self.equity_curve is not None:
            daily_returns = self.equity_curve.pct_change()
            stats['daily_volatility'] = daily_returns.std()
            stats['annualized_volatility'] = daily_returns.std() * np.sqrt(252)
            
        # Statistiques sur les trades
        if self.trades:
            profitable_trades = sum(1 for t in self.trades if t.realized_pnl > 0)
            total_trades = len(self.trades)
            stats['win_rate'] = profitable_trades / total_trades if total_trades > 0 else 0
            
        return stats

@dataclass
class FeatureSet:
    """Ensemble de caractéristiques pour l'entraînement."""
    # Champs obligatoires
    features: np.ndarray
    feature_names: List[str]
    timestamps: pd.DatetimeIndex
    symbol: str
    timeframe: str
    
    # Champs optionnels
    metadata: Optional[Dict[str, Any]] = None 

@dataclass
class Position:
    """Information sur une position de trading."""
    
    # Paramètres obligatoires
    entry_price: float
    entry_time: datetime
    size: float
    symbol: str
    
    # Paramètres optionnels (avec valeurs par défaut)
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    
    # Propriétés calculées
    @property
    def profit_pct(self) -> float:
        """Calcule le pourcentage de profit/perte."""
        if self.exit_price is None:
            return 0.0
        return ((self.exit_price - self.entry_price) / self.entry_price) * 100
    
    @property
    def profit_amount(self) -> float:
        """Calcule le montant du profit/perte."""
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.size
    
    @property
    def duration(self) -> float:
        """Calcule la durée de la position en heures."""
        if self.exit_time is None:
            return 0.0
        return (self.exit_time - self.entry_time).total_seconds() / 3600

@dataclass
class Signal(BaseData):
    """Signal généré par une stratégie de trading."""
    symbol: str
    timestamp: datetime
    side: str  # "buy" ou "sell"
    strength: float  # Force du signal (normalisée entre -1 et 1)
    source: str  # Source du signal (nom de la stratégie ou de l'indicateur)
    timeframe: Optional[str] = None
    price: Optional[float] = None
    confidence: Optional[float] = None  # Niveau de confiance (0 à 1)
    expiration: Optional[datetime] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceMetrics(BaseData):
    """Métriques de performance pour les stratégies de trading."""
    # Métriques de rentabilité
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    profit_factor: float
    
    # Métriques de risque
    max_drawdown: float
    drawdown_duration: int
    volatility: float
    value_at_risk: float
    
    # Métriques de trading
    total_trades: int
    win_rate: float
    profit_ratio: float
    avg_trade_duration: float
    largest_win: float
    largest_loss: float
    
    # Données supplémentaires
    start_date: datetime
    end_date: datetime
    strategy_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None