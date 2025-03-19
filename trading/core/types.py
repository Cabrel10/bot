"""Types de base pour le système de trading.

Ce module définit les types de base et les configurations pour le système de trading,
incluant :
- Types de données fondamentaux
- Configurations de traitement
- Résultats de prétraitement
- Validation et sérialisation
- Gestion des métadonnées
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Any, TypeVar, Protocol, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import json
from pathlib import Path
from abc import ABC, abstractmethod

# Type générique pour les données
T = TypeVar('T')

class Validatable(Protocol):
    """Protocol pour les objets validables."""
    def validate(self) -> Tuple[bool, List[str]]:
        """Valide l'objet et retourne (is_valid, errors)."""
        ...

class Serializable(Protocol):
    """Protocol pour les objets sérialisables."""
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'objet en dictionnaire."""
        ...
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Serializable':
        """Crée une instance depuis un dictionnaire."""
        ...

@dataclass
class BaseData:
    """Classe de base pour les données."""
    metadata: Dict[str, Any] = field(default_factory=dict, kw_only=True)
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Ajoute une métadonnée."""
        self.metadata[key] = value
        
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Récupère une métadonnée."""
        return self.metadata.get(key, default)
        
    def save(self, path: Union[str, Path]) -> None:
        """Sauvegarde les données."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(asdict(self), f, default=self._json_serialize)
            
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BaseData':
        """Charge les données."""
        path = Path(path)
        
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
        
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
        return str(obj)

@dataclass
class PreprocessingResult(BaseData, Validatable, Serializable):
    """Résultat du prétraitement des données."""
    # Champs obligatoires (sans valeur par défaut)
    raw_data: pd.DataFrame
    processed_data: pd.DataFrame
    features: List[str]
    
    # Champs optionnels (avec valeur par défaut)
    timestamps: Optional[pd.DatetimeIndex] = None
    feature_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validation après initialisation."""
        if self.timestamps is None:
            self.timestamps = self.raw_data.index
            
        # Initialisation des métadonnées des features
        for feature in self.features:
            if feature not in self.feature_metadata:
                self.feature_metadata[feature] = {
                    'type': str(self.processed_data[feature].dtype),
                    'missing_ratio': self._calculate_missing_ratio(feature),
                    'statistics': self._calculate_feature_statistics(feature)
                }

    def validate(self) -> Tuple[bool, List[str]]:
        """Valide l'intégrité des données."""
        errors = []
        
        try:
            # Vérification des dimensions
            if len(self.raw_data) != len(self.processed_data):
                errors.append(
                    "Dimensions incohérentes entre données brutes et traitées"
                )
                
            # Vérification des features
            missing_features = [
                f for f in self.features
                if f not in self.processed_data.columns
            ]
            if missing_features:
                errors.append(
                    f"Features manquantes: {missing_features}"
                )
                
            # Vérification des timestamps
            if not self.timestamps.is_monotonic_increasing:
                errors.append("Timestamps non monotones")
                
            # Vérification des valeurs manquantes
            if self.processed_data.isnull().any().any():
                errors.append("Valeurs manquantes dans les données traitées")
                
        except Exception as e:
            errors.append(f"Erreur de validation: {str(e)}")
            
        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            'raw_data': self.raw_data.to_dict('records'),
            'processed_data': self.processed_data.to_dict('records'),
            'features': self.features,
            'timestamps': [t.isoformat() for t in self.timestamps],
            'feature_metadata': self.feature_metadata,
            'statistics': self.statistics,
            'metadata': self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PreprocessingResult':
        """Crée une instance depuis un dictionnaire."""
        # Conversion des DataFrames
        raw_data = pd.DataFrame(data['raw_data'])
        processed_data = pd.DataFrame(data['processed_data'])
        
        # Conversion des timestamps
        timestamps = pd.DatetimeIndex([
            pd.to_datetime(t) for t in data['timestamps']
        ])
        
        return cls(
            raw_data=raw_data,
            processed_data=processed_data,
            features=data['features'],
            timestamps=timestamps,
            feature_metadata=data.get('feature_metadata', {}),
            statistics=data.get('statistics', {}),
            metadata=data.get('metadata', {})
        )
        
    def _calculate_missing_ratio(self, feature: str) -> float:
        """Calcule le ratio de valeurs manquantes."""
        return (
            self.processed_data[feature].isnull().sum() /
            len(self.processed_data)
        )
        
    def _calculate_feature_statistics(self, feature: str) -> Dict[str, float]:
        """Calcule les statistiques d'une feature."""
        series = self.processed_data[feature]
        return {
            'mean': series.mean(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'median': series.median(),
            'skew': series.skew(),
            'kurtosis': series.kurtosis()
        }
        
    def get_feature_importance(self) -> pd.Series:
        """Calcule l'importance des features."""
        # À implémenter selon la méthode choisie
        pass
        
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Calcule la matrice de corrélation des features."""
        return self.processed_data[self.features].corr()

@dataclass
class ProcessingConfig(BaseData, Validatable, Serializable):
    """Configuration pour le traitement des données."""
    # Paramètres généraux
    feature_engineering: bool = True
    normalization: bool = True
    handle_missing: bool = True
    handle_outliers: bool = True
    technical_indicators: bool = True
    
    # Paramètres de normalisation
    normalization_method: str = 'standard'  # 'standard', 'minmax', 'robust'
    normalization_params: Dict[str, Any] = field(default_factory=dict)
    
    # Gestion des valeurs manquantes
    missing_threshold: float = 0.1
    missing_strategy: str = 'interpolate'  # 'interpolate', 'ffill', 'bfill', 'drop'
    interpolation_method: str = 'linear'
    
    # Gestion des outliers
    outlier_method: str = 'zscore'  # 'zscore', 'iqr', 'isolation_forest'
    outlier_threshold: float = 3.0
    
    # Indicateurs techniques
    indicators: List[str] = field(default_factory=list)
    indicator_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Fenêtres temporelles
    window_sizes: List[int] = field(default_factory=lambda: [5, 10, 20])
    
    def __post_init__(self):
        """Validation après initialisation."""
        # Validation des paramètres
        if self.normalization_method not in ['standard', 'minmax', 'robust']:
            raise ValueError(f"Méthode de normalisation invalide: {self.normalization_method}")
            
        if self.missing_strategy not in ['interpolate', 'ffill', 'bfill', 'drop']:
            raise ValueError(f"Stratégie de gestion des valeurs manquantes invalide: {self.missing_strategy}")
            
        if self.outlier_method not in ['zscore', 'iqr', 'isolation_forest']:
            raise ValueError(f"Méthode de détection des outliers invalide: {self.outlier_method}")
            
        if not 0 <= self.missing_threshold <= 1:
            raise ValueError("Le seuil de valeurs manquantes doit être entre 0 et 1")
            
        if self.outlier_threshold <= 0:
            raise ValueError("Le seuil des outliers doit être positif")

    def validate(self) -> Tuple[bool, List[str]]:
        """Valide la configuration."""
        errors = []
        
        try:
            # Vérification des fenêtres temporelles
            if any(w <= 0 for w in self.window_sizes):
                errors.append("Les tailles de fenêtre doivent être positives")
                
            # Vérification des paramètres des indicateurs
            for indicator in self.indicators:
                if indicator not in self.indicator_params:
                    errors.append(f"Paramètres manquants pour l'indicateur: {indicator}")
                    
            # Vérification des paramètres de normalisation
            if self.normalization:
                required_params = {
                    'standard': ['with_mean', 'with_std'],
                    'minmax': ['feature_range'],
                    'robust': ['quantile_range']
                }
                
                required = required_params.get(self.normalization_method, [])
                missing_params = [
                    p for p in required
                    if p not in self.normalization_params
                ]
                
                if missing_params:
                    errors.append(
                        f"Paramètres de normalisation manquants: {missing_params}"
                    )
                    
        except Exception as e:
            errors.append(f"Erreur de validation: {str(e)}")
            
        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            'feature_engineering': self.feature_engineering,
            'normalization': self.normalization,
            'handle_missing': self.handle_missing,
            'handle_outliers': self.handle_outliers,
            'technical_indicators': self.technical_indicators,
            'normalization_method': self.normalization_method,
            'normalization_params': self.normalization_params,
            'missing_threshold': self.missing_threshold,
            'missing_strategy': self.missing_strategy,
            'interpolation_method': self.interpolation_method,
            'outlier_method': self.outlier_method,
            'outlier_threshold': self.outlier_threshold,
            'indicators': self.indicators,
            'indicator_params': self.indicator_params,
            'window_sizes': self.window_sizes,
            'metadata': self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingConfig':
        """Crée une instance depuis un dictionnaire."""
        return cls(
            feature_engineering=data.get('feature_engineering', True),
            normalization=data.get('normalization', True),
            handle_missing=data.get('handle_missing', True),
            handle_outliers=data.get('handle_outliers', True),
            technical_indicators=data.get('technical_indicators', True),
            normalization_method=data.get('normalization_method', 'standard'),
            normalization_params=data.get('normalization_params', {}),
            missing_threshold=data.get('missing_threshold', 0.1),
            missing_strategy=data.get('missing_strategy', 'interpolate'),
            interpolation_method=data.get('interpolation_method', 'linear'),
            outlier_method=data.get('outlier_method', 'zscore'),
            outlier_threshold=data.get('outlier_threshold', 3.0),
            indicators=data.get('indicators', []),
            indicator_params=data.get('indicator_params', {}),
            window_sizes=data.get('window_sizes', [5, 10, 20]),
            metadata=data.get('metadata', {})
        )

# Constants for order types
ORDER_TYPE_MARKET = "market"
ORDER_TYPE_LIMIT = "limit"
ORDER_TYPE_STOP = "stop"
ORDER_TYPE_STOP_LIMIT = "stop_limit"

# Constants for order sides
ORDER_SIDE_BUY = "buy"
ORDER_SIDE_SELL = "sell"

# Constants for timeframes
TIMEFRAME_1M = "1m"
TIMEFRAME_5M = "5m"
TIMEFRAME_15M = "15m"
TIMEFRAME_30M = "30m"
TIMEFRAME_1H = "1h"
TIMEFRAME_2H = "2h"
TIMEFRAME_4H = "4h"
TIMEFRAME_6H = "6h"
TIMEFRAME_8H = "8h"
TIMEFRAME_12H = "12h"
TIMEFRAME_1D = "1d"
TIMEFRAME_3D = "3d"
TIMEFRAME_1W = "1w"
TIMEFRAME_1MONTH = "1M"
TIMEFRAME_1M_MONTH = "1M"  # Alias pour 1 mois 