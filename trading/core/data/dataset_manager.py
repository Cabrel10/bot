"""
Module de gestion des datasets avec support avancé pour les indicateurs techniques,
la visualisation et le traitement des données futures.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import yaml
from abc import ABC, abstractmethod
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from trading.core.exceptions import DatasetError
from trading.utils.logging import setup_logger

@dataclass
class DatasetConfig:
    """Configuration pour la gestion des datasets."""
    base_path: Path
    cache_enabled: bool = True
    use_parquet: bool = True
    compression: str = 'snappy'
    outlier_threshold: float = 3.0
    missing_threshold: float = 0.1
    visualization_backend: str = 'plotly'
    futures_enabled: bool = False
    save_formats: List[str] = None

class TechnicalIndicator(ABC):
    """Classe de base abstraite pour les indicateurs techniques."""
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcule l'indicateur technique."""
        pass

class MomentumIndicators(TechnicalIndicator):
    """Classe pour les indicateurs de momentum."""
    
    def calculate_rsi(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calcule le RSI."""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def calculate_macd(self, data: pd.DataFrame, 
                      fast: int = 12, 
                      slow: int = 26, 
                      signal: int = 9) -> pd.DataFrame:
        """Calcule le MACD."""
        exp1 = data['close'].ewm(span=fast, adjust=False).mean()
        exp2 = data['close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return pd.DataFrame({
            'MACD': macd,
            'Signal': signal_line,
            'Histogram': macd - signal_line
        })
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcule tous les indicateurs de momentum."""
        result = data.copy()
        result['RSI'] = self.calculate_rsi(data)
        macd_data = self.calculate_macd(data)
        result = pd.concat([result, macd_data], axis=1)
        return result

class TrendIndicators(TechnicalIndicator):
    """Classe pour les indicateurs de tendance."""
    
    def calculate_sma(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calcule la moyenne mobile simple."""
        return data['close'].rolling(window=window).mean()
        
    def calculate_ema(self, data: pd.DataFrame, span: int = 20) -> pd.Series:
        """Calcule la moyenne mobile exponentielle."""
        return data['close'].ewm(span=span, adjust=False).mean()
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcule tous les indicateurs de tendance."""
        result = data.copy()
        result['SMA_20'] = self.calculate_sma(data, 20)
        result['SMA_50'] = self.calculate_sma(data, 50)
        result['EMA_20'] = self.calculate_ema(data, 20)
        return result

class VolatilityIndicators(TechnicalIndicator):
    """Classe pour les indicateurs de volatilité."""
    
    def calculate_bollinger(self, data: pd.DataFrame, 
                          window: int = 20, 
                          num_std: float = 2) -> pd.DataFrame:
        """Calcule les bandes de Bollinger."""
        sma = data['close'].rolling(window=window).mean()
        std = data['close'].rolling(window=window).std()
        return pd.DataFrame({
            'BB_middle': sma,
            'BB_upper': sma + (std * num_std),
            'BB_lower': sma - (std * num_std)
        })
        
    def calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calcule l'ATR."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=window).mean()
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcule tous les indicateurs de volatilité."""
        result = data.copy()
        bollinger = self.calculate_bollinger(data)
        result = pd.concat([result, bollinger], axis=1)
        result['ATR'] = self.calculate_atr(data)
        return result

class VolumeIndicators(TechnicalIndicator):
    """Classe pour les indicateurs de volume."""
    
    def calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calcule l'On-Balance Volume."""
        obv = pd.Series(0, index=data.index)
        obv.iloc[0] = data['volume'].iloc[0]
        
        price_diff = data['close'].diff()
        obv[price_diff > 0] = data['volume'][price_diff > 0]
        obv[price_diff < 0] = -data['volume'][price_diff < 0]
        
        return obv.cumsum()
        
    def calculate_vwap(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calcule le VWAP."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        dollar_volume = typical_price * data['volume']
        cumsum_dv = dollar_volume.rolling(window=window).sum()
        cumsum_vol = data['volume'].rolling(window=window).sum()
        return cumsum_dv / cumsum_vol
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcule tous les indicateurs de volume."""
        result = data.copy()
        result['OBV'] = self.calculate_obv(data)
        result['VWAP'] = self.calculate_vwap(data)
        return result

class FuturesIndicators(TechnicalIndicator):
    """Classe pour les indicateurs spécifiques aux futures."""
    
    def calculate_basis(self, data: pd.DataFrame) -> pd.Series:
        """Calcule le basis (différence entre prix spot et futures)."""
        if 'index_price' not in data.columns:
            raise ValueError("La colonne 'index_price' est requise pour le calcul du basis")
        return data['close'] - data['index_price']
        
    def calculate_funding_rate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcule les features liées au funding rate."""
        if 'funding_rate' not in data.columns:
            raise ValueError("La colonne 'funding_rate' est requise")
            
        result = pd.DataFrame()
        result['funding_rate_change'] = data['funding_rate'].diff()
        result['funding_rate_ma'] = data['funding_rate'].rolling(window=20).mean()
        return result
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcule tous les indicateurs futures."""
        result = data.copy()
        try:
            result['basis'] = self.calculate_basis(data)
            funding_features = self.calculate_funding_rate_features(data)
            result = pd.concat([result, funding_features], axis=1)
        except ValueError as e:
            logging.warning(f"Impossible de calculer certains indicateurs futures: {str(e)}")
        return result

class DatasetManager:
    """Gestionnaire de datasets avec support avancé pour les indicateurs techniques."""
    
    def __init__(self, config: Optional[Union[Dict, DatasetConfig]] = None):
        """
        Initialise le gestionnaire de datasets.
        
        Args:
            config: Configuration du gestionnaire
        """
        self.logger = setup_logger(__name__)
        
        if isinstance(config, dict):
            self.config = self._create_config_from_dict(config)
        elif isinstance(config, DatasetConfig):
            self.config = config
        else:
            self.config = DatasetConfig(base_path=Path('data'))
            
        self._setup_paths()
        self._initialize_indicators()
        self._initialize_scalers()
        
    def _create_config_from_dict(self, config_dict: Dict) -> DatasetConfig:
        """Crée une configuration à partir d'un dictionnaire."""
        return DatasetConfig(
            base_path=Path(config_dict.get('base_path', 'data')),
            cache_enabled=config_dict.get('cache_enabled', True),
            use_parquet=config_dict.get('use_parquet', True),
            compression=config_dict.get('compression', 'snappy'),
            outlier_threshold=config_dict.get('outlier_threshold', 3.0),
            missing_threshold=config_dict.get('missing_threshold', 0.1),
            visualization_backend=config_dict.get('visualization_backend', 'plotly'),
            futures_enabled=config_dict.get('futures_enabled', False),
            save_formats=config_dict.get('save_formats', ['parquet'])
        )
        
    def _setup_paths(self) -> None:
        """Configure les chemins de données."""
        self.data_path = self.config.base_path
        self.cache_path = self.data_path / 'cache'
        self.processed_path = self.data_path / 'processed'
        self.plots_path = self.data_path / 'plots'
        
        for path in [self.data_path, self.cache_path, 
                    self.processed_path, self.plots_path]:
            path.mkdir(parents=True, exist_ok=True)
            
    def _initialize_indicators(self) -> None:
        """Initialise les indicateurs techniques."""
        self.indicators = {
            'momentum': MomentumIndicators(),
            'trend': TrendIndicators(),
            'volatility': VolatilityIndicators(),
            'volume': VolumeIndicators()
        }
        
        if self.config.futures_enabled:
            self.indicators['futures'] = FuturesIndicators()
            
    def _initialize_scalers(self) -> None:
        """Initialise les scalers disponibles."""
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(feature_range=(-1, 1)),
            'robust': RobustScaler()
        }
        
    def validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Valide et nettoie les données.
        
        Args:
            data: DataFrame à valider
            
        Returns:
            DataFrame validé
        
        Raises:
            DatasetError: Si les données ne sont pas valides
        """
        try:
            # Vérification des colonnes requises
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_cols):
                raise DatasetError(f"Colonnes manquantes. Requises: {required_cols}")
                
            # Vérification des données manquantes
            missing_ratio = data[required_cols].isnull().sum() / len(data)
            if (missing_ratio > self.config.missing_threshold).any():
                problematic_cols = missing_ratio[
                    missing_ratio > self.config.missing_threshold
                ].index.tolist()
                raise DatasetError(
                    f"Trop de valeurs manquantes dans: {problematic_cols}"
                )
                
            # Vérification des valeurs négatives
            for col in required_cols:
                if (data[col] < 0).any():
                    raise DatasetError(f"Valeurs négatives détectées dans {col}")
                    
            # Vérification de la cohérence OHLC
            invalid_ohlc = (
                (data['high'] < data['low']) |
                (data['high'] < data['open']) |
                (data['high'] < data['close']) |
                (data['low'] > data['open']) |
                (data['low'] > data['close'])
            )
            
            if invalid_ohlc.any():
                raise DatasetError("Incohérence dans les données OHLC")
                
            return data
            
        except Exception as e:
            raise DatasetError(f"Erreur lors de la validation: {str(e)}")
            
    def handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Gère les valeurs aberrantes.
        
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
                
                # Remplacement des outliers par les bornes
                result.loc[result[col] < lower_bound, col] = lower_bound
                result.loc[result[col] > upper_bound, col] = upper_bound
                
            return result
            
        except Exception as e:
            raise DatasetError(f"Erreur lors du traitement des outliers: {str(e)}")
            
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute les indicateurs techniques au DataFrame.
        
        Args:
            data: DataFrame source
            
        Returns:
            DataFrame avec indicateurs
        """
        try:
            result = data.copy()
            
            for indicator_type, indicator in self.indicators.items():
                try:
                    result = indicator.calculate(result)
                except Exception as e:
                    self.logger.warning(
                        f"Erreur lors du calcul des indicateurs {indicator_type}: {str(e)}"
                    )
                    
            return result
            
        except Exception as e:
            raise DatasetError(f"Erreur lors du calcul des indicateurs: {str(e)}")
            
    def create_visualization(self,
                           data: pd.DataFrame,
                           plot_type: str = 'candlestick',
                           **kwargs) -> Any:
        """
        Crée une visualisation personnalisée.
        
        Args:
            data: DataFrame à visualiser
            plot_type: Type de graphique
            **kwargs: Paramètres de personnalisation
            
        Returns:
            Objet graphique
        """
        try:
            if self.config.visualization_backend != 'plotly':
                raise ValueError(f"Backend non supporté: {self.config.visualization_backend}")
                
            if plot_type == 'candlestick':
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.7, 0.3]
                )
                
                fig.add_trace(
                    go.Candlestick(
                        x=data.index,
                        open=data['open'],
                        high=data['high'],
                        low=data['low'],
                        close=data['close'],
                        name='OHLC'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data['volume'],
                        name='Volume'
                    ),
                    row=2, col=1
                )
                
            elif plot_type == 'line':
                fig = go.Figure()
                for col in kwargs.get('columns', ['close']):
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data[col],
                            mode='lines',
                            name=col
                        )
                    )
                    
            else:
                raise ValueError(f"Type de graphique non supporté: {plot_type}")
                
            # Personnalisation du graphique
            fig.update_layout(
                title=kwargs.get('title', 'Analyse technique'),
                xaxis_title=kwargs.get('xaxis_title', 'Date'),
                yaxis_title=kwargs.get('yaxis_title', 'Prix'),
                template=kwargs.get('template', 'plotly_dark')
            )
            
            return fig
            
        except Exception as e:
            raise DatasetError(f"Erreur lors de la création du graphique: {str(e)}")
            
    def save_dataset(self, 
                    data: pd.DataFrame,
                    symbol: str,
                    timeframe: str) -> None:
        """
        Sauvegarde le dataset dans les formats configurés.
        
        Args:
            data: DataFrame à sauvegarder
            symbol: Symbole de trading
            timeframe: Timeframe des données
        """
        try:
            base_filename = f"{symbol.replace('/', '_')}_{timeframe}"
            
            for format in self.config.save_formats:
                if format == 'parquet':
                    filepath = self.processed_path / f"{base_filename}.parquet"
                    data.to_parquet(
                        filepath,
                        compression=self.config.compression
                    )
                elif format == 'csv':
                    filepath = self.processed_path / f"{base_filename}.csv"
                    data.to_csv(filepath)
                elif format == 'feather':
                    filepath = self.processed_path / f"{base_filename}.feather"
                    data.to_feather(filepath)
                else:
                    self.logger.warning(f"Format non supporté: {format}")
                    
            self.logger.info(f"Dataset sauvegardé dans {self.processed_path}")
            
        except Exception as e:
            raise DatasetError(f"Erreur lors de la sauvegarde: {str(e)}")
            
    def process_data(self,
                    data: pd.DataFrame,
                    symbol: str,
                    timeframe: str,
                    handle_outliers: bool = True) -> pd.DataFrame:
        """
        Traite les données complètes.
        
        Args:
            data: DataFrame à traiter
            symbol: Symbole de trading
            timeframe: Timeframe des données
            handle_outliers: Si True, gère les outliers
            
        Returns:
            DataFrame traité
        """
        try:
            # Validation
            validated_data = self.validate_data(data)
            
            # Gestion des outliers
            if handle_outliers:
                validated_data = self.handle_outliers(validated_data)
                
            # Ajout des indicateurs
            processed_data = self.add_technical_indicators(validated_data)
            
            # Sauvegarde
            self.save_dataset(processed_data, symbol, timeframe)
            
            return processed_data
            
        except Exception as e:
            raise DatasetError(f"Erreur lors du traitement: {str(e)}")
            
    def save_scaler(self, scaler: Any, name: str) -> None:
        """
        Sauvegarde un scaler.
        
        Args:
            scaler: Scaler à sauvegarder
            name: Nom du scaler
        """
        try:
            filepath = self.cache_path / f"scaler_{name}.joblib"
            joblib.dump(scaler, filepath)
            self.logger.info(f"Scaler sauvegardé: {filepath}")
            
        except Exception as e:
            raise DatasetError(f"Erreur lors de la sauvegarde du scaler: {str(e)}")
            
    def load_scaler(self, name: str) -> Any:
        """
        Charge un scaler.
        
        Args:
            name: Nom du scaler
            
        Returns:
            Scaler chargé
        """
        try:
            filepath = self.cache_path / f"scaler_{name}.joblib"
            if not filepath.exists():
                raise FileNotFoundError(f"Scaler non trouvé: {filepath}")
                
            scaler = joblib.load(filepath)
            self.logger.info(f"Scaler chargé: {filepath}")
        return scaler 
            
        except Exception as e:
            raise DatasetError(f"Erreur lors du chargement du scaler: {str(e)}")
