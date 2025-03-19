"""
Module de gestion des données avec support avancé pour les indicateurs techniques et la visualisation.
"""

from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
import joblib
from pathlib import Path

from trading.utils.logging.logger import TradingLogger
from trading.core.exceptions import DataError
from .data_validation import DataValidator, ValidationConfig, ValidationLevel

@dataclass
class DataConfig:
    """Configuration pour la gestion des données."""
    base_path: Path
    cache_enabled: bool = True
    use_parquet: bool = True
    compression: str = 'snappy'
    validation_level: ValidationLevel = ValidationLevel.FULL
    missing_data_strategy: str = 'interpolate'
    technical_indicators: List[str] = None
    visualization_backend: str = 'plotly'
    train_test_split: float = 0.8
    validation_split: float = 0.1
    random_state: int = 42
    futures_enabled: bool = False

class DataManager:
    """Gestionnaire de données avec support avancé pour les indicateurs techniques et la visualisation."""
    
    def __init__(self, config: Optional[Union[Dict, DataConfig]] = None):
        """
        Initialise le gestionnaire de données.
        
        Args:
            config: Configuration du gestionnaire
        """
        self.logger = TradingLogger()
        
        if isinstance(config, dict):
            self.config = self._create_config_from_dict(config)
        elif isinstance(config, DataConfig):
            self.config = config
        else:
            self.config = DataConfig(base_path=Path('data'))
            
        self.validator = DataValidator(ValidationConfig(
            level=self.config.validation_level
        ))
        
        self._setup_paths()
        self._initialize_technical_indicators()
        
    def _create_config_from_dict(self, config_dict: Dict) -> DataConfig:
        """Crée une configuration à partir d'un dictionnaire."""
        return DataConfig(
            base_path=Path(config_dict.get('base_path', 'data')),
            cache_enabled=config_dict.get('cache_enabled', True),
            use_parquet=config_dict.get('use_parquet', True),
            compression=config_dict.get('compression', 'snappy'),
            validation_level=ValidationLevel[config_dict.get('validation_level', 'FULL').upper()],
            missing_data_strategy=config_dict.get('missing_data_strategy', 'interpolate'),
            technical_indicators=config_dict.get('technical_indicators'),
            visualization_backend=config_dict.get('visualization_backend', 'plotly'),
            train_test_split=config_dict.get('train_test_split', 0.8),
            validation_split=config_dict.get('validation_split', 0.1),
            random_state=config_dict.get('random_state', 42),
            futures_enabled=config_dict.get('futures_enabled', False)
        )
        
    def _setup_paths(self) -> None:
        """Configure les chemins de données."""
        self.data_path = self.config.base_path
        self.cache_path = self.data_path / 'cache'
        self.processed_path = self.data_path / 'processed'
        self.raw_path = self.data_path / 'raw'
        
        for path in [self.data_path, self.cache_path, self.processed_path, self.raw_path]:
            path.mkdir(parents=True, exist_ok=True)
            
    def _initialize_technical_indicators(self) -> None:
        """Initialise les indicateurs techniques disponibles."""
        self.available_indicators = {
            # Tendance
            'SMA': self._calculate_sma,
            'EMA': self._calculate_ema,
            'MACD': self._calculate_macd,
            'ADX': self._calculate_adx,
            
            # Momentum
            'RSI': self._calculate_rsi,
            'Stochastic': self._calculate_stochastic,
            'ROC': self._calculate_roc,
            'MFI': self._calculate_mfi,
            
            # Volatilité
            'Bollinger': self._calculate_bollinger,
            'ATR': self._calculate_atr,
            'Volatility': self._calculate_volatility,
            
            # Volume
            'OBV': self._calculate_obv,
            'VWAP': self._calculate_vwap,
            'ADL': self._calculate_adl,
            
            # Personnalisés
            'Custom_Momentum': self._calculate_custom_momentum,
            'Volatility_Ratio': self._calculate_volatility_ratio
        }
        
        self.indicator_params = {
            'SMA': {'window': 20},
            'EMA': {'span': 20},
            'MACD': {'fast': 12, 'slow': 26, 'signal': 9},
            'RSI': {'window': 14},
            'Bollinger': {'window': 20, 'num_std': 2},
            'ATR': {'window': 14},
            'Stochastic': {'window': 14, 'smooth_window': 3},
            'VWAP': {'reset_period': 'D'}
        }
        
    def load_data(self, 
                 symbol: str,
                 start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None,
                 timeframe: str = '1h',
                 data_type: str = 'spot') -> pd.DataFrame:
        """
        Charge les données pour un symbole.
        
        Args:
            symbol: Symbole de trading
            start_date: Date de début
            end_date: Date de fin
            timeframe: Intervalle temporel
            data_type: Type de données ('spot', 'futures')
        
        Returns:
            DataFrame avec les données
        """
        try:
            # Construction du chemin du fichier
            filename = f"{symbol}_{timeframe}_{data_type}"
            if self.config.use_parquet:
                file_path = self.raw_path / f"{filename}.parquet"
            else:
                file_path = self.raw_path / f"{filename}.csv"
                
            # Vérification du cache
            if self.config.cache_enabled:
                cached_data = self._check_cache(filename)
                if cached_data is not None:
                    return self._filter_date_range(cached_data, start_date, end_date)
                    
            # Chargement des données
            if file_path.exists():
                if self.config.use_parquet:
                    data = pd.read_parquet(file_path)
                else:
                    data = pd.read_csv(file_path, parse_dates=['timestamp'])
                    data.set_index('timestamp', inplace=True)
                    
                # Validation des données
                validation_result = self.validator.validate_market_data(data, data_type)
                if not validation_result.is_valid:
                    self.logger.warning(
                        f"Problèmes de validation détectés: {validation_result.errors}"
                    )
                    
                # Gestion des données manquantes
                data = self._handle_missing_data(data)
                
                # Mise en cache
                if self.config.cache_enabled:
                    self._cache_data(filename, data)
                    
                return self._filter_date_range(data, start_date, end_date)
                
            else:
                raise FileNotFoundError(f"Fichier non trouvé: {file_path}")
                
        except Exception as e:
            raise DataError(f"Erreur lors du chargement des données: {str(e)}")
            
    def prepare_features(self, 
                        data: pd.DataFrame,
                        indicators: Optional[List[str]] = None,
                        custom_features: Optional[Dict[str, callable]] = None) -> pd.DataFrame:
        """
        Prépare les features pour l'entraînement.
        
        Args:
            data: DataFrame source
            indicators: Liste des indicateurs à calculer
            custom_features: Fonctions de features personnalisées
        
        Returns:
            DataFrame avec les features
        """
        try:
            features = data.copy()
            
            # Calcul des indicateurs techniques
            if indicators:
                for indicator in indicators:
                    if indicator in self.available_indicators:
                        params = self.indicator_params.get(indicator, {})
                        features = self.available_indicators[indicator](features, **params)
                    else:
                        self.logger.warning(f"Indicateur non disponible: {indicator}")
                        
            # Calcul des features personnalisées
            if custom_features:
                for name, func in custom_features.items():
                    try:
                        features[name] = func(features)
                    except Exception as e:
                        self.logger.error(f"Erreur lors du calcul de {name}: {str(e)}")
                        
            # Gestion des valeurs manquantes
            features = self._handle_missing_data(features)
            
            return features
            
        except Exception as e:
            raise DataError(f"Erreur lors de la préparation des features: {str(e)}")
            
    def split_data(self, 
                   data: pd.DataFrame,
                   target_column: str = 'close',
                   sequence_length: int = 60) -> Tuple[Dict[str, np.ndarray], ...]:
        """
        Divise les données en ensembles d'entraînement, validation et test.
        
        Args:
            data: DataFrame à diviser
            target_column: Colonne cible
            sequence_length: Longueur des séquences
            
        Returns:
            Tuple de dictionnaires contenant les données divisées
        """
        try:
            # Création des séquences
            sequences = self._create_sequences(data, sequence_length)
            
            # Séparation des features et de la cible
            X = sequences[:, :-1, :]
            y = sequences[:, -1, data.columns.get_loc(target_column)]
            
            # Division des données
            total_samples = len(X)
            train_size = int(total_samples * self.config.train_test_split)
            val_size = int(total_samples * self.config.validation_split)
            
            # Indices de division
            indices = np.arange(total_samples)
            np.random.seed(self.config.random_state)
            np.random.shuffle(indices)
            
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]
            
            # Création des ensembles
            train_data = {
                'X': X[train_indices],
                'y': y[train_indices]
            }
            
            val_data = {
                'X': X[val_indices],
                'y': y[val_indices]
            }
            
            test_data = {
                'X': X[test_indices],
                'y': y[test_indices]
            }
            
            return train_data, val_data, test_data
            
        except Exception as e:
            raise DataError(f"Erreur lors de la division des données: {str(e)}")
            
    def visualize_data(self,
                      data: pd.DataFrame,
                      plot_type: str = 'candlestick',
                      indicators: Optional[List[str]] = None,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> Any:
        """
        Visualise les données avec les indicateurs.
        
        Args:
            data: DataFrame à visualiser
            plot_type: Type de graphique
            indicators: Liste des indicateurs à afficher
            start_date: Date de début
            end_date: Date de fin
        
        Returns:
            Objet graphique
        """
        try:
            # Filtrage des données
            plot_data = self._filter_date_range(data, start_date, end_date)
            
            if self.config.visualization_backend == 'plotly':
                # Création du graphique principal
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.7, 0.3]
                )
                
                # Ajout du graphique principal
                if plot_type == 'candlestick':
                    fig.add_trace(
                        go.Candlestick(
                            x=plot_data.index,
                            open=plot_data['open'],
                            high=plot_data['high'],
                            low=plot_data['low'],
                            close=plot_data['close'],
                            name='OHLC'
                        ),
                        row=1, col=1
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=plot_data.index,
                            y=plot_data['close'],
                            mode='lines',
                            name='Prix'
                        ),
                        row=1, col=1
                    )
            
            # Ajout des indicateurs
                if indicators:
                    for indicator in indicators:
                        if indicator in self.available_indicators:
                            indicator_data = self.available_indicators[indicator](
                                plot_data,
                                **self.indicator_params.get(indicator, {})
                            )
                            if isinstance(indicator_data, pd.Series):
                                fig.add_trace(
                                    go.Scatter(
                                        x=plot_data.index,
                                        y=indicator_data,
                                        mode='lines',
                                        name=indicator
                                    ),
                                    row=1, col=1
                                )
                                
                # Ajout du volume
                fig.add_trace(
                    go.Bar(
                        x=plot_data.index,
                        y=plot_data['volume'],
                        name='Volume'
                    ),
                    row=2, col=1
                )
                
                # Mise en forme
                fig.update_layout(
                    title='Analyse technique',
                    xaxis_title='Date',
                    yaxis_title='Prix',
                    yaxis2_title='Volume',
                    showlegend=True
                )
                
                return fig
                
            else:
                raise ValueError(f"Backend de visualisation non supporté: {self.config.visualization_backend}")
                
        except Exception as e:
            raise DataError(f"Erreur lors de la visualisation: {str(e)}")
            
    def _handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Gère les données manquantes selon la stratégie configurée."""
        if data.isnull().sum().sum() == 0:
            return data
            
        strategy = self.config.missing_data_strategy
        if strategy == 'drop':
            return data.dropna()
        elif strategy == 'interpolate':
            return data.interpolate(method='time')
        elif strategy == 'forward':
            return data.fillna(method='ffill').fillna(method='bfill')
        elif strategy == 'mean':
            return data.fillna(data.mean())
        else:
            raise ValueError(f"Stratégie de gestion des données manquantes invalide: {strategy}")
            
    def _create_sequences(self, data: pd.DataFrame, sequence_length: int) -> np.ndarray:
        """Crée des séquences pour l'apprentissage."""
        sequences = []
        for i in range(len(data) - sequence_length):
            sequence = data.iloc[i:i + sequence_length + 1].values
            sequences.append(sequence)
        return np.array(sequences)
        
    def _check_cache(self, key: str) -> Optional[pd.DataFrame]:
        """Vérifie si les données sont en cache."""
        if not self.config.cache_enabled:
            return None
            
        cache_file = self.cache_path / f"{key}_cache.joblib"
        if cache_file.exists():
            try:
                return joblib.load(cache_file)
            except Exception as e:
                self.logger.warning(f"Erreur lors du chargement du cache: {str(e)}")
                return None
        return None
        
    def _cache_data(self, key: str, data: pd.DataFrame) -> None:
        """Met les données en cache."""
        if not self.config.cache_enabled:
            return
            
        cache_file = self.cache_path / f"{key}_cache.joblib"
        try:
            joblib.dump(data, cache_file)
        except Exception as e:
            self.logger.warning(f"Erreur lors de la mise en cache: {str(e)}")
            
    def _filter_date_range(self,
                          data: pd.DataFrame,
                          start_date: Optional[datetime],
                          end_date: Optional[datetime]) -> pd.DataFrame:
        """Filtre les données selon la plage de dates."""
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        return data
        
    # Méthodes de calcul des indicateurs techniques
    def _calculate_sma(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calcule la moyenne mobile simple."""
        return data['close'].rolling(window=window).mean()
        
    def _calculate_ema(self, data: pd.DataFrame, span: int = 20) -> pd.Series:
        """Calcule la moyenne mobile exponentielle."""
        return data['close'].ewm(span=span, adjust=False).mean()
        
    def _calculate_macd(self,
                       data: pd.DataFrame,
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
        
    def _calculate_rsi(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calcule le RSI."""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _calculate_bollinger(self,
                           data: pd.DataFrame,
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
        
    def _calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calcule l'ATR."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=window).mean()
        
    def _calculate_stochastic(self,
                            data: pd.DataFrame,
                            window: int = 14,
                            smooth_window: int = 3) -> pd.DataFrame:
        """Calcule l'oscillateur stochastique."""
        low_min = data['low'].rolling(window=window).min()
        high_max = data['high'].rolling(window=window).max()
        k = 100 * (data['close'] - low_min) / (high_max - low_min)
        d = k.rolling(window=smooth_window).mean()
        return pd.DataFrame({
            'Stoch_K': k,
            'Stoch_D': d
        })
        
    def _calculate_roc(self, data: pd.DataFrame, window: int = 12) -> pd.Series:
        """Calcule le taux de variation."""
        return ((data['close'] - data['close'].shift(window)) / 
                data['close'].shift(window) * 100)
                
    def _calculate_mfi(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calcule le Money Flow Index."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']
        
        positive_flow = pd.Series(0, index=data.index)
        negative_flow = pd.Series(0, index=data.index)
        
        # Calcul des flux positifs et négatifs
        price_diff = typical_price.diff()
        positive_flow[price_diff > 0] = money_flow[price_diff > 0]
        negative_flow[price_diff < 0] = money_flow[price_diff < 0]
        
        positive_mf = positive_flow.rolling(window=window).sum()
        negative_mf = negative_flow.rolling(window=window).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
        
    def _calculate_volatility(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calcule la volatilité."""
        log_returns = np.log(data['close'] / data['close'].shift(1))
        return log_returns.rolling(window=window).std() * np.sqrt(252)
        
    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calcule l'On-Balance Volume."""
        obv = pd.Series(0, index=data.index)
        obv.iloc[0] = data['volume'].iloc[0]
        
        price_diff = data['close'].diff()
        obv[price_diff > 0] = data['volume'][price_diff > 0]
        obv[price_diff < 0] = -data['volume'][price_diff < 0]
        
        return obv.cumsum()
        
    def _calculate_vwap(self, data: pd.DataFrame, reset_period: str = 'D') -> pd.Series:
        """Calcule le VWAP."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        dollar_volume = typical_price * data['volume']
        
        if reset_period:
            cumsum_dv = dollar_volume.groupby(pd.Grouper(freq=reset_period)).cumsum()
            cumsum_vol = data['volume'].groupby(pd.Grouper(freq=reset_period)).cumsum()
        else:
            cumsum_dv = dollar_volume.cumsum()
            cumsum_vol = data['volume'].cumsum()
            
        return cumsum_dv / cumsum_vol
        
    def _calculate_adl(self, data: pd.DataFrame) -> pd.Series:
        """Calcule l'Accumulation/Distribution Line."""
        clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        adl = clv * data['volume']
        return adl.cumsum()
        
    def _calculate_adx(self, data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Calcule l'ADX."""
        # Calcul du True Range
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        # Calcul du Directional Movement
        up_move = data['high'] - data['high'].shift()
        down_move = data['low'].shift() - data['low']
        
        pos_dm = pd.Series(0, index=data.index)
        neg_dm = pd.Series(0, index=data.index)
        
        pos_dm[(up_move > down_move) & (up_move > 0)] = up_move
        neg_dm[(down_move > up_move) & (down_move > 0)] = down_move
        
        # Calcul des indicateurs directionnels
        tr_14 = true_range.rolling(window=window).sum()
        pos_di_14 = 100 * (pos_dm.rolling(window=window).sum() / tr_14)
        neg_di_14 = 100 * (neg_dm.rolling(window=window).sum() / tr_14)
        
        # Calcul de l'ADX
        dx = 100 * np.abs(pos_di_14 - neg_di_14) / (pos_di_14 + neg_di_14)
        adx = dx.rolling(window=window).mean()
        
        return pd.DataFrame({
            'ADX': adx,
            '+DI': pos_di_14,
            '-DI': neg_di_14
        })
        
    def _calculate_custom_momentum(self, data: pd.DataFrame, window: int = 10) -> pd.Series:
        """Calcule un indicateur de momentum personnalisé."""
        close_diff = data['close'].diff(window)
        volume_diff = data['volume'].diff(window)
        return close_diff * volume_diff
        
    def _calculate_volatility_ratio(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calcule un ratio de volatilité personnalisé."""
        returns = np.log(data['close'] / data['close'].shift(1))
        volatility = returns.rolling(window=window).std()
        avg_volume = data['volume'].rolling(window=window).mean()
        return volatility * avg_volume 