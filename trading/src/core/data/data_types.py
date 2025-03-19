"""
Module définissant les types de données utilisés dans le système de trading.
"""

from typing import TypedDict, List, Dict, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Types de base pour les données OHLCV
class OHLCVData(TypedDict):
    """Structure de données OHLCV."""
    timestamp: List[datetime]
    open: List[float]
    high: List[float]
    low: List[float]
    close: List[float]
    volume: List[float]

class TradeData(TypedDict):
    """Structure de données pour les trades."""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' ou 'sell'
    price: float
    amount: float
    cost: float
    fee: Optional[Dict[str, float]]

class OrderBookData(TypedDict):
    """Structure de données pour le carnet d'ordres."""
    timestamp: datetime
    bids: List[List[float]]  # [[prix, quantité], ...]
    asks: List[List[float]]  # [[prix, quantité], ...]
    symbol: str

class TechnicalIndicators(TypedDict):
    """Structure pour les indicateurs techniques."""
    rsi: List[float]
    macd: List[float]
    macd_signal: List[float]
    macd_hist: List[float]
    ema_9: List[float]
    ema_21: List[float]
    bollinger_upper: List[float]
    bollinger_middle: List[float]
    bollinger_lower: List[float]

@dataclass
class ProcessedData:
    """Données traitées avec indicateurs et normalisation."""
    raw_data: OHLCVData
    indicators: Dict[str, List[float]]
    normalized_data: pd.DataFrame
    metadata: Optional[Dict] = None

@dataclass
class MarketState:
    """État du marché à un instant T."""
    timestamp: datetime
    symbol: str
    current_price: float
    bid_price: float
    ask_price: float
    spread: float
    volume_24h: float
    price_change_24h: float
    high_24h: float
    low_24h: float

@dataclass
class TrainingData:
    """Données préparées pour l'entraînement."""
    X: np.ndarray  # Features
    y: np.ndarray  # Labels/Targets
    feature_names: List[str]
    timestamps: np.ndarray
    symbol: str
    timeframe: str

@dataclass
class BacktestResult:
    """Résultats d'un backtest."""
    trades: List[TradeData]
    pnl: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    start_date: datetime
    end_date: datetime
    symbol: str
    strategy_name: str

@dataclass
class ModelPrediction:
    """Prédiction d'un modèle."""
    timestamp: datetime
    symbol: str
    prediction_type: str  # 'direction', 'price', 'probability'
    value: Union[float, int, str]
    confidence: float
    horizon: str  # '1h', '4h', '1d', etc.
    model_name: str

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
    source: str
    version: str

class ValidationResult(TypedDict):
    """Résultat de validation des données."""
    is_valid: bool
    missing_values: Dict[str, int]
    outliers: Dict[str, List[int]]
    data_quality_score: float
    errors: List[str]

@dataclass
class FeatureSet:
    """Ensemble de features pour un modèle."""
    price_features: List[str]
    volume_features: List[str]
    technical_features: List[str]
    custom_features: List[str]
    target_feature: str
    feature_importance: Optional[Dict[str, float]] = None

@dataclass
class DataStats:
    """Statistiques descriptives des données."""
    count: int
    mean: Dict[str, float]
    std: Dict[str, float]
    min: Dict[str, float]
    max: Dict[str, float]
    missing_ratio: Dict[str, float]
    correlation_matrix: pd.DataFrame

class DataTransformConfig(TypedDict):
    """Configuration pour la transformation des données."""
    normalization_method: str  # 'min_max', 'standard', etc.
    feature_engineering: Dict[str, bool]
    sequence_length: int
    target_type: str
    split_ratios: Dict[str, float]
    random_seed: int

@dataclass
class MarketData:
    """Données de marché de base."""
    timestamp: datetime
    symbol: str
    exchange: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: Optional[int] = None
    vwap: Optional[float] = None

@dataclass
class FuturesData(MarketData):
    """Données spécifiques aux contrats futures."""
    funding_rate: float
    mark_price: float
    index_price: float
    open_interest: float
    next_funding_time: datetime
    predicted_funding_rate: Optional[float] = None
    basis: Optional[float] = None
    basis_percent: Optional[float] = None

@dataclass
class OrderBookData:
    """Données du carnet d'ordres."""
    timestamp: datetime
    symbol: str
    exchange: str
    bids: List[List[float]]  # [[prix, quantité], ...]
    asks: List[List[float]]  # [[prix, quantité], ...]
    last_update_id: Optional[int] = None

@dataclass
class TradeData:
    """Données d'une transaction."""
    timestamp: datetime
    symbol: str
    exchange: str
    price: float
    amount: float
    side: str  # 'buy' ou 'sell'
    trade_id: str
    maker: Optional[bool] = None
    taker: Optional[bool] = None
    fee: Optional[float] = None
    fee_currency: Optional[str] = None

@dataclass
class TechnicalIndicators:
    """Indicateurs techniques calculés."""
    timestamp: datetime
    symbol: str
    # Momentum
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    stochastic_k: Optional[float] = None
    stochastic_d: Optional[float] = None
    
    # Trend
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    adx: Optional[float] = None
    
    # Volatility
    atr: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None
    bollinger_bandwidth: Optional[float] = None
    
    # Volume
    obv: Optional[float] = None
    volume_sma: Optional[float] = None
    volume_ema: Optional[float] = None
    mfi: Optional[float] = None

@dataclass
class MarketMetrics:
    """Métriques de marché calculées."""
    timestamp: datetime
    symbol: str
    volatility: float
    liquidity_score: float
    spread: float
    depth: float
    volume_profile: Dict[str, float]
    price_momentum: float
    trend_strength: float
    market_regime: str
    correlation_matrix: Optional[Dict[str, float]] = None

@dataclass
class TradingSignal:
    """Signal de trading généré."""
    timestamp: datetime
    symbol: str
    strategy_name: str
    signal_type: str  # 'buy', 'sell', 'close'
    confidence: float
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timeframe: Optional[str] = None
    expiration: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class PositionInfo:
    """Information sur une position ouverte."""
    symbol: str
    side: str
    entry_price: float
    current_price: float
    quantity: float
    leverage: float
    unrealized_pnl: float
    realized_pnl: float
    margin: float
    liquidation_price: Optional[float] = None
    entry_time: Optional[datetime] = None
    update_time: Optional[datetime] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class DataValidationResult:
    """Résultat de la validation des données."""
    is_valid: bool
    timestamp: datetime
    data_type: str
    validation_checks: List[Dict[str, Any]]
    errors: List[str]
    warnings: List[str]
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ProcessedData:
    """Données traitées prêtes pour l'entraînement."""
    features: np.ndarray
    targets: np.ndarray
    feature_names: List[str]
    target_names: List[str]
    timestamps: np.ndarray
    scaler: Any
    metadata: Dict[str, Any]

@dataclass
class ModelPrediction:
    """Prédiction générée par un modèle."""
    timestamp: datetime
    symbol: str
    model_name: str
    prediction_type: str
    value: Union[float, str]
    probability: Optional[float] = None
    confidence_interval: Optional[tuple] = None
    features_importance: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

class DataTypeRegistry:
    """Registre des types de données disponibles."""
    
    _types = {
        'market_data': MarketData,
        'futures_data': FuturesData,
        'orderbook_data': OrderBookData,
        'trade_data': TradeData,
        'technical_indicators': TechnicalIndicators,
        'market_metrics': MarketMetrics,
        'trading_signal': TradingSignal,
        'position_info': PositionInfo,
        'backtest_result': BacktestResult,
        'validation_result': DataValidationResult,
        'processed_data': ProcessedData,
        'model_prediction': ModelPrediction
    }
    
    @classmethod
    def get_type(cls, type_name: str) -> type:
        """Récupère un type de données par son nom."""
        if type_name not in cls._types:
            raise ValueError(f"Type de données non reconnu: {type_name}")
        return cls._types[type_name]
    
    @classmethod
    def register_type(cls, type_name: str, data_type: type) -> None:
        """Enregistre un nouveau type de données."""
        if type_name in cls._types:
            raise ValueError(f"Le type {type_name} existe déjà")
        cls._types[type_name] = data_type
    
    @classmethod
    def list_types(cls) -> List[str]:
        """Liste tous les types de données disponibles."""
        return list(cls._types.keys())

# Exemple d'utilisation
def create_example_data() -> None:
    """Crée des exemples de données pour démonstration."""
    # Exemple OHLCV
    ohlcv_data = OHLCVData(
        timestamp=[datetime.now()],
        open=[50000.0],
        high=[51000.0],
        low=[49000.0],
        close=[50500.0],
        volume=[100.0]
    )

    # Exemple d'indicateurs techniques
    indicators = TechnicalIndicators(
        rsi=[60.0],
        macd=[100.0],
        macd_signal=[90.0],
        macd_hist=[10.0],
        ema_9=[50200.0],
        ema_21=[50100.0],
        bollinger_upper=[51000.0],
        bollinger_middle=[50000.0],
        bollinger_lower=[49000.0]
    )

    # Exemple de données de trading
    trade = TradeData(
        timestamp=datetime.now(),
        symbol="BTC/USDT",
        side="buy",
        price=50000.0,
        amount=1.0,
        cost=50000.0,
        fee={"cost": 25.0, "currency": "USDT"}
    )

    # Exemple d'état du marché
    market_state = MarketState(
        timestamp=datetime.now(),
        symbol="BTC/USDT",
        current_price=50000.0,
        bid_price=49990.0,
        ask_price=50010.0,
        spread=20.0,
        volume_24h=1000000.0,
        price_change_24h=0.05,
        high_24h=51000.0,
        low_24h=49000.0
    )

    print("Exemples de données créés avec succès")
    return ohlcv_data, indicators, trade, market_state

if __name__ == "__main__":
    # Test de création des structures de données
    example_data = create_example_data()
    print("Types de données validés et fonctionnels")

    # Création d'instances de test
    market_data = MarketData(
        timestamp=datetime.now(),
        symbol="BTCUSDT",
        exchange="binance",
        open=50000.0,
        high=51000.0,
        low=49000.0,
        close=50500.0,
        volume=100.0
    )
    
    futures_data = FuturesData(
        timestamp=datetime.now(),
        symbol="BTCUSDT",
        exchange="binance",
        open=50000.0,
        high=51000.0,
        low=49000.0,
        close=50500.0,
        volume=100.0,
        funding_rate=0.0001,
        mark_price=50500.0,
        index_price=50450.0,
        open_interest=1000.0,
        next_funding_time=datetime.now()
    )
    
    # Test du registre
    print("Types disponibles:", DataTypeRegistry.list_types())
    market_data_type = DataTypeRegistry.get_type('market_data')
    assert isinstance(market_data, market_data_type)
    print("Tests passés avec succès")

class OrderData(TypedDict):
    """Structure de données pour les ordres."""
    timestamp: datetime
    symbol: str
    order_type: str  # 'market', 'limit', 'stop', 'stop_limit'
    side: str  # 'buy' ou 'sell'
    price: float
    amount: float
    status: str  # 'pending', 'filled', 'canceled', 'rejected'
    filled_amount: float
    remaining_amount: float
    average_price: Optional[float]
    cost: Optional[float]
    fee: Optional[Dict[str, float]]
    id: str
class OrderData:
    def __init__(self, order_id, symbol, quantity, price):
        self.order_id = order_id
        self.symbol = symbol
        self.quantity = quantity
        self.price = price