"""
Configuration des exchanges et des paires de trading supportées.
Supporte la configuration via fichier YAML et inclut le support des futures.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import logging
from trading.utils.logging import setup_logger

logger = setup_logger(__name__)

@dataclass
class ExchangeLimits:
    """Limites de l'exchange."""
    rate_limit: int
    max_requests_per_minute: int
    max_symbols_per_request: int
    max_leverage: int = 100  # Pour les futures
    min_leverage: int = 1    # Pour les futures

@dataclass
class FuturesConfig:
    """Configuration des contrats futures."""
    enabled: bool = False
    margin_type: str = 'ISOLATED'  # ou 'CROSSED'
    position_mode: str = 'ONE_WAY'  # ou 'HEDGE'
    leverage_tiers: Dict[str, List[Dict]] = None
    funding_intervals: Dict[str, int] = None
    risk_limits: Dict[str, Dict] = None

@dataclass
class ExchangeConfig:
    """Configuration complète d'un exchange."""
    name: str
    pairs: List[str]
    timeframes: List[str]
    limits: ExchangeLimits
    futures: FuturesConfig = None
    api_config: Dict = None

SUPPORTED_EXCHANGES = {
    'binance': ExchangeConfig(
        name='Binance',
        pairs=[
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
            'ADA/USDT', 'AVAX/USDT', 'DOT/USDT', 'MATIC/USDT', 'LINK/USDT',
            'UNI/USDT', 'ATOM/USDT', 'LTC/USDT', 'DOGE/USDT', 'SHIB/USDT'
        ],
        timeframes=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w'],
        limits=ExchangeLimits(
            rate_limit=1200,
            max_requests_per_minute=1200,
            max_symbols_per_request=100,
            max_leverage=125,
            min_leverage=1
        ),
        futures=FuturesConfig(
            enabled=True,
            margin_type='ISOLATED',
            position_mode='HEDGE',
            leverage_tiers={
                'BTC/USDT': [
                    {'tier': 1, 'leverage': 125, 'min_notional': 0, 'max_notional': 50000},
                    {'tier': 2, 'leverage': 100, 'min_notional': 50000, 'max_notional': 250000}
                ]
            },
            funding_intervals={'BTC/USDT': 8, 'ETH/USDT': 8},
            risk_limits={
                'BTC/USDT': {
                    'max_position_size': 100,
                    'maintenance_margin_rate': 0.0075
                }
            }
        )
    ),
    'kucoin': {
        'name': 'KuCoin',
        'pairs': [
            'BTC/USDT', 'ETH/USDT', 'KCS/USDT', 'SOL/USDT', 'XRP/USDT',
            'ADA/USDT', 'DOT/USDT', 'MATIC/USDT', 'LINK/USDT', 'UNI/USDT',
            'ATOM/USDT', 'LTC/USDT', 'DOGE/USDT', 'TRX/USDT', 'VET/USDT'
        ],
        'timeframes': ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '1w'],
        'limits': {
            'rate_limit': 1000,
            'max_requests_per_minute': 600,
            'max_symbols_per_request': 50
        }
    },
    'bybit': ExchangeConfig(
        name='Bybit',
        pairs=[
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT',
            'ADA/USDT', 'AVAX/USDT', 'MATIC/USDT', 'LINK/USDT', 'DOT/USDT'
        ],
        timeframes=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w', '1M'],
        limits=ExchangeLimits(
            rate_limit=1000,
            max_requests_per_minute=600,
            max_symbols_per_request=50,
            max_leverage=100,
            min_leverage=1
        ),
        futures=FuturesConfig(
            enabled=True,
            margin_type='ISOLATED',
            position_mode='BOTH',
            leverage_tiers={
                'BTC/USDT': [
                    {'tier': 1, 'leverage': 100, 'min_notional': 0, 'max_notional': 100000},
                    {'tier': 2, 'leverage': 75, 'min_notional': 100000, 'max_notional': 500000}
                ]
            },
            funding_intervals={'BTC/USDT': 8, 'ETH/USDT': 8},
            risk_limits={
                'BTC/USDT': {
                    'max_position_size': 150,
                    'maintenance_margin_rate': 0.01
                }
            }
        )
    )
}

# Caractéristiques techniques requises
REQUIRED_FEATURES = [
    'open', 'high', 'low', 'close', 'volume'
]

# Indicateurs techniques par défaut
TECHNICAL_FEATURES = {
    'momentum': [
        'rsi_14',
        'macd',
        'macd_signal',
        'macd_hist',
        'stoch_k',
        'stoch_d',
        'mom_14',
        'tsi'
    ],
    'trend': [
        'sma_20',
        'sma_50',
        'sma_200',
        'ema_12',
        'ema_26',
        'supertrend',
        'adx'
    ],
    'volatility': [
        'atr_14',
        'bollinger_upper',
        'bollinger_middle',
        'bollinger_lower',
        'keltner_upper',
        'keltner_middle',
        'keltner_lower',
        'true_range'
    ],
    'volume': [
        'obv',
        'vwap',
        'volume_sma_20',
        'cmf',
        'mfi'
    ],
    'futures': [  # Indicateurs spécifiques aux futures
        'funding_rate',
        'next_funding_time',
        'mark_price',
        'index_price',
        'basis',
        'open_interest',
        'long_short_ratio'
    ]
}

# Configuration par défaut
DEFAULT_TIMEFRAME = '1h'
DEFAULT_EXCHANGE = 'binance'
DEFAULT_PAIRS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']

# Configuration des limites de téléchargement
DOWNLOAD_LIMITS = {
    'max_days_per_request': 30,
    'max_candles_per_request': 1000,
    'rate_limit_sleep': 0.5  # secondes
}

# Configuration de la validation des données
DATA_VALIDATION = {
    'min_candles': 100,
    'max_missing_values_pct': 0.01,
    'min_volume': 0,
    'check_price_coherence': True,
    'check_timestamp_sequence': True,
    'check_funding_rate': True,  # Pour les futures
    'check_mark_price': True     # Pour les futures
}

# Configuration du stockage des données
STORAGE_CONFIG = {
    'base_path': 'data',
    'raw_data_path': 'data/raw',
    'processed_data_path': 'data/processed',
    'plots_path': 'data/plots',
    'futures_data_path': 'data/futures',  # Nouveau chemin pour les données futures
    'use_parquet': True,
    'compression': 'snappy'
}

def load_config_from_yaml(config_path: Union[str, Path]) -> Dict:
    """
    Charge la configuration depuis un fichier YAML.
    
    Args:
        config_path: Chemin vers le fichier de configuration
        
    Returns:
        Configuration chargée
    """
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Fichier de configuration non trouvé: {config_path}")
            return {}
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return config
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration: {str(e)}")
        return {}

def save_config_to_yaml(config: Dict, config_path: Union[str, Path]) -> None:
    """
    Sauvegarde la configuration dans un fichier YAML.
    
    Args:
        config: Configuration à sauvegarder
        config_path: Chemin de sauvegarde
    """
    try:
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        logger.info(f"Configuration sauvegardée dans {config_path}")
        
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde de la configuration: {str(e)}")

def get_exchange_config(exchange_id: str) -> Optional[ExchangeConfig]:
    """
    Récupère la configuration d'un exchange.
    
    Args:
        exchange_id: Identifiant de l'exchange
        
    Returns:
        Configuration de l'exchange ou None si non trouvé
    """
    return SUPPORTED_EXCHANGES.get(exchange_id)

def validate_exchange_config(config: Dict) -> bool:
    """
    Valide une configuration d'exchange.
    
    Args:
        config: Configuration à valider
        
    Returns:
        True si la configuration est valide
    """
    try:
        required_fields = ['name', 'pairs', 'timeframes', 'limits']
        if not all(field in config for field in required_fields):
            logger.error(f"Champs requis manquants: {required_fields}")
            return False
            
        if config.get('futures', {}).get('enabled', False):
            required_futures_fields = ['margin_type', 'position_mode', 'leverage_tiers']
            futures_config = config.get('futures', {})
            if not all(field in futures_config for field in required_futures_fields):
                logger.error(f"Champs futures requis manquants: {required_futures_fields}")
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors de la validation de la configuration: {str(e)}")
        return False 