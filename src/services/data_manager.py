"""
Gestionnaire de données pour le système de trading.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass

@dataclass
class DataConfig:
    """Configuration pour la gestion des données."""
    base_path: str = "data"
    cache_size: int = 1000
    update_interval: int = 60
    supported_timeframes: List[str] = None
    
    def __post_init__(self):
        if self.supported_timeframes is None:
            self.supported_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]

class DataManager:
    """Gestionnaire de données pour le système de trading."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.base_path = Path(config.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure la journalisation."""
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Récupère les données historiques.
        
        Args:
            symbol: Paire de trading
            timeframe: Intervalle de temps
            start_date: Date de début
            end_date: Date de fin
            
        Returns:
            DataFrame avec les données OHLCV
        """
        try:
            file_path = self.base_path / f"{symbol}_{timeframe}.csv"
            if not file_path.exists():
                self.logger.warning(f"Fichier de données non trouvé: {file_path}")
                return pd.DataFrame()
                
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
            return df[mask]
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des données: {e}")
            return pd.DataFrame()
    
    def update_market_data(
        self,
        symbol: str,
        timeframe: str,
        new_data: pd.DataFrame
    ) -> bool:
        """
        Met à jour les données de marché.
        
        Args:
            symbol: Paire de trading
            timeframe: Intervalle de temps
            new_data: Nouvelles données
            
        Returns:
            True si la mise à jour a réussi
        """
        try:
            file_path = self.base_path / f"{symbol}_{timeframe}.csv"
            
            if file_path.exists():
                existing_data = pd.read_csv(file_path)
                existing_data['timestamp'] = pd.to_datetime(existing_data['timestamp'])
                
                # Fusion des données
                combined_data = pd.concat([existing_data, new_data])
                combined_data = combined_data.drop_duplicates(subset=['timestamp'])
                combined_data = combined_data.sort_values('timestamp')
                
                # Limite de cache
                if len(combined_data) > self.config.cache_size:
                    combined_data = combined_data.tail(self.config.cache_size)
                
                combined_data.to_csv(file_path, index=False)
            else:
                new_data.to_csv(file_path, index=False)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la mise à jour des données: {e}")
            return False
    
    def get_latest_data(
        self,
        symbol: str,
        timeframe: str,
        n_points: int = 100
    ) -> pd.DataFrame:
        """
        Récupère les dernières données.
        
        Args:
            symbol: Paire de trading
            timeframe: Intervalle de temps
            n_points: Nombre de points à récupérer
            
        Returns:
            DataFrame avec les dernières données
        """
        try:
            file_path = self.base_path / f"{symbol}_{timeframe}.csv"
            if not file_path.exists():
                return pd.DataFrame()
                
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df.tail(n_points)
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des dernières données: {e}")
            return pd.DataFrame()
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Valide les données.
        
        Args:
            df: DataFrame à valider
            
        Returns:
            True si les données sont valides
        """
        try:
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            # Vérification des colonnes requises
            if not all(col in df.columns for col in required_columns):
                self.logger.error("Colonnes requises manquantes")
                return False
            
            # Vérification des types de données
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                self.logger.error("Format de timestamp invalide")
                return False
            
            # Vérification des valeurs nulles
            if df[required_columns].isnull().any().any():
                self.logger.error("Valeurs nulles détectées")
                return False
            
            # Vérification de la cohérence OHLC
            if not (df['high'] >= df['low']).all():
                self.logger.error("Incohérence OHLC détectée")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la validation des données: {e}")
            return False 