"""
Gestionnaire de configuration pour le système de trading.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass, asdict
import os

@dataclass
class ModelConfig:
    """Configuration du modèle hybride."""
    cnn_config: Dict[str, Any]
    gna_config: Dict[str, Any]
    hybrid_config: Dict[str, Any]

@dataclass
class TradingConfig:
    """Configuration du trading."""
    initial_balance: float
    position_size: float
    leverage: float
    stop_loss: float
    take_profit: float
    fee_rate: float

@dataclass
class DataConfig:
    """Configuration des données."""
    symbols: list[str]
    timeframes: list[str]
    start_date: str
    end_date: str
    batch_size: int

class ConfigManager:
    """Gestionnaire de configuration pour le système de trading."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._setup_logging()
        self._ensure_config_dir()
        
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
    
    def _ensure_config_dir(self):
        """Crée le répertoire de configuration s'il n'existe pas."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model_config(
        self,
        config: ModelConfig,
        version: str = "latest"
    ) -> bool:
        """
        Sauvegarde la configuration du modèle.
        
        Args:
            config: Configuration du modèle
            version: Version de la configuration
            
        Returns:
            True si la sauvegarde a réussi
        """
        try:
            config_path = self.config_dir / f"model_config_{version}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(asdict(config), f)
            
            self.logger.info(f"Configuration du modèle sauvegardée: {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde de la configuration: {e}")
            return False
    
    def load_model_config(
        self,
        version: str = "latest"
    ) -> Optional[ModelConfig]:
        """
        Charge la configuration du modèle.
        
        Args:
            version: Version de la configuration
            
        Returns:
            Configuration du modèle ou None si erreur
        """
        try:
            config_path = self.config_dir / f"model_config_{version}.yaml"
            if not config_path.exists():
                self.logger.warning(f"Configuration non trouvée: {config_path}")
                return None
                
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                
            return ModelConfig(**config_dict)
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de la configuration: {e}")
            return None
    
    def save_trading_config(
        self,
        config: TradingConfig,
        version: str = "latest"
    ) -> bool:
        """
        Sauvegarde la configuration du trading.
        
        Args:
            config: Configuration du trading
            version: Version de la configuration
            
        Returns:
            True si la sauvegarde a réussi
        """
        try:
            config_path = self.config_dir / f"trading_config_{version}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(asdict(config), f)
            
            self.logger.info(f"Configuration du trading sauvegardée: {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde de la configuration: {e}")
            return False
    
    def load_trading_config(
        self,
        version: str = "latest"
    ) -> Optional[TradingConfig]:
        """
        Charge la configuration du trading.
        
        Args:
            version: Version de la configuration
            
        Returns:
            Configuration du trading ou None si erreur
        """
        try:
            config_path = self.config_dir / f"trading_config_{version}.yaml"
            if not config_path.exists():
                self.logger.warning(f"Configuration non trouvée: {config_path}")
                return None
                
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                
            return TradingConfig(**config_dict)
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de la configuration: {e}")
            return None
    
    def save_data_config(
        self,
        config: DataConfig,
        version: str = "latest"
    ) -> bool:
        """
        Sauvegarde la configuration des données.
        
        Args:
            config: Configuration des données
            version: Version de la configuration
            
        Returns:
            True si la sauvegarde a réussi
        """
        try:
            config_path = self.config_dir / f"data_config_{version}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(asdict(config), f)
            
            self.logger.info(f"Configuration des données sauvegardée: {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde de la configuration: {e}")
            return False
    
    def load_data_config(
        self,
        version: str = "latest"
    ) -> Optional[DataConfig]:
        """
        Charge la configuration des données.
        
        Args:
            version: Version de la configuration
            
        Returns:
            Configuration des données ou None si erreur
        """
        try:
            config_path = self.config_dir / f"data_config_{version}.yaml"
            if not config_path.exists():
                self.logger.warning(f"Configuration non trouvée: {config_path}")
                return None
                
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                
            return DataConfig(**config_dict)
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de la configuration: {e}")
            return None
    
    def get_available_versions(self) -> list[str]:
        """
        Récupère les versions disponibles des configurations.
        
        Returns:
            Liste des versions disponibles
        """
        try:
            versions = set()
            for file in self.config_dir.glob("*.yaml"):
                version = file.stem.split("_")[-1]
                versions.add(version)
            return sorted(list(versions))
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des versions: {e}")
            return []
    
    def delete_config(self, version: str) -> bool:
        """
        Supprime une version de configuration.
        
        Args:
            version: Version à supprimer
            
        Returns:
            True si la suppression a réussi
        """
        try:
            for file in self.config_dir.glob(f"*_{version}.yaml"):
                file.unlink()
            
            self.logger.info(f"Configuration version {version} supprimée")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la suppression de la configuration: {e}")
            return False 