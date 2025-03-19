import numpy as np
import pandas as pd
from typing import Dict, Any
import logging

class RiskManager:
    """Gestionnaire des risques du système de trading."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.risk_config = config.get('risk', {})
        self.base_risk_factor = self.risk_config.get('base_risk_factor', 0.8)
        self.market_risk_weight = self.risk_config.get('market_risk_weight', 0.4)
        self.position_risk_weight = self.risk_config.get('position_risk_weight', 0.3)
        self.volatility_risk_weight = self.risk_config.get('volatility_risk_weight', 0.3)
        
    async def adjust_signal(self, raw_signal: float) -> float:
        """Ajuste le signal en fonction du risque."""
        try:
            # Calcul du facteur de risque
            risk_factor = self.base_risk_factor
            
            # Application du facteur de risque
            adjusted_signal = raw_signal * risk_factor
            
            # Normalisation entre -1 et 1
            return max(-1.0, min(1.0, adjusted_signal))
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajustement du signal: {str(e)}")
            return 0.0
            
    async def calculate_risk_score(self, data: Dict) -> float:
        """Calcule le score de risque global."""
        try:
            # Calcul des composantes de risque
            market_risk = await self._assess_market_risk(data)
            position_risk = await self._assess_position_risk(data)
            volatility_risk = await self._assess_volatility_risk(data)
            
            # Calcul du score pondéré
            risk_score = (
                market_risk * self.market_risk_weight +
                position_risk * self.position_risk_weight +
                volatility_risk * self.volatility_risk_weight
            )
            
            return float(risk_score)
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul du score de risque: {str(e)}")
            return 0.5
            
    async def _assess_market_risk(self, data: Dict) -> float:
        """Évalue le risque de marché."""
        try:
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            
            # Calcul de la volatilité récente
            if 'close' in data.columns:
                returns = np.log(data['close']).diff().dropna()
                volatility = returns.std()
                return float(min(1.0, volatility * 10))  # Normalisation
            return 0.5
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'évaluation du risque de marché: {str(e)}")
            return 0.5
            
    async def _assess_position_risk(self, data: Dict) -> float:
        """Évalue le risque lié aux positions."""
        try:
            # Simulation d'évaluation du risque de position
            return 0.5  # Valeur par défaut pour le moment
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'évaluation du risque de position: {str(e)}")
            return 0.5
            
    async def _assess_volatility_risk(self, data: Dict) -> float:
        """Évalue le risque de volatilité."""
        try:
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            
            if 'high' in data.columns and 'low' in data.columns:
                # Calcul de la volatilité basée sur High-Low
                hl_volatility = (data['high'] - data['low']).mean() / data['low'].mean()
                return float(min(1.0, hl_volatility * 5))  # Normalisation
            return 0.5
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'évaluation du risque de volatilité: {str(e)}")
            return 0.5
            
    async def cleanup(self):
        """Nettoyage des ressources."""
        # Pas de ressources à nettoyer pour le moment
        pass 