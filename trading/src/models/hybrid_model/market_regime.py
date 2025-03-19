import numpy as np
import pandas as pd
from typing import Dict, Any
import logging
from scipy import stats
import ruptures

class MarketRegimeManager:
    """Gestionnaire des régimes de marché."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.volatility_threshold = config.get('volatility_threshold', 0.02)
        self.volume_threshold = config.get('volume_threshold', 1.5)
        self.trend_threshold = config.get('trend_threshold', 0.01)
        
    async def detect_regime(self, data: Dict) -> str:
        """Détecte le régime de marché actuel."""
        try:
            # Conversion en DataFrame si nécessaire
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            
            # Calcul des métriques
            volatility = await self.calculate_volatility(data)
            volume_score = await self.analyze_volume(data)
            trend = await self.detect_trend(data)
            
            # Détermination du régime
            if volatility > self.volatility_threshold:
                return "volatile"
            elif abs(trend) > self.trend_threshold:
                return "trending"
            else:
                return "ranging"
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la détection du régime: {str(e)}")
            return "unknown"
            
    async def calculate_volatility(self, data: Dict) -> float:
        """Calcule la volatilité du marché."""
        try:
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            
            if 'close' not in data.columns:
                raise ValueError("Données de prix manquantes")
                
            # Calcul de la volatilité sur les prix de clôture
            returns = np.log(data['close']).diff().dropna()
            volatility = returns.std()
            
            return float(volatility)
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul de la volatilité: {str(e)}")
            return 0.0
            
    async def analyze_volume(self, data: Dict) -> float:
        """Analyse le volume de trading."""
        try:
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            
            if 'volume' not in data.columns:
                raise ValueError("Données de volume manquantes")
                
            # Calcul du score de volume normalisé
            volume = data['volume'].values
            volume_ma = np.mean(volume)
            volume_std = np.std(volume)
            
            if volume_std == 0:
                return 0.5
                
            volume_score = (volume[-1] - volume_ma) / volume_std
            
            # Normalisation entre 0 et 1
            return float(1 / (1 + np.exp(-volume_score)))
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse du volume: {str(e)}")
            return 0.5
            
    async def detect_trend(self, data: Dict) -> float:
        """Détecte la tendance du marché."""
        try:
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            
            if 'close' not in data.columns:
                raise ValueError("Données de prix manquantes")
                
            # Calcul de la pente de la régression linéaire
            x = np.arange(len(data))
            y = data['close'].values
            slope, _, _, _, _ = stats.linregress(x, y)
            
            return float(slope)
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la détection de la tendance: {str(e)}")
            return 0.0
            
    async def cleanup(self):
        """Nettoyage des ressources."""
        # Pas de ressources à nettoyer pour le moment
        pass 