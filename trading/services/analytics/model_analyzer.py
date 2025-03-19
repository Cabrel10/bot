from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ..models.base_model import BaseModel
from ..core.data_types import MarketData
from ..utils.logger import TradingLogger

class ModelAnalyzer:
    """Analyseur avancé de modèles de trading."""
    
    def __init__(self, 
                 model: BaseModel,
                 data: MarketData,
                 save_dir: str = "reports/model_analysis"):
        """
        Initialise l'analyseur de modèle.
        
        Args:
            model: Modèle à analyser
            data: Données de marché
            save_dir: Répertoire de sauvegarde
        """
        self.model = model
        self.data = data
        self.save_dir = save_dir
        self.logger = TradingLogger()
        
        # Cache pour les calculs intensifs
        self._prediction_cache = {}
        self._sensitivity_cache = {}
        
    def analyze_model_behavior(self) -> Dict[str, Union[float, Dict]]:
        """Analyse complète du comportement du modèle."""
        try:
            results = {
                'prediction_stability': self._analyze_prediction_stability(),
                'feature_importance': self._analyze_feature_importance(),
                'market_regime_sensitivity': self._analyze_market_regime_sensitivity(),
                'decision_boundaries': self._analyze_decision_boundaries(),
                'confidence_metrics': self._analyze_confidence_metrics()
            }
            return results
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse du modèle: {str(e)}")
            raise
            
    def _analyze_prediction_stability(self) -> Dict[str, float]:
        """Analyse la stabilité des prédictions."""
        predictions = []
        for _ in range(10):  # Multiple runs
            pred = self.model.predict(self.data.get_features())
            predictions.append(pred)
            
        predictions = np.array(predictions)
        return {
            'mean_std': float(np.mean(np.std(predictions, axis=0))),
            'prediction_range': float(np.ptp(predictions)),
            'consistency_score': float(np.mean(np.corrcoef(predictions)[0, 1:]))
        }
        
    def _analyze_feature_importance(self) -> Dict[str, float]:
        """Analyse l'importance des features."""
        base_pred = self.model.predict(self.data.get_features())
        importance_scores = {}
        
        for feature in self.data.get_feature_names():
            perturbed_data = self.data.copy()
            perturbed_data[feature] = np.random.permutation(perturbed_data[feature])
            new_pred = self.model.predict(perturbed_data.get_features())
            importance_scores[feature] = np.mean(np.abs(base_pred - new_pred))
            
        return importance_scores
        
    def _analyze_market_regime_sensitivity(self) -> Dict[str, Dict[str, float]]:
        """Analyse la sensibilité aux régimes de marché."""
        volatility = self._calculate_volatility(self.data)
        trend = self._calculate_trend(self.data)
        
        regimes = {
            'high_volatility': volatility > np.percentile(volatility, 75),
            'low_volatility': volatility < np.percentile(volatility, 25),
            'uptrend': trend > 0,
            'downtrend': trend < 0
        }
        
        sensitivity = {}
        for regime_name, regime_mask in regimes.items():
            regime_data = self.data[regime_mask]
            if len(regime_data) > 0:
                predictions = self.model.predict(regime_data.get_features())
                sensitivity[regime_name] = {
                    'mean_prediction': float(np.mean(predictions)),
                    'std_prediction': float(np.std(predictions)),
                    'confidence': float(np.mean(np.abs(predictions)))
                }
                
        return sensitivity
        
    def _analyze_decision_boundaries(self) -> Dict[str, np.ndarray]:
        """Analyse les frontières de décision."""
        features = self.data.get_features()
        predictions = self.model.predict(features)
        
        boundaries = {
            'decision_thresholds': np.percentile(predictions, [25, 50, 75]),
            'boundary_stability': self._calculate_boundary_stability(features, predictions)
        }
        
        return boundaries
        
    def _analyze_confidence_metrics(self) -> Dict[str, float]:
        """Analyse les métriques de confiance."""
        predictions = self.model.predict(self.data.get_features())
        actual = self.data.get_target()
        
        return {
            'prediction_confidence': float(np.mean(np.abs(predictions))),
            'prediction_bias': float(np.mean(predictions)),
            'prediction_variance': float(np.var(predictions)),
            'accuracy_score': float(np.mean(np.sign(predictions) == np.sign(actual)))
        }
        
    def generate_interactive_analysis(self) -> go.Figure:
        """Génère une analyse interactive."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Stabilité des Prédictions',
                'Importance des Features',
                'Sensibilité aux Régimes',
                'Métriques de Confiance'
            )
        )
        
        # Stabilité des prédictions
        stability = self._analyze_prediction_stability()
        fig.add_trace(
            go.Bar(
                x=list(stability.keys()),
                y=list(stability.values()),
                name='Stabilité'
            ),
            row=1, col=1
        )
        
        # Importance des features
        importance = self._analyze_feature_importance()
        fig.add_trace(
            go.Bar(
                x=list(importance.keys()),
                y=list(importance.values()),
                name='Importance'
            ),
            row=1, col=2
        )
        
        # Sensibilité aux régimes
        sensitivity = self._analyze_market_regime_sensitivity()
        regime_names = list(sensitivity.keys())
        mean_predictions = [s['mean_prediction'] for s in sensitivity.values()]
        fig.add_trace(
            go.Bar(
                x=regime_names,
                y=mean_predictions,
                name='Sensibilité'
            ),
            row=2, col=1
        )
        
        # Métriques de confiance
        confidence = self._analyze_confidence_metrics()
        fig.add_trace(
            go.Bar(
                x=list(confidence.keys()),
                y=list(confidence.values()),
                name='Confiance'
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True)
        return fig
        
    def _calculate_volatility(self, data: MarketData, window: int = 20) -> np.ndarray:
        """Calcule la volatilité."""
        returns = np.diff(np.log(data['close']))
        volatility = pd.Series(returns).rolling(window).std().fillna(0).values
        return np.pad(volatility, (1, 0), 'edge')
        
    def _calculate_trend(self, data: MarketData, window: int = 20) -> np.ndarray:
        """Calcule la tendance."""
        prices = data['close']
        ma = pd.Series(prices).rolling(window).mean().fillna(method='bfill').values
        return (prices - ma) / ma
        
    def _calculate_boundary_stability(self, 
                                   features: np.ndarray,
                                   predictions: np.ndarray,
                                   n_samples: int = 100) -> np.ndarray:
        """Calcule la stabilité des frontières de décision."""
        thresholds = np.linspace(np.min(predictions), np.max(predictions), n_samples)
        stability = np.zeros_like(thresholds)
        
        for i, threshold in enumerate(thresholds):
            decisions = predictions > threshold
            stability[i] = np.mean(np.abs(np.diff(decisions)))
            
        return stability 