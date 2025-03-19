"""
Métriques de performance et de risque pour le modèle hybride.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from dataclasses import dataclass
from scipy import stats

@dataclass
class PerformanceMetrics:
    """Métriques de performance du modèle."""
    returns: np.ndarray
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    omega_ratio: float
    var_95: float
    cvar_95: float

class PerformanceAnalyzer:
    """Analyseur de performance pour le modèle hybride."""
    
    def __init__(self):
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
    
    def calculate_metrics(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.02
    ) -> PerformanceMetrics:
        """
        Calcule les métriques de performance.
        
        Args:
            returns: Tableau des rendements
            risk_free_rate: Taux sans risque
            
        Returns:
            Métriques de performance
        """
        try:
            # Ratio de Sharpe
            sharpe = self._calculate_sharpe_ratio(returns, risk_free_rate)
            
            # Ratio de Sortino
            sortino = self._calculate_sortino_ratio(returns, risk_free_rate)
            
            # Drawdown maximum
            max_dd = self._calculate_max_drawdown(returns)
            
            # Taux de réussite
            win_rate = self._calculate_win_rate(returns)
            
            # Facteur de profit
            profit_factor = self._calculate_profit_factor(returns)
            
            # Ratio de Calmar
            calmar = self._calculate_calmar_ratio(returns)
            
            # Ratio d'Omega
            omega = self._calculate_omega_ratio(returns)
            
            # Value at Risk et Conditional VaR
            var_95, cvar_95 = self._calculate_var_cvar(returns)
            
            return PerformanceMetrics(
                returns=returns,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                max_drawdown=max_dd,
                win_rate=win_rate,
                profit_factor=profit_factor,
                calmar_ratio=calmar,
                omega_ratio=omega,
                var_95=var_95,
                cvar_95=cvar_95
            )
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul des métriques: {e}")
            raise
    
    def _calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float
    ) -> float:
        """Calcule le ratio de Sharpe."""
        excess_returns = returns - risk_free_rate/252  # Annualisé
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    def _calculate_sortino_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float
    ) -> float:
        """Calcule le ratio de Sortino."""
        excess_returns = returns - risk_free_rate/252
        downside_std = np.sqrt(np.mean(np.minimum(excess_returns, 0)**2))
        return np.sqrt(252) * np.mean(excess_returns) / downside_std
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calcule le drawdown maximum."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return np.min(drawdowns)
    
    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """Calcule le taux de réussite."""
        return np.mean(returns > 0)
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calcule le facteur de profit."""
        gains = np.sum(returns[returns > 0])
        losses = abs(np.sum(returns[returns < 0]))
        return gains / losses if losses != 0 else float('inf')
    
    def _calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """Calcule le ratio de Calmar."""
        annual_return = np.mean(returns) * 252
        max_dd = self._calculate_max_drawdown(returns)
        return annual_return / abs(max_dd) if max_dd != 0 else float('inf')
    
    def _calculate_omega_ratio(self, returns: np.ndarray) -> float:
        """Calcule le ratio d'Omega."""
        gains = np.sum(np.maximum(returns, 0))
        losses = abs(np.sum(np.minimum(returns, 0)))
        return gains / losses if losses != 0 else float('inf')
    
    def _calculate_var_cvar(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calcule la Value at Risk et la Conditional VaR."""
        var = np.percentile(returns, (1 - confidence_level) * 100)
        cvar = np.mean(returns[returns <= var])
        return var, cvar
    
    def analyze_robustness(
        self,
        returns: np.ndarray,
        window_sizes: List[int] = [30, 60, 90, 180]
    ) -> Dict[str, List[float]]:
        """
        Analyse la robustesse des performances.
        
        Args:
            returns: Tableau des rendements
            window_sizes: Tailles des fenêtres d'analyse
            
        Returns:
            Métriques de robustesse
        """
        try:
            results = {
                'sharpe_ratios': [],
                'sortino_ratios': [],
                'max_drawdowns': [],
                'win_rates': []
            }
            
            for window in window_sizes:
                # Calcul des métriques sur des fenêtres glissantes
                metrics = self._calculate_rolling_metrics(returns, window)
                
                # Statistiques sur les métriques
                results['sharpe_ratios'].append(np.mean(metrics['sharpe']))
                results['sortino_ratios'].append(np.mean(metrics['sortino']))
                results['max_drawdowns'].append(np.mean(metrics['max_dd']))
                results['win_rates'].append(np.mean(metrics['win_rate']))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse de robustesse: {e}")
            raise
    
    def _calculate_rolling_metrics(
        self,
        returns: np.ndarray,
        window: int
    ) -> Dict[str, np.ndarray]:
        """Calcule les métriques sur des fenêtres glissantes."""
        n_windows = len(returns) - window + 1
        
        metrics = {
            'sharpe': np.zeros(n_windows),
            'sortino': np.zeros(n_windows),
            'max_dd': np.zeros(n_windows),
            'win_rate': np.zeros(n_windows)
        }
        
        for i in range(n_windows):
            window_returns = returns[i:i+window]
            
            metrics['sharpe'][i] = self._calculate_sharpe_ratio(window_returns, 0.02)
            metrics['sortino'][i] = self._calculate_sortino_ratio(window_returns, 0.02)
            metrics['max_dd'][i] = self._calculate_max_drawdown(window_returns)
            metrics['win_rate'][i] = self._calculate_win_rate(window_returns)
        
        return metrics
    
    def analyze_market_regimes(
        self,
        returns: np.ndarray,
        volatility: np.ndarray,
        n_regimes: int = 3
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyse les performances par régime de marché.
        
        Args:
            returns: Tableau des rendements
            volatility: Tableau de la volatilité
            n_regimes: Nombre de régimes
            
        Returns:
            Métriques par régime
        """
        try:
            # Clustering des régimes basé sur la volatilité
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_regimes)
            regimes = kmeans.fit_predict(volatility.reshape(-1, 1))
            
            results = {}
            for regime in range(n_regimes):
                regime_returns = returns[regimes == regime]
                
                metrics = self.calculate_metrics(regime_returns)
                results[f'regime_{regime}'] = {
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'sortino_ratio': metrics.sortino_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'win_rate': metrics.win_rate,
                    'profit_factor': metrics.profit_factor
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse des régimes: {e}")
            raise 