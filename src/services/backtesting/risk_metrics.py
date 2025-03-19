"""
Module de calcul des métriques de risque avancées.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class RiskMetrics:
    """Métriques de risque calculées."""
    var_95: float  # Value at Risk à 95%
    cvar_95: float  # Conditional Value at Risk à 95%
    sortino_ratio: float  # Ratio de Sortino
    calmar_ratio: float  # Ratio de Calmar
    omega_ratio: float  # Ratio d'Omega
    ulcer_index: float  # Index d'Ulcer
    pain_index: float  # Index de douleur
    recovery_factor: float  # Facteur de récupération
    risk_reward_ratio: float  # Ratio risque/récompense
    max_drawdown: float  # Drawdown maximum
    avg_drawdown: float  # Drawdown moyen
    drawdown_duration: timedelta  # Durée moyenne des drawdowns
    win_rate: float  # Taux de réussite
    profit_factor: float  # Facteur de profit
    sharpe_ratio: float  # Ratio de Sharpe
    volatility: float  # Volatilité
    skewness: float  # Asymétrie
    kurtosis: float  # Kurtosis

class RiskAnalyzer:
    """Analyseur de risque pour les stratégies de trading."""
    
    def __init__(self, returns: pd.Series, trades: pd.DataFrame):
        """
        Initialise l'analyseur de risque.
        
        Args:
            returns: Série des rendements
            trades: DataFrame des trades
        """
        self.returns = returns
        self.trades = trades
        self._validate_data()
    
    def _validate_data(self):
        """Valide les données d'entrée."""
        if self.returns.empty:
            raise ValueError("La série des rendements est vide")
        if self.trades.empty:
            raise ValueError("Le DataFrame des trades est vide")
    
    def calculate_metrics(self) -> RiskMetrics:
        """Calcule toutes les métriques de risque."""
        return RiskMetrics(
            var_95=self._calculate_var(0.95),
            cvar_95=self._calculate_cvar(0.95),
            sortino_ratio=self._calculate_sortino_ratio(),
            calmar_ratio=self._calculate_calmar_ratio(),
            omega_ratio=self._calculate_omega_ratio(),
            ulcer_index=self._calculate_ulcer_index(),
            pain_index=self._calculate_pain_index(),
            recovery_factor=self._calculate_recovery_factor(),
            risk_reward_ratio=self._calculate_risk_reward_ratio(),
            max_drawdown=self._calculate_max_drawdown(),
            avg_drawdown=self._calculate_avg_drawdown(),
            drawdown_duration=self._calculate_drawdown_duration(),
            win_rate=self._calculate_win_rate(),
            profit_factor=self._calculate_profit_factor(),
            sharpe_ratio=self._calculate_sharpe_ratio(),
            volatility=self._calculate_volatility(),
            skewness=self._calculate_skewness(),
            kurtosis=self._calculate_kurtosis()
        )
    
    def _calculate_var(self, confidence: float) -> float:
        """Calcule la Value at Risk."""
        return np.percentile(self.returns, (1 - confidence) * 100)
    
    def _calculate_cvar(self, confidence: float) -> float:
        """Calcule la Conditional Value at Risk."""
        var = self._calculate_var(confidence)
        return self.returns[self.returns <= var].mean()
    
    def _calculate_sortino_ratio(self) -> float:
        """Calcule le ratio de Sortino."""
        downside_returns = self.returns[self.returns < 0]
        if len(downside_returns) == 0:
            return np.inf
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        if downside_std == 0:
            return np.inf
        return np.sqrt(252) * (self.returns.mean() / downside_std)
    
    def _calculate_calmar_ratio(self) -> float:
        """Calcule le ratio de Calmar."""
        max_dd = self._calculate_max_drawdown()
        if max_dd == 0:
            return np.inf
        return np.sqrt(252) * (self.returns.mean() / abs(max_dd))
    
    def _calculate_omega_ratio(self) -> float:
        """Calcule le ratio d'Omega."""
        threshold = 0
        gains = self.returns[self.returns > threshold].sum()
        losses = abs(self.returns[self.returns < threshold].sum())
        if losses == 0:
            return np.inf
        return gains / losses
    
    def _calculate_ulcer_index(self) -> float:
        """Calcule l'index d'Ulcer."""
        drawdowns = self._calculate_drawdowns()
        return np.sqrt(np.mean(drawdowns ** 2))
    
    def _calculate_pain_index(self) -> float:
        """Calcule l'index de douleur."""
        drawdowns = self._calculate_drawdowns()
        return np.mean(drawdowns)
    
    def _calculate_recovery_factor(self) -> float:
        """Calcule le facteur de récupération."""
        total_profit = self.trades[self.trades["pnl"] > 0]["pnl"].sum()
        max_dd = self._calculate_max_drawdown()
        if max_dd == 0:
            return np.inf
        return total_profit / abs(max_dd)
    
    def _calculate_risk_reward_ratio(self) -> float:
        """Calcule le ratio risque/récompense."""
        avg_win = self.trades[self.trades["pnl"] > 0]["pnl"].mean()
        avg_loss = abs(self.trades[self.trades["pnl"] < 0]["pnl"].mean())
        if avg_loss == 0:
            return np.inf
        return avg_win / avg_loss
    
    def _calculate_max_drawdown(self) -> float:
        """Calcule le drawdown maximum."""
        cumulative_returns = (1 + self.returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        return drawdowns.min()
    
    def _calculate_avg_drawdown(self) -> float:
        """Calcule le drawdown moyen."""
        drawdowns = self._calculate_drawdowns()
        return np.mean(drawdowns)
    
    def _calculate_drawdown_duration(self) -> timedelta:
        """Calcule la durée moyenne des drawdowns."""
        drawdowns = self._calculate_drawdowns()
        if len(drawdowns) == 0:
            return timedelta(0)
        
        # Calcul des périodes de drawdown
        is_drawdown = drawdowns < 0
        drawdown_periods = []
        current_period = 0
        
        for is_dd in is_drawdown:
            if is_dd:
                current_period += 1
            elif current_period > 0:
                drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        if not drawdown_periods:
            return timedelta(0)
        
        avg_periods = np.mean(drawdown_periods)
        return timedelta(days=int(avg_periods))
    
    def _calculate_drawdowns(self) -> np.ndarray:
        """Calcule les drawdowns."""
        cumulative_returns = (1 + self.returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        return (cumulative_returns - rolling_max) / rolling_max
    
    def _calculate_win_rate(self) -> float:
        """Calcule le taux de réussite."""
        winning_trades = len(self.trades[self.trades["pnl"] > 0])
        return winning_trades / len(self.trades)
    
    def _calculate_profit_factor(self) -> float:
        """Calcule le facteur de profit."""
        gross_profit = self.trades[self.trades["pnl"] > 0]["pnl"].sum()
        gross_loss = abs(self.trades[self.trades["pnl"] < 0]["pnl"].sum())
        if gross_loss == 0:
            return np.inf
        return gross_profit / gross_loss
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calcule le ratio de Sharpe."""
        if len(self.returns) < 2:
            return 0.0
        return np.sqrt(252) * (self.returns.mean() / self.returns.std())
    
    def _calculate_volatility(self) -> float:
        """Calcule la volatilité."""
        return np.sqrt(252) * self.returns.std()
    
    def _calculate_skewness(self) -> float:
        """Calcule l'asymétrie."""
        return self.returns.skew()
    
    def _calculate_kurtosis(self) -> float:
        """Calcule la kurtosis."""
        return self.returns.kurtosis()
    
    def get_risk_report(self) -> Dict[str, float]:
        """Génère un rapport complet des métriques de risque."""
        metrics = self.calculate_metrics()
        return {
            "Value at Risk (95%)": metrics.var_95,
            "Conditional VaR (95%)": metrics.cvar_95,
            "Ratio de Sortino": metrics.sortino_ratio,
            "Ratio de Calmar": metrics.calmar_ratio,
            "Ratio d'Omega": metrics.omega_ratio,
            "Index d'Ulcer": metrics.ulcer_index,
            "Index de douleur": metrics.pain_index,
            "Facteur de récupération": metrics.recovery_factor,
            "Ratio risque/récompense": metrics.risk_reward_ratio,
            "Drawdown maximum": metrics.max_drawdown,
            "Drawdown moyen": metrics.avg_drawdown,
            "Durée moyenne des drawdowns": str(metrics.drawdown_duration),
            "Taux de réussite": metrics.win_rate,
            "Facteur de profit": metrics.profit_factor,
            "Ratio de Sharpe": metrics.sharpe_ratio,
            "Volatilité": metrics.volatility,
            "Asymétrie": metrics.skewness,
            "Kurtosis": metrics.kurtosis
        } 