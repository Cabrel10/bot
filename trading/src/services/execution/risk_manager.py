from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import yaml
import json
from dataclasses import dataclass

from ..data.data_types import TradeData, PositionData
from ..utils.logger import TradingLogger

@dataclass
class RiskMetrics:
    """Métriques de risque calculées."""
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    var_95: float  # Value at Risk à 95%
    exposure: float
    leverage: float
    profit_factor: float
    win_rate: float

class RiskManager:
    """Gestionnaire de risque pour le trading."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialise le gestionnaire de risque.
        
        Args:
            config_path: Chemin vers la configuration
        """
        self.logger = TradingLogger()
        self.config = self._load_config(config_path)
        self.risk_metrics: Optional[RiskMetrics] = None
        self.position_history: List[PositionData] = []
        self.trade_history: List[TradeData] = []

    def _load_config(self, config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Charge la configuration."""
        if not config_path:
            config_path = Path(__file__).parent / 'config' / 'risk_config.yaml'
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def check_order_risk(self,
                        symbol: str,
                        side: str,
                        amount: float,
                        price: float) -> Tuple[bool, str]:
        """Vérifie si un ordre respecte les critères de risque.
        
        Args:
            symbol: Symbole à trader
            side: Direction ('buy' ou 'sell')
            amount: Quantité
            price: Prix
            
        Returns:
            (autorisation, raison)
        """
        try:
            # Vérification de l'exposition
            current_exposure = self._calculate_current_exposure()
            new_exposure = current_exposure + (amount * price)
            
            if new_exposure > self.config['limits']['max_exposure']:
                return False, "Exposition maximale dépassée"
            
            # Vérification du levier
            if self._calculate_leverage(new_exposure) > self.config['limits']['max_leverage']:
                return False, "Levier maximal dépassé"
            
            # Vérification de la concentration
            symbol_exposure = self._calculate_symbol_exposure(symbol)
            if symbol_exposure > self.config['limits']['max_symbol_exposure']:
                return False, "Exposition maximale par symbole dépassée"
            
            # Vérification de la perte maximale
            if self._check_max_loss_limit():
                return False, "Limite de perte maximale atteinte"
            
            return True, "OK"

        except Exception as e:
            self.logger.log_error(e, {'action': 'check_order_risk'})
            return False, f"Erreur: {str(e)}"

    def update_metrics(self, equity_curve: pd.Series) -> RiskMetrics:
        """Met à jour les métriques de risque.
        
        Args:
            equity_curve: Série temporelle de l'equity
            
        Returns:
            Métriques de risque calculées
        """
        try:
            returns = equity_curve.pct_change().dropna()
            
            self.risk_metrics = RiskMetrics(
                max_drawdown=self._calculate_max_drawdown(equity_curve),
                volatility=self._calculate_volatility(returns),
                sharpe_ratio=self._calculate_sharpe_ratio(returns),
                sortino_ratio=self._calculate_sortino_ratio(returns),
                var_95=self._calculate_var(returns, 0.95),
                exposure=self._calculate_current_exposure(),
                leverage=self._calculate_leverage(),
                profit_factor=self._calculate_profit_factor(),
                win_rate=self._calculate_win_rate()
            )
            
            return self.risk_metrics

        except Exception as e:
            self.logger.log_error(e, {'action': 'update_metrics'})
            raise

    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calcule le drawdown maximal."""
        rolling_max = equity_curve.expanding().max()
        drawdowns = equity_curve / rolling_max - 1.0
        return abs(drawdowns.min())

    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calcule la volatilité annualisée."""
        return returns.std() * np.sqrt(252)  # 252 jours de trading

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calcule le ratio de Sharpe."""
        rf_rate = self.config['metrics']['risk_free_rate']
        excess_returns = returns - (rf_rate / 252)
        return np.sqrt(252) * (excess_returns.mean() / returns.std())

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calcule le ratio de Sortino."""
        rf_rate = self.config['metrics']['risk_free_rate']
        excess_returns = returns - (rf_rate / 252)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        return np.sqrt(252) * (excess_returns.mean() / downside_std)

    def _calculate_var(self, returns: pd.Series, confidence: float) -> float:
        """Calcule la Value at Risk."""
        return abs(np.percentile(returns, (1 - confidence) * 100))

    def _calculate_current_exposure(self) -> float:
        """Calcule l'exposition actuelle."""
        return sum(pos.value for pos in self.position_history if pos.is_open)

    def _calculate_leverage(self, new_exposure: Optional[float] = None) -> float:
        """Calcule le levier actuel."""
        total_equity = self.config['account']['equity']
        exposure = new_exposure if new_exposure else self._calculate_current_exposure()
        return exposure / total_equity

    def _calculate_symbol_exposure(self, symbol: str) -> float:
        """Calcule l'exposition pour un symbole."""
        return sum(
            pos.value 
            for pos in self.position_history 
            if pos.is_open and pos.symbol == symbol
        )

    def _check_max_loss_limit(self) -> bool:
        """Vérifie si la limite de perte est atteinte."""
        daily_pnl = self._calculate_daily_pnl()
        return abs(daily_pnl) > self.config['limits']['max_daily_loss']

    def _calculate_daily_pnl(self) -> float:
        """Calcule le P&L quotidien."""
        today = datetime.now().date()
        daily_trades = [
            trade for trade in self.trade_history 
            if trade.exit_time.date() == today
        ]
        return sum(trade.pnl for trade in daily_trades)

    def _calculate_profit_factor(self) -> float:
        """Calcule le facteur de profit."""
        profits = sum(t.pnl for t in self.trade_history if t.pnl > 0)
        losses = abs(sum(t.pnl for t in self.trade_history if t.pnl < 0))
        return profits / losses if losses > 0 else float('inf')

    def _calculate_win_rate(self) -> float:
        """Calcule le taux de trades gagnants."""
        if not self.trade_history:
            return 0.0
        winning_trades = sum(1 for t in self.trade_history if t.pnl > 0)
        return winning_trades / len(self.trade_history)

    def get_risk_report(self) -> Dict[str, Any]:
        """Génère un rapport de risque complet."""
        if not self.risk_metrics:
            return {"error": "Aucune métrique disponible"}
            
        return {
            "metrics": {
                "max_drawdown": self.risk_metrics.max_drawdown,
                "volatility": self.risk_metrics.volatility,
                "sharpe_ratio": self.risk_metrics.sharpe_ratio,
                "sortino_ratio": self.risk_metrics.sortino_ratio,
                "var_95": self.risk_metrics.var_95,
                "exposure": self.risk_metrics.exposure,
                "leverage": self.risk_metrics.leverage,
                "profit_factor": self.risk_metrics.profit_factor,
                "win_rate": self.risk_metrics.win_rate
            },
            "limits": {
                "exposure": self._calculate_current_exposure(),
                "max_allowed": self.config['limits']['max_exposure'],
                "daily_pnl": self._calculate_daily_pnl(),
                "max_daily_loss": self.config['limits']['max_daily_loss']
            },
            "timestamp": datetime.now().isoformat()
        }

# Exemple d'utilisation
if __name__ == "__main__":
    # Création du gestionnaire
    risk_manager = RiskManager()
    
    # Exemple d'equity curve
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    equity = pd.Series(np.random.normal(0, 0.01, len(dates)).cumsum(), index=dates)
    
    try:
        # Mise à jour des métriques
        metrics = risk_manager.update_metrics(equity)
        
        # Vérification d'un ordre
        allowed, reason = risk_manager.check_order_risk(
            symbol="BTC/USDT",
            side="buy",
            amount=0.1,
            price=50000
        )
        
        print("Ordre autorisé:", allowed)
        print("Raison:", reason)
        
        # Affichage du rapport
        print("\nRapport de risque:")
        print(json.dumps(risk_manager.get_risk_report(), indent=2))
        
    except Exception as e:
        print(f"Erreur: {e}") 