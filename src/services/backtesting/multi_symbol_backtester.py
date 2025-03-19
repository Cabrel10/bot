"""
Backtester avancé avec support multi-symboles et ordres limites.
"""

from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from ...models.hybrid_model.model import HybridModel
from ...core.data_types import OrderType, OrderSide, TimeFrame

@dataclass
class BacktestConfig:
    """Configuration du backtest."""
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    initial_balance: float = 10000.0
    commission_rate: float = 0.001
    slippage: float = 0.0001
    use_limit_orders: bool = True
    max_positions: int = 3
    risk_per_trade: float = 0.02
    timeframe: TimeFrame = TimeFrame.HOUR_1

@dataclass
class BacktestResult:
    """Résultats du backtest."""
    trades: pd.DataFrame
    metrics: Dict[str, float]
    equity_curve: pd.DataFrame
    positions: pd.DataFrame
    orders: pd.DataFrame
    symbol_performance: Dict[str, Dict[str, float]]

class MultiSymbolBacktester:
    """Backtester avancé avec support multi-symboles."""
    
    def __init__(self, config: BacktestConfig):
        """Initialisation du backtester."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure la journalisation."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def run(self, model: HybridModel, data: Dict[str, pd.DataFrame]) -> BacktestResult:
        """
        Exécute le backtest sur plusieurs symboles.
        
        Args:
            model: Modèle hybride à tester
            data: Dictionnaire {symbole: DataFrame} des données historiques
        
        Returns:
            BacktestResult: Résultats du backtest
        """
        self.logger.info(f"Démarrage du backtest sur {len(data)} symboles")
        
        # Validation des données
        self._validate_data(data)
        
        # Initialisation des structures de données
        trades = []
        positions = []
        orders = []
        equity_curve = []
        
        # État du portfolio
        portfolio = {
            "cash": self.config.initial_balance,
            "positions": {},
            "equity": self.config.initial_balance
        }
        
        # Boucle principale du backtest
        with ThreadPoolExecutor() as executor:
            for current_time in self._get_timepoints(data):
                # 1. Mise à jour des données de marché
                market_data = self._get_market_data(data, current_time)
                
                # 2. Mise à jour des positions existantes
                self._update_positions(portfolio, market_data)
                
                # 3. Génération des signaux pour chaque symbole
                signals = self._generate_signals(model, market_data, executor)
                
                # 4. Exécution des ordres
                new_orders = self._execute_orders(signals, portfolio, market_data)
                orders.extend(new_orders)
                
                # 5. Mise à jour de l'equity curve
                equity = self._calculate_equity(portfolio, market_data)
                equity_curve.append({
                    "timestamp": current_time,
                    "equity": equity,
                    "cash": portfolio["cash"]
                })
                
                # 6. Journalisation des positions fermées
                closed_trades = self._log_closed_trades(portfolio)
                trades.extend(closed_trades)
                
                # 7. Sauvegarde de l'état des positions
                positions.append(self._get_positions_snapshot(portfolio, current_time))
        
        # Compilation des résultats
        results = BacktestResult(
            trades=pd.DataFrame(trades),
            metrics=self._calculate_metrics(trades, equity_curve),
            equity_curve=pd.DataFrame(equity_curve),
            positions=pd.DataFrame(positions),
            orders=pd.DataFrame(orders),
            symbol_performance=self._calculate_symbol_performance(trades)
        )
        
        self.logger.info("Backtest terminé")
        return results
    
    def _validate_data(self, data: Dict[str, pd.DataFrame]):
        """Valide les données d'entrée."""
        required_columns = ["open", "high", "low", "close", "volume"]
        for symbol, df in data.items():
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Colonnes manquantes pour {symbol}: {missing}")
    
    def _get_timepoints(self, data: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        """Retourne une liste unifiée des points temporels."""
        all_times = pd.concat([df.index for df in data.values()])
        return all_times.unique().sort_values()
    
    def _get_market_data(self, data: Dict[str, pd.DataFrame], 
                        timestamp) -> Dict[str, pd.Series]:
        """Récupère les données de marché pour un instant donné."""
        return {
            symbol: df.loc[timestamp] if timestamp in df.index else None
            for symbol, df in data.items()
        }
    
    def _generate_signals(self, model: HybridModel, market_data: Dict[str, pd.Series],
                         executor) -> Dict[str, Dict]:
        """Génère les signaux de trading en parallèle."""
        futures = {
            symbol: executor.submit(model.predict, data)
            for symbol, data in market_data.items()
            if data is not None
        }
        return {
            symbol: future.result()
            for symbol, future in futures.items()
        }
    
    def _execute_orders(self, signals: Dict[str, Dict], portfolio: Dict,
                       market_data: Dict[str, pd.Series]) -> List[Dict]:
        """Exécute les ordres basés sur les signaux."""
        orders = []
        for symbol, signal in signals.items():
            if self._can_trade(portfolio, symbol):
                order = self._create_order(symbol, signal, portfolio, market_data[symbol])
                if order:
                    self._apply_order(order, portfolio)
                    orders.append(order)
        return orders
    
    def _can_trade(self, portfolio: Dict, symbol: str) -> bool:
        """Vérifie si un nouveau trade est possible."""
        current_positions = len(portfolio["positions"])
        return (current_positions < self.config.max_positions and
                symbol not in portfolio["positions"])
    
    def _create_order(self, symbol: str, signal: Dict, portfolio: Dict,
                     market_data: pd.Series) -> Optional[Dict]:
        """Crée un ordre basé sur le signal."""
        if not signal or not market_data:
            return None
            
        direction = signal.get("direction", 0)
        if abs(direction) < 0.5:  # Seuil minimal pour un signal
            return None
            
        # Calcul de la taille de position
        position_size = self._calculate_position_size(
            portfolio["cash"],
            market_data["close"],
            self.config.risk_per_trade
        )
        
        return {
            "symbol": symbol,
            "type": OrderType.LIMIT if self.config.use_limit_orders else OrderType.MARKET,
            "side": OrderSide.BUY if direction > 0 else OrderSide.SELL,
            "quantity": position_size,
            "price": market_data["close"],
            "timestamp": market_data.name,
            "status": "pending"
        }
    
    def _apply_order(self, order: Dict, portfolio: Dict):
        """Applique un ordre au portfolio."""
        cost = order["quantity"] * order["price"]
        commission = cost * self.config.commission_rate
        
        if order["side"] == OrderSide.BUY:
            portfolio["cash"] -= (cost + commission)
            portfolio["positions"][order["symbol"]] = {
                "quantity": order["quantity"],
                "entry_price": order["price"],
                "entry_time": order["timestamp"]
            }
        else:
            portfolio["cash"] += (cost - commission)
            if order["symbol"] in portfolio["positions"]:
                del portfolio["positions"][order["symbol"]]
    
    def _calculate_position_size(self, cash: float, price: float,
                               risk_ratio: float) -> float:
        """Calcule la taille de position optimale."""
        return (cash * risk_ratio) / price
    
    def _update_positions(self, portfolio: Dict, market_data: Dict[str, pd.Series]):
        """Met à jour les positions existantes."""
        for symbol, position in list(portfolio["positions"].items()):
            if symbol in market_data and market_data[symbol] is not None:
                current_price = market_data[symbol]["close"]
                # Mise à jour de la valeur de la position
                position["current_value"] = position["quantity"] * current_price
                position["unrealized_pnl"] = (
                    position["current_value"] -
                    (position["quantity"] * position["entry_price"])
                )
    
    def _calculate_equity(self, portfolio: Dict, market_data: Dict[str, pd.Series]) -> float:
        """Calcule l'equity totale."""
        positions_value = sum(
            pos["current_value"]
            for pos in portfolio["positions"].values()
        )
        return portfolio["cash"] + positions_value
    
    def _log_closed_trades(self, portfolio: Dict) -> List[Dict]:
        """Enregistre les trades fermés."""
        closed_trades = []
        for symbol, position in list(portfolio["positions"].items()):
            if "exit_price" in position:
                trade = {
                    "symbol": symbol,
                    "entry_time": position["entry_time"],
                    "exit_time": position["exit_time"],
                    "entry_price": position["entry_price"],
                    "exit_price": position["exit_price"],
                    "quantity": position["quantity"],
                    "pnl": position["realized_pnl"],
                    "return": position["return"]
                }
                closed_trades.append(trade)
                del portfolio["positions"][symbol]
        return closed_trades
    
    def _get_positions_snapshot(self, portfolio: Dict, timestamp) -> Dict:
        """Crée un snapshot des positions actuelles."""
        return {
            "timestamp": timestamp,
            "positions": portfolio["positions"].copy(),
            "cash": portfolio["cash"],
            "equity": self._calculate_equity(portfolio, {})
        }
    
    def _calculate_metrics(self, trades: List[Dict],
                         equity_curve: List[Dict]) -> Dict[str, float]:
        """Calcule les métriques de performance."""
        if not trades or not equity_curve:
            return {}
            
        df_trades = pd.DataFrame(trades)
        df_equity = pd.DataFrame(equity_curve)
        
        returns = df_equity["equity"].pct_change().dropna()
        
        return {
            "total_trades": len(trades),
            "winning_trades": len(df_trades[df_trades["pnl"] > 0]),
            "profit_factor": abs(df_trades[df_trades["pnl"] > 0]["pnl"].sum() /
                               df_trades[df_trades["pnl"] < 0]["pnl"].sum()),
            "sharpe_ratio": self._calculate_sharpe_ratio(returns),
            "max_drawdown": self._calculate_max_drawdown(df_equity["equity"]),
            "total_return": (df_equity["equity"].iloc[-1] / 
                           df_equity["equity"].iloc[0] - 1) * 100
        }
    
    def _calculate_symbol_performance(self, trades: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Calcule les performances par symbole."""
        df_trades = pd.DataFrame(trades)
        performance = {}
        
        for symbol in df_trades["symbol"].unique():
            symbol_trades = df_trades[df_trades["symbol"] == symbol]
            performance[symbol] = {
                "total_trades": len(symbol_trades),
                "winning_trades": len(symbol_trades[symbol_trades["pnl"] > 0]),
                "total_pnl": symbol_trades["pnl"].sum(),
                "avg_return": symbol_trades["return"].mean()
            }
            
        return performance
    
    @staticmethod
    def _calculate_sharpe_ratio(returns: pd.Series) -> float:
        """Calcule le ratio de Sharpe."""
        if len(returns) < 2:
            return 0.0
        return np.sqrt(252) * (returns.mean() / returns.std())
    
    @staticmethod
    def _calculate_max_drawdown(equity: pd.Series) -> float:
        """Calcule le drawdown maximum."""
        peak = equity.expanding(min_periods=1).max()
        drawdown = (equity - peak) / peak
        return drawdown.min() * 100 