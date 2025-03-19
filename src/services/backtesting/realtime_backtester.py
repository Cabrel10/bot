"""
Module de backtesting en temps réel avec support des flux de données en direct.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from ...models.hybrid_model.model import HybridModel
from ...core.data_types import OrderType, OrderSide, TimeFrame
from ...core.exchanges.base import BaseExchange
from .multi_symbol_backtester import MultiSymbolBacktester, BacktestConfig, BacktestResult

@dataclass
class RealtimeConfig(BacktestConfig):
    """Configuration du backtest en temps réel."""
    update_interval: int = 60  # Intervalle de mise à jour en secondes
    buffer_size: int = 1000    # Nombre de points de données à conserver
    max_latency: float = 1.0   # Latence maximale acceptable en secondes
    use_websocket: bool = True # Utiliser WebSocket pour les mises à jour
    save_history: bool = True  # Sauvegarder l'historique des trades

class RealtimeBacktester(MultiSymbolBacktester):
    """Backtester en temps réel avec support des flux de données en direct."""
    
    def __init__(self, config: RealtimeConfig, exchange: BaseExchange):
        """Initialisation du backtester en temps réel."""
        super().__init__(config)
        self.exchange = exchange
        self.config = config
        self.data_buffer = {}
        self.last_update = {}
        self.is_running = False
        self._setup_realtime_logging()
        
    def _setup_realtime_logging(self):
        """Configure la journalisation en temps réel."""
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    async def start(self, model: HybridModel):
        """Démarre le backtest en temps réel."""
        self.is_running = True
        self.logger.info("Démarrage du backtest en temps réel")
        
        # Initialisation des buffers de données
        for symbol in self.config.symbols:
            self.data_buffer[symbol] = []
            self.last_update[symbol] = datetime.now()
            
        # Démarrage des tâches asynchrones
        tasks = [
            self._update_market_data(),
            self._process_signals(model),
            self._monitor_performance()
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Erreur lors du backtest en temps réel: {e}")
            self.stop()
    
    def stop(self):
        """Arrête le backtest en temps réel."""
        self.is_running = False
        self.logger.info("Arrêt du backtest en temps réel")
        
        # Sauvegarde des résultats si configuré
        if self.config.save_history:
            self._save_results()
    
    async def _update_market_data(self):
        """Met à jour les données de marché en temps réel."""
        while self.is_running:
            try:
                for symbol in self.config.symbols:
                    # Vérification de la latence
                    if (datetime.now() - self.last_update[symbol]).total_seconds() > self.config.max_latency:
                        self.logger.warning(f"Latence élevée pour {symbol}")
                    
                    # Récupération des nouvelles données
                    if self.config.use_websocket:
                        data = await self.exchange.get_websocket_data(symbol, self.config.timeframe)
                    else:
                        data = await self.exchange.get_klines(symbol, self.config.timeframe, limit=1)
                    
                    # Mise à jour du buffer
                    self.data_buffer[symbol].append(data)
                    if len(self.data_buffer[symbol]) > self.config.buffer_size:
                        self.data_buffer[symbol].pop(0)
                    
                    self.last_update[symbol] = datetime.now()
                
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                self.logger.error(f"Erreur lors de la mise à jour des données: {e}")
                await asyncio.sleep(1)
    
    async def _process_signals(self, model: HybridModel):
        """Traite les signaux de trading en temps réel."""
        while self.is_running:
            try:
                # Préparation des données pour le modèle
                market_data = self._prepare_market_data()
                
                # Génération des signaux
                signals = await self._generate_signals(model, market_data)
                
                # Exécution des ordres
                await self._execute_orders(signals, market_data)
                
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                self.logger.error(f"Erreur lors du traitement des signaux: {e}")
                await asyncio.sleep(1)
    
    async def _monitor_performance(self):
        """Surveille les performances en temps réel."""
        while self.is_running:
            try:
                # Calcul des métriques
                metrics = self._calculate_realtime_metrics()
                
                # Affichage des métriques
                self._display_realtime_metrics(metrics)
                
                # Vérification des conditions d'arrêt
                if self._should_stop(metrics):
                    self.logger.info("Conditions d'arrêt atteintes")
                    self.stop()
                
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                self.logger.error(f"Erreur lors du monitoring: {e}")
                await asyncio.sleep(1)
    
    def _prepare_market_data(self) -> Dict[str, pd.DataFrame]:
        """Prépare les données de marché pour le modèle."""
        market_data = {}
        for symbol in self.config.symbols:
            if self.data_buffer[symbol]:
                df = pd.DataFrame(self.data_buffer[symbol])
                df.set_index('timestamp', inplace=True)
                market_data[symbol] = df
        return market_data
    
    async def _generate_signals(self, model: HybridModel,
                              market_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Génère les signaux de trading en temps réel."""
        signals = {}
        with ThreadPoolExecutor() as executor:
            for symbol, data in market_data.items():
                if not data.empty:
                    signals[symbol] = await asyncio.get_event_loop().run_in_executor(
                        executor, model.predict, data
                    )
        return signals
    
    async def _execute_orders(self, signals: Dict[str, Dict],
                            market_data: Dict[str, pd.DataFrame]):
        """Exécute les ordres en temps réel."""
        for symbol, signal in signals.items():
            if self._can_trade(symbol):
                order = self._create_order(symbol, signal, market_data[symbol])
                if order:
                    try:
                        # Simulation de l'exécution de l'ordre
                        executed_order = await self._simulate_order_execution(order, market_data[symbol])
                        self._apply_order(executed_order)
                        self._log_trade(executed_order)
                    except Exception as e:
                        self.logger.error(f"Erreur lors de l'exécution de l'ordre: {e}")
    
    async def _simulate_order_execution(self, order: Dict,
                                     market_data: pd.DataFrame) -> Dict:
        """Simule l'exécution d'un ordre en temps réel."""
        if order["type"] == OrderType.MARKET:
            # Exécution immédiate au prix du marché
            order["executed_price"] = market_data["close"].iloc[-1]
            order["executed_time"] = datetime.now()
        else:
            # Simulation d'ordre limite
            order["executed_price"] = await self._simulate_limit_order(order, market_data)
            order["executed_time"] = datetime.now()
        
        order["status"] = "executed"
        return order
    
    async def _simulate_limit_order(self, order: Dict,
                                  market_data: pd.DataFrame) -> float:
        """Simule l'exécution d'un ordre limite."""
        # Logique de simulation d'ordre limite
        if order["side"] == OrderSide.BUY:
            return min(order["price"], market_data["high"].iloc[-1])
        else:
            return max(order["price"], market_data["low"].iloc[-1])
    
    def _calculate_realtime_metrics(self) -> Dict[str, float]:
        """Calcule les métriques en temps réel."""
        return {
            "equity": self._calculate_equity(),
            "open_positions": len(self.portfolio["positions"]),
            "total_trades": len(self.trades),
            "win_rate": self._calculate_win_rate(),
            "sharpe_ratio": self._calculate_sharpe_ratio(),
            "max_drawdown": self._calculate_max_drawdown()
        }
    
    def _display_realtime_metrics(self, metrics: Dict[str, float]):
        """Affiche les métriques en temps réel."""
        self.logger.info(f"Equity: {metrics['equity']:.2f}")
        self.logger.info(f"Positions ouvertes: {metrics['open_positions']}")
        self.logger.info(f"Win Rate: {metrics['win_rate']:.2f}%")
        self.logger.info(f"Ratio de Sharpe: {metrics['sharpe_ratio']:.2f}")
        self.logger.info(f"Drawdown Max: {metrics['max_drawdown']:.2f}%")
    
    def _should_stop(self, metrics: Dict[str, float]) -> bool:
        """Vérifie si le backtest doit être arrêté."""
        # Conditions d'arrêt personnalisables
        return (
            metrics["max_drawdown"] < -20.0 or  # Drawdown maximum de 20%
            metrics["equity"] < self.config.initial_balance * 0.5  # Perte de 50%
        )
    
    def _save_results(self):
        """Sauvegarde les résultats du backtest."""
        results = BacktestResult(
            trades=pd.DataFrame(self.trades),
            metrics=self._calculate_realtime_metrics(),
            equity_curve=pd.DataFrame(self.equity_curve),
            positions=pd.DataFrame(self.positions),
            orders=pd.DataFrame(self.orders),
            symbol_performance=self._calculate_symbol_performance(self.trades)
        )
        
        # Sauvegarde dans un fichier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results.trades.to_csv(f"results/realtime_trades_{timestamp}.csv")
        results.equity_curve.to_csv(f"results/realtime_equity_{timestamp}.csv") 