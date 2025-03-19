"""
Gestionnaire de risques dynamique pour l'ajustement des paramètres de trading.

Ce module implémente un gestionnaire de risques sophistiqué qui :
- Calcule et surveille les métriques de risque en temps réel
- Détecte les régimes de marché en utilisant des modèles de Markov
- Ajuste dynamiquement les limites de position
- Effectue des stress tests avec simulation Monte Carlo
- Gère les risques de liquidité et de contrepartie
- Implémente des stratégies de couverture adaptatives

Classes:
    RiskConfig: Configuration du gestionnaire de risques
    DynamicRiskManager: Gestionnaire de risques principal
    
Exemple d'utilisation:
    ```python
    # Configuration
    config = RiskConfig(
        max_position_size=100000,
        max_drawdown=0.2,
        var_calculation_method='monte_carlo'
    )
    
    # Initialisation
    risk_manager = DynamicRiskManager(config)
    
    # Mise à jour avec nouvelles données
    await risk_manager.update_risk_metrics(market_data)
    
    # Récupération des métriques
    metrics = risk_manager.get_risk_metrics()
    limits = risk_manager.get_current_limits()
    ```
"""

from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy import stats
import hmmlearn.hmm as hmm
from sklearn.preprocessing import StandardScaler
import warnings
import json
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from trading.utils.logging import TradingLogger
from trading.core.data_types import MarketData
from trading.utils.validation import (
    validate_positive,
    validate_probability,
    validate_market_data
)
from trading.utils.metrics import (
    calculate_drawdown,
    calculate_var_es,
    calculate_liquidity_metrics
)
from trading.utils.time_series import (
    detect_outliers,
    handle_missing_values,
    detect_regime_hmm
)

@dataclass
class RiskConfig:
    """Configuration du gestionnaire de risques.
    
    Attributes:
        max_position_size (float): Taille maximale de position autorisée
        max_drawdown (float): Drawdown maximal acceptable
        volatility_window (int): Fenêtre pour le calcul de la volatilité
        risk_free_rate (float): Taux sans risque annualisé
        confidence_level (float): Niveau de confiance pour VaR/ES
        max_leverage (float): Levier maximal autorisé
        position_sizing_method (str): Méthode de sizing ('kelly', 'fixed', 'volatility')
        var_calculation_method (str): Méthode de calcul VaR
        regime_detection_method (str): Méthode de détection de régime
        stress_test_scenarios (int): Nombre de scénarios pour stress tests
        min_trading_volume (float): Volume minimal pour trading
        rebalancing_interval (timedelta): Intervalle entre mises à jour
        metrics_history_size (int): Taille de l'historique des métriques
        alert_thresholds (Dict[str, float]): Seuils pour les alertes
    """
    
    # Paramètres de base
    max_position_size: float
    max_drawdown: float
    
    # Paramètres de calcul
    volatility_window: int = 20
    risk_free_rate: float = 0.02
    confidence_level: float = 0.95
    max_leverage: float = 3.0
    
    # Méthodes de calcul
    position_sizing_method: str = 'kelly'
    var_calculation_method: str = 'historical'  # 'historical', 'parametric', 'monte_carlo'
    regime_detection_method: str = 'hmm'  # 'hmm', 'threshold', 'clustering'
    
    # Paramètres de stress test
    stress_test_scenarios: int = 1000
    stress_test_confidence: float = 0.99
    extreme_scenarios_ratio: float = 0.25
    
    # Paramètres de liquidité
    min_trading_volume: float = 1000
    max_position_volume_ratio: float = 0.1
    slippage_threshold: float = 0.02
    
    # Paramètres de mise à jour
    rebalancing_interval: timedelta = field(default_factory=lambda: timedelta(hours=1))
    metrics_history_size: int = 100
    
    # Paramètres de notification
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'drawdown': 0.1,
        'var': 0.05,
        'volatility': 0.3
    })
    
    def __post_init__(self):
        """Valide les paramètres de configuration."""
        self._validate_parameters()
        
    def _validate_parameters(self):
        """Valide tous les paramètres de configuration."""
        # Validation des valeurs positives
        for param, value in {
            'max_position_size': self.max_position_size,
            'max_drawdown': self.max_drawdown,
            'volatility_window': self.volatility_window,
            'risk_free_rate': self.risk_free_rate,
            'max_leverage': self.max_leverage,
            'stress_test_scenarios': self.stress_test_scenarios,
            'min_trading_volume': self.min_trading_volume
        }.items():
            validate_positive(value, param)
            
        # Validation des probabilités
        for param, value in {
            'confidence_level': self.confidence_level,
            'stress_test_confidence': self.stress_test_confidence,
            'extreme_scenarios_ratio': self.extreme_scenarios_ratio,
            'max_position_volume_ratio': self.max_position_volume_ratio
        }.items():
            validate_probability(value, param)
            
        # Validation des méthodes
        if self.position_sizing_method not in ['kelly', 'fixed', 'volatility']:
            raise ValueError(f"Méthode de sizing invalide: {self.position_sizing_method}")
            
        if self.var_calculation_method not in ['historical', 'parametric', 'monte_carlo']:
            raise ValueError(f"Méthode de VaR invalide: {self.var_calculation_method}")
            
        if self.regime_detection_method not in ['hmm', 'threshold', 'clustering']:
            raise ValueError(f"Méthode de détection de régime invalide: {self.regime_detection_method}")
            
        # Validation des seuils d'alerte
        for metric, threshold in self.alert_thresholds.items():
            validate_positive(threshold, f"alert_threshold_{metric}")

class DynamicRiskManager:
    """
    Gestionnaire de risques dynamique qui ajuste les paramètres
    en fonction des conditions de marché.
    
    Attributes:
        config (RiskConfig): Configuration du gestionnaire
        logger (TradingLogger): Logger personnalisé
        
    Methods:
        update_risk_metrics: Met à jour les métriques de risque
        get_current_limits: Retourne les limites de position actuelles
        get_risk_metrics: Retourne les métriques de risque actuelles
        get_market_regime: Retourne le régime de marché actuel
        get_stress_test_results: Retourne les résultats des stress tests
        get_active_alerts: Retourne les alertes actives
        get_metrics_history: Retourne l'historique des métriques
    """
    
    def __init__(self, config: RiskConfig):
        """
        Initialise le gestionnaire de risques.
        
        Args:
            config: Configuration du gestionnaire de risques
            
        Raises:
            ValueError: Si la configuration est invalide
        """
        self.config = config
        self.logger = TradingLogger()
        
        # Métriques et états
        self._risk_metrics: Dict[str, float] = {}
        self._position_limits: Dict[str, float] = {}
        self._market_regime: Optional[str] = None
        self._last_update: Optional[datetime] = None
        self._stress_test_results: Dict[str, float] = {}
        
        # Historique des métriques
        self._metrics_history: Dict[str, List[float]] = {
            'volatility': [],
            'var': [],
            'expected_shortfall': [],
            'max_drawdown': [],
            'sharpe_ratio': [],
            'sortino_ratio': [],
            'calmar_ratio': [],
            'information_ratio': []
        }
        
        # Modèles
        self._regime_model = self._initialize_regime_model()
        self._scaler = StandardScaler()
        
        # État de liquidité
        self._liquidity_state: Dict[str, float] = {}
        
        # Alertes actives
        self._active_alerts: List[Dict[str, Any]] = []
        
        # Pool de threads pour les calculs parallèles
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        
    def _initialize_regime_model(self) -> Optional[hmm.GaussianHMM]:
        """
        Initialise le modèle de détection de régime.
        
        Returns:
            Optional[hmm.GaussianHMM]: Modèle HMM initialisé ou None
        """
        if self.config.regime_detection_method == 'hmm':
            return hmm.GaussianHMM(
                n_components=3,  # Bearish, Neutral, Bullish
                covariance_type="full",
                n_iter=100
            )
        return None
        
    async def update_risk_metrics(self, market_data: MarketData) -> None:
        """
        Met à jour les métriques de risque.
        
        Args:
            market_data: Données de marché actuelles
            
        Raises:
            ValueError: Si les données sont invalides
        """
        try:
            # Validation des données
            validate_market_data(market_data)
            current_time = datetime.now()
            
            # Vérification de l'intervalle de mise à jour
            if (self._last_update and 
                current_time - self._last_update < self.config.rebalancing_interval):
                return
                
            # Calculs parallèles
            tasks = [
                self._calculate_risk_metrics(market_data),
                self._detect_market_regime(market_data),
                self._analyze_liquidity(market_data),
                self._run_stress_tests(market_data)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Mise à jour des états
            self._risk_metrics = results[0]
            self._market_regime = results[1]
            self._liquidity_state = results[2]
            self._stress_test_results = results[3]
            
            # Mise à jour de l'historique
            self._update_metrics_history()
            
            # Mise à jour des limites de position
            self._position_limits = await self._calculate_position_limits()
            
            # Vérification des alertes
            await self._check_alerts()
            
            self._last_update = current_time
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la mise à jour des métriques de risque: {e}")
            raise
            
    def _update_metrics_history(self):
        """Met à jour l'historique des métriques."""
        for metric, value in self._risk_metrics.items():
            if metric in self._metrics_history:
                self._metrics_history[metric].append(value)
                if len(self._metrics_history[metric]) > self.config.metrics_history_size:
                    self._metrics_history[metric].pop(0)
                    
    async def _analyze_liquidity(self, market_data: MarketData) -> Dict[str, float]:
        """
        Analyse la liquidité du marché.
        
        Returns:
            Dict contenant les métriques de liquidité
        """
        volume_ma = np.mean(market_data.volume[-20:])
        price_impact = np.abs(market_data.high - market_data.low) / market_data.volume
        
        return {
            'avg_volume': volume_ma,
            'price_impact': np.mean(price_impact),
            'bid_ask_spread': np.mean(market_data.high - market_data.low) / market_data.close[-1],
            'volume_ratio': market_data.volume[-1] / volume_ma
        }
        
    async def _check_alerts(self):
        """Vérifie et génère les alertes si nécessaire."""
        for metric, threshold in self.config.alert_thresholds.items():
            if metric in self._risk_metrics:
                current_value = self._risk_metrics[metric]
                if abs(current_value) > threshold:
                    alert = {
                        'timestamp': datetime.now(),
                        'metric': metric,
                        'value': current_value,
                        'threshold': threshold
                    }
                    self._active_alerts.append(alert)
                    self.logger.warning(
                        f"Alerte: {metric} = {current_value:.4f} > {threshold:.4f}"
                    )
                    
    async def _calculate_risk_metrics(self, market_data: MarketData) -> Dict[str, float]:
        """
        Calcule les métriques de risque principales.
        
        Returns:
            Dict contenant les métriques de risque
        """
        returns = self._calculate_returns(market_data.close)
        
        # Calcul VaR et ES selon la méthode configurée
        var, es = calculate_var_es(
            returns,
            method=self.config.var_calculation_method,
            confidence_level=self.config.confidence_level
        )
        
        metrics = {
            'volatility': self._calculate_volatility(returns),
            'var': var,
            'expected_shortfall': es,
            'beta': self._calculate_beta(returns, market_data),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': calculate_drawdown(market_data.close)
        }
        
        # Métriques de liquidité
        metrics.update(await self._analyze_liquidity(market_data))
        
        return metrics
        
    async def _detect_market_regime(self, market_data: MarketData) -> str:
        """
        Détecte le régime de marché actuel.
        
        Returns:
            str: Le régime de marché détecté
        """
        if self.config.regime_detection_method == 'hmm':
            return await self._detect_regime_hmm(market_data)
        elif self.config.regime_detection_method == 'threshold':
            return await self._detect_regime_threshold(market_data)
        else:
            return await self._detect_regime_clustering(market_data)
            
    async def _detect_regime_hmm(self, market_data: MarketData) -> str:
        """Détecte le régime avec un modèle HMM."""
        returns = self._calculate_returns(market_data.close)
        features = np.column_stack([
            returns,
            self._calculate_volatility(returns),
            market_data.volume[-len(returns):] / np.mean(market_data.volume)
        ])
        
        # Normalisation
        features_scaled = self._scaler.fit_transform(features)
        
        # Entraînement et prédiction
        self._regime_model.fit(features_scaled)
        regime = self._regime_model.predict(features_scaled)[-1]
        
        # Mapping des états
        regime_map = {0: 'bearish', 1: 'neutral', 2: 'bullish'}
        base_regime = regime_map[regime]
        
        # Ajout du niveau de risque
        volatility = self._risk_metrics['volatility']
        if volatility > np.percentile(self._metrics_history['volatility'], 75):
            return f'high_risk_{base_regime}'
        elif volatility < np.percentile(self._metrics_history['volatility'], 25):
            return f'low_risk_{base_regime}'
        else:
            return base_regime
            
    async def _calculate_position_limits(self) -> Dict[str, float]:
        """
        Calcule les limites de position basées sur le risque.
        
        Returns:
            Dict contenant les limites de position
        """
        base_limit = self.config.max_position_size
        
        # Ajustement selon le régime de marché
        regime_adjustments = {
            'high_risk_bearish': 0.5,
            'high_risk_bullish': 0.7,
            'bearish': 0.6,
            'bullish': 0.9,
            'low_risk_bullish': 1.0,
            'low_risk_bearish': 0.8,
            'neutral': 0.9
        }
        
        regime_factor = regime_adjustments.get(self._market_regime, 0.5)
        
        # Ajustement selon les métriques de risque
        risk_factor = min(
            1.0,
            (self.config.max_drawdown / self._risk_metrics['max_drawdown'])
            if self._risk_metrics['max_drawdown'] > 0 else 1.0
        )
        
        # Ajustement selon la liquidité
        liquidity_factor = min(
            1.0,
            self._liquidity_state['avg_volume'] / self.config.min_trading_volume
        )
        
        # Calcul de la taille optimale selon la méthode configurée
        if self.config.position_sizing_method == 'kelly':
            position_limit = base_limit * self._calculate_kelly_fraction()
        elif self.config.position_sizing_method == 'volatility':
            vol_factor = 1.0 / max(self._risk_metrics['volatility'], 0.01)
            position_limit = base_limit * min(vol_factor, 1.0)
        else:  # 'fixed'
            position_limit = base_limit
        
        # Application des facteurs d'ajustement
        adjusted_limit = position_limit * regime_factor * risk_factor * liquidity_factor
        
        return {
            'max_position': min(adjusted_limit, base_limit),
            'max_leverage': min(
                self.config.max_leverage,
                self.config.max_leverage * risk_factor
            ),
            'liquidity_factor': liquidity_factor,
            'risk_factor': risk_factor,
            'regime_factor': regime_factor
        }
        
    async def _run_stress_tests(self, market_data: MarketData) -> Dict[str, float]:
        """
        Exécute des tests de stress sur le portefeuille.
        
        Returns:
            Dict contenant les résultats des stress tests
        """
        results = {
            'var_99': 0.0,
            'expected_shortfall_99': 0.0,
            'max_drawdown_stress': 0.0,
            'worst_case_loss': 0.0,
            'liquidity_stress': 0.0
        }
        
        try:
            returns = self._calculate_returns(market_data.close)
            
            # Génération de scénarios Monte Carlo
            scenarios = await self._generate_monte_carlo_scenarios(
                returns,
                n_scenarios=self.config.stress_test_scenarios
            )
            
            # Calcul des métriques de stress
            results['var_99'] = np.percentile(scenarios, 1)
            results['expected_shortfall_99'] = np.mean(
                scenarios[scenarios < results['var_99']]
            )
            
            # Simulation de drawdown
            for scenario in scenarios:
                scenario_prices = market_data.close[-1] * np.exp(np.cumsum(scenario))
                drawdown = calculate_drawdown(scenario_prices)
                results['max_drawdown_stress'] = max(
                    results['max_drawdown_stress'],
                    drawdown
                )
            
            # Simulation de stress de liquidité
            volume_scenarios = np.random.lognormal(
                np.log(market_data.volume).mean(),
                np.log(market_data.volume).std(),
                size=len(scenarios)
            )
            results['liquidity_stress'] = np.percentile(
                volume_scenarios,
                1
            ) / np.mean(market_data.volume)
            
            results['worst_case_loss'] = np.min(scenarios)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erreur lors des stress tests: {e}")
            return results
            
    async def _generate_monte_carlo_scenarios(
        self,
        returns: np.ndarray,
        n_scenarios: int
    ) -> np.ndarray:
        """
        Génère des scénarios Monte Carlo.
        
        Args:
            returns: Historique des rendements
            n_scenarios: Nombre de scénarios à générer
            
        Returns:
            np.ndarray: Scénarios générés
        """
        # Paramètres du modèle
        mu = np.mean(returns)
        sigma = np.std(returns)
        
        # Scénarios normaux
        normal_scenarios = np.random.normal(
            loc=mu,
            scale=sigma,
            size=(n_scenarios, len(returns))
        )
        
        # Scénarios extrêmes
        n_extreme = int(n_scenarios * self.config.extreme_scenarios_ratio)
        extreme_scenarios = np.random.normal(
            loc=mu,
            scale=sigma * 3,  # Plus volatile
            size=(n_extreme, len(returns))
        )
        
        # Combinaison des scénarios
        all_scenarios = np.vstack([normal_scenarios, extreme_scenarios])
        
        return all_scenarios
        
    def get_current_limits(self) -> Dict[str, float]:
        """Retourne les limites de position actuelles."""
        return self._position_limits
        
    def get_risk_metrics(self) -> Dict[str, float]:
        """Retourne les métriques de risque actuelles."""
        return self._risk_metrics
        
    def get_market_regime(self) -> str:
        """Retourne le régime de marché actuel."""
        return self._market_regime
        
    def get_stress_test_results(self) -> Dict[str, float]:
        """Retourne les résultats des derniers stress tests."""
        return self._stress_test_results
        
    def get_active_alerts(self) -> List[Dict]:
        """Retourne les alertes actives."""
        return self._active_alerts
        
    def get_metrics_history(self) -> Dict[str, List[float]]:
        """Retourne l'historique des métriques."""
        return self._metrics_history
        
    def save_state(self, path: str):
        """
        Sauvegarde l'état du gestionnaire de risques.
        
        Args:
            path: Chemin de sauvegarde
        """
        state = {
            'risk_metrics': self._risk_metrics,
            'position_limits': self._position_limits,
            'market_regime': self._market_regime,
            'metrics_history': self._metrics_history,
            'active_alerts': self._active_alerts,
            'last_update': self._last_update.isoformat() if self._last_update else None
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
            
    def load_state(self, path: str):
        """
        Charge l'état du gestionnaire de risques.
        
        Args:
            path: Chemin du fichier d'état
        """
        with open(path, 'r') as f:
            state = json.load(f)
            
        self._risk_metrics = state['risk_metrics']
        self._position_limits = state['position_limits']
        self._market_regime = state['market_regime']
        self._metrics_history = state['metrics_history']
        self._active_alerts = state['active_alerts']
        self._last_update = datetime.fromisoformat(state['last_update']) if state['last_update'] else None