"""
Tests unitaires pour le module de métriques de risque.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.services.backtesting.risk_metrics import RiskAnalyzer, RiskMetrics

@pytest.fixture
def sample_returns():
    """Crée une série de rendements de test."""
    return pd.Series([
        0.01, -0.02, 0.03, -0.01, 0.02,
        0.01, -0.03, 0.02, -0.01, 0.01
    ])

@pytest.fixture
def sample_trades():
    """Crée un DataFrame de trades de test."""
    return pd.DataFrame({
        "timestamp": pd.date_range(start="2024-01-01", periods=10, freq="D"),
        "symbol": ["BTC/USDT"] * 10,
        "pnl": [100, -50, 150, -30, 80, 40, -100, 60, -20, 30],
        "return": [0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.03, 0.02, -0.01, 0.01]
    })

@pytest.fixture
def risk_analyzer(sample_returns, sample_trades):
    """Crée une instance de RiskAnalyzer."""
    return RiskAnalyzer(sample_returns, sample_trades)

def test_initialization(risk_analyzer, sample_returns, sample_trades):
    """Teste l'initialisation de RiskAnalyzer."""
    assert len(risk_analyzer.returns) == len(sample_returns)
    assert len(risk_analyzer.trades) == len(sample_trades)
    assert isinstance(risk_analyzer.returns, pd.Series)
    assert isinstance(risk_analyzer.trades, pd.DataFrame)

def test_validation_empty_data():
    """Teste la validation des données vides."""
    with pytest.raises(ValueError, match="La série des rendements est vide"):
        RiskAnalyzer(pd.Series(), pd.DataFrame())
    
    with pytest.raises(ValueError, match="Le DataFrame des trades est vide"):
        RiskAnalyzer(pd.Series([1, 2, 3]), pd.DataFrame())

def test_var_calculation(risk_analyzer):
    """Teste le calcul de la Value at Risk."""
    var = risk_analyzer._calculate_var(0.95)
    assert isinstance(var, float)
    assert var <= 0  # La VaR devrait être négative pour un niveau de confiance de 95%

def test_cvar_calculation(risk_analyzer):
    """Teste le calcul de la Conditional Value at Risk."""
    cvar = risk_analyzer._calculate_cvar(0.95)
    assert isinstance(cvar, float)
    assert cvar <= risk_analyzer._calculate_var(0.95)  # CVaR devrait être inférieure à la VaR

def test_sortino_ratio(risk_analyzer):
    """Teste le calcul du ratio de Sortino."""
    ratio = risk_analyzer._calculate_sortino_ratio()
    assert isinstance(ratio, float)
    assert ratio >= 0

def test_calmar_ratio(risk_analyzer):
    """Teste le calcul du ratio de Calmar."""
    ratio = risk_analyzer._calculate_calmar_ratio()
    assert isinstance(ratio, float)
    assert ratio >= 0

def test_omega_ratio(risk_analyzer):
    """Teste le calcul du ratio d'Omega."""
    ratio = risk_analyzer._calculate_omega_ratio()
    assert isinstance(ratio, float)
    assert ratio >= 0

def test_ulcer_index(risk_analyzer):
    """Teste le calcul de l'index d'Ulcer."""
    index = risk_analyzer._calculate_ulcer_index()
    assert isinstance(index, float)
    assert index >= 0

def test_pain_index(risk_analyzer):
    """Teste le calcul de l'index de douleur."""
    index = risk_analyzer._calculate_pain_index()
    assert isinstance(index, float)
    assert index <= 0  # L'index de douleur devrait être négatif

def test_recovery_factor(risk_analyzer):
    """Teste le calcul du facteur de récupération."""
    factor = risk_analyzer._calculate_recovery_factor()
    assert isinstance(factor, float)
    assert factor >= 0

def test_risk_reward_ratio(risk_analyzer):
    """Teste le calcul du ratio risque/récompense."""
    ratio = risk_analyzer._calculate_risk_reward_ratio()
    assert isinstance(ratio, float)
    assert ratio >= 0

def test_max_drawdown(risk_analyzer):
    """Teste le calcul du drawdown maximum."""
    max_dd = risk_analyzer._calculate_max_drawdown()
    assert isinstance(max_dd, float)
    assert max_dd <= 0  # Le drawdown devrait être négatif

def test_avg_drawdown(risk_analyzer):
    """Teste le calcul du drawdown moyen."""
    avg_dd = risk_analyzer._calculate_avg_drawdown()
    assert isinstance(avg_dd, float)
    assert avg_dd <= 0  # Le drawdown moyen devrait être négatif

def test_drawdown_duration(risk_analyzer):
    """Teste le calcul de la durée des drawdowns."""
    duration = risk_analyzer._calculate_drawdown_duration()
    assert isinstance(duration, timedelta)
    assert duration.days >= 0

def test_win_rate(risk_analyzer):
    """Teste le calcul du taux de réussite."""
    win_rate = risk_analyzer._calculate_win_rate()
    assert isinstance(win_rate, float)
    assert 0 <= win_rate <= 1

def test_profit_factor(risk_analyzer):
    """Teste le calcul du facteur de profit."""
    factor = risk_analyzer._calculate_profit_factor()
    assert isinstance(factor, float)
    assert factor >= 0

def test_sharpe_ratio(risk_analyzer):
    """Teste le calcul du ratio de Sharpe."""
    ratio = risk_analyzer._calculate_sharpe_ratio()
    assert isinstance(ratio, float)

def test_volatility(risk_analyzer):
    """Teste le calcul de la volatilité."""
    vol = risk_analyzer._calculate_volatility()
    assert isinstance(vol, float)
    assert vol >= 0

def test_skewness(risk_analyzer):
    """Teste le calcul de l'asymétrie."""
    skew = risk_analyzer._calculate_skewness()
    assert isinstance(skew, float)

def test_kurtosis(risk_analyzer):
    """Teste le calcul de la kurtosis."""
    kurt = risk_analyzer._calculate_kurtosis()
    assert isinstance(kurt, float)

def test_risk_report(risk_analyzer):
    """Teste la génération du rapport de risque."""
    report = risk_analyzer.get_risk_report()
    assert isinstance(report, dict)
    assert len(report) == 18  # Nombre de métriques dans le rapport
    assert all(isinstance(value, (float, str)) for value in report.values())

def test_edge_cases(risk_analyzer):
    """Teste les cas limites."""
    # Test avec des rendements constants
    constant_returns = pd.Series([0.01] * 10)
    constant_trades = pd.DataFrame({
        "timestamp": pd.date_range(start="2024-01-01", periods=10, freq="D"),
        "symbol": ["BTC/USDT"] * 10,
        "pnl": [100] * 10,
        "return": [0.01] * 10
    })
    analyzer = RiskAnalyzer(constant_returns, constant_trades)
    
    # Vérification des métriques
    assert analyzer._calculate_volatility() == 0
    assert analyzer._calculate_skewness() == 0
    assert analyzer._calculate_kurtosis() == 0
    assert analyzer._calculate_win_rate() == 1.0
    assert analyzer._calculate_profit_factor() == np.inf 