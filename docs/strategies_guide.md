# Guide des Stratégies de Trading

## Table des Matières
1. [Introduction](#introduction)
2. [Stratégies Disponibles](#strategies)
3. [Création de Stratégies](#creation)
4. [Optimisation](#optimization)
5. [Backtesting](#backtesting)

## Introduction <a name="introduction"></a>

Les stratégies de trading sont le cœur du système. Chaque stratégie doit implémenter l'interface `BaseStrategy` et définir sa logique de trading.

### Structure de Base
```python
from trading.core.strategies import BaseStrategy

class CustomStrategy(BaseStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.required_indicators = ['rsi', 'macd']

    def analyze(self, candle):
        # Logique de trading
        return self.generate_signal()
```

## Stratégies Disponibles <a name="strategies"></a>

### 1. Moving Average Crossover (MA Cross)
```python
class MACrossStrategy(BaseStrategy):
    """
    Stratégie basée sur le croisement de moyennes mobiles
    - Achat : MA rapide croise MA lente vers le haut
    - Vente : MA rapide croise MA lente vers le bas
    """
    def __init__(self, fast_period=12, slow_period=26):
        self.fast_period = fast_period
        self.slow_period = slow_period
```

Configuration type :
```yaml
strategy:
  name: MACrossStrategy
  params:
    fast_period: 12
    slow_period: 26
    signal_period: 9
```

### 2. RSI + MACD
```python
class RSIMACDStrategy(BaseStrategy):
    """
    Stratégie combinant RSI et MACD
    - Achat : RSI < 30 et MACD croise signal vers le haut
    - Vente : RSI > 70 ou MACD croise signal vers le bas
    """
    def __init__(self, rsi_period=14, macd_fast=12, macd_slow=26):
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
```

## Création de Stratégies <a name="creation"></a>

### 1. Structure Requise
```python
class NewStrategy(BaseStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "NewStrategy"
        self.description = "Description de la stratégie"
        
    def analyze(self, candle):
        """Analyse une bougie et retourne un signal"""
        return self.generate_signal()
        
    def generate_signal(self):
        """Génère un signal de trading"""
        return {
            'type': 'BUY',  # ou 'SELL'
            'price': price,
            'volume': volume,
            'confidence': confidence
        }
```

### 2. Indicateurs Techniques
```python
from trading.plugins.indicators import RSI, MACD, BB

class TechnicalStrategy(BaseStrategy):
    def __init__(self):
        self.rsi = RSI(period=14)
        self.macd = MACD(fast=12, slow=26, signal=9)
        self.bb = BB(period=20, std_dev=2)
```

### 3. Gestion des Risques
```python
def analyze(self, candle):
    signal = self.generate_signal()
    
    # Validation du signal
    if not self.validate_signal(signal):
        return None
        
    # Gestion des risques
    signal = self.apply_risk_management(signal)
    
    return signal
```

## Optimisation <a name="optimization"></a>

### 1. Optimisation des Paramètres
```python
from trading.core.optimization import StrategyOptimizer

optimizer = StrategyOptimizer(
    strategy_class=MACrossStrategy,
    param_grid={
        'fast_period': range(5, 30),
        'slow_period': range(20, 50)
    }
)

best_params = optimizer.optimize(data)
```

### 2. Validation Croisée
```python
from trading.core.validation import CrossValidator

validator = CrossValidator(
    strategy=strategy,
    data=data,
    n_splits=5
)

scores = validator.validate()
```

## Backtesting <a name="backtesting"></a>

### 1. Configuration du Backtest
```python
from trading.core.backtest import Backtester

backtester = Backtester(
    strategy=strategy,
    data=data,
    initial_capital=10000,
    commission=0.001
)
```

### 2. Exécution et Analyse
```python
results = backtester.run()

# Analyse des résultats
print(f"Profit total : {results.total_profit}")
print(f"Sharpe Ratio : {results.sharpe_ratio}")
print(f"Max Drawdown : {results.max_drawdown}")
```

### 3. Visualisation
```python
from trading.visualization import StrategyVisualizer

visualizer = StrategyVisualizer(results)
visualizer.plot_equity_curve()
visualizer.plot_drawdown()
visualizer.plot_monthly_returns()
```

## Bonnes Pratiques

1. **Validation des Signaux**
```python
def validate_signal(self, signal):
    if signal['volume'] > self.max_position_size:
        return False
    if signal['price'] < self.min_price:
        return False
    return True
```

2. **Gestion des Erreurs**
```python
def analyze(self, candle):
    try:
        signal = self.generate_signal()
        return signal
    except Exception as e:
        self.logger.error(f"Erreur dans l'analyse : {e}")
        return None
```

3. **Documentation**
```python
class MyStrategy(BaseStrategy):
    """
    Ma Stratégie de Trading
    
    Parameters
    ----------
    param1 : int
        Description du paramètre 1
    param2 : float
        Description du paramètre 2
        
    Attributes
    ----------
    attr1 : type
        Description de l'attribut 1
    """
```