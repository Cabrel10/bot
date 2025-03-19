# Documentation Complète du Trading Bot

## Table des Matières
1. [Introduction](#introduction)
2. [Architecture du Projet](#architecture)
3. [Configuration](#configuration)
4. [Composants Principaux](#composants)
5. [Guide d'Installation](#installation)
6. [Guide d'Utilisation](#utilisation)
7. [API Reference](#api)

## Introduction <a name="introduction"></a>

Le Trading Bot est une plateforme complète de trading algorithmique qui combine plusieurs approches :
- Trading basé sur les indicateurs techniques
- Apprentissage automatique (ML) et Deep Learning
- Algorithmes génétiques pour l'optimisation
- Analyse en temps réel des marchés

### Objectifs du Projet
- Automatiser les stratégies de trading
- Optimiser les performances via ML
- Fournir une interface unifiée
- Assurer une gestion des risques robuste

## Architecture du Projet <a name="architecture"></a>

### Structure des Dossiers
```
trading/
├── core/                  # Logique métier principale
│   ├── data/             # Gestion des données
│   ├── models/           # Modèles de trading
│   └── strategies/       # Stratégies de trading
├── exchange/             # Intégration des exchanges
│   ├── binance/         # Client Binance
│   └── bitget/          # Client Bitget
├── plugins/              # Extensions et indicateurs
└── config/              # Configuration
```

### Fichiers de Configuration Essentiels
- `config/execution_config.yaml`: Configuration d'exécution
- `config/order_config.yaml`: Paramètres des ordres
- `config/position_config.yaml`: Gestion des positions
- `config/risk_config.yaml`: Gestion des risques

## Configuration <a name="configuration"></a>

### Configuration de l'Exécution
```yaml
execution:
  max_open_trades: 3
  stake_amount: 100
  dry_run: true
  exchange:
    name: "binance"
    key: ""
    secret: ""
```

### Configuration des Ordres
```yaml
order:
  max_retries: 3
  retry_delay: 10
  default_type: "LIMIT"
  time_in_force: "GTC"
```

### Configuration des Positions
```yaml
position:
  max_open_positions: 5
  leverage: 1
  margin_type: "ISOLATED"
```

## Composants Principaux <a name="composants"></a>

### 1. Gestion des Données
- Module: `trading/core/data/`
- Fonctions principales:
  - Collecte de données historiques
  - Prétraitement des données
  - Validation et nettoyage

### 2. Stratégies de Trading
- Module: `trading/core/strategies/`
- Types de stratégies:
  - Moyennes mobiles croisées
  - RSI avec MACD
  - Analyse de tendance

### 3. Gestion des Risques
- Module: `trading/core/risk/`
- Fonctionnalités:
  - Stop-loss dynamique
  - Take-profit adaptatif
  - Gestion de l'exposition

## Guide d'Installation <a name="installation"></a>

1. Prérequis
```bash
Python 3.8+
conda
```

2. Installation
```bash
# Activer l'environnement virtuel
conda activate trading_env

# Installer dépendances
pip install -r requirements.txt
```

## Guide d'Utilisation <a name="utilisation"></a>

### 1. Configuration Initiale
1. Copier `.env.example` vers `.env`
2. Configurer les clés API
3. Ajuster les paramètres dans `config/`

### 2. Lancement du Bot
```bash
python run_app.py
```

### 3. Interface Web
- URL: http://localhost:8501
- Sections:
  - Dashboard principal
  - Configuration
  - Monitoring
  - Backtesting

## API Reference <a name="api"></a>

### Client Exchange
```python
from trading.exchange.bitget import BitgetClient

client = BitgetClient(api_key="", secret="")
```

### Stratégies
```python
from trading.core.strategies import MACrossStrategy

strategy = MACrossStrategy(
    fast_period=12,
    slow_period=26
)
```

### Indicateurs
```python
from trading.plugins.indicators import RSI, MACD

rsi = RSI(period=14)
macd = MACD(fast=12, slow=26, signal=9)
```