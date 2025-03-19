# Modèle Hybride de Trading

Un système de trading avancé combinant un réseau de neurones convolutifs (CNN) et un algorithme génétique (GNA) pour la prédiction et l'optimisation des stratégies de trading.

## Fonctionnalités

### Modèle Hybride
- Architecture CNN avec attention et blocs ResNet
- Algorithme génétique avec gestion des risques
- Optimisation continue des paramètres
- Adaptation aux conditions de marché

### Analyse des Données
- Prétraitement avancé des données
- Indicateurs techniques
- Analyse des sentiments
- Prédiction de volatilité

### Gestion des Risques
- Détection d'anomalies
- Détection de fraude
- Métriques de performance avancées
- Analyse des régimes de marché

### Backtesting
- Simulation en temps réel
- Support multi-symboles
- Gestion des ordres
- Métriques de performance

## Installation

1. Cloner le repository :
```bash
git clone https://github.com/votre-username/trading-hybrid-model.git
cd trading-hybrid-model
```

2. Créer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

4. Configurer les variables d'environnement :
```bash
cp .env.example .env
# Éditer .env avec vos clés API et configurations
```

## Structure du Projet

```
trading-hybrid-model/
├── src/
│   ├── models/
│   │   └── hybrid_model/
│   │       ├── hybrid_model.py
│   │       ├── anomaly_detection.py
│   │       ├── advanced_metrics.py
│   │       ├── continuous_optimization.py
│   │       ├── sentiment_volatility.py
│   │       └── tests/
│   │           └── test_hybrid_model.py
│   └── services/
│       ├── data_manager.py
│       ├── exchange_manager.py
│       └── config_manager.py
├── config/
│   ├── model_config.yaml
│   ├── trading_config.yaml
│   └── data_config.yaml
├── data/
│   ├── historical/
│   └── realtime/
├── logs/
├── tests/
├── requirements.txt
└── README.md
```

## Utilisation

### Configuration

1. Configurer les paramètres du modèle dans `config/model_config.yaml`
2. Configurer les paramètres de trading dans `config/trading_config.yaml`
3. Configurer les paramètres de données dans `config/data_config.yaml`

### Entraînement

```python
from src.models.hybrid_model.hybrid_model import HybridModel
from src.services.data_manager import DataManager
from src.services.config_manager import ConfigManager

# Charger la configuration
config = ConfigManager.load_model_config()

# Initialiser le gestionnaire de données
data_manager = DataManager()

# Initialiser le modèle
model = HybridModel(config)

# Entraîner le modèle
model.train(data_manager.get_historical_data())
```

### Prédiction

```python
# Obtenir des prédictions
predictions = model.predict(data_manager.get_latest_data())
```

### Backtesting

```python
from src.services.backtesting import BacktestingService

# Initialiser le service de backtesting
backtesting = BacktestingService(model, data_manager)

# Exécuter le backtesting
results = backtesting.run()
```

## Tests

Exécuter les tests unitaires :
```bash
pytest src/models/hybrid_model/tests/
```

## Contribution

1. Fork le repository
2. Créer une branche pour votre fonctionnalité
3. Commiter vos changements
4. Pousser vers la branche
5. Créer une Pull Request

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## Contact

Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue ou à me contacter directement.