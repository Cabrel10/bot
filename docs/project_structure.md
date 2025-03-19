# Nouvelle Structure du Projet

```
trading/
├── core/                    # Composants principaux
│   ├── data/               # Gestion des données
│   ├── models/             # Modèles (NN et GA)
│   └── strategies/         # Stratégies de trading
├── infrastructure/         # Infrastructure technique
│   ├── monitoring/        # Grafana, Tensorboard configs
│   ├── docker/           # Fichiers Docker optimisés
│   └── redis/           # Configuration Redis
├── services/             # Services applicatifs
│   ├── backtesting/     # Service de backtesting
│   ├── execution/       # Service d'exécution
│   └── analytics/       # Service d'analyse
├── utils/               # Utilitaires
│   ├── logging/        # Configuration des logs
│   └── config/         # Gestion de la configuration
├── tests/              # Tests unitaires et d'intégration
├── docs/               # Documentation
│   ├── api/           # Documentation API
│   ├── models/        # Documentation des modèles
│   └── setup/         # Guides d'installation
└── scripts/           # Scripts utilitaires
    ├── setup/        # Scripts d'installation
    └── maintenance/  # Scripts de maintenance
```

## Fichiers de Configuration
- `config/`
  - `trading_config.yaml`    # Configuration principale
  - `model_config.yaml`      # Configuration des modèles
  - `monitoring_config.yaml` # Configuration monitoring

## Dépendances
- `requirements/`
  - `base.txt`      # Dépendances de base
  - `dev.txt`       # Dépendances de développement
  - `prod.txt`      # Dépendances de production

## Docker
- `docker/`
  - `Dockerfile.base`     # Image de base
  - `Dockerfile.dev`      # Image de développement
  - `Dockerfile.prod`     # Image de production
  - `docker-compose.dev.yml`
  - `docker-compose.prod.yml`
