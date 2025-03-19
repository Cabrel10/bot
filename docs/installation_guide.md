# Guide d'Installation Détaillé

## Prérequis Système

### Système d'exploitation
- Linux (recommandé)
- Windows 10/11
- macOS 10.15+

### Logiciels Requis
- Python 3.8+ 
- Git
- Docker (optionnel)
- MongoDB (optionnel)

## Installation Étape par Étape

### 1. Préparation de l'Environnement

```bash
# Créer le dossier du projet
mkdir trading_bot
cd trading_bot

# Cloner le repository
git clone [URL_DU_REPO]
cd [NOM_DU_REPO]

# Créer l'environnement virtuel
conda create -n trading_env python=3.8

# Activer l'environnement virtuel
conda activate trading_env
```

### 2. Installation des Dépendances

```bash
# Mettre à jour pip
pip install --upgrade pip

# Installer les dépendances de base
pip install -r requirements.txt

# Installer les dépendances optionnelles
pip install -r requirements/dev.txt  # Pour le développement
pip install -r requirements/prod.txt # Pour la production
```

### 3. Configuration du Projet

1. Copier les fichiers de configuration
```bash
cp config/example.env .env
cp config/example.execution_config.yaml config/execution_config.yaml
cp config/example.order_config.yaml config/order_config.yaml
```

2. Éditer le fichier .env
```env
EXCHANGE_API_KEY=votre_clé_api
EXCHANGE_SECRET_KEY=votre_clé_secrète
ENVIRONMENT=development
LOG_LEVEL=INFO
```

3. Configurer les paramètres dans config/execution_config.yaml
```yaml
execution:
  max_open_trades: 3
  stake_amount: 100
  dry_run: true
```

### 4. Installation de la Base de Données (Optionnel)

Si vous utilisez MongoDB :
```bash
# Sur Ubuntu
sudo apt-get install mongodb

# Sur macOS avec Homebrew
brew install mongodb-community

# Démarrer MongoDB
sudo systemctl start mongodb
```

### 5. Configuration Docker (Optionnel)

```bash
# Construire l'image
docker build -t trading_bot .

# Lancer le conteneur
docker-compose up -d
```

### 6. Vérification de l'Installation

```bash
# Tester l'installation
python -m pytest tests/

# Lancer le bot en mode test
python run_app.py --dry-run
```

## Structure des Fichiers de Configuration

### execution_config.yaml
```yaml
execution:
  max_open_trades: 3
  stake_amount: 100
  dry_run: true
  exchange:
    name: "binance"
    key: "${EXCHANGE_API_KEY}"
    secret: "${EXCHANGE_SECRET_KEY}"
```

### order_config.yaml
```yaml
order:
  max_retries: 3
  retry_delay: 10
  default_type: "LIMIT"
  time_in_force: "GTC"
```

### position_config.yaml
```yaml
position:
  max_open_positions: 5
  leverage: 1
  margin_type: "ISOLATED"
```

## Dépannage

### Problèmes Courants

1. **Erreur de dépendances**
```bash
pip install --upgrade setuptools wheel
pip install -r requirements.txt --no-cache-dir
```

2. **Erreur de connexion à l'exchange**
- Vérifier les clés API
- Vérifier la connexion internet
- Vérifier les restrictions IP

3. **Erreur de base de données**
```bash
# Vérifier le statut de MongoDB
sudo systemctl status mongodb
# Redémarrer si nécessaire
sudo systemctl restart mongodb
```

## Mise à Jour

Pour mettre à jour le bot :
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

## Désinstallation

```bash
# Désactiver l'environnement virtuel
conda deactivate

# Supprimer l'environnement virtuel
conda remove -n trading_env --all

# Supprimer le dossier du projet
cd ..
rm -rf trading_bot
```