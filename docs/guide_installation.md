# Guide d'Installation du Système de Trading

## Prérequis

### Pour Windows
- Python 3.8 ou supérieur
- Git (optionnel)
- Accès administrateur pour l'installation des dépendances

### Pour Linux
- Python 3.8 ou supérieur
- pip3
- Accès sudo pour l'installation des dépendances

## Installation sous Windows

1. Téléchargez le projet dans le dossier de votre choix

2. Ouvrez une invite de commande (PowerShell ou CMD) en tant qu'administrateur

3. Naviguez vers le dossier du projet

4. Exécutez le script d'installation :
   ```batch
   install_windows.bat
   ```

5. Le script effectuera automatiquement :
   - La vérification de l'installation de Python
   - La création de la structure des dossiers
   - La création de l'environnement virtuel
   - L'installation des dépendances
   - La vérification de l'installation

6. Pour activer l'environnement virtuel :
   ```batch
   venv\Scripts\activate.bat
   ```

## Installation sous Linux

1. Téléchargez le projet dans le dossier de votre choix

2. Ouvrez un terminal

3. Naviguez vers le dossier du projet

4. Rendez le script d'installation exécutable :
   ```bash
   chmod +x install_linux.sh
   ```

5. Exécutez le script d'installation :
   ```bash
   ./install_linux.sh
   ```

6. Le script effectuera automatiquement :
   - La vérification de l'installation de Python et pip
   - La création de la structure des dossiers
   - La configuration des permissions
   - La création de l'environnement virtuel
   - L'installation des dépendances
   - La vérification de l'installation

7. Pour activer l'environnement virtuel :
   ```bash
   source venv/bin/activate
   ```

## Structure des Dossiers

```
├── logs/
│   ├── neural_network/
│   ├── mlflow/
│   └── trading/
├── data/
│   ├── raw/
│   ├── processed/
│   └── backtest/
├── .secret/
├── monitoring/
│   ├── prometheus/
│   └── grafana/
├── mlruns/
├── models/
└── config/
```

## Démarrage du Système

1. Activez l'environnement virtuel (voir étapes précédentes)

2. Lancez le système :
   ```bash
   python run.py
   ```

## Résolution des Problèmes

### Problèmes Courants sous Windows

1. **Python non trouvé** :
   - Vérifiez que Python est installé
   - Vérifiez que Python est dans le PATH système

2. **Erreurs de permissions** :
   - Exécutez l'invite de commande en tant qu'administrateur

### Problèmes Courants sous Linux

1. **Erreurs de permissions** :
   - Vérifiez que vous avez les droits d'exécution sur le script
   - Utilisez sudo pour les commandes nécessitant des privilèges

2. **Python ou pip non trouvé** :
   - Installez Python : `sudo apt-get install python3`
   - Installez pip : `sudo apt-get install python3-pip`

## Notes Importantes

- Gardez vos clés API et informations sensibles dans le dossier `.secret`
- Consultez régulièrement les logs pour le suivi des opérations
- Sauvegardez régulièrement vos configurations personnalisées

## Configuration de l'environnement

```bash
# Activer l'environnement virtuel
conda activate trading_env

# Installer les dépendances
pip install -r requirements.txt
```