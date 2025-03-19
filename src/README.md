# Démo du Système de Trading Hybride

Ce répertoire contient une démonstration simplifiée du système de trading hybride.

## Contenu de la démo

La démo illustre les fonctionnalités suivantes :

1. **Génération de données de test** - Création de données de séries temporelles réalistes avec tendances et cycles
2. **Analyse technique** - Calcul d'indicateurs techniques (RSI, MACD, Bandes de Bollinger, etc.)
3. **Détection d'anomalies** - Identification de valeurs aberrantes dans les données
4. **Analyse de sentiment** - Simulation d'analyse de sentiment
5. **Prédiction de prix** - Utilisation d'un modèle LSTM pour prédire les prix futurs

## Comment exécuter la démo

```bash
# Assurez-vous d'être dans le répertoire src
cd src

# Activer l'environnement Conda
conda activate trading_env

# Exécuter la démo
python demo.py
```

## Résultats de la démo

La démo génère les résultats suivants :

1. **Indicateurs techniques** calculés sur les données générées
2. **Anomalies détectées** dans les données
3. **Analyse de sentiment** simulée pour les périodes de temps
4. **Modèle LSTM** entraîné pour la prédiction de prix
5. **Graphique des prédictions** sauvegardé dans le dossier `results/`

## Architecture du système complet

Le système complet comprend les modules suivants :

1. **Modèle hybride** (CNN + GNA)
   - `hybrid_model.py` - Implémentation du modèle hybride

2. **Détection d'anomalies**
   - `anomaly_detection.py` - Détection d'anomalies et de fraude

3. **Métriques avancées**
   - `advanced_metrics.py` - Calcul de métriques de performance et de risque

4. **Optimisation continue**
   - `continuous_optimization.py` - Optimisation continue du modèle

5. **Analyse de sentiment et volatilité**
   - `sentiment_volatility.py` - Analyse de sentiment et prédiction de volatilité

6. **Sauvegarde et restauration**
   - `save_restore.py` - Sauvegarde et restauration des modèles

Les configurations et paramètres du système se trouvent dans le dossier `config/`. 