# Guide de Visualisation des Modèles

## 📊 Visualisation des Stratégies

### Composants Visuels

1. **Diversité Génétique**
   ```python
   from trading.visualization import StrategyVisualizer
   
   visualizer = StrategyVisualizer()
   visualizer.plot_genetic_diversity(
       population_dna=population_matrix,
       gene_labels=['EMA', 'RSI', 'Stop Loss']
   )
   ```
   - Carte thermique des gènes
   - Identification des clusters
   - Analyse de la diversité

2. **Distribution des Gènes**
   ```python
   gene_values = {
       'EMA Fast': [12, 15, 20, 25],
       'EMA Slow': [50, 75, 100, 150],
       'RSI Period': [14, 21, 28]
   }
   visualizer.plot_gene_distribution(gene_values)
   ```
   - Histogrammes par gène
   - Statistiques descriptives
   - Détection des anomalies

3. **Évolution du Fitness**
   ```python
   fitness_history = [0.5, 0.6, 0.7, 0.8, 0.85]
   visualizer.plot_fitness_evolution(
       fitness_history,
       window=3  # Moyenne mobile
   )
   ```
   - Progression temporelle
   - Moyenne mobile
   - Points de convergence

### Paramètres Temporels

1. **Configuration**
   ```python
   from trading.models import TemporalParameters
   
   params = TemporalParameters(
       window_size=60,      # 60 périodes
       update_frequency='1D',  # Mise à jour quotidienne
       prediction_horizon=5,   # Prédiction à 5 périodes
       train_test_split=0.8    # 80% entraînement, 20% test
   )
   ```

2. **Création des Fenêtres**
   ```python
   # Fenêtres glissantes
   X, y = params.create_rolling_windows(market_data)
   
   # Division train/test
   X_train, X_test, y_train, y_test = params.split_data(X, y)
   ```

3. **Optimisation**
   ```python
   # Test de différentes tailles de fenêtres
   window_results = params.evaluate_window_sizes(
       data=market_data,
       model=trading_model,
       window_sizes=[30, 60, 90]
   )
   
   # Optimisation complète
   best_params = params.optimize_parameters(
       data=market_data,
       model=trading_model,
       param_ranges={
           'window_size': [30, 60, 90],
           'update_frequency': ['1D', '4H', '1H'],
           'prediction_horizon': [3, 5, 7]
       }
   )
   ```

## 🎨 Visualisations Avancées

### 1. SHAP Values
```python
from trading.visualization import AdvancedVisualizer

visualizer = AdvancedVisualizer()
visualizer.plot_shap_values(
    model=trained_model,
    X=test_data,
    feature_names=['price', 'volume', 'rsi', 'macd']
)
```

### 2. Grad-CAM
```python
# Visualisation de l'attention du modèle
visualizer.plot_gradcam(
    model=neural_model,
    layer_name='conv2d_1',
    X=input_data
)
```

### 3. Front de Pareto
```python
# Optimisation multi-objectifs
visualizer.plot_pareto_front(
    results={
        'risk': risk_values,
        'return': return_values,
        'stability': stability_values
    },
    objectives=['risk', 'return', 'stability']
)
```

### 4. Timeline des Générations
```python
# Analyse temporelle de l'évolution
visualizer.plot_generation_timeline(
    generation_times=execution_times,
    generation_metrics={
        'fitness': fitness_history,
        'diversity': diversity_scores
    }
)
```

### 5. Arbre Évolutif
```python
# Visualisation de l'évolution des solutions
visualizer.plot_evolution_tree(
    population_history=populations,
    fitness_history=fitness_scores
)
```

### 6. Analyse de Robustesse
```python
# Test sur différents seeds
visualizer.plot_robustness_analysis(
    seed_results={
        42: {'sharpe': 1.5, 'drawdown': -0.1},
        123: {'sharpe': 1.4, 'drawdown': -0.12},
        456: {'sharpe': 1.6, 'drawdown': -0.09}
    },
    metrics=['sharpe', 'drawdown']
)
```

### 7. Corrélations des Features
```python
# Analyse des corrélations
visualizer.plot_feature_correlations(
    data=market_data,
    target='price_direction'
)
```

## 📈 Bonnes Pratiques

1. **Sauvegarde des Visualisations**
   - Les graphiques sont automatiquement sauvegardés dans `logs/visualization/strategies`
   - Format de nommage : `{type}_{timestamp}.png`
   - Résolution adaptée pour la présentation

2. **Paramètres Temporels**
   - Commencer avec des valeurs standards (60 périodes, mise à jour quotidienne)
   - Optimiser progressivement selon les résultats
   - Valider sur différentes périodes de marché

3. **Analyse des Résultats**
   - Examiner la convergence du fitness
   - Vérifier la diversité génétique
   - Identifier les paramètres optimaux

## �� Points Clés

1. **Visualisation**
   - Utiliser des graphiques appropriés pour chaque métrique
   - Maintenir une cohérence visuelle
   - Sauvegarder systématiquement les résultats

2. **Paramètres**
   - Adapter les fenêtres au marché
   - Équilibrer réactivité et stabilité
   - Valider sur plusieurs périodes

3. **Optimisation**
   - Procéder par étapes
   - Éviter le surapprentissage
   - Documenter les résultats

## 📈 Bonnes Pratiques de Visualisation

1. **Organisation**
   - Sauvegarder les visualisations dans des dossiers dédiés
   - Utiliser des timestamps pour le versioning
   - Maintenir une hiérarchie claire

2. **Interactivité**
   - Utiliser Plotly pour les graphiques interactifs
   - Permettre le zoom et l'exploration
   - Ajouter des tooltips informatifs

3. **Clarté**
   - Choisir des palettes de couleurs appropriées
   - Ajouter des titres et légendes explicites
   - Optimiser la taille des graphiques

4. **Performance**
   - Limiter la taille des données visualisées
   - Utiliser des formats de fichiers optimisés
   - Nettoyer la mémoire après chaque plot

## 🔍 Exemples d'Utilisation

### 1. Analyse Complète d'une Stratégie
```python
# Initialisation
visualizer = StrategyVisualizer()
temporal_params = TemporalParameters()

# Préparation des données
X, y = temporal_params.create_rolling_windows(market_data)
X_train, X_test, y_train, y_test = temporal_params.split_data(X, y)

# Entraînement et visualisation
model.train(X_train, y_train)
visualizer.plot_fitness_evolution(model.fitness_history)

# Analyse des prédictions
predictions = {
    '1D': model.predict(X_test[:, :, 0]),
    '3D': model.predict(X_test[:, :, 1]),
    '5D': model.predict(X_test[:, :, 2])
}
visualizer.plot_prediction_cascade(
    predictions=predictions,
    actual_values=y_test,
    timestamps=test_dates
)
```

### 2. Optimisation des Paramètres
```python
# Configuration des plages
param_ranges = {
    'window_size': [30, 60, 90],
    'update_frequency': ['1D', '4H', '1H'],
    'prediction_horizon': [3, 5, 7]
}

# Optimisation
best_params = temporal_params.optimize_parameters(
    data=market_data,
    model=trading_model,
    param_ranges=param_ranges
)

# Visualisation des résultats
performances = {
    'Sharpe Ratio': [0.8, 1.2, 1.5],
    'Max Drawdown': [-0.15, -0.12, -0.10],
    'Win Rate': [0.55, 0.58, 0.60]
}
visualizer.plot_temporal_performance(
    performances=performances,
    time_windows=['1M', '3M', '6M']
)
```

### 3. Analyse Complète d'un Modèle
```python
# Initialisation
visualizer = AdvancedVisualizer()

# SHAP Analysis
visualizer.plot_shap_values(model, X_test, feature_names)

# Robustesse
visualizer.plot_robustness_analysis(seed_results, metrics)

# Corrélations
visualizer.plot_feature_correlations(data, 'target')
```

### 4. Suivi d'Évolution
```python
# Timeline
visualizer.plot_generation_timeline(
    times,
    {'fitness': fitness_history, 'diversity': diversity_history}
)

# Arbre évolutif
visualizer.plot_evolution_tree(population_history, fitness_history)
```

### 5. Optimisation Multi-objectifs
```python
# Front de Pareto
visualizer.plot_pareto_front(
    results={
        'risk': risks,
        'return': returns,
        'stability': stabilities
    },
    objectives=['risk', 'return', 'stability']
)
```

## 📈 Points Clés

1. **Choix des Visualisations**
   - Adapter le type de graphique aux données
   - Privilégier la clarté sur l'esthétique
   - Assurer la cohérence des styles

2. **Automatisation**
   - Créer des pipelines de visualisation
   - Sauvegarder systématiquement
   - Documenter les paramètres

3. **Interprétation**
   - Ajouter des annotations explicatives
   - Identifier les patterns importants
   - Faciliter la comparaison 