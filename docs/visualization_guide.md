# Guide de Visualisation

## 🎨 Visualiseur Avancé

### Initialisation

```python
from trading.visualization import AdvancedVisualizer

visualizer = AdvancedVisualizer(save_dir="logs/visualization/advanced")
```

## 📊 Types de Visualisations

### 1. Analyse SHAP

```python
# Visualisation de l'importance des features
visualizer.plot_shap_values(model, X, feature_names)
```

### 2. Grad-CAM

```python
# Visualisation de l'attention du modèle
visualizer.plot_gradcam(model, "conv_layer", X)
```

### 3. Front de Pareto

```python
# Visualisation multi-objectifs
visualizer.plot_pareto_front(
    results={"return": returns, "risk": risks},
    objectives=["return", "risk"]
)
```

### 4. Timeline des Générations

```python
# Analyse temporelle des générations
visualizer.plot_generation_timeline(
    generation_times=times,
    generation_metrics={"fitness": fitness_values}
)
```

### 5. Arbre Évolutif

```python
# Visualisation de l'évolution des solutions
visualizer.plot_evolution_tree(
    population_history=populations,
    fitness_history=fitness_values
)
```

### 6. Analyse de Robustesse

```python
# Analyse de sensibilité aux seeds
visualizer.plot_robustness_analysis(
    seed_results=results,
    metrics=["sharpe_ratio", "max_drawdown"]
)
```

### 7. Corrélations des Features

```python
# Analyse des corrélations
visualizer.plot_feature_correlations(
    data=features_df,
    target="returns"
)
```

## 🎯 Bonnes Pratiques

### 1. Configuration des Graphiques
- Utilisation de styles cohérents
- Palettes de couleurs adaptées
- Dimensions optimales

### 2. Sauvegarde
- Format approprié (PNG, HTML)
- Nommage clair avec timestamp
- Organisation des dossiers

### 3. Interactivité
- Graphiques interactifs avec Plotly
- Zoom et sélection
- Tooltips informatifs

## 🔍 Cas d'Utilisation

### 1. Analyse de Performance
- Distribution des rendements
- Courbes de drawdown
- Métriques de risque

### 2. Diagnostic de Modèle
- Importance des features
- Cartes d'attention
- Analyse d'erreurs

### 3. Optimisation
- Fronts de Pareto
- Convergence des algorithmes
- Analyse de sensibilité

## 🛠️ Personnalisation

### 1. Styles
```python
# Configuration personnalisée
visualizer._setup_style()
plt.style.use('seaborn')
sns.set_palette("viridis")
```

### 2. Formats de Sortie
- PNG pour rapports statiques
- HTML pour dashboards interactifs
- SVG pour publication

### 3. Dimensions
- Adaptation à l'écran
- Résolution pour impression
- Ratio d'aspect optimal

## 📈 Intégration

### 1. Reporting Automatique
- Génération de rapports
- Dashboards en temps réel
- Alertes visuelles

### 2. Pipeline de Trading
- Monitoring en direct
- Analyse post-trade
- Validation de stratégie

### 3. Documentation
- Exemples reproductibles
- Guides interactifs
- Notebooks tutoriels