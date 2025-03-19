"""
Modèle hybride CNN + GNA amélioré pour le trading.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import pandas as pd
from sklearn.preprocessing import StandardScaler
# import talib
from abc import ABC, abstractmethod

@dataclass
class CNNConfig:
    """Configuration du CNN."""
    input_shape: Tuple[int, int]
    num_classes: int
    architecture: str = "resnet"  # resnet, inception, efficientnet
    dropout_rate: float = 0.3
    l2_reg: float = 0.01
    use_attention: bool = True
    use_batch_norm: bool = True

@dataclass
class GNAConfig:
    """Configuration du GNA."""
    population_size: int
    num_generations: int
    mutation_rate: float
    crossover_rate: float
    elite_size: int
    risk_constraints: Dict[str, float]
    fitness_weights: Dict[str, float]

@dataclass
class HybridConfig:
    """Configuration du modèle hybride."""
    cnn_config: CNNConfig
    gna_config: GNAConfig
    fusion_method: str = "weighted"  # weighted, voting, stacking
    adaptation_rate: float = 0.1
    online_learning: bool = True

class AttentionLayer(layers.Layer):
    """Couche d'attention pour le CNN."""
    
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention = layers.Dense(1, activation='softmax')
        
    def call(self, inputs):
        attention_weights = self.attention(inputs)
        return inputs * attention_weights

class ResNetBlock(layers.Layer):
    """Bloc ResNet pour le CNN."""
    
    def __init__(self, filters, kernel_size=3, **kwargs):
        super(ResNetBlock, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters, kernel_size, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.add = layers.Add()
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.add([x, inputs])

class CNNModel:
    """Modèle CNN amélioré avec différentes architectures."""
    
    def __init__(self, config: CNNConfig):
        self.config = config
        self.model = self._build_model()
        
    def _build_model(self) -> models.Model:
        """Construit le modèle CNN selon l'architecture spécifiée."""
        inputs = layers.Input(shape=self.config.input_shape)
        
        if self.config.architecture == "resnet":
            x = self._build_resnet(inputs)
        elif self.config.architecture == "inception":
            x = self._build_inception(inputs)
        else:  # efficientnet
            x = self._build_efficientnet(inputs)
            
        if self.config.use_attention:
            x = AttentionLayer()(x)
            
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        x = layers.Dense(128, activation='relu', 
                        kernel_regularizer=regularizers.l2(self.config.l2_reg))(x)
        outputs = layers.Dense(self.config.num_classes, activation='softmax')(x)
        
        return models.Model(inputs=inputs, outputs=outputs)
    
    def _build_resnet(self, inputs):
        """Construit une architecture ResNet."""
        x = layers.Conv2D(64, 3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        for filters in [64, 128, 256]:
            x = ResNetBlock(filters)(x)
            x = layers.MaxPooling2D(2)(x)
            
        return x
    
    def _build_inception(self, inputs):
        """Construit une architecture Inception."""
        # Implémentation de l'architecture Inception
        pass
    
    def _build_efficientnet(self, inputs):
        """Construit une architecture EfficientNet."""
        # Implémentation de l'architecture EfficientNet
        pass

class GeneticAlgorithm:
    """Algorithme génétique amélioré avec gestion des risques."""
    
    def __init__(self, config: GNAConfig):
        self.config = config
        self.population = self._initialize_population()
        self.best_solution = None
        self.best_fitness = float('-inf')
        
    def _initialize_population(self) -> np.ndarray:
        """Initialise la population avec des contraintes de risque."""
        population = np.random.randn(
            self.config.population_size,
            self.config.num_parameters
        )
        return self._apply_risk_constraints(population)
    
    def _apply_risk_constraints(self, population: np.ndarray) -> np.ndarray:
        """Applique les contraintes de risque à la population."""
        for i in range(len(population)):
            # Vérification du drawdown maximum
            if self._calculate_max_drawdown(population[i]) > self.config.risk_constraints['max_drawdown']:
                population[i] = self._adjust_for_risk(population[i])
                
            # Vérification de la volatilité
            if self._calculate_volatility(population[i]) > self.config.risk_constraints['max_volatility']:
                population[i] = self._adjust_for_risk(population[i])
                
        return population
    
    def _calculate_fitness(self, solution: np.ndarray) -> float:
        """Calcule la fitness avec pondération des objectifs."""
        returns = self._calculate_returns(solution)
        risk = self._calculate_risk(solution)
        
        fitness = (
            self.config.fitness_weights['returns'] * returns +
            self.config.fitness_weights['risk'] * (1 - risk)
        )
        
        return fitness
    
    def evolve(self):
        """Évolue la population sur une génération."""
        # Évaluation de la fitness
        fitness_scores = np.array([self._calculate_fitness(sol) for sol in self.population])
        
        # Sélection des meilleurs individus (élitisme)
        elite_indices = np.argsort(fitness_scores)[-self.config.elite_size:]
        elite = self.population[elite_indices]
        
        # Sélection des parents
        parents = self._select_parents(fitness_scores)
        
        # Croisement et mutation
        offspring = self._crossover(parents)
        offspring = self._mutate(offspring)
        
        # Application des contraintes de risque
        offspring = self._apply_risk_constraints(offspring)
        
        # Mise à jour de la population
        self.population = np.vstack([elite, offspring])
        
        # Mise à jour de la meilleure solution
        current_best_idx = np.argmax(fitness_scores)
        if fitness_scores[current_best_idx] > self.best_fitness:
            self.best_solution = self.population[current_best_idx]
            self.best_fitness = fitness_scores[current_best_idx]

class HybridModel:
    """Modèle hybride amélioré combinant CNN et GNA."""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.cnn = CNNModel(config.cnn_config)
        self.gna = GeneticAlgorithm(config.gna_config)
        self.fusion_model = self._build_fusion_model()
        self.scaler = StandardScaler()
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure la journalisation."""
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _build_fusion_model(self) -> models.Model:
        """Construit le modèle de fusion selon la méthode spécifiée."""
        if self.config.fusion_method == "weighted":
            return self._build_weighted_fusion()
        elif self.config.fusion_method == "voting":
            return self._build_voting_fusion()
        else:  # stacking
            return self._build_stacking_fusion()
    
    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Prétraite les données avec des techniques avancées."""
        # Calcul des indicateurs techniques
        data = self._add_technical_indicators(data)
        
        # Normalisation
        scaled_data = self.scaler.fit_transform(data)
        
        # Création des séquences
        sequences = self._create_sequences(scaled_data)
        
        return sequences
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ajoute des indicateurs techniques aux données."""
        self.logger.info("Ajout des indicateurs techniques...")
        
        # Copie des données pour éviter les avertissements SettingWithCopyWarning
        df = data.copy()
        
        # Indicateurs de tendance
        # RSI
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        # MACD
        def calculate_macd(prices, fast=12, slow=26, signal=9):
            ema_fast = prices.ewm(span=fast, adjust=False).mean()
            ema_slow = prices.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
        
        # Bollinger Bands
        def calculate_bollinger_bands(prices, window=20, num_std=2):
            rolling_mean = prices.rolling(window=window).mean()
            rolling_std = prices.rolling(window=window).std()
            upper_band = rolling_mean + (rolling_std * num_std)
            lower_band = rolling_mean - (rolling_std * num_std)
            return rolling_mean, upper_band, lower_band
        
        # Calcul des indicateurs
        df['rsi'] = calculate_rsi(df['close'])
        macd, signal, hist = calculate_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        ma20, upper, lower = calculate_bollinger_bands(df['close'])
        df['ma20'] = ma20
        df['upper_band'] = upper
        df['lower_band'] = lower
        
        # Moyennes mobiles simples
        df['sma5'] = df['close'].rolling(window=5).mean()
        df['sma10'] = df['close'].rolling(window=10).mean()
        df['sma30'] = df['close'].rolling(window=30).mean()
        
        # Indicateurs de volatilité
        df['volatility'] = df['close'].rolling(window=20).std()
        
        # Suppression des valeurs NaN
        df.dropna(inplace=True)
        
        return df
    
    def _create_sequences(self, data: np.ndarray, seq_length: int = 10) -> np.ndarray:
        """Crée des séquences pour l'entraînement."""
        sequences = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:(i + seq_length)])
        return np.array(sequences)
    
    def train(self, data: pd.DataFrame, labels: np.ndarray):
        """Entraîne le modèle hybride."""
        # Prétraitement des données
        X = self.preprocess_data(data)
        
        # Entraînement du CNN
        self.cnn.model.fit(
            X, labels,
            epochs=10,
            batch_size=32,
            validation_split=0.2
        )
        
        # Entraînement du GNA
        for generation in range(self.config.gna_config.num_generations):
            self.gna.evolve()
            if generation % 10 == 0:
                self.logger.info(f"Génération {generation}: Meilleure fitness = {self.gna.best_fitness}")
        
        # Entraînement du modèle de fusion
        cnn_predictions = self.cnn.model.predict(X)
        gna_predictions = self._get_gna_predictions(X)
        
        fusion_input = np.concatenate([cnn_predictions, gna_predictions], axis=1)
        self.fusion_model.fit(
            fusion_input, labels,
            epochs=5,
            batch_size=32
        )
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Fait des prédictions avec le modèle hybride."""
        # Prétraitement des données
        X = self.preprocess_data(data)
        
        # Prédictions du CNN
        cnn_predictions = self.cnn.model.predict(X)
        
        # Prédictions du GNA
        gna_predictions = self._get_gna_predictions(X)
        
        # Fusion des prédictions
        fusion_input = np.concatenate([cnn_predictions, gna_predictions], axis=1)
        predictions = self.fusion_model.predict(fusion_input)
        
        return predictions
    
    def adapt_to_market(self, new_data: pd.DataFrame, new_labels: np.ndarray):
        """Adapte le modèle aux nouvelles conditions de marché."""
        if not self.config.online_learning:
            return
            
        # Mise à jour du CNN
        X = self.preprocess_data(new_data)
        self.cnn.model.fit(
            X, new_labels,
            epochs=1,
            batch_size=32,
            verbose=0
        )
        
        # Mise à jour du GNA
        self.gna.population = self._adapt_gna_population(new_data)
        
        # Mise à jour du modèle de fusion
        cnn_predictions = self.cnn.model.predict(X)
        gna_predictions = self._get_gna_predictions(X)
        
        fusion_input = np.concatenate([cnn_predictions, gna_predictions], axis=1)
        self.fusion_model.fit(
            fusion_input, new_labels,
            epochs=1,
            batch_size=32,
            verbose=0
        ) 