"""
Implémentation du modèle hybride combinant algorithme génétique et réseau neuronal.
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
from abc import ABC, abstractmethod
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
import ruptures
import gc
import yaml
import json
import logging

from ..common.model_interface import ModelInterface, EnsembleModel
from ..genetic_algorithm_model.model import GeneticAlgorithmModel
from ..neural_network_model.model import NeuralNetworkModel
from ...data.data_types import ProcessedData, ValidationResult, FeatureSet
from .optimizer import HybridOptimizer
from .params import HybridModelParams
from .meta_learner import MetaLearner
from .neat_evolution import NEATEvolution
from .synthetic_data import LightGAN

class HybridModel(EnsembleModel, ModelInterface):
    """Modèle hybride combinant plusieurs approches."""
    
    def __init__(self, config: Dict):
        """Initialisation du modèle hybride."""
        super().__init__()
        self.config = config
        self.params = self._init_params()
        self.logger = self._setup_logger()
        self.normalizer = self.AdaptiveNormalizer()
        self._setup_components()
        
    def _setup_logger(self):
        """Configure le logger pour le modèle."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _setup_components(self):
        """Initialisation des composants avancés."""
        self.meta_learner = MetaLearner(self.params)
        self.neat_evolution = NEATEvolution(self.params)
        self.synthetic_data_gen = LightGAN(self.params)
        
        # Composants de base
        self.nn_model = self._create_neural_network()
        self.ga_model = self._create_genetic_algorithm()

    def _initialize_detectors(self):
        """Initialise les détecteurs de régime et de changement."""
        self.change_point_detector = ruptures.Pelt(model="rbf")
        self.regime_detector = self._create_regime_detector()

    def _create_neural_network(self):
        """Création du réseau neuronal optimisé."""
        input_shape = (self.params.get('sequence_length', 30), len(self.config['features']))
        
        inputs = tf.keras.layers.Input(shape=input_shape)
        x = self.normalizer(inputs)
        x = tf.keras.layers.LSTM(32, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(16)(x)
        x = tf.keras.layers.Dense(16)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
        outputs = tf.keras.layers.Dense(1, activation='linear')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.params.get('learning_rate', 0.001)),
            loss='mse',
            metrics=['mae']
        )
        return model

    def _create_data_pipeline(self):
        """Création du pipeline de données optimisé."""
        return lambda data: tf.data.Dataset.from_tensor_slices(data)\
            .cache()\
            .prefetch(tf.data.AUTOTUNE)\
            .batch(self.batch_size)

    def train(self, data: pd.DataFrame, **kwargs):
        """Processus d'entraînement complet."""
        # 1. Génération de données synthétiques si nécessaire
        if self.params['use_synthetic_data']:
            synthetic_data = self.synthetic_data_gen.generate_data(
                self.params['synthetic_samples']
            )
            data = self._combine_data(data, synthetic_data)

        # 2. Évolution de l'architecture si nécessaire
        if self.params['evolve_architecture']:
            self.neat_evolution.evolve(data)
            self._update_architecture()

        # 3. Meta-learning et fine-tuning
        self.meta_learner.pre_train({
            'main': data,
            **kwargs.get('additional_instruments', {})
        })
        
        # 4. Entraînement final
        self._train_final_model(data)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Prédiction avec gestion des coûts cachés."""
        # Préparation des données
        processed_data = self._preprocess_prediction_data(data)
        
        # Prédictions des modèles
        ga_pred = self.ga_model.predict(processed_data)
        nn_pred = self.nn_model.predict(processed_data)
        
        # Application des coûts cachés
        combined_pred = self._combine_predictions(ga_pred, nn_pred)
        final_pred = self._apply_trading_costs(combined_pred)
        
        return final_pred

    def _apply_trading_costs(self, predictions: np.ndarray) -> np.ndarray:
        """Application des coûts de trading."""
        spread = self.params['trading']['spread']
        latency = self.params['trading']['latency']
        
        # Ajustement pour le spread
        adjusted_pred = predictions * (1 - spread)
        
        # Simulation de latence
        return self._simulate_latency(adjusted_pred, latency)

    def _check_data_drift(self, train_data: pd.DataFrame, 
                         live_data: pd.DataFrame) -> bool:
        """Détection de la dérive des données."""
        return ks_2samp(train_data['close'], 
                       live_data['close']).pvalue < 0.01

    def save(self, path: str):
        """Sauvegarde optimisée du modèle."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarde du modèle neuronal quantifié
        converter = tf.lite.TFLiteConverter.from_keras_model(self.nn_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open(path / 'model.tflite', 'wb') as f:
            f.write(tflite_model)
            
        # Sauvegarde des autres composants
        self._save_components(path)

    def validate(self, data: ProcessedData) -> ValidationResult:
        """
        Valide les données d'entrée pour les deux modèles.

        Args:
            data: Données à valider

        Returns:
            ValidationResult: Résultat de la validation
        """
        ga_validation = self.ga_model.validate(data)
        nn_validation = self.nn_model.validate(data)

        return ValidationResult(
            is_valid=ga_validation.is_valid and nn_validation.is_valid,
            errors=ga_validation.errors + nn_validation.errors,
            warnings=ga_validation.warnings + nn_validation.warnings
        )

    def get_features(self) -> FeatureSet:
        """
        Retourne l'ensemble des features utilisées par le modèle.

        Returns:
            FeatureSet: Description des features
        """
        return self.nn_model.get_features()

    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Retourne les hyperparamètres du modèle hybride.

        Returns:
            Dict[str, Any]: Hyperparamètres
        """
        return {
            'genetic': self.ga_model.get_hyperparameters(),
            'neural': self.nn_model.get_hyperparameters(),
            'weights': self.model_weights
        }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retourne les informations sur le modèle hybride.

        Returns:
            Dict[str, Any]: Informations du modèle
        """
        return {
            'type': 'hybrid',
            'components': ['genetic_algorithm', 'neural_network'],
            'is_trained': self.is_trained,
            'last_training': datetime.now().isoformat(),
            'performance_metrics': self.evaluate(None) if self.is_trained else None
        }

    def is_ready(self) -> bool:
        """
        Vérifie si le modèle est prêt pour les prédictions.

        Returns:
            bool: True si le modèle est entraîné et opérationnel
        """
        return self.is_trained and all(model.is_ready() for model in self.models.values())

    def evaluate(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Évalue les performances du modèle hybride.

        Args:
            data: DataFrame contenant les données d'évaluation

        Returns:
            Dict[str, float]: Métriques de performance
        """
        if data is None:
            return {}

        metrics = {}
        for name, model in self.models.items():
            model_metrics = model.evaluate(data)
            metrics.update({f"{name}_{k}": v for k, v in model_metrics.items()})

        return metrics

    def get_parameters(self) -> Dict[str, Any]:
        """
        Retourne les paramètres du modèle.

        Returns:
            Dict[str, Any]: Paramètres du modèle
        """
        return self.params.to_dict()

    def add_model(self, model: ModelInterface) -> None:
        """
        Ajoute un modèle à l'ensemble.

        Args:
            model: Modèle à ajouter
        """
        model_name = f"additional_{len(self.models)}"
        self.models[model_name] = model
        # Répartition équitable des poids
        weight = 1.0 / len(self.models)
        self.model_weights = {name: weight for name in self.models.keys()}

    def remove_model(self, model_name: str) -> None:
        """
        Retire un modèle de l'ensemble.

        Args:
            model_name: Nom du modèle à retirer
        """
        if model_name in ['genetic', 'neural']:
            raise ValueError("Impossible de retirer les modèles de base")
        
        if model_name in self.models:
            del self.models[model_name]
            del self.model_weights[model_name]
            # Répartition équitable des poids
            weight = 1.0 / len(self.models)
            self.model_weights = {name: weight for name in self.models.keys()}

    def get_model_weights(self) -> Dict[str, float]:
        """
        Retourne les poids des modèles dans l'ensemble.

        Returns:
            Dict[str, float]: Poids par modèle
        """
        return self.model_weights.copy()

    def _update_training_history(self, data: pd.DataFrame) -> None:
        """
        Met à jour l'historique d'entraînement.

        Args:
            data: Données d'entraînement
        """
        metrics = self.evaluate(data)
        self.training_history['loss'].append(metrics.get('neural_loss', 0.0))
        self.training_history['accuracy'].append(metrics.get('accuracy', 0.0))
        self.training_history['genetic_fitness'].append(metrics.get('genetic_fitness', 0.0))
        self.training_history['neural_loss'].append(metrics.get('neural_loss', 0.0))

    def _adjust_weights(self, validation_data: pd.DataFrame) -> None:
        """Ajuste dynamiquement les poids des modèles basés sur leurs performances."""
        ga_score = self.ga_model.evaluate(validation_data)['accuracy']
        nn_score = self.nn_model.evaluate(validation_data)['accuracy']
        
        total_score = ga_score + nn_score
        self.model_weights = {
            'genetic': ga_score / total_score,
            'neural': nn_score / total_score
        }
        
    def _update_performance_tracking(self, data: pd.DataFrame) -> None:
        """Met à jour le suivi des performances."""
        ga_metrics = self.ga_model.evaluate(data)
        nn_metrics = self.nn_model.evaluate(data)
        combined_pred = self.predict(data)
        
        self.performance_tracker['genetic'].append(ga_metrics)
        self.performance_tracker['neural'].append(nn_metrics)
        self.performance_tracker['combined'].append({
            'timestamp': datetime.now(),
            'accuracy': self._calculate_accuracy(combined_pred, data)
        })

    def _create_regime_detector(self):
        """Crée un détecteur de régime de marché."""
        return ruptures.KernelCPD(kernel="rbf")

    def _detect_market_regime(self, data: np.ndarray) -> str:
        """Détecte le régime de marché actuel."""
        # Utilisation de ruptures pour la détection
        algo = ruptures.Binseg(model="rbf").fit(data)
        change_points = algo.predict(n_bkps=3)
        
        # Analyse des segments pour déterminer le régime
        if len(change_points) > 1:
            last_segment = data[change_points[-2]:change_points[-1]]
            volatility = np.std(last_segment)
            trend = np.polyfit(np.arange(len(last_segment)), last_segment, 1)[0]
            
            if volatility > self.params.volatility_threshold:
                return "volatile"
            elif abs(trend) > self.params.trend_threshold:
                return "trending"
            else:
                return "ranging"
        
        return "unknown"

    async def cleanup(self):
        """Nettoie les ressources utilisées par le modèle."""
        try:
            # Nettoyage du modèle neuronal
            tf.keras.backend.clear_session()
            
            # Nettoyage des autres composants
            if hasattr(self, 'synthetic_data_gen'):
                await self.synthetic_data_gen.cleanup()
            
            # Libération de la mémoire
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"Erreur lors du nettoyage: {str(e)}")

    def detect_regime_changes(self, data):
        """Détecte les changements de régime de manière synchrone."""
        try:
            if isinstance(data, pd.DataFrame):
                signal = data['close'].values
            else:
                signal = np.array(data)
                
            # Normalisation des données
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
            
            # Détection des points de changement
            algo = ruptures.Pelt(model="rbf").fit(signal.reshape(-1, 1))
            change_points = algo.predict(pen=10)
            
            return change_points
        except Exception as e:
            self.logger.error(f"Erreur lors de la détection des régimes: {str(e)}")
            return []

    def preprocess_data(self, data):
        """Prétraitement synchrone des données."""
        try:
            # Conversion en DataFrame si nécessaire
            if not isinstance(data, pd.DataFrame):
                if isinstance(data, dict):
                    data = pd.DataFrame([data])
                else:
                    data = pd.DataFrame(data)
            
            # Normalisation
            numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
            if not numeric_columns.empty:
                data[numeric_columns] = (data[numeric_columns] - data[numeric_columns].mean()) / (data[numeric_columns].std() + 1e-8)
            
            # Gestion des valeurs manquantes
            data = data.ffill().bfill()
            
            # Conversion en ndarray
            if 'timestamp' in data.columns:
                data = data.drop('timestamp', axis=1)
            return data.values
            
        except Exception as e:
            self.logger.error(f"Erreur lors du prétraitement: {str(e)}")
            return None

    class AdaptiveNormalizer(tf.keras.layers.Layer):
        """Couche de normalisation adaptative."""
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            
        def call(self, inputs):
            mean, variance = tf.nn.moments(inputs, axes=[1], keepdims=True)
            return (inputs - mean) / (tf.sqrt(variance) + 1e-6)

    def _add_technical_features(self, data):
        """Ajoute des features techniques aux données.
        
        Args:
            data: DataFrame contenant les données
            
        Returns:
            DataFrame avec les features techniques ajoutées
        """
        try:
            if 'close' in data.columns:
                # Moyennes mobiles
                data['SMA_20'] = data['close'].rolling(window=20).mean()
                data['SMA_50'] = data['close'].rolling(window=50).mean()
                
                # Volatilité
                data['volatility'] = data['close'].rolling(window=20).std()
                
                # RSI
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                data['RSI'] = 100 - (100 / (1 + rs))
            
            return data
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajout des features techniques: {str(e)}")
            raise

    def _init_params(self) -> Dict:
        """Initialise les paramètres du modèle."""
        return {
            'meta_learning': self.config.get('meta_learning', {
                'base_lr': 0.001,
                'adaptation_steps': 3
            }),
            'neat': self.config.get('neat', {
                'population_size': 20,
                'generations': 5
            }),
            'gan': self.config.get('gan', {
                'latent_dim': 100,
                'generator_dim': 128,
                'discriminator_dim': 128,
                'learning_rate': 0.0002,
                'beta1': 0.5
            }),
            'features': self.config.get('features', ['close', 'volume', 'high', 'low', 'open']),
            'timeframes': self.config.get('timeframes', ['1h']),
            'batch_size': self.config.get('batch_size', 32)
        }

    def _create_genetic_algorithm(self):
        """Création de l'algorithme génétique."""
        ga_config = {
            'population_size': self.params.get('neat', {}).get('population_size', 20),
            'generations': self.params.get('neat', {}).get('generations', 5),
            'input_dim': len(self.config['features']),
            'output_dim': 1,
            'fitness_threshold': 0.95,
            'fitness_criterion': 'max',
            'activation_functions': ['relu', 'tanh', 'sigmoid'],
            'hidden_nodes': [32, 16],
            'mutation_rate': 0.2,
            'crossover_rate': 0.7,
            'no_fitness_termination': False,
            'single_structural_mutation': False,
            'bias_init_type': 'gaussian',
            'response_init_type': 'gaussian',
            'weight_init_type': 'gaussian',
            'enabled_rate_to_true_add': 0.0,
            'enabled_rate_to_false_add': 0.0,
            'min_species_size': 2
        }
        
        return GeneticAlgorithmModel(ga_config)

class GeneticAlgorithmModel(ModelInterface):
    """Implémentation de l'algorithme génétique pour le trading."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = self._build_model()
        self.is_trained = False
        self.population = []
        self.best_individual = None
        self.generation = 0
        self.fitness_history = []

    def _build_model(self):
        """Construction du modèle génétique."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.config['hidden_nodes'][0],
                activation='relu',
                input_dim=self.config['input_dim']
            ),
            tf.keras.layers.Dense(
                self.config['hidden_nodes'][1],
                activation='relu'
            ),
            tf.keras.layers.Dense(
                self.config['output_dim'],
                activation='tanh'
            )
        ])
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        return model

    def _predict_impl(self, data: np.ndarray) -> np.ndarray:
        """Implémentation de la prédiction."""
        if not self.is_trained:
            raise RuntimeError("Le modèle n'est pas entraîné")
        return self.model.predict(data)

    def _preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Prétraitement des données pour l'algorithme génétique."""
        if isinstance(data, pd.DataFrame):
            # Sélection des features
            features = self.config.get('features', ['close', 'volume', 'high', 'low'])
            data = data[features].values
            
            # Normalisation
            data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
            
            return data
        return data

    def _train_impl(self, data: pd.DataFrame, **kwargs):
        """Implémentation de l'entraînement."""
        processed_data = self._preprocess_data(data)
        
        # Initialisation de la population
        self.population = self._initialize_population()
        
        for gen in range(self.config['generations']):
            # Évaluation
            fitness_scores = self._evaluate_population(processed_data)
            
            # Sélection des meilleurs individus
            parents = self._select_parents(fitness_scores)
            
            # Création de la nouvelle génération
            new_population = []
            while len(new_population) < self.config['population_size']:
                parent1, parent2 = np.random.choice(parents, size=2, replace=False)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)
            
            self.population = new_population
            self.generation += 1
            
            # Mise à jour du meilleur individu
            best_idx = np.argmax(fitness_scores)
            self.best_individual = self.population[best_idx]
            self.fitness_history.append(max(fitness_scores))
            
            if max(fitness_scores) >= self.config['fitness_threshold']:
                break
        
        self.is_trained = True

    def _initialize_population(self):
        """Initialisation de la population."""
        return [self._create_individual() for _ in range(self.config['population_size'])]

    def _create_individual(self):
        """Création d'un individu."""
        weights = []
        for layer in self.model.layers:
            if hasattr(layer, 'kernel'):
                w = np.random.normal(0, 0.1, layer.kernel.shape)
                weights.append(w)
        return weights

    def _evaluate_population(self, data):
        """Évaluation de la population."""
        scores = []
        for individual in self.population:
            self._set_weights(individual)
            pred = self.model.predict(data)
            score = -np.mean((pred - data[:, -1])**2)  # Negative MSE
            scores.append(score)
        return np.array(scores)

    def _select_parents(self, fitness_scores):
        """Sélection des parents."""
        num_parents = self.config['population_size'] // 2
        parent_indices = np.argsort(fitness_scores)[-num_parents:]
        return [self.population[i] for i in parent_indices]

    def _crossover(self, parent1, parent2):
        """Croisement de deux parents."""
        child = []
        for w1, w2 in zip(parent1, parent2):
            mask = np.random.random(w1.shape) < 0.5
            child_w = np.where(mask, w1, w2)
            child.append(child_w)
        return child

    def _mutate(self, individual):
        """Mutation d'un individu."""
        mutated = []
        for w in individual:
            mask = np.random.random(w.shape) < self.config['mutation_rate']
            mutation = np.random.normal(0, 0.1, w.shape)
            mutated_w = np.where(mask, w + mutation, w)
            mutated.append(mutated_w)
        return mutated

    def _set_weights(self, weights):
        """Application des poids au modèle."""
        weight_layers = [layer for layer in self.model.layers if hasattr(layer, 'kernel')]
        for layer, w in zip(weight_layers, weights):
            layer.set_weights([w, layer.bias.numpy()])

    def save(self, path: str):
        """Sauvegarde du modèle."""
        if not self.is_trained:
            raise RuntimeError("Le modèle n'est pas entraîné")
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarde du modèle
        self.model.save(save_path / 'genetic_model')
        
        # Sauvegarde de la configuration et des métriques
        config_data = {
            'config': self.config,
            'generation': self.generation,
            'fitness_history': self.fitness_history,
            'is_trained': self.is_trained
        }
        
        with open(save_path / 'genetic_config.json', 'w') as f:
            json.dump(config_data, f)

    def load(self, path: str):
        """Chargement du modèle."""
        load_path = Path(path)
        
        # Chargement du modèle
        self.model = tf.keras.models.load_model(load_path / 'genetic_model')
        
        # Chargement de la configuration et des métriques
        with open(load_path / 'genetic_config.json', 'r') as f:
            config_data = json.load(f)
        
        self.config = config_data['config']
        self.generation = config_data['generation']
        self.fitness_history = config_data['fitness_history']
        self.is_trained = config_data['is_trained']

    def validate(self, data: ProcessedData) -> ValidationResult:
        """Validation des données."""
        errors = []
        warnings = []
        
        # Vérification des features requises
        required_features = self.config.get('features', ['close', 'volume', 'high', 'low'])
        missing_features = [f for f in required_features if f not in data.columns]
        
        if missing_features:
            errors.append(f"Features manquantes: {missing_features}")
        
        # Vérification des valeurs manquantes
        if data.isnull().any().any():
            warnings.append("Les données contiennent des valeurs manquantes")
        
        # Vérification de la taille minimale des données
        min_samples = 100  # Exemple de taille minimale
        if len(data) < min_samples:
            warnings.append(f"Jeu de données trop petit. Minimum recommandé: {min_samples}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
