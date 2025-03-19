"""
Gestionnaire de transfer learning pour l'adaptation des modèles.
"""
from typing import Dict, List, Optional, Union
import tensorflow as tf
import numpy as np
from dataclasses import dataclass
import logging
from datetime import datetime

from src.core.models.neural_network import NeuralNetworkModel
from src.utils.logging import TradingLogger

@dataclass
class TransferConfig:
    """Configuration pour le transfer learning."""
    source_model_path: str
    target_data_ratio: float = 0.3
    fine_tuning_epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping_patience: int = 3
    layers_to_freeze: List[str] = None
    adaptation_threshold: float = 0.15

class ModelAdapter:
    """Gestionnaire d'adaptation des modèles."""
    
    def __init__(self, config: TransferConfig):
        self.config = config
        self.logger = TradingLogger()
        self._adaptation_history = []
        self._performance_metrics = {}
        self._current_adaptation = None

    async def adapt_model(self, 
                         source_model: NeuralNetworkModel,
                         target_data: Dict[str, np.ndarray]) -> NeuralNetworkModel:
        """
        Adapte un modèle aux nouvelles conditions de marché.
        
        Args:
            source_model: Modèle source à adapter
            target_data: Nouvelles données cibles
            
        Returns:
            NeuralNetworkModel: Modèle adapté
        """
        try:
            self._current_adaptation = {
                'start_time': datetime.now(),
                'source_model': source_model.name,
                'target_data_shape': {k: v.shape for k, v in target_data.items()},
                'config': self.config.__dict__
            }
            
            # Chargement du modèle source
            self.logger.info(f"Adaptation du modèle {source_model.name} aux nouvelles données")
            
            # Préparation des données
            X_train, y_train, X_val, y_val = self._prepare_data(target_data)
            
            # Gel des couches spécifiées
            self._freeze_layers(source_model)
            
            # Fine-tuning sur les nouvelles données
            adapted_model = await self._fine_tune(source_model, X_train, y_train, X_val, y_val)
            
            # Évaluation de l'adaptation
            adaptation_metrics = self._evaluate_adaptation(adapted_model, X_val, y_val)
            
            # Mise à jour de l'historique
            self._current_adaptation.update({
                'end_time': datetime.now(),
                'metrics': adaptation_metrics,
                'status': 'success'
            })
            self._adaptation_history.append(self._current_adaptation)
            
            return adapted_model
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'adaptation du modèle: {str(e)}")
            if self._current_adaptation:
                self._current_adaptation.update({
                    'end_time': datetime.now(),
                    'error': str(e),
                    'status': 'failed'
                })
                self._adaptation_history.append(self._current_adaptation)
            raise
    
    def _prepare_data(self, data: Dict[str, np.ndarray]) -> tuple:
        """Prépare les données pour l'adaptation."""
        X = data.get('features')
        y = data.get('targets')
        
        if X is None or y is None:
            raise ValueError("Les données doivent contenir 'features' et 'targets'")
            
        # Conversion en tenseurs TensorFlow
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        
        # Split train/validation
        val_size = int(len(X) * self.config.validation_split)
        indices = tf.range(start=0, limit=tf.shape(X)[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)
        
        train_indices = shuffled_indices[val_size:]
        val_indices = shuffled_indices[:val_size]
        
        X_train = tf.gather(X, train_indices)
        y_train = tf.gather(y, train_indices)
        X_val = tf.gather(X, val_indices)
        y_val = tf.gather(y, val_indices)
        
        return X_train, y_train, X_val, y_val
    
    def _freeze_layers(self, model: NeuralNetworkModel) -> None:
        """Gèle certaines couches du modèle."""
        if not self.config.layers_to_freeze:
            return
            
        for layer in model.model.layers:
            if any(name in layer.name for name in self.config.layers_to_freeze):
                layer.trainable = False
                self.logger.info(f"Couche {layer.name} gelée")
    
    async def _fine_tune(self, 
                        model: NeuralNetworkModel,
                        X_train, y_train, X_val, y_val) -> NeuralNetworkModel:
        """Fine-tune le modèle sur les nouvelles données."""
        # Sauvegarde du modèle original
        original_weights = model.model.get_weights()
        
        # Configuration de l'optimiseur avec un learning rate plus faible
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        model.model.compile(
            optimizer=optimizer,
            loss=model.config.get('loss', 'mse'),
            metrics=model.config.get('metrics', ['mae'])
        )
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.early_stopping_patience,
            restore_best_weights=True
        )
        
        # Entraînement
        history = model.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.fine_tuning_epochs,
            batch_size=self.config.batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Vérification de l'amélioration
        if self._check_improvement(history):
            self.logger.info("Adaptation réussie, modèle amélioré")
            return model
        else:
            self.logger.warning("Adaptation non concluante, restauration du modèle original")
            model.model.set_weights(original_weights)
            return model
    
    def _check_improvement(self, history) -> bool:
        """Vérifie si l'adaptation a amélioré le modèle."""
        val_loss = history.history['val_loss']
        initial_loss = val_loss[0]
        final_loss = val_loss[-1]
        
        improvement = (initial_loss - final_loss) / initial_loss
        self.logger.info(f"Amélioration relative: {improvement:.2%}")
        
        return improvement > self.config.adaptation_threshold
    
    def _evaluate_adaptation(self, model: NeuralNetworkModel, X_val, y_val) -> Dict:
        """Évalue les performances du modèle adapté."""
        evaluation = model.model.evaluate(X_val, y_val, verbose=0)
        metrics = {}
        
        for i, metric_name in enumerate(model.model.metrics_names):
            metrics[metric_name] = float(evaluation[i])
            
        self._performance_metrics[model.name] = metrics
        return metrics
    
    def get_adaptation_history(self) -> List[Dict]:
        """Retourne l'historique des adaptations."""
        return self._adaptation_history
    
    def get_performance_metrics(self) -> Dict:
        """Retourne les métriques de performance."""
        return self._performance_metrics