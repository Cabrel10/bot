"""
Implémentation du réseau neuronal pour le trading.

Ce module implémente un réseau neuronal spécialisé pour le trading avec:
- Support des séries temporelles (LSTM)
- Mécanisme d'attention
- Optimisation des performances
- Support multi-GPU
- Validation croisée temporelle
- Support des contrats futures et des volumes
- Monitoring avancé avec TensorBoard
"""

from typing import List, Dict, Optional, Union, Tuple, Any
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, Concatenate, Conv1D, LayerNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, BinaryCrossentropy
from dataclasses import dataclass, field
import logging
from pathlib import Path
from datetime import datetime
import json
from sklearn.model_selection import TimeSeriesSplit

@dataclass
class NeuralNetworkParams:
    """
    Paramètres du réseau neuronal.
    
    Attributes:
        input_size (int): Taille de l'entrée
        hidden_sizes (List[int]): Tailles des couches cachées
        output_size (int): Taille de la sortie
        learning_rate (float): Taux d'apprentissage
        batch_size (int): Taille des batchs
        epochs (int): Nombre d'époques
        dropout_rate (float): Taux de dropout
        early_stopping_patience (int): Patience pour l'arrêt précoce
        validation_split (float): Proportion de validation
        optimizer (str): Type d'optimiseur
        loss_function (str): Fonction de perte
        use_batch_norm (bool): Utilise la normalisation par batch
        use_residual (bool): Utilise les connexions résiduelles
        activation (str): Fonction d'activation
        use_futures (bool): Active le support des futures
        use_volume (bool): Active le support des volumes
        n_gpu (int): Nombre de GPUs à utiliser
        use_amp (bool): Active l'entraînement en précision mixte
        
    Raises:
        ValidationError: Si les paramètres sont invalides
    """
    input_size: int
    hidden_sizes: List[int] = None
    output_size: int = 1
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    dropout_rate: float = 0.2
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    optimizer: str = 'adam'
    loss_function: str = 'mse'
    use_batch_norm: bool = True
    use_residual: bool = True
    activation: str = 'relu'
    use_futures: bool = False
    use_volume: bool = False
    n_gpu: int = 0
    use_amp: bool = False
    
    def __post_init__(self):
        """Valide les paramètres après l'initialisation."""
        if self.input_size <= 0:
            raise ValueError("input_size doit être > 0")
            
        if self.hidden_sizes is None:
            self.hidden_sizes = [64, 32]
        elif not all(size > 0 for size in self.hidden_sizes):
            raise ValueError("Toutes les tailles cachées doivent être > 0")
            
        if self.output_size <= 0:
            raise ValueError("output_size doit être > 0")
            
        if not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate doit être entre 0 et 1")
            
        if self.batch_size <= 0:
            raise ValueError("batch_size doit être > 0")
            
        if self.epochs <= 0:
            raise ValueError("epochs doit être > 0")
            
        if not 0 <= self.dropout_rate < 1:
            raise ValueError("dropout_rate doit être entre 0 et 1")
            
        if not 0 < self.validation_split < 1:
            raise ValueError("validation_split doit être entre 0 et 1")
            
        if self.n_gpu < 0:
            raise ValueError("n_gpu doit être >= 0")
            
        if self.n_gpu > 0 and len(tf.config.list_physical_devices('GPU')) == 0:
            raise RuntimeError("GPU demandé mais aucun GPU n'est disponible")

class TimeSeriesBlock(tf.keras.layers.Layer):
    """
    Bloc de traitement des séries temporelles.
    Combine LSTM et TCN pour capturer les dépendances temporelles.
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 dropout_rate: float = 0.2,
                 **kwargs):
        super().__init__(**kwargs)
        
        # LSTM bidirectionnel
        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=hidden_size // 2,  # Divisé par 2 car bidirectionnel
                return_sequences=True
            )
        )
        
        # TCN (Temporal Convolutional Network)
        self.conv1 = tf.keras.layers.Conv1D(hidden_size, kernel_size=3, padding='same')
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.activation1 = tf.keras.layers.Activation('relu')
        self.conv2 = tf.keras.layers.Conv1D(hidden_size, kernel_size=3, padding='same', dilation_rate=2)
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.activation2 = tf.keras.layers.Activation('relu')
        
        # Fusion et normalisation
        self.fusion = tf.keras.layers.Dense(hidden_size)
        self.norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
        # Connexion résiduelle
        self.use_residual = input_size == hidden_size

    def call(self, inputs, training=None):
        """Propagation avant."""
        # LSTM
        lstm_out = self.lstm(inputs)
        
        # TCN
        tcn_out = self.conv1(inputs)
        tcn_out = self.batch_norm1(tcn_out, training=training)
        tcn_out = self.activation1(tcn_out)
        tcn_out = self.conv2(tcn_out)
        tcn_out = self.batch_norm2(tcn_out, training=training)
        tcn_out = self.activation2(tcn_out)
        
        # Fusion
        combined = tf.keras.layers.Concatenate()([lstm_out, tcn_out])
        out = self.fusion(combined)
        
        # Normalisation et dropout
        out = self.norm(out)
        out = self.dropout(out, training=training)
        
        # Connexion résiduelle
        if self.use_residual:
            out = out + inputs
            
        return out

class AttentionBlock(tf.keras.layers.Layer):
    """
    Bloc d'attention multi-têtes avec normalisation et connexion résiduelle.
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 4, **kwargs):
        super().__init__(**kwargs)
        
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_size // num_heads
        )
        
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size * 4, activation='relu'),
            tf.keras.layers.Dense(hidden_size)
        ])

    def call(self, inputs, training=None):
        """Propagation avant avec attention et FFN."""
        # Self-attention
        attended = self.attention(inputs, inputs)
        out = self.norm1(attended + inputs)
        
        # Feed-forward
        ffn_out = self.ffn(out)
        out = self.norm2(ffn_out + out)
        
        return out

class FuturesBlock(tf.keras.layers.Layer):
    """
    Bloc de traitement des données futures.
    """
    
    def __init__(self, input_size: int, hidden_size: int, **kwargs):
        super().__init__(**kwargs)
        
        self.price_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation='relu'),
            tf.keras.layers.LayerNormalization()
        ])
        
        self.volume_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation='relu'),
            tf.keras.layers.LayerNormalization()
        ])
        
        self.fusion = tf.keras.layers.Dense(hidden_size)

    def call(self, price, volume, training=None):
        """Propagation avant."""
        price_encoded = self.price_encoder(price)
        volume_encoded = self.volume_encoder(volume)
        
        # Fusion des caractéristiques
        combined = tf.keras.layers.Concatenate()([price_encoded, volume_encoded])
        out = self.fusion(combined)
        
        return out

class TradingNeuralNetwork(tf.keras.Model):
    """
    Réseau neuronal spécialisé pour le trading.
    """
    
    def __init__(self, params: NeuralNetworkParams, **kwargs):
        super().__init__(**kwargs)
        self.params = params
        
        # Couches de traitement temporel
        self.time_series_blocks = []
        for i, size in enumerate(params.hidden_sizes):
            self.time_series_blocks.append(
                TimeSeriesBlock(
                    input_size=params.input_size if i == 0 else params.hidden_sizes[i-1],
                    hidden_size=size,
                    dropout_rate=params.dropout_rate
                )
            )
        
        # Bloc d'attention
        self.attention = AttentionBlock(params.hidden_sizes[-1])
        
        # Support des futures et volumes
        if params.use_futures:
            self.futures_block = FuturesBlock(
                params.input_size,
                params.hidden_sizes[-1]
            )
        
        # Couches de prédiction
        input_size = params.hidden_sizes[-1]
        if params.use_futures:
            input_size *= 2
        
        self.prediction_layers = tf.keras.Sequential()
        self.prediction_layers.add(tf.keras.layers.Dense(params.hidden_sizes[-1] // 2, activation=self._get_activation()))
        self.prediction_layers.add(tf.keras.layers.BatchNormalization())
        self.prediction_layers.add(tf.keras.layers.Dropout(params.dropout_rate))
        self.prediction_layers.add(tf.keras.layers.Dense(params.output_size, activation='tanh'))

    def _get_activation(self):
        """Retourne la fonction d'activation configurée."""
        if self.params.activation == 'relu':
            return 'relu'
        elif self.params.activation == 'leaky_relu':
            return tf.keras.layers.LeakyReLU(alpha=0.1)
        elif self.params.activation == 'elu':
            return 'elu'
        elif self.params.activation == 'selu':
            return 'selu'
        else:
            return 'relu'

    def call(self, inputs, futures_data=None, training=None):
        """
        Propagation avant.
        
        Args:
            inputs: Données d'entrée
            futures_data: Tuple optionnel (prix, volume) pour les futures
            training: Booléen indiquant si le modèle est en mode entraînement
            
        Returns:
            tf.Tensor: Prédictions
        """
        x = inputs
        
        # Traitement des séries temporelles
        for block in self.time_series_blocks:
            x = block(x, training=training)
        
        # Attention
        x = self.attention(x, training=training)
        
        # Traitement des futures si disponibles
        if self.params.use_futures and futures_data is not None:
            price_data, volume_data = futures_data
            futures_features = self.futures_block(price_data, volume_data, training=training)
            x = tf.keras.layers.Concatenate()([x, futures_features])
        
        # Prédiction finale (utilise la dernière séquence pour la prédiction)
        x = x[:, -1, :]  # Prend le dernier pas de temps
        predictions = self.prediction_layers(x, training=training)
        
        return predictions

class TradingNeuralNetworkModel:
    """
    Gestionnaire du modèle neuronal pour le trading.
    Gère l'entraînement, la validation et l'inférence.
    """
    
    def __init__(self, params: NeuralNetworkParams):
        """
        Initialise le modèle.
        
        Args:
            params: Paramètres de configuration
            
        Raises:
            ValueError: Si les paramètres sont invalides
            RuntimeError: Si le GPU est demandé mais non disponible
        """
        self.params = params
        self.model = TradingNeuralNetwork(params)
        
        # Configuration multi-GPU si demandé
        if params.n_gpu > 1:
            self.strategy = tf.distribute.MirroredStrategy()
            with self.strategy.scope():
                self.model = TradingNeuralNetwork(params)
                self.model.compile(
                    optimizer=self._get_optimizer(),
                    loss=self._get_criterion()
                )
        else:
            self.model.compile(
                optimizer=self._get_optimizer(),
                loss=self._get_criterion()
            )
        
        # TensorBoard
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=f"logs/trading_nn_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            histogram_freq=1
        )
        
        # Historique
        self.history = None
        
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def _get_optimizer(self):
        """Configure l'optimiseur."""
        if self.params.optimizer.lower() == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate)
        elif self.params.optimizer.lower() == 'rmsprop':
            return tf.keras.optimizers.RMSprop(learning_rate=self.params.learning_rate)
        elif self.params.optimizer.lower() == 'sgd':
            return tf.keras.optimizers.SGD(learning_rate=self.params.learning_rate)
        else:
            return tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate)

    def _get_criterion(self):
        """Configure la fonction de perte."""
        if self.params.loss_function.lower() == 'mse':
            return tf.keras.losses.MeanSquaredError()
        elif self.params.loss_function.lower() == 'mae':
            return tf.keras.losses.MeanAbsoluteError()
        elif self.params.loss_function.lower() == 'binary_crossentropy':
            return tf.keras.losses.BinaryCrossentropy()
        else:
            return tf.keras.losses.MeanSquaredError()

    def _setup_logging(self):
        """Configure le système de logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def train(self, 
             train_data: Tuple[np.ndarray, np.ndarray],
             val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
             futures_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
             ):
        """
        Entraîne le modèle avec validation croisée temporelle.
        
        Args:
            train_data: Tuple (X_train, y_train)
            val_data: Tuple optionnel (X_val, y_val)
            futures_data: Tuple optionnel (prix, volume) pour les futures
            
        Returns:
            Dict: Historique d'entraînement
            
        Raises:
            ValueError: Si les données sont invalides
            RuntimeError: Si l'entraînement échoue
        """
        X_train, y_train = train_data
        
        if X_train.ndim != 3:
            raise ValueError(f"X_train doit être 3D (batch, time_steps, features), mais a la forme {X_train.shape}")
            
        if y_train.ndim != 2 and y_train.ndim != 1:
            raise ValueError(f"y_train doit être 1D ou 2D, mais a la forme {y_train.shape}")
            
        # Préparation des données de validation
        if val_data is None:
            # Utilise une partie des données d'entraînement pour la validation
            val_split = int(len(X_train) * (1 - self.params.validation_split))
            X_val, y_val = X_train[val_split:], y_train[val_split:]
            X_train, y_train = X_train[:val_split], y_train[:val_split]
        else:
            X_val, y_val = val_data
            
        # Callbacks
        callbacks_list = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.params.early_stopping_patience,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='best_model.keras',
                monitor='val_loss',
                save_best_only=True
            ),
            self.tensorboard_callback
        ]
        
        # Entraînement
        try:
            if futures_data is not None and self.params.use_futures:
                self.history = self.model.fit(
                    [X_train, futures_data[0], futures_data[1]], y_train,
                    validation_data=([X_val, futures_data[0], futures_data[1]], y_val),
                    epochs=self.params.epochs,
                    batch_size=self.params.batch_size,
                    callbacks=callbacks_list,
                    verbose=1
                )
            else:
                self.history = self.model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=self.params.epochs,
                    batch_size=self.params.batch_size,
                    callbacks=callbacks_list,
                    verbose=1
                )
                
            self.logger.info("Entraînement terminé avec succès")
            return self.history.history
            
        except Exception as e:
            self.logger.error(f"Erreur pendant l'entraînement: {str(e)}")
            raise RuntimeError(f"Erreur pendant l'entraînement: {str(e)}")

    def predict(self,
               X: np.ndarray,
               futures_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
               ):
        """
        Effectue des prédictions sur de nouvelles données.
        
        Args:
            X: Données d'entrée
            futures_data: Données des futures optionnelles
            
        Returns:
            np.ndarray: Prédictions
            
        Raises:
            ValueError: Si les données sont invalides
        """
        if X.ndim != 3:
            raise ValueError(f"X doit être 3D (batch, time_steps, features), mais a la forme {X.shape}")
            
        try:
            if futures_data is not None and self.params.use_futures:
                predictions = self.model.predict([X, futures_data[0], futures_data[1]])
            else:
                predictions = self.model.predict(X)
                
            return predictions
            
        except Exception as e:
            self.logger.error(f"Erreur pendant la prédiction: {str(e)}")
            raise RuntimeError(f"Erreur pendant la prédiction: {str(e)}")

    def load_model(self, path: str):
        """
        Charge un modèle sauvegardé.
        
        Args:
            path: Chemin vers le fichier du modèle
            
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            RuntimeError: Si le chargement échoue
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Le fichier modèle {path} n'existe pas")
            
        try:
            self.model = tf.keras.models.load_model(path)
            self.logger.info(f"Modèle chargé depuis {path}")
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
            raise RuntimeError(f"Erreur lors du chargement du modèle: {str(e)}")

    def save_model(self, path: str):
        """
        Sauvegarde le modèle.
        
        Args:
            path: Chemin où sauvegarder le modèle
            
        Raises:
            RuntimeError: Si la sauvegarde échoue
        """
        try:
            self.model.save(path)
            self.logger.info(f"Modèle sauvegardé à {path}")
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde du modèle: {str(e)}")
            raise RuntimeError(f"Erreur lors de la sauvegarde du modèle: {str(e)}")