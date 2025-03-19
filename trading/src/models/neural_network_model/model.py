from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime
import os
from tensorflow.keras import mixed_precision
from tensorflow.data.experimental import AUTOTUNE
import psutil
import multiprocessing

class NeuralNetworkModel:
    """Neural network model for time series prediction using LSTM architecture."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize avec optimisations."""
        self._setup_compute_optimizations()
        self.config = config or self._default_config()
        self.model = None
        self.history = None
        self._build_model()

    def _setup_compute_optimizations(self):
        """Configure les optimisations de calcul."""
        # Optimisation CPU multi-coeurs
        cpu_count = multiprocessing.cpu_count()
        tf.config.threading.set_inter_op_parallelism_threads(cpu_count)
        tf.config.threading.set_intra_op_parallelism_threads(cpu_count)

        # Activation de la mise en cache XLA
        tf.config.optimizer.set_jit(True)

        # Configuration de la mémoire GPU si disponible
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        # Optimisation de la précision mixte
        mixed_precision.set_global_policy('mixed_float16')

    def _default_config(self) -> Dict:
        """Configuration optimisée pour 16GB RAM."""
        return {
            'input_shape': (60, 15),
            'lstm_layers': [
                {'units': 128, 'return_sequences': True},
                {'units': 64, 'return_sequences': True},
                {'units': 32, 'return_sequences': False}
            ],
            'dense_layers': [
                {'units': 32},
                {'units': 16},
                {'units': 1}
            ],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 256,  # Augmenté pour de meilleures performances
            'epochs': 100,
            'early_stopping_patience': 10,
            'model_path': 'models/saved_models',
            'memory_management': {
                'prefetch_buffer_size': AUTOTUNE,
                'cache_data': True,
                'max_memory_percent': 75  # Limite d'utilisation RAM
            }
        }

    def _create_data_pipeline(self, X: np.ndarray, y: np.ndarray) -> tf.data.Dataset:
        """Crée un pipeline de données optimisé."""
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        # Optimisations du pipeline
        dataset = dataset.cache() if self.config['memory_management']['cache_data'] else dataset
        dataset = dataset.batch(self.config['batch_size'])
        dataset = dataset.prefetch(self.config['memory_management']['prefetch_buffer_size'])
        
        return dataset

    def _build_model(self) -> None:
        """Build optimisé du modèle."""
        # Optimisations TensorFlow pour CPU
        tf.config.threading.set_inter_op_parallelism_threads(2)  # Limite les threads
        tf.config.threading.set_intra_op_parallelism_threads(2)
        
        model = Sequential()

        # LSTM layers
        for i, lstm_config in enumerate(self.config['lstm_layers']):
            if i == 0:
                model.add(LSTM(
                    units=lstm_config['units'],
                    return_sequences=lstm_config['return_sequences'],
                    input_shape=self.config['input_shape']
                ))
            else:
                model.add(LSTM(
                    units=lstm_config['units'],
                    return_sequences=lstm_config['return_sequences']
                ))
            model.add(BatchNormalization())
            model.add(Dropout(self.config['dropout_rate']))

        # Dense layers
        for dense_config in self.config['dense_layers'][:-1]:
            model.add(Dense(
                units=dense_config['units'],
                activation='relu'
            ))
            model.add(Dropout(self.config['dropout_rate']))

        # Output layer
        model.add(Dense(
            units=self.config['dense_layers'][-1]['units'],
            activation='linear'
        ))

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )

        self.model = model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Entraînement avec pipeline optimisé."""
        # Vérification de la mémoire disponible
        mem = psutil.virtual_memory()
        if mem.percent > self.config['memory_management']['max_memory_percent']:
            raise MemoryError("Utilisation mémoire trop élevée")

        # Création des pipelines de données
        train_dataset = self._create_data_pipeline(X_train, y_train)
        val_dataset = self._create_data_pipeline(X_val, y_val)

        # Configuration des callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    self.config['model_path'],
                    f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5'
                ),
                monitor='val_loss',
                save_best_only=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            # Moniteur de ressources personnalisé
            ResourceMonitor(
                memory_limit=self.config['memory_management']['max_memory_percent']
            )
        ]

        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config['epochs'],
            callbacks=callbacks,
            verbose=1,
            use_multiprocessing=True,
            workers=multiprocessing.cpu_count()
        )

        return self.history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model.

        Args:
            X: Input features

        Returns:
            Model predictions
        """
        if self.model is None:
            raise ValueError("Model not initialized or trained.")

        return self.model.predict(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on test data.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not initialized or trained.")

        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        return {
            'loss': loss,
            'mae': mae
        }

    def save_model(self, filepath: str) -> None:
        """Save the model to disk.

        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save.")

        self.model.save(filepath)

    def load_model(self, filepath: str) -> None:
        """Load a saved model from disk.

        Args:
            filepath: Path to the saved model
        """
        self.model = load_model(filepath)

    def get_model_summary(self) -> str:
        """Get a string summary of the model architecture.

        Returns:
            Model summary string
        """
        if self.model is None:
            raise ValueError("Model not initialized.")

        # Create a string buffer to capture the summary
        import io
        buffer = io.StringIO()
        self.model.summary(print_fn=lambda x: buffer.write(x + '\n'))
        return buffer.getvalue()

    def update_config(self, new_config: Dict) -> None:
        """Update the model configuration and rebuild the model.

        Args:
            new_config: New configuration dictionary
        """
        self.config.update(new_config)
        self._build_model()

class ResourceMonitor(tf.keras.callbacks.Callback):
    """Moniteur de ressources personnalisé."""
    def __init__(self, memory_limit: float):
        super().__init__()
        self.memory_limit = memory_limit

    def on_epoch_end(self, epoch, logs=None):
        mem = psutil.virtual_memory()
        if mem.percent > self.memory_limit:
            print(f"\nAttention: Utilisation mémoire élevée ({mem.percent}%)")

# Example usage
if __name__ == "__main__":
    # Create model with default configuration
    model = NeuralNetworkModel()

    # Generate some dummy data
    sequence_length, n_features = 60, 15
    n_samples = 1000

    X = np.random.random((n_samples, sequence_length, n_features))
    y = np.random.random((n_samples, 1))

    # Split data
    train_size = int(0.8 * n_samples)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    try:
        # Train model
        history = model.train(X_train, y_train, X_val, y_val)
        print("Training completed successfully")

        # Make predictions
        predictions = model.predict(X_val)
        print("Predictions shape:", predictions.shape)

        # Evaluate model
        metrics = model.evaluate(X_val, y_val)
        print("Evaluation metrics:", metrics)

        # Save model
        model.save_model('example_model.h5')
        print("Model saved successfully")

    except Exception as e:
        print(f"Error during model training: {e}")
