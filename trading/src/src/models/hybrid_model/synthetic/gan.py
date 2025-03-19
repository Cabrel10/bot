import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Sequential, Model
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
import asyncio
import gc

class CryptoGAN:
    """Classe pour générer des données synthétiques en utilisant un GAN."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.latent_dim = config.get('latent_dim', 100)
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.gan = self._build_gan()
        self.logger = logging.getLogger(__name__)

    def _build_generator(self):
        inputs = Input(shape=(self.latent_dim,))
        x = Dense(256)(inputs)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        x = Dense(512)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        x = Dense(1024)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        outputs = Dense(self.config.get('n_features', 5), activation='tanh')(x)
        
        return Model(inputs, outputs)

    def _build_discriminator(self):
        inputs = Input(shape=(self.config.get('n_features', 5),))
        x = Dense(512)(inputs)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(256)(x)
        x = LeakyReLU(alpha=0.2)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs)

    def _build_gan(self):
        self.discriminator.trainable = False
        gan_input = Input(shape=(self.latent_dim,))
        x = self.generator(gan_input)
        gan_output = self.discriminator(x)
        return Model(gan_input, gan_output)

    def generate_data(self, real_data: Dict, n_samples: int = 1000, regime: str = None) -> pd.DataFrame:
        """Génère des données synthétiques adaptées au régime actuel."""
        try:
            # Préparation des données réelles
            if isinstance(real_data, dict):
                if isinstance(next(iter(real_data.values())), dict):
                    # Si les valeurs sont des dictionnaires, on prend leurs valeurs
                    real_df = pd.DataFrame({k: list(v.values()) if isinstance(v, dict) else [v] 
                                          for k, v in real_data.items()})
                else:
                    real_df = pd.DataFrame([real_data])
            else:
                real_df = pd.DataFrame(real_data)

            # Vérification des colonnes requises
            required_columns = ['close', 'high', 'low', 'open', 'volume']
            missing_columns = [col for col in required_columns if col not in real_df.columns]
            if missing_columns:
                self.logger.warning(f"Colonnes manquantes: {missing_columns}")
                for col in missing_columns:
                    real_df[col] = 0.0

            # Génération du bruit latent
            noise = tf.random.normal([n_samples, self.latent_dim])
            
            # Ajout du conditionnement sur le régime si spécifié
            if regime:
                regime_embedding = self._get_regime_embedding(regime)
                noise = tf.concat([noise, regime_embedding], axis=1)

            # Génération des données
            with tf.device('/CPU:0'):  # Force l'utilisation du CPU
                generated_data = self.generator.predict(noise, batch_size=32)
            
            # Conversion en DataFrame avec les bonnes colonnes
            synthetic_df = pd.DataFrame(
                generated_data,
                columns=required_columns[:generated_data.shape[1]]
            )
            
            # Post-traitement pour assurer la cohérence des données
            synthetic_df = self._post_process_data(synthetic_df, real_df)
            
            return synthetic_df

        except Exception as e:
            self.logger.error(f"Erreur lors de la génération de données: {str(e)}")
            return pd.DataFrame()

    def _get_regime_embedding(self, regime: str) -> tf.Tensor:
        """Crée un embedding pour le régime de marché."""
        regime_map = {
            'volatile': [1.0, 0.0, 0.0],
            'trending': [0.0, 1.0, 0.0],
            'ranging': [0.0, 0.0, 1.0],
            'unknown': [0.33, 0.33, 0.33]
        }
        embedding = regime_map.get(regime, regime_map['unknown'])
        return tf.tile(tf.constant([embedding]), [self.latent_dim, 1])

    def _post_process_data(self, synthetic_df: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:
        """Post-traitement des données synthétiques pour assurer leur cohérence."""
        try:
            # Assurer que high >= low
            if 'high' in synthetic_df.columns and 'low' in synthetic_df.columns:
                mask = synthetic_df['high'] < synthetic_df['low']
                synthetic_df.loc[mask, ['high', 'low']] = synthetic_df.loc[mask, ['low', 'high']].values

            # Assurer que le volume est positif
            if 'volume' in synthetic_df.columns:
                synthetic_df['volume'] = np.abs(synthetic_df['volume'])

            # Normalisation des données dans la plage des données réelles
            for col in synthetic_df.columns:
                if col in real_df.columns:
                    real_values = real_df[col].values if isinstance(real_df[col], pd.Series) else real_df[col]
                    min_val = np.min(real_values)
                    max_val = np.max(real_values)
                    if min_val != max_val:
                        synthetic_df[col] = min_val + (max_val - min_val) * (synthetic_df[col] - synthetic_df[col].min()) / (synthetic_df[col].max() - synthetic_df[col].min())
                    else:
                        synthetic_df[col] = min_val

            return synthetic_df

        except Exception as e:
            self.logger.error(f"Erreur lors du post-traitement: {str(e)}")
            return synthetic_df

    def cleanup(self):
        """Nettoie les ressources utilisées par le GAN."""
        try:
            if hasattr(self, 'generator'):
                tf.keras.backend.clear_session()
            
            # Libération de la mémoire
            gc.collect()
            
        except Exception as e:
            logging.error(f"Erreur lors du nettoyage du GAN: {str(e)}") 