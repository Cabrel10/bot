import tensorflow as tf
from typing import Dict, List
import numpy as np
import pandas as pd

class LightGAN:
    """Implémentation légère d'un GAN pour la génération de données synthétiques."""
    
    def __init__(self, config: Dict):
        """Initialisation du GAN."""
        self.config = config
        self.latent_dim = config['gan']['latent_dim']
        self.generator_dim = config['gan']['generator_dim']
        self.discriminator_dim = config['gan']['discriminator_dim']
        self.learning_rate = config['gan']['learning_rate']
        self.beta1 = config['gan']['beta1']
        
        # Construction des modèles
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        
        # Optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta1)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta1)

    def _build_generator(self) -> tf.keras.Model:
        """Construction du générateur."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.generator_dim, input_shape=(self.latent_dim,)),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dense(self.generator_dim * 2),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dense(self.generator_dim * 4),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dense(len(self.config['features']), activation='tanh')
        ])
        return model

    def _build_discriminator(self) -> tf.keras.Model:
        """Construction du discriminateur."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.discriminator_dim, 
                                input_shape=(len(self.config['features']),)),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(self.discriminator_dim // 2),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model

    def generate_data(self, n_samples: int) -> pd.DataFrame:
        """Génère des données synthétiques."""
        noise = tf.random.normal([n_samples, self.latent_dim])
        generated_data = self.generator(noise, training=False)
        
        # Conversion en DataFrame
        df = pd.DataFrame(
            generated_data.numpy(),
            columns=self.config['features']
        )
        return df

    @tf.function
    def _train_step(self, real_data: tf.Tensor):
        """Une étape d'entraînement du GAN."""
        batch_size = tf.shape(real_data)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = self.generator(noise, training=True)
            
            real_output = self.discriminator(real_data, training=True)
            fake_output = self.discriminator(generated_data, training=True)
            
            gen_loss = self._generator_loss(fake_output)
            disc_loss = self._discriminator_loss(real_output, fake_output)
        
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        self.generator_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables))
        
        return gen_loss, disc_loss

    def _generator_loss(self, fake_output):
        """Calcul de la perte du générateur."""
        return tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                tf.ones_like(fake_output), fake_output))

    def _discriminator_loss(self, real_output, fake_output):
        """Calcul de la perte du discriminateur."""
        real_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                tf.ones_like(real_output), real_output))
        fake_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                tf.zeros_like(fake_output), fake_output))
        return real_loss + fake_loss

    def train(self, real_data: pd.DataFrame):
        """Entraînement du GAN."""
        # Implémentation d'entraînement léger 