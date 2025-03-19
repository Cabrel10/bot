"""
Analyse des sentiments et de la volatilité pour le modèle hybride.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from datetime import datetime, timedelta

@dataclass
class SentimentVolatilityConfig:
    """Configuration pour l'analyse des sentiments et de la volatilité."""
    window_size: int = 24  # heures
    min_samples: int = 100
    confidence_threshold: float = 0.7
    sentiment_threshold: float = 0.3
    volatility_threshold: float = 0.2
    max_sequence_length: int = 100
    embedding_dim: int = 100
    lstm_units: int = 64
    dense_units: int = 32
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50
    early_stopping_patience: int = 10
    min_delta: float = 0.001

class SentimentVolatilityAnalyzer:
    """Analyseur de sentiments et de volatilité."""
    
    def __init__(self, config: SentimentVolatilityConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.sentiment_model = self._build_sentiment_model()
        self.volatility_model = self._build_volatility_model()
        self._setup_nltk()
        self._setup_logging()
        
    def _setup_nltk(self):
        """Configure NLTK."""
        try:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
        except Exception as e:
            self.logger.error(f"Erreur lors de la configuration de NLTK: {e}")
            raise
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
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
    
    def _build_sentiment_model(self) -> models.Model:
        """Construit le modèle d'analyse des sentiments."""
        model = models.Sequential([
            layers.Embedding(
                input_dim=10000,
                output_dim=self.config.embedding_dim,
                input_length=self.config.max_sequence_length
            ),
            layers.LSTM(self.config.lstm_units),
            layers.Dense(self.config.dense_units, activation='relu'),
            layers.Dropout(self.config.dropout_rate),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_volatility_model(self) -> models.Model:
        """Construit le modèle de prédiction de volatilité."""
        model = models.Sequential([
            layers.LSTM(
                self.config.lstm_units,
                input_shape=(self.config.window_size, 1),
                return_sequences=True
            ),
            layers.LSTM(self.config.lstm_units),
            layers.Dense(self.config.dense_units, activation='relu'),
            layers.Dropout(self.config.dropout_rate),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def preprocess_text(self, text: str) -> str:
        """
        Prétraite le texte pour l'analyse des sentiments.
        
        Args:
            text: Texte à prétraiter
            
        Returns:
            Texte prétraité
        """
        try:
            # Conversion en minuscules
            text = text.lower()
            
            # Suppression des caractères spéciaux
            text = re.sub(r'[^\w\s]', '', text)
            
            # Tokenization
            tokens = word_tokenize(text)
            
            # Suppression des stop words et lemmatization
            tokens = [
                self.lemmatizer.lemmatize(token)
                for token in tokens
                if token not in self.stop_words
            ]
            
            return ' '.join(tokens)
            
        except Exception as e:
            self.logger.error(f"Erreur lors du prétraitement du texte: {e}")
            raise
    
    def analyze_sentiment(
        self,
        text: str,
        confidence_threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Analyse le sentiment d'un texte.
        
        Args:
            text: Texte à analyser
            confidence_threshold: Seuil de confiance (optionnel)
            
        Returns:
            Dictionnaire des scores de sentiment
        """
        try:
            # Prétraitement
            processed_text = self.preprocess_text(text)
            
            # Analyse avec TextBlob
            blob = TextBlob(processed_text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Analyse avec le modèle
            sequence = self._text_to_sequence(processed_text)
            model_score = self.sentiment_model.predict(sequence)[0][0]
            
            # Combinaison des scores
            threshold = confidence_threshold or self.config.confidence_threshold
            if abs(polarity) > threshold and abs(model_score - 0.5) > threshold:
                final_score = (polarity + (model_score - 0.5) * 2) / 2
            else:
                final_score = polarity
            
            return {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'model_score': model_score,
                'final_score': final_score,
                'confidence': abs(final_score)
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse des sentiments: {e}")
            raise
    
    def predict_volatility(
        self,
        data: pd.DataFrame,
        window_size: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Prédit la volatilité future.
        
        Args:
            data: Données historiques
            window_size: Taille de la fenêtre (optionnel)
            
        Returns:
            Dictionnaire des prédictions de volatilité
        """
        try:
            if len(data) < self.config.min_samples:
                self.logger.warning(
                    f"Pas assez d'échantillons pour la prédiction: {len(data)} < {self.config.min_samples}"
                )
                return {}
            
            # Préparation des données
            window = window_size or self.config.window_size
            scaled_data = self.scaler.fit_transform(data[['close']])
            
            # Création des séquences
            sequences = []
            for i in range(len(scaled_data) - window):
                sequences.append(scaled_data[i:i+window])
            
            sequences = np.array(sequences)
            
            # Prédiction
            predictions = self.volatility_model.predict(sequences)
            
            # Calcul des métriques
            current_volatility = np.std(scaled_data[-window:])
            predicted_volatility = predictions[-1][0]
            volatility_change = (predicted_volatility - current_volatility) / current_volatility
            
            return {
                'current_volatility': current_volatility,
                'predicted_volatility': predicted_volatility,
                'volatility_change': volatility_change,
                'confidence': 1 - abs(volatility_change)
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la prédiction de volatilité: {e}")
            raise
    
    def analyze_market_regime(
        self,
        data: pd.DataFrame,
        sentiment_scores: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Analyse le régime de marché.
        
        Args:
            data: Données de marché
            sentiment_scores: Scores de sentiment
            
        Returns:
            Dictionnaire des caractéristiques du régime
        """
        try:
            # Calcul de la volatilité
            volatility = np.std(data['close'].pct_change())
            
            # Calcul des scores de sentiment moyens
            avg_polarity = np.mean([s['polarity'] for s in sentiment_scores])
            avg_subjectivity = np.mean([s['subjectivity'] for s in sentiment_scores])
            
            # Détermination du régime
            if volatility > self.config.volatility_threshold:
                regime = 'high_volatility'
            elif volatility < self.config.volatility_threshold / 2:
                regime = 'low_volatility'
            else:
                regime = 'medium_volatility'
            
            # Analyse des sentiments
            if avg_polarity > self.config.sentiment_threshold:
                sentiment = 'bullish'
            elif avg_polarity < -self.config.sentiment_threshold:
                sentiment = 'bearish'
            else:
                sentiment = 'neutral'
            
            return {
                'regime': regime,
                'sentiment': sentiment,
                'volatility': volatility,
                'avg_polarity': avg_polarity,
                'avg_subjectivity': avg_subjectivity,
                'confidence': 1 - abs(avg_polarity)
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse du régime: {e}")
            raise
    
    def _text_to_sequence(self, text: str) -> np.ndarray:
        """
        Convertit le texte en séquence numérique.
        
        Args:
            text: Texte à convertir
            
        Returns:
            Séquence numérique
        """
        # À implémenter selon le tokenizer utilisé
        # Pour l'instant, retourne une séquence aléatoire
        return np.random.randint(0, 10000, (1, self.config.max_sequence_length))
    
    def train_sentiment_model(
        self,
        texts: List[str],
        labels: np.ndarray,
        validation_split: float = 0.2
    ):
        """
        Entraîne le modèle d'analyse des sentiments.
        
        Args:
            texts: Textes d'entraînement
            labels: Labels d'entraînement
            validation_split: Proportion de validation
        """
        try:
            # Préparation des données
            processed_texts = [self.preprocess_text(text) for text in texts]
            sequences = np.array([self._text_to_sequence(text) for text in processed_texts])
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.early_stopping_patience,
                    min_delta=self.config.min_delta,
                    restore_best_weights=True
                )
            ]
            
            # Entraînement
            self.sentiment_model.fit(
                sequences,
                labels,
                validation_split=validation_split,
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                callbacks=callbacks,
                verbose=0
            )
            
            self.logger.info("Modèle de sentiment entraîné avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'entraînement du modèle de sentiment: {e}")
            raise
    
    def train_volatility_model(
        self,
        data: pd.DataFrame,
        window_size: Optional[int] = None,
        validation_split: float = 0.2
    ):
        """
        Entraîne le modèle de prédiction de volatilité.
        
        Args:
            data: Données d'entraînement
            window_size: Taille de la fenêtre (optionnel)
            validation_split: Proportion de validation
        """
        try:
            if len(data) < self.config.min_samples:
                self.logger.warning(
                    f"Pas assez d'échantillons pour l'entraînement: {len(data)} < {self.config.min_samples}"
                )
                return
            
            # Préparation des données
            window = window_size or self.config.window_size
            scaled_data = self.scaler.fit_transform(data[['close']])
            
            # Création des séquences
            sequences = []
            targets = []
            for i in range(len(scaled_data) - window):
                sequences.append(scaled_data[i:i+window])
                targets.append(scaled_data[i+window])
            
            sequences = np.array(sequences)
            targets = np.array(targets)
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.early_stopping_patience,
                    min_delta=self.config.min_delta,
                    restore_best_weights=True
                )
            ]
            
            # Entraînement
            self.volatility_model.fit(
                sequences,
                targets,
                validation_split=validation_split,
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                callbacks=callbacks,
                verbose=0
            )
            
            self.logger.info("Modèle de volatilité entraîné avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'entraînement du modèle de volatilité: {e}")
            raise 