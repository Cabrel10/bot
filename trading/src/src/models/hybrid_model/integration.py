from trading.models.hybrid_model.meta_learning.meta_trader import CryptoMetaTrader
from trading.models.hybrid_model.neat_evolution import NEATEvolution
from trading.models.hybrid_model.synthetic.gan import CryptoGAN
from typing import Dict, List, Optional, Union
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from .model import HybridModel
from tensorflow import keras
import asyncio
import gc

class DataPipeline:
    """Pipeline de données pour le traitement des données crypto."""
    
    def __init__(self, features: List[str], timeframes: List[str], batch_size: int):
        self.features = features
        self.timeframes = timeframes
        self.batch_size = batch_size
        
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Traite les données brutes."""
        processed_data = data.copy()
        # Assurez-vous que toutes les colonnes requises sont présentes
        for feature in self.features:
            if feature not in processed_data.columns:
                raise ValueError(f"Feature {feature} not found in data")
        
        # Suppression des valeurs manquantes
        processed_data = processed_data.dropna()
        
        return processed_data
        
    def create_batches(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """Crée des lots de données."""
        n_samples = len(data)
        return [data[i:i + self.batch_size] for i in range(0, n_samples, self.batch_size)]

class CryptoTradingSystem:
    """Système intégré de trading crypto combinant Meta-Learning, NEAT et GAN."""
    
    def __init__(self, config: Dict):
        """Initialisation du système de trading."""
        self.config = config
        self.logger = self._setup_logger()
        self.performance_metrics = {}
        self.monitoring_data = {}
        self._initialize_components()
        self._setup_monitoring()

    def _setup_logger(self):
        """Configure le logger pour le système."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _setup_monitoring(self):
        """Configure le système de monitoring."""
        self.monitoring_data = {
            'performance': {
                'returns': [],
                'sharpe_ratio': [],
                'max_drawdown': [],
                'win_rate': []
            },
            'model_metrics': {
                'prediction_accuracy': [],
                'confidence_scores': [],
                'regime_changes': []
            },
            'risk_metrics': {
                'var': [],
                'position_exposure': [],
                'volatility': []
            },
            'system_health': {
                'last_update': None,
                'errors': [],
                'warnings': [],
                'component_status': {}
            }
        }
        
        # Initialisation des métriques de performance
        self.performance_metrics = {
            'cumulative_return': 0.0,
            'current_drawdown': 0.0,
            'trades_count': 0,
            'successful_trades': 0
        }

    def _initialize_components(self):
        """Initialisation des composants avec leurs interactions."""
        try:
            # Composants principaux
            self.meta_trader = CryptoMetaTrader(self.config.get('meta_learning', {}))
            self.neat_evolution = NEATEvolution(self.config.get('neat', {}))
            self.gan = CryptoGAN(self.config.get('gan', {
                'latent_dim': 100,
                'generator_dim': 128,
                'discriminator_dim': 128,
                'learning_rate': 0.0002,
                'beta1': 0.5
            }))

            # Pipeline de données
            self.data_pipeline = DataPipeline(
                features=self.config.get('features', []),
                timeframes=self.config.get('timeframes', []),
                batch_size=self.config.get('batch_size', 32)
            )
            
            # Gestionnaire de régimes de marché
            self.regime_manager = MarketRegimeManager(self.config)
            
            # Gestionnaire de risques
            self.risk_manager = RiskManager(self.config)
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation des composants: {str(e)}")
            raise

    async def process_market_update(self, market_data: Dict) -> Dict[str, Union[float, str]]:
        """Traitement en temps réel des mises à jour du marché."""
        try:
            # 1. Détection du régime de marché
            current_regime = self.regime_manager.detect_regime(market_data)
            
            # 2. Génération de données synthétiques si nécessaire
            if self.config.get('use_synthetic', False):
                synthetic_data = await self._generate_synthetic_data(market_data)
                enhanced_data = self._enhance_dataset(market_data, synthetic_data)
            else:
                enhanced_data = market_data

            # 3. Adaptation du modèle au régime actuel
            if current_regime != self.regime_manager.last_regime:
                await self._adapt_to_new_regime(current_regime, enhanced_data)

            # 4. Prédiction et gestion des risques
            try:
                prediction = await self._make_prediction(enhanced_data)
                adjusted_signal = self.risk_manager.adjust_signal(prediction)
                confidence = 0.8  # Valeur fixe pour le moment
                risk_score = self.risk_manager.calculate_risk_score(market_data)
                
                return {
                    'signal': float(adjusted_signal),
                    'confidence': float(confidence),
                    'regime': current_regime,
                    'risk_score': float(risk_score)
                }
            except Exception as e:
                self.logger.error(f"Erreur lors de la prédiction: {str(e)}")
                return {
                    'signal': 0.0,
                    'confidence': 0.0,
                    'regime': current_regime,
                    'risk_score': 1.0
                }

        except Exception as e:
            self.logger.error(f"Erreur dans le traitement: {str(e)}")
            return {
                'signal': 0.0,
                'confidence': 0.0,
                'regime': 'ranging',
                'risk_score': 1.0
            }

    async def _adapt_to_new_regime(self, regime: str, data: Dict):
        """Adaptation du modèle à un nouveau régime de marché."""
        try:
            # Évolution de l'architecture si nécessaire
            if self.config.get('evolve_on_regime_change', False):
                evolved_genome = await asyncio.to_thread(
                    self.neat_evolution.evolve,
                    data,
                    generations=self.config.get('evolution_generations', 10)
                )
                if evolved_genome:
                    await asyncio.to_thread(self.meta_trader.update_architecture, evolved_genome)

            # Fine-tuning rapide sur le nouveau régime
            await asyncio.to_thread(
                self.meta_trader.quick_adapt,
                data,
                steps=self.config.get('adaptation_steps', 100),
                learning_rate=self.config.get('adaptation_lr', 0.001)
            )
        except Exception as e:
            self.logger.error(f"Erreur lors de l'adaptation au nouveau régime: {str(e)}")

    def _enhance_dataset(self, real_data: Dict, synthetic_data: pd.DataFrame) -> Dict:
        """Combine les données réelles et synthétiques."""
        try:
            if synthetic_data.empty:
                return real_data
                
            if isinstance(real_data, dict):
                real_df = pd.DataFrame([real_data])
            else:
                real_df = pd.DataFrame(real_data)
                
            # Combine les données en gardant les timestamps uniques
            combined = pd.concat([real_df, synthetic_data]).drop_duplicates()
            return combined.to_dict('records')[0]
        except Exception as e:
            self.logger.error(f"Erreur lors de la fusion des données: {str(e)}")
            return real_data

    async def cleanup(self):
        """Nettoyage des ressources."""
        try:
            # Nettoyage des composants
            if hasattr(self, 'gan'):
                self.gan.cleanup()
            if hasattr(self, 'meta_trader'):
                await asyncio.to_thread(self.meta_trader.cleanup)
            
            # Nettoyage de TensorFlow
            tf.keras.backend.clear_session()
            
        except Exception as e:
            self.logger.error(f"Erreur lors du nettoyage: {str(e)}")
            raise

    async def _generate_synthetic_data(self, data: Dict) -> pd.DataFrame:
        """Génère des données synthétiques basées sur les données réelles."""
        try:
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame([data])
            
            # Utilisation du GAN pour générer des données synthétiques
            synthetic_data = await asyncio.to_thread(
                self.gan.generate_data,
                data,
                n_samples=self.config.get('synthetic_samples', 100)
            )
            
            return synthetic_data
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération de données synthétiques: {str(e)}")
            return pd.DataFrame()

    async def _make_prediction(self, data: Dict) -> float:
        """Fait une prédiction basée sur les données du marché."""
        try:
            # Prétraitement des données
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                df = pd.DataFrame(data)
            
            # Normalisation des données
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            if not numeric_columns.empty:
                df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].mean()) / (df[numeric_columns].std() + 1e-8)
            
            # Prédiction via le meta-trader
            prediction = await asyncio.to_thread(
                self.meta_trader.predict,
                df
            )
            
            if isinstance(prediction, (list, np.ndarray)):
                return float(prediction[0])
            return float(prediction)
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la prédiction: {str(e)}")
            return 0.0

class MarketRegimeManager:
    """Gestionnaire des régimes de marché."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.current_regime = None
        self.regime_history = []
        self.transition_probs = {}
        self.HIGH_VOL_THRESHOLD = 0.02  # 2% de volatilité
        self.STRONG_TREND_THRESHOLD = 0.01  # 1% de tendance
        self.last_regime = None
        self.window_size = 20

    def _calculate_volatility(self, data: Dict) -> float:
        """Calcule la volatilité des prix."""
        try:
            if isinstance(data, dict) and 'close' in data:
                if isinstance(data['close'], (dict, pd.Series)):
                    prices = pd.Series(data['close'].values())
                else:
                    prices = pd.Series([data['close']])
                return prices.pct_change().std() if len(prices) > 1 else 0.0
            return 0.0
        except Exception as e:
            logging.error(f"Erreur dans le calcul de la volatilité: {str(e)}")
            return 0.0

    def _analyze_volume(self, data: Dict) -> float:
        """Analyse le volume des transactions."""
        try:
            if isinstance(data, dict) and 'volume' in data:
                if isinstance(data['volume'], (dict, pd.Series)):
                    volumes = pd.Series(data['volume'].values())
                else:
                    volumes = pd.Series([data['volume']])
                return volumes.pct_change().mean() if len(volumes) > 1 else 0.0
            return 0.0
        except Exception as e:
            logging.error(f"Erreur dans l'analyse du volume: {str(e)}")
            return 0.0

    def _detect_trend(self, data: Dict) -> float:
        """Détecte la tendance des prix."""
        try:
            if isinstance(data, dict) and 'close' in data:
                if isinstance(data['close'], (dict, pd.Series)):
                    prices = pd.Series(data['close'].values())
                else:
                    prices = pd.Series([data['close']])
                
                if len(prices) > 1:
                    x = np.arange(len(prices))
                    try:
                        slope, _ = np.polyfit(x, prices, 1)
                        return slope / (prices.mean() + 1e-8)  # Évite la division par zéro
                    except np.linalg.LinAlgError:
                        return 0.0
                return 0.0
            return 0.0
        except Exception as e:
            logging.error(f"Erreur dans la détection de tendance: {str(e)}")
            return 0.0

    def detect_regime(self, data: Dict) -> str:
        """Détection du régime de marché actuel."""
        try:
            volatility = self._calculate_volatility(data)
            volume = self._analyze_volume(data)
            trend = self._detect_trend(data)
            
            regime = self._classify_regime(volatility, volume, trend)
            self.last_regime = self.current_regime
            self.current_regime = regime
            self.regime_history.append(regime)
            
            return regime
        except Exception as e:
            logging.error(f"Erreur lors de la détection du régime: {str(e)}")
            return 'ranging'  # Régime par défaut plus sûr que 'unknown'

    def _classify_regime(self, volatility: float, volume: float, trend: float) -> str:
        """Classification du régime basée sur plusieurs métriques."""
        try:
            if volatility > self.HIGH_VOL_THRESHOLD:
                return 'volatile'
            elif abs(trend) > self.STRONG_TREND_THRESHOLD:
                return 'trending'
            else:
                return 'ranging'
        except Exception as e:
            logging.error(f"Erreur lors de la classification du régime: {str(e)}")
            return 'ranging'  # Régime par défaut

class RiskManager:
    """Gestionnaire de risque pour le système de trading."""
    
    def __init__(self, config=None):
        self.default_config = {
            'base_risk_factor': 0.8,
            'market_risk_weight': 0.4,
            'position_risk_weight': 0.3,
            'volatility_risk_weight': 0.3,
            'max_position_size': 1.0,
            'risk_free_rate': 0.02
        }
        self.config = {**self.default_config, **(config or {})}
        self.logger = logging.getLogger(__name__)
        self.current_risk_score = 0.0
        self.position_sizes = {}
        self.risk_history = []
        self.max_position_size = self.config['max_position_size']
        self.risk_free_rate = self.config['risk_free_rate']

    def _assess_market_risk(self) -> float:
        """Évalue le risque de marché."""
        return min(1.0, self.current_risk_score)

    def _assess_position_risk(self) -> float:
        """Évalue le risque des positions."""
        total_exposure = sum(self.position_sizes.values())
        return min(1.0, total_exposure / self.max_position_size)

    def _assess_volatility_risk(self) -> float:
        """Évalue le risque de volatilité."""
        if not self.risk_history:
            return 0.5
        recent_risks = self.risk_history[-20:]  # Utilise les 20 dernières mesures
        return min(1.0, np.std(recent_risks))

    def adjust_signal(self, raw_signal: float) -> float:
        """Ajustement du signal en fonction du risque."""
        try:
            risk_factor = self._calculate_risk_factor()
            adjusted_signal = raw_signal * risk_factor
            
            # Mise à jour de l'historique
            self.risk_history.append(risk_factor)
            return adjusted_signal
        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajustement du signal: {str(e)}")
            return 0.0

    def _calculate_risk_factor(self) -> float:
        """Calcul du facteur de risque basé sur plusieurs métriques."""
        try:
            market_risk = self._assess_market_risk()
            position_risk = self._assess_position_risk()
            volatility_risk = self._assess_volatility_risk()
            
            return min(
                1.0,
                self.config['base_risk_factor'] * (
                    market_risk * self.config['market_risk_weight'] +
                    position_risk * self.config['position_risk_weight'] +
                    volatility_risk * self.config['volatility_risk_weight']
                )
            )
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul du facteur de risque: {str(e)}")
            return 0.5

    def calculate_risk_score(self, market_data: Dict) -> float:
        """Calcule le score de risque basé sur les données du marché."""
        try:
            if isinstance(market_data, dict):
                # Calcul de la volatilité
                if 'close' in market_data:
                    prices = pd.Series(market_data['close'] if isinstance(market_data['close'], (list, np.ndarray)) else [market_data['close']])
                    volatility = prices.pct_change().std() if len(prices) > 1 else 0.0
                else:
                    volatility = 0.0

                # Calcul du volume relatif
                if 'volume' in market_data:
                    volume = pd.Series(market_data['volume'] if isinstance(market_data['volume'], (list, np.ndarray)) else [market_data['volume']])
                    volume_risk = volume.pct_change().abs().mean() if len(volume) > 1 else 0.0
                else:
                    volume_risk = 0.0

                # Score de risque combiné
                risk_score = (
                    volatility * self.config['volatility_risk_weight'] +
                    volume_risk * self.config['market_risk_weight'] +
                    self._assess_position_risk() * self.config['position_risk_weight']
                )

                self.current_risk_score = min(1.0, max(0.0, risk_score))
                return self.current_risk_score
            
            return 0.5  # Valeur par défaut si les données ne sont pas valides
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul du score de risque: {str(e)}")
            return 0.5  # Valeur par défaut en cas d'erreur

class CryptoMetaTrader:
    """Gestionnaire de trading basé sur le meta-learning."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = self._build_model()
        self.optimizer = None
        self.logger = logging.getLogger(__name__)

    def _build_model(self):
        """Construction du modèle de base."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='tanh')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Fait une prédiction sur les données d'entrée."""
        try:
            if self.model is None:
                raise ValueError("Le modèle n'est pas initialisé")
                
            # Préparation des données
            features = data[self.config.get('features', ['close', 'high', 'low', 'open', 'volume'])]
            
            # Prédiction
            prediction = self.model.predict(features)
            return prediction
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la prédiction: {str(e)}")
            return np.array([0.0])

    def get_confidence(self) -> float:
        """Retourne le niveau de confiance du modèle."""
        return 0.8  # Valeur par défaut pour le moment

    async def cleanup(self):
        """Nettoie les ressources utilisées par le trader."""
        try:
            if self.model is not None:
                tf.keras.backend.clear_session()
            
            # Libération de la mémoire
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"Erreur lors du nettoyage du MetaTrader: {str(e)}") 