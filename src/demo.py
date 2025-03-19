"""
Démo du système de trading hybride.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Utiliser le backend non-graphique
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Génère des données de test pour la démonstration."""
    dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='h')
    
    # Simuler un cours avec tendance et cycles
    t = np.linspace(0, 4*np.pi, n_samples)
    trend = np.cumsum(np.random.normal(0, 0.01, n_samples)) + np.linspace(0, 10, n_samples)
    cycle1 = 3 * np.sin(t)
    cycle2 = 2 * np.sin(t/4)
    noise = np.random.normal(0, 1, n_samples)
    
    # Prix de clôture
    close_prices = 100 + trend + cycle1 + cycle2 + noise
    
    # Prix d'ouverture, haut et bas
    open_prices = close_prices - np.random.normal(0, 1, n_samples)
    high_prices = np.maximum(close_prices, open_prices) + np.random.exponential(0.5, n_samples)
    low_prices = np.minimum(close_prices, open_prices) - np.random.exponential(0.5, n_samples)
    
    # Volume
    volume = np.random.exponential(1000, n_samples) * (1 + 0.5 * np.sin(t/8))
    
    # Création du DataFrame
    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)
    
    return data

def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des indicateurs techniques aux données."""
    logger.info("Ajout des indicateurs techniques...")
    
    # Copie des données pour éviter les avertissements SettingWithCopyWarning
    df = data.copy()
    
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

def create_lstm_model(seq_length: int, features: int) -> tf.keras.Model:
    """Crée un modèle LSTM pour la prédiction de prix."""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(seq_length, features), return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def prepare_sequences(data: pd.DataFrame, seq_length: int = 10) -> tuple:
    """Prépare les séquences pour l'entraînement du modèle LSTM."""
    # Sélection des features
    features = ['close', 'volume', 'rsi', 'macd', 'macd_signal', 'volatility']
    dataset = data[features].values
    
    # Normalisation
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataset)
    
    # Création des séquences
    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i+seq_length])
        y.append(scaled_data[i+seq_length, 0])  # Prédire le prochain prix de clôture
    
    return np.array(X), np.array(y), scaler

def plot_predictions(data, predictions, title="Prédiction de prix"):
    """Sauvegarde les prédictions vs les valeurs réelles."""
    plt.figure(figsize=(14, 7))
    plt.plot(data.index[-len(predictions):], data['close'].values[-len(predictions):], label='Réel')
    plt.plot(data.index[-len(predictions):], predictions, label='Prédiction', alpha=0.7)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Prix')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Sauvegarde sans affichage
    plt.savefig('prediction.png')
    plt.close()
    
    logger.info("Graphique sauvegardé dans 'prediction.png'")

def detect_anomalies(data: pd.DataFrame) -> np.ndarray:
    """Détecte les anomalies dans les données."""
    logger.info("Détection des anomalies...")
    
    # Z-score pour la détection simple
    def z_score_detection(series, threshold=3.0):
        mean = series.mean()
        std = series.std()
        z_scores = abs((series - mean) / std)
        return z_scores > threshold
    
    # Détection sur les prix et volumes
    price_anomalies = z_score_detection(data['close'])
    volume_anomalies = z_score_detection(data['volume'])
    
    # Combinaison
    combined_anomalies = price_anomalies | volume_anomalies
    
    logger.info(f"Nombre d'anomalies détectées: {combined_anomalies.sum()}")
    return combined_anomalies

def sentiment_analysis_simulation(n_samples: int) -> pd.Series:
    """Simule une analyse de sentiment pour la démo."""
    logger.info("Simulation d'analyse de sentiment...")
    
    # Simuler un sentiment de base avec une tendance et du bruit
    t = np.linspace(0, 4*np.pi, n_samples)
    trend = np.cumsum(np.random.normal(0, 0.01, n_samples))
    cycle = 0.3 * np.sin(t/2)
    noise = np.random.normal(0, 0.1, n_samples)
    
    sentiment = 0.5 + trend + cycle + noise
    # Normaliser entre -1 et 1
    sentiment = np.clip(sentiment, -1, 1)
    
    return pd.Series(sentiment)

def main():
    """Fonction principale de la démo."""
    logger.info("Démarrage de la démo du système de trading hybride...")
    
    # Génération des données
    data = generate_test_data(1000)
    logger.info(f"Données générées: {data.shape}")
    
    # Ajout des indicateurs techniques
    data_with_indicators = add_technical_indicators(data)
    logger.info(f"Données avec indicateurs: {data_with_indicators.shape}")
    
    # Détection des anomalies
    anomalies = detect_anomalies(data_with_indicators)
    
    # Simulation d'analyse de sentiment
    sentiment = sentiment_analysis_simulation(len(data_with_indicators))
    data_with_indicators['sentiment'] = sentiment
    
    # Création d'un modèle de prédiction simple
    seq_length = 20
    
    # Préparation des données
    X, y, scaler = prepare_sequences(data_with_indicators, seq_length)
    n_features = X.shape[2]
    
    # Séparation train/test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Création et entraînement du modèle
    logger.info("Création et entraînement du modèle...")
    model = create_lstm_model(seq_length, n_features)
    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        ]
    )
    
    # Prédictions
    logger.info("Génération des prédictions...")
    predictions = model.predict(X_test)
    
    # Conversion des prédictions normalisées en valeurs réelles
    # On récupère les lignes originales pour reconstruire les features complètes
    original_features = np.zeros((len(predictions), 6))  # Nombre de features sélectionnées
    
    # On remplit avec les valeurs du dernier timestep de chaque séquence
    for i in range(len(X_test)):
        original_features[i] = X_test[i, -1, :]
    
    # On remplace la prédiction (première colonne - prix de clôture)
    original_features[:, 0] = predictions.flatten()
    
    # On inverse la normalisation
    predictions_inverse = scaler.inverse_transform(original_features)[:, 0]
    
    # Obtenir les vraies valeurs
    true_prices = data_with_indicators['close'].values[split+seq_length:]
    
    # Affichage des résultats
    rmse = np.sqrt(np.mean((predictions_inverse - true_prices[:len(predictions_inverse)]) ** 2))
    logger.info(f"RMSE: {rmse}")
    
    # Plot des prédictions
    plot_predictions(
        data_with_indicators.iloc[split:],
        predictions_inverse,
        "Prédiction des prix avec LSTM"
    )
    
    logger.info("Démo terminée avec succès!")

if __name__ == "__main__":
    main() 