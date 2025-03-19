"""
Interface principale Streamlit pour le trading bot.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys
import os
import yaml
import json
from datetime import datetime, timedelta

# Assurez-vous que le répertoire racine du projet est dans le chemin Python
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.core.data.data_acquisition import BatchDataAcquisition, MarketDataAcquisition
from src.core.models.neural_network import NeuralNetworkParams, TradingNeuralNetworkModel

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonction pour charger la configuration
def load_config(config_path):
    """Charge la configuration depuis un fichier YAML."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        st.error(f"Erreur lors du chargement de la configuration: {str(e)}")
        return {}

# Fonction pour afficher les données de marché
def display_market_data(data):
    """Affiche les données de marché sous forme de graphique."""
    if data is None or data.empty:
        st.warning("Aucune donnée disponible.")
        return
    
    fig = go.Figure()
    
    # Graphique des prix
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name="OHLC"
    ))
    
    # Volume en bas du graphique
    if 'volume' in data.columns:
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['volume'],
            name="Volume",
            marker_color='rgba(0, 0, 255, 0.3)',
            yaxis="y2"
        ))
    
    # Mise en page
    fig.update_layout(
        title="Données de marché",
        xaxis_title="Date",
        yaxis_title="Prix",
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        height=600,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Fonction pour afficher les prédictions du modèle
def display_predictions(data, predictions):
    """Affiche les prédictions du modèle avec les données réelles."""
    if data is None or data.empty or predictions is None:
        st.warning("Aucune prédiction disponible.")
        return
    
    # Préparer les données pour l'affichage
    df = data.copy()
    df['prediction'] = np.nan
    
    # Ajouter les prédictions aux données
    prediction_dates = df.index[-len(predictions):]
    for i, date in enumerate(prediction_dates):
        df.loc[date, 'prediction'] = predictions[i][0]
    
    # Créer le graphique
    fig = go.Figure()
    
    # Données réelles
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['close'],
        mode='lines',
        name="Prix réel",
        line=dict(color='blue')
    ))
    
    # Prédictions
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['prediction'],
        mode='lines',
        name="Prédiction",
        line=dict(color='red', dash='dash')
    ))
    
    # Mise en page
    fig.update_layout(
        title="Prédictions du modèle",
        xaxis_title="Date",
        yaxis_title="Prix",
        height=500,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Fonction principale
def main():
    """Fonction principale de l'interface."""
    st.title("Trading Bot Dashboard")
    
    # Sidebar pour la configuration
    st.sidebar.title("Configuration")
    
    # Sélection de l'exchange
    exchange = st.sidebar.selectbox(
        "Exchange",
        ["binance", "kucoin", "ftx", "coinbase"]
    )
    
    # Sélection du symbole
    symbol = st.sidebar.text_input("Symbole", "BTC/USDT")
    
    # Sélection de l'intervalle
    timeframe = st.sidebar.selectbox(
        "Intervalle",
        ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    )
    
    # Sélection de la période
    days = st.sidebar.slider("Nombre de jours", 1, 30, 7)
    
    # Bouton pour récupérer les données
    if st.sidebar.button("Récupérer les données"):
        with st.spinner("Récupération des données en cours..."):
            try:
                # Créer une instance de MarketDataAcquisition
                market_data = MarketDataAcquisition()
                
                # Calculer les dates
                end_time = datetime.now()
                start_time = end_time - timedelta(days=days)
                
                # Récupérer les données
                data = market_data.fetch_data(
                    symbol=symbol,
                    data_type='ohlcv',
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time,
                    exchange=exchange
                )
                
                # Convertir en DataFrame
                if isinstance(data, pd.DataFrame):
                    # Stocker les données dans la session
                    st.session_state.market_data = data
                    st.success("Données récupérées avec succès!")
                else:
                    st.error("Erreur: Les données récupérées ne sont pas au format attendu.")
            except Exception as e:
                st.error(f"Erreur lors de la récupération des données: {str(e)}")
    
    # Onglets pour les différentes sections
    tab1, tab2, tab3 = st.tabs(["Données de marché", "Prédictions", "Configuration"])
    
    with tab1:
        st.header("Données de marché")
        
        if 'market_data' in st.session_state:
            display_market_data(st.session_state.market_data)
        else:
            st.info("Utilisez le panneau de configuration pour récupérer des données.")
    
    with tab2:
        st.header("Prédictions du modèle")
        
        # Paramètres du modèle
        st.subheader("Paramètres du modèle")
        
        col1, col2 = st.columns(2)
        
        with col1:
            input_size = st.number_input("Taille d'entrée", min_value=1, value=10)
            hidden_sizes = st.text_input("Tailles des couches cachées (séparées par des virgules)", "64,32")
            learning_rate = st.number_input("Taux d'apprentissage", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
        
        with col2:
            epochs = st.number_input("Nombre d'époques", min_value=1, value=100)
            batch_size = st.number_input("Taille des batchs", min_value=1, value=32)
            dropout_rate = st.number_input("Taux de dropout", min_value=0.0, max_value=0.9, value=0.2, format="%.1f")
        
        # Bouton pour entraîner le modèle
        if st.button("Entraîner le modèle"):
            if 'market_data' not in st.session_state:
                st.warning("Veuillez d'abord récupérer des données de marché.")
            else:
                with st.spinner("Entraînement du modèle en cours..."):
                    try:
                        # Préparer les données
                        data = st.session_state.market_data
                        
                        # Convertir les tailles des couches cachées
                        hidden_sizes_list = [int(x.strip()) for x in hidden_sizes.split(",")]
                        
                        # Créer les paramètres du modèle
                        params = NeuralNetworkParams(
                            input_size=input_size,
                            hidden_sizes=hidden_sizes_list,
                            output_size=1,
                            learning_rate=learning_rate,
                            batch_size=batch_size,
                            epochs=epochs,
                            dropout_rate=dropout_rate
                        )
                        
                        # Créer et entraîner le modèle
                        model = TradingNeuralNetworkModel(params)
                        
                        # Préparer les données d'entraînement (exemple simplifié)
                        # Dans un cas réel, il faudrait faire plus de prétraitement
                        X, y = [], []
                        for i in range(len(data) - input_size):
                            X.append(data.iloc[i:i+input_size][['open', 'high', 'low', 'close', 'volume']].values)
                            y.append(data.iloc[i+input_size]['close'])
                        
                        X = np.array(X)
                        y = np.array(y).reshape(-1, 1)
                        
                        # Entraîner le modèle
                        history = model.train((X, y))
                        
                        # Faire des prédictions
                        predictions = model.predict(X)
                        
                        # Stocker les prédictions
                        st.session_state.predictions = predictions
                        
                        st.success("Modèle entraîné avec succès!")
                        
                        # Afficher les métriques d'entraînement
                        st.subheader("Métriques d'entraînement")
                        if history and 'loss' in history:
                            st.line_chart(pd.DataFrame({
                                'loss': history['loss'],
                                'val_loss': history['val_loss']
                            }))
                    except Exception as e:
                        st.error(f"Erreur lors de l'entraînement du modèle: {str(e)}")
        
        # Afficher les prédictions
        if 'market_data' in st.session_state and 'predictions' in st.session_state:
            display_predictions(st.session_state.market_data, st.session_state.predictions)
    
    with tab3:
        st.header("Configuration du système")
        
        # Chemin vers le fichier de configuration
        config_path = st.text_input(
            "Chemin vers le fichier de configuration",
            value=str(project_root / "config" / "trading_config.yaml")
        )
        
        # Bouton pour charger la configuration
        if st.button("Charger la configuration"):
            config = load_config(config_path)
            if config:
                st.session_state.config = config
                st.success("Configuration chargée avec succès!")
        
        # Afficher la configuration
        if 'config' in st.session_state:
            st.subheader("Configuration actuelle")
            st.json(st.session_state.config)
        
        # Paramètres de l'API
        st.subheader("Paramètres de l'API")
        
        api_key = st.text_input("Clé API", type="password")
        api_secret = st.text_input("Secret API", type="password")
        
        if st.button("Sauvegarder les clés API"):
            # Dans un cas réel, il faudrait stocker ces clés de manière sécurisée
            st.session_state.api_key = api_key
            st.session_state.api_secret = api_secret
            st.success("Clés API sauvegardées!")

if __name__ == "__main__":
    main()
