import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
import os
import numpy as np
from datetime import datetime, timedelta
import json
import plotly.express as px

# Ajout du chemin racine au PYTHONPATH
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configuration de la page avec thème personnalisé
st.set_page_config(
    page_title="Bot de Trading Hybride CNN+GNA",
    page_icon="🧬",
    layout="wide"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .success { color: #28a745; }
    .danger { color: #dc3545; }
    .warning { color: #ffc107; }
    .info { color: #17a2b8; }
    
    .metric-card {
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 5px;
        background-color: #f8f9fa;
    }
    
    .alert {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .alert-success {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }
    
    .alert-danger {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Barre latérale avec navigation
with st.sidebar:
    st.title("Navigation")
    page = st.radio(
        "Sélectionnez une page",
        ["Trading Live", "Configuration Modèle", "Entraînement", "Backtesting", "Analyse Performance"]
    )

# Pages
if page == "Trading Live":
    st.title("🤖 Trading Live - Modèle Hybride")
    
    # Configuration du trading
    with st.expander("⚙️ Configuration du Trading", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            model_version = st.selectbox(
                "Version du modèle",
                ["latest", "20240319_123456", "20240318_234567"],
                help="Sélectionnez la version du modèle à utiliser pour le trading"
            )
            
            trading_pair = st.selectbox(
                "Paire de trading",
                ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
            )
        
        with col2:
            mode = st.radio(
                "Mode d'exécution",
                ["Simulation", "Live Trading"],
                help="Simulation: pas d'ordres réels, Live: trading réel"
            )
            
            timeframe = st.selectbox(
                "Timeframe",
                ["1m", "5m", "15m", "1h", "4h", "1d"]
            )
            
        with col3:
            leverage = st.number_input("Levier", 1, 20, 1)
            position_size = st.slider("Taille position (%)", 1, 100, 10)

    # Prix en temps réel et métriques
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        current_price = 42150.75
        st.metric(
            "Prix actuel",
            f"${current_price:,.2f}",
            "↑ $150.25 (0.36%)",
            help="Prix actuel de la paire sélectionnée"
        )
    with col2:
        st.metric(
            "Signal CNN",
            "0.75",
            "↑ 0.05",
            help="Signal généré par le réseau de neurones (0-1)"
        )
    with col3:
        st.metric(
            "Signal GNA",
            "0.82",
            "↑ 0.03",
            help="Signal généré par l'algorithme génétique (0-1)"
        )
    with col4:
        signal_hybrid = 0.78
        delta = 0.04
        st.metric(
            "Signal Hybride",
            f"{signal_hybrid:.2f}",
            f"↑ {delta:.2f}",
            help="Signal combiné CNN+GNA (0-1)"
        )

    # Alertes
    if signal_hybrid > 0.75:
        st.markdown(
            '<div class="alert alert-success">🔔 Signal d\'achat fort détecté! Confiance: 85%</div>',
            unsafe_allow_html=True
        )

    # Graphique avec signaux
    st.subheader("Analyse Technique & Signaux")
    
    # Données exemple pour le graphique
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1H")
    prices = np.random.normal(42000, 1000, 100).cumsum()
    
    fig = go.Figure()
    
    # Chandeliers
    fig.add_candlestick(
        x=dates,
        open=prices,
        high=prices + np.random.uniform(100, 200, 100),
        low=prices - np.random.uniform(100, 200, 100),
        close=prices + np.random.uniform(-100, 100, 100),
        name="Prix"
    )
    
    # Signaux d'achat/vente
    buy_signals = np.random.choice([True, False], size=100, p=[0.1, 0.9])
    sell_signals = np.random.choice([True, False], size=100, p=[0.1, 0.9])
    
    fig.add_scatter(
        x=dates[buy_signals],
        y=prices[buy_signals],
        mode="markers",
        marker=dict(symbol="triangle-up", size=15, color="green"),
        name="Signaux d'achat"
    )
    
    fig.add_scatter(
        x=dates[sell_signals],
        y=prices[sell_signals],
        mode="markers",
        marker=dict(symbol="triangle-down", size=15, color="red"),
        name="Signaux de vente"
    )
    
    fig.update_layout(
        title=f"{trading_pair} - {timeframe}",
        yaxis_title="Prix (USDT)",
        xaxis_title="Date",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Ordres en cours et historique
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Ordres en cours")
        orders_df = pd.DataFrame({
            "ID": ["#1234", "#1235"],
            "Type": ["LIMIT BUY", "STOP LOSS"],
            "Prix": [41500, 40000],
            "Quantité": [0.1, 0.1],
            "Statut": ["En attente", "En attente"]
        })
        st.dataframe(orders_df, use_container_width=True)

    with col2:
        st.subheader("📈 Positions ouvertes")
        positions_df = pd.DataFrame({
            "ID": ["#1233"],
            "Type": ["LONG"],
            "Prix entrée": [42000],
            "Prix actuel": [42150.75],
            "P/L": ["+0.36%"]
        })
        st.dataframe(positions_df, use_container_width=True)

    # Historique des trades
    st.subheader("📜 Historique des trades")
    history_df = pd.DataFrame({
        "Date": pd.date_range(start="2024-03-19 00:00", periods=5, freq="1H"),
        "Type": ["BUY", "SELL", "BUY", "SELL", "BUY"],
        "Prix": [42000, 42500, 42300, 42800, 42150],
        "Quantité": [0.1, 0.1, 0.15, 0.15, 0.1],
        "P/L": ["--", "+$50", "--", "+$75", "--"],
        "Signaux": ["CNN: 0.82, GNA: 0.85", "CNN: 0.15, GNA: 0.20",
                   "CNN: 0.78, GNA: 0.80", "CNN: 0.18, GNA: 0.15",
                   "CNN: 0.88, GNA: 0.82"]
    })
    st.dataframe(history_df, use_container_width=True)

    # Métriques de session
    st.subheader("📊 Métriques de la session")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Trades total", "15")
    with col2:
        st.metric("Trades gagnants", "11 (73.3%)")
    with col3:
        st.metric("Profit net", "+$325.50")
    with col4:
        st.metric("ROI session", "+3.25%")

elif page == "Configuration Modèle":
    st.title("🧬 Configuration du Modèle Hybride")
    
    # Boutons d'import/export
    col1, col2 = st.columns([1, 4])
    with col1:
        st.download_button(
            "📥 Exporter config",
            data=json.dumps({
                "cnn_config": {"layers": [64, 128, 256]},
                "gna_config": {"population_size": 100}
            }, indent=2),
            file_name="model_config.json",
            mime="application/json"
        )
    with col2:
        uploaded_file = st.file_uploader("📤 Importer configuration", type="json")
        if uploaded_file is not None:
            st.success("Configuration importée avec succès!")
    
    tab1, tab2 = st.tabs(["Réseau de Neurones (CNN)", "Algorithme Génétique"])
    
    with tab1:
        st.subheader("Architecture CNN")
        
        # Configuration des couches CNN avec validation
        st.write("🔄 Couches de convolution")
        col1, col2 = st.columns(2)
        with col1:
            n_conv_layers = st.number_input(
                "Nombre de couches conv",
                1, 5, 3,
                help="Nombre de couches de convolution (1-5)"
            )
            
            # Validation des filtres
            filters_input = st.text_input(
                "Filtres par couche",
                "64,128,256",
                help="Nombre de filtres par couche, séparés par des virgules"
            )
            try:
                filters = [int(x) for x in filters_input.split(",")]
                if len(filters) != n_conv_layers:
                    st.warning(f"⚠️ Le nombre de filtres ({len(filters)}) ne correspond pas au nombre de couches ({n_conv_layers})")
            except ValueError:
                st.error("❌ Format invalide pour les filtres. Utilisez des nombres séparés par des virgules")
            
            kernel_sizes = st.text_input(
                "Tailles des noyaux",
                "3,3,3",
                help="Taille des noyaux de convolution par couche"
            )
        
        with col2:
            activation = st.selectbox(
                "Fonction d'activation",
                ["relu", "leaky_relu", "elu"],
                help="Fonction d'activation pour les couches de convolution"
            )
            pooling = st.selectbox(
                "Type de pooling",
                ["max", "average", "none"],
                help="Type de pooling après chaque couche de convolution"
            )
            dropout_rate = st.slider(
                "Taux de dropout",
                0.0, 0.5, 0.2,
                help="Taux de dropout pour la régularisation"
            )
        
        # Visualisation de l'architecture
        st.write("📊 Visualisation de l'architecture")
        
        # Création d'un graphique simple pour visualiser l'architecture
        fig = go.Figure()
        
        # Ajout des couches
        layer_x = 0
        for i in range(n_conv_layers):
            # Couche de convolution
            fig.add_shape(
                type="rect",
                x0=layer_x, y0=0,
                x1=layer_x + 0.8, y1=filters[i]/64,
                line=dict(color="blue"),
                fillcolor="lightblue",
                opacity=0.7
            )
            fig.add_annotation(
                x=layer_x + 0.4, y=filters[i]/128,
                text=f"Conv{i+1}\n{filters[i]}",
                showarrow=False
            )
            layer_x += 1
            
            # Pooling
            if pooling != "none":
                fig.add_shape(
                    type="rect",
                    x0=layer_x, y0=0,
                    x1=layer_x + 0.4, y1=filters[i]/64,
                    line=dict(color="green"),
                    fillcolor="lightgreen",
                    opacity=0.7
                )
                fig.add_annotation(
                    x=layer_x + 0.2, y=filters[i]/128,
                    text="Pool",
                    showarrow=False
                )
                layer_x += 0.6
        
        fig.update_layout(
            title="Architecture du CNN",
            showlegend=False,
            width=800,
            height=300,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        st.plotly_chart(fig)
        
        st.write("🔄 Couches denses")
        col1, col2 = st.columns(2)
        with col1:
            n_dense_layers = st.number_input(
                "Nombre de couches denses",
                1, 3, 2,
                help="Nombre de couches entièrement connectées"
            )
            dense_units = st.text_input(
                "Unités par couche",
                "128,64",
                help="Nombre d'unités par couche dense"
            )
            
            # Validation des unités denses
            try:
                dense_units_list = [int(x) for x in dense_units.split(",")]
                if len(dense_units_list) != n_dense_layers:
                    st.warning(f"⚠️ Le nombre d'unités ({len(dense_units_list)}) ne correspond pas au nombre de couches denses ({n_dense_layers})")
            except ValueError:
                st.error("❌ Format invalide pour les unités. Utilisez des nombres séparés par des virgules")
        
        with col2:
            dense_activation = st.selectbox(
                "Activation dense",
                ["relu", "tanh", "sigmoid"],
                help="Fonction d'activation pour les couches denses"
            )
            final_activation = st.selectbox(
                "Activation finale",
                ["sigmoid", "softmax", "linear"],
                help="Fonction d'activation de la couche de sortie"
            )

    with tab2:
        st.subheader("Paramètres Génétiques")
        
        col1, col2 = st.columns(2)
        with col1:
            population_size = st.number_input(
                "Taille de la population",
                50, 1000, 100,
                help="Nombre d'individus dans la population"
            )
            n_generations = st.number_input(
                "Nombre de générations",
                10, 1000, 100,
                help="Nombre de générations pour l'évolution"
            )
            mutation_rate = st.slider(
                "Taux de mutation",
                0.01, 0.5, 0.1,
                help="Probabilité de mutation des gènes"
            )
        
        with col2:
            crossover_rate = st.slider(
                "Taux de croisement",
                0.1, 1.0, 0.8,
                help="Probabilité de croisement entre individus"
            )
            elitism_rate = st.slider(
                "Taux d'élitisme",
                0.0, 0.3, 0.1,
                help="Proportion des meilleurs individus préservés"
            )
            selection_method = st.selectbox(
                "Méthode de sélection",
                ["tournament", "roulette", "rank"],
                help="Méthode de sélection des parents"
            )
        
        # Visualisation de la configuration génétique
        st.write("📊 Visualisation de la configuration génétique")
        
        # Création d'un graphique pour visualiser les paramètres
        fig = go.Figure()
        
        # Ajout des barres pour les différents paramètres
        params = {
            "Population": population_size/1000,
            "Générations": n_generations/1000,
            "Mutation": mutation_rate,
            "Croisement": crossover_rate,
            "Élitisme": elitism_rate
        }
        
        fig.add_trace(go.Bar(
            x=list(params.keys()),
            y=list(params.values()),
            marker_color=['rgb(158,202,225)', 'rgb(94,158,217)', 
                        'rgb(32,119,180)', 'rgb(12,94,157)', 
                        'rgb(8,64,129)']
        ))
        
        fig.update_layout(
            title="Paramètres de l'algorithme génétique",
            yaxis_title="Valeur normalisée",
            height=300
        )
        
        st.plotly_chart(fig)
        
        st.write("⚙️ Configuration des gènes")
        gene_params = st.text_area(
            "Paramètres des gènes (JSON)",
            '''{
    "price_weight": {"min": 0, "max": 1},
    "volume_weight": {"min": 0, "max": 1},
    "trend_weight": {"min": 0, "max": 1},
    "momentum_weight": {"min": 0, "max": 1},
    "volatility_weight": {"min": 0, "max": 1}
}''',
            help="Configuration des gènes au format JSON"
        )
        
        # Validation du JSON
        try:
            gene_config = json.loads(gene_params)
            st.success("✅ Configuration des gènes valide")
            
            # Affichage des gènes sous forme de tableau
            gene_df = pd.DataFrame([
                {"Gène": k, "Min": v["min"], "Max": v["max"]}
                for k, v in gene_config.items()
            ])
            st.dataframe(gene_df, use_container_width=True)
            
        except json.JSONDecodeError:
            st.error("❌ Format JSON invalide")

    # Bouton de sauvegarde avec validation
    if st.button("💾 Sauvegarder la configuration"):
        # Vérification de la validité de la configuration
        config_valid = True
        validation_messages = []
        
        # Validation CNN
        try:
            filters = [int(x) for x in filters_input.split(",")]
            if len(filters) != n_conv_layers:
                config_valid = False
                validation_messages.append("Nombre de filtres incorrect")
        except ValueError:
            config_valid = False
            validation_messages.append("Format des filtres invalide")
        
        try:
            dense_units_list = [int(x) for x in dense_units.split(",")]
            if len(dense_units_list) != n_dense_layers:
                config_valid = False
                validation_messages.append("Nombre d'unités denses incorrect")
        except ValueError:
            config_valid = False
            validation_messages.append("Format des unités denses invalide")
        
        # Validation GNA
        try:
            gene_config = json.loads(gene_params)
            for gene, params in gene_config.items():
                if "min" not in params or "max" not in params:
                    config_valid = False
                    validation_messages.append(f"Configuration invalide pour le gène {gene}")
        except json.JSONDecodeError:
            config_valid = False
            validation_messages.append("Format JSON des gènes invalide")
        
        if config_valid:
            st.success("✅ Configuration sauvegardée avec succès!")
        else:
            st.error("❌ Erreurs de validation:\n" + "\n".join(validation_messages))

elif page == "Entraînement":
    st.title("🎯 Entraînement du Modèle Hybride")
    
    # Configuration de l'entraînement
    with st.expander("⚙️ Configuration de l'entraînement", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            train_pairs = st.multiselect(
                "Paires d'entraînement",
                ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"],
                ["BTC/USDT", "ETH/USDT"],
                help="Sélectionnez les paires pour l'entraînement"
            )
            start_date = st.date_input(
                "Date de début",
                datetime.now() - timedelta(days=30),
                help="Date de début des données d'entraînement"
            )
            batch_size = st.number_input(
                "Batch Size",
                16, 512, 32,
                help="Taille des lots pour l'entraînement"
            )
            initial_balance = st.number_input(
                "Balance initiale (USDT)",
                100, 1000000, 100,
                help="Capital initial pour l'entraînement"
            )
            
        with col2:
            timeframes = st.multiselect(
                "Timeframes",
                ["1m", "5m", "15m", "1h", "4h", "1d"],
                ["15m", "1h", "4h"],
                help="Intervalles de temps pour l'analyse"
            )
            end_date = st.date_input(
                "Date de fin",
                datetime.now(),
                help="Date de fin des données d'entraînement"
            )
            epochs = st.number_input(
                "Epochs",
                1, 1000, 100,
                help="Nombre d'époques d'entraînement"
            )
    
    # Paramètres avancés
    with st.expander("⚙️ Paramètres avancés"):
        col1, col2, col3 = st.columns(3)
        with col1:
            learning_rate = st.number_input(
                "Learning Rate",
                0.0001, 0.1, 0.001,
                format="%.4f",
                help="Taux d'apprentissage pour le CNN"
            )
            early_stopping = st.number_input(
                "Early Stopping (epochs)",
                5, 50, 10,
                help="Arrêt anticipé si pas d'amélioration"
            )
            
        with col2:
            validation_split = st.slider(
                "Validation Split",
                0.1, 0.3, 0.2,
                help="Proportion des données pour la validation"
            )
            data_augmentation = st.checkbox(
                "Augmentation des données",
                True,
                help="Activer l'augmentation des données"
            )
            
        with col3:
            save_best_only = st.checkbox(
                "Sauvegarder meilleur modèle",
                True,
                help="Ne sauvegarder que le meilleur modèle"
            )
            checkpoint_frequency = st.number_input(
                "Fréquence de sauvegarde (epochs)",
                1, 10, 5,
                help="Fréquence de sauvegarde du modèle"
            )
    
    # Boutons de contrôle
    col1, col2, col3 = st.columns(3)
    with col1:
        start_training = st.button("▶️ Démarrer l'entraînement", type="primary")
    with col2:
        pause_training = st.button("⏸️ Pause", disabled=True)
    with col3:
        stop_training = st.button("⏹️ Arrêter", disabled=True)
    
    if start_training:
        # Validation des paramètres d'entraînement
        if not train_pairs:
            st.error("❌ Veuillez sélectionner au moins une paire d'entraînement")
            st.stop()
            
        if start_date >= end_date:
            st.error("❌ La date de début doit être antérieure à la date de fin")
            st.stop()
            
        if batch_size > initial_balance:
            st.error("❌ Le batch size ne peut pas être supérieur à la balance initiale")
            st.stop()
            
        st.info("🚀 Entraînement en cours...")
        
        # Barre de progression
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Métriques en temps réel
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Loss CNN",
                "0.245",
                "↓ 0.015",
                help="Perte du réseau de neurones"
            )
        with col2:
            st.metric(
                "Fitness GNA",
                "0.823",
                "↑ 0.054",
                help="Score de fitness de l'algorithme génétique"
            )
        with col3:
            st.metric(
                "Précision",
                "78.5%",
                "↑ 1.2%",
                help="Précision globale du modèle"
            )
        with col4:
            st.metric(
                "Temps restant",
                "45 min",
                help="Temps estimé pour terminer l'entraînement"
            )
        
        # Graphiques d'entraînement
        tab1, tab2, tab3 = st.tabs(["Loss", "Métriques", "Poids"])
        
        with tab1:
            st.subheader("📊 Évolution des pertes")
            loss_data = pd.DataFrame({
                "loss": [0.5, 0.4, 0.3, 0.25, 0.24],
                "val_loss": [0.55, 0.45, 0.35, 0.30, 0.28]
            })
            fig = px.line(
                loss_data,
                title="Évolution des pertes d'entraînement et de validation",
                labels={"value": "Loss", "index": "Epoch"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("📈 Métriques d'entraînement")
            metrics_data = pd.DataFrame({
                "accuracy": [0.65, 0.70, 0.75, 0.78, 0.785],
                "val_accuracy": [0.60, 0.65, 0.70, 0.75, 0.78],
                "f1_score": [0.62, 0.68, 0.73, 0.76, 0.78]
            })
            fig = px.line(
                metrics_data,
                title="Évolution des métriques",
                labels={"value": "Score", "index": "Epoch"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Matrice de confusion
            st.subheader("🎯 Matrice de confusion")
            confusion_matrix = np.array([
                [150, 25, 10],
                [20, 180, 15],
                [5, 15, 160]
            ])
            fig = px.imshow(
                confusion_matrix,
                labels=dict(x="Prédiction", y="Valeur réelle"),
                title="Matrice de confusion"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("🔍 Visualisation des poids")
            # Distribution des poids
            weights = np.random.normal(0, 0.1, 1000)
            fig = px.histogram(
                x=weights,
                title="Distribution des poids du modèle",
                labels={"x": "Valeur du poids", "y": "Fréquence"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Heatmap des poids
            weight_matrix = np.random.rand(10, 10)
            fig = px.imshow(
                weight_matrix,
                title="Heatmap des poids de la première couche",
                labels=dict(x="Neurone sortant", y="Neurone entrant")
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Logs d'entraînement
        st.subheader("📝 Logs d'entraînement")
        logs_df = pd.DataFrame({
            "Epoch": range(1, 6),
            "Loss": [0.5, 0.4, 0.3, 0.25, 0.24],
            "Val Loss": [0.55, 0.45, 0.35, 0.30, 0.28],
            "Accuracy": ["65%", "70%", "75%", "78%", "78.5%"],
            "Val Accuracy": ["60%", "65%", "70%", "75%", "78%"],
            "Temps": ["2:30", "2:25", "2:20", "2:15", "2:10"]
        })
        st.dataframe(logs_df, use_container_width=True)
        
        # Alertes et notifications
        if 0.24 < 0.25:  # Condition d'exemple
            st.markdown(
                '<div class="alert alert-success">🎉 Nouveau meilleur modèle! Loss: 0.24</div>',
                unsafe_allow_html=True
            )
        
        # Boutons d'action
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 Sauvegarder le modèle",
                data=b"model_data",  # À remplacer par les vraies données
                file_name="model.h5",
                mime="application/octet-stream"
            )
        with col2:
            if st.button("📊 Générer rapport"):
                st.success("Rapport généré avec succès!")

elif page == "Backtesting":
    st.title("📊 Backtesting du Modèle Hybride")
    
    # Configuration du backtest
    with st.expander("⚙️ Configuration du backtest", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            model_version = st.selectbox(
                "Version du modèle à tester",
                ["latest", "20240319_123456", "20240318_234567"],
                help="Sélectionnez la version du modèle à évaluer"
            )
            test_pairs = st.multiselect(
                "Paires à tester",
                ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"],
                ["BTC/USDT"],
                help="Sélectionnez les paires pour le backtest"
            )
            
        with col2:
            start_date = st.date_input(
                "Période de test - Début",
                datetime.now() - timedelta(days=30),
                help="Date de début du backtest"
            )
            end_date = st.date_input(
                "Période de test - Fin",
                datetime.now(),
                help="Date de fin du backtest"
            )
    
    # Paramètres avancés
    with st.expander("⚙️ Paramètres avancés"):
        col1, col2, col3 = st.columns(3)
        with col1:
            initial_balance = st.number_input(
                "Balance initiale (USDT)",
                1000, 1000000, 10000,
                help="Capital initial pour le backtest"
            )
            position_size = st.slider(
                "Taille de position (%)",
                1, 100, 10,
                help="Pourcentage du capital par trade"
            )
            
        with col2:
            leverage = st.number_input(
                "Levier",
                1, 20, 1,
                help="Levier de trading"
            )
            fee_rate = st.number_input(
                "Frais (%)",
                0.0, 1.0, 0.1,
                help="Frais de trading en pourcentage"
            )
            
        with col3:
            stop_loss = st.number_input(
                "Stop Loss (%)",
                0.1, 10.0, 2.0,
                help="Stop loss en pourcentage"
            )
            take_profit = st.number_input(
                "Take Profit (%)",
                0.1, 20.0, 4.0,
                help="Take profit en pourcentage"
            )
    
    # Boutons de contrôle
    col1, col2 = st.columns(2)
    with col1:
        start_backtest = st.button("▶️ Lancer le Backtest", type="primary")
    with col2:
        if st.button("📥 Exporter résultats"):
            st.success("Résultats exportés avec succès!")
    
    if start_backtest:
        # Validation des paramètres de backtest
        if not test_pairs:
            st.error("❌ Veuillez sélectionner au moins une paire pour le backtest")
            st.stop()
            
        if start_date >= end_date:
            st.error("❌ La date de début doit être antérieure à la date de fin")
            st.stop()
            
        if position_size > 100:
            st.error("❌ La taille de position ne peut pas dépasser 100%")
            st.stop()
            
        if leverage * position_size > 100:
            st.error("❌ Le risque total (levier * taille position) ne peut pas dépasser 100%")
            st.stop()
            
        st.info("🚀 Backtest en cours...")
        
        # Barre de progression
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "ROI",
                "+45.8%",
                "↑ 2.3%",
                help="Retour sur investissement"
            )
        with col2:
            st.metric(
                "Sharpe Ratio",
                "2.34",
                help="Ratio de Sharpe (risque/récompense)"
            )
        with col3:
            st.metric(
                "Max Drawdown",
                "-15.2%",
                help="Perte maximale en pourcentage"
            )
        with col4:
            st.metric(
                "Profit Factor",
                "1.85",
                help="Ratio profit/pertes"
            )
        
        # Graphiques de performance
        tab1, tab2, tab3 = st.tabs(["Performance", "Trades", "Analyse"])
        
        with tab1:
            st.subheader("📈 Évolution du capital")
            
            # Courbe de capital
            balance_data = pd.DataFrame({
                "balance": [10000, 11000, 10500, 12000, 14580],
                "drawdown": [0, 0, -4.5, 0, 0]
            })
            fig = go.Figure()
            
            # Balance
            fig.add_trace(go.Scatter(
                y=balance_data["balance"],
                name="Balance",
                line=dict(color="blue")
            ))
            
            # Drawdown
            fig.add_trace(go.Scatter(
                y=balance_data["drawdown"],
                name="Drawdown",
                line=dict(color="red"),
                yaxis="y2"
            ))
            
            fig.update_layout(
                title="Évolution du capital et drawdown",
                yaxis2=dict(
                    overlaying="y",
                    side="right",
                    range=[-20, 0],
                    title="Drawdown (%)"
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution des rendements
            st.subheader("📊 Distribution des rendements")
            returns = np.random.normal(0.02, 0.05, 1000)
            fig = px.histogram(
                x=returns,
                title="Distribution des rendements",
                labels={"x": "Rendement (%)", "y": "Fréquence"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("📝 Historique des trades")
            
            # Tableau des trades
            trades_df = pd.DataFrame({
                "Date": pd.date_range(start="2024-01-01", periods=10),
                "Type": ["Long", "Short", "Long", "Short", "Long", "Short", "Long", "Short", "Long", "Short"],
                "Entrée": [42000, 43500, 44000, 45000, 44500, 46000, 45500, 47000, 46500, 48000],
                "Sortie": [43500, 42000, 45000, 44000, 46000, 44500, 47000, 45500, 48000, 46500],
                "Quantité": [0.1, 0.1, 0.15, 0.15, 0.1, 0.1, 0.15, 0.15, 0.1, 0.1],
                "Profit": ["+3.57%", "-3.45%", "+2.27%", "-2.22%", "+3.37%", "-3.26%", "+3.30%", "-3.19%", "+3.23%", "-3.12%"],
                "Signal CNN": [0.82, 0.15, 0.78, 0.18, 0.88, 0.12, 0.85, 0.15, 0.90, 0.10],
                "Signal GNA": [0.85, 0.20, 0.80, 0.15, 0.82, 0.18, 0.88, 0.12, 0.92, 0.08]
            })
            st.dataframe(trades_df, use_container_width=True)
            
            # Analyse des trades
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Trades total", "156")
            with col2:
                st.metric("Trades gagnants", "98 (62.8%)")
            with col3:
                st.metric("Trades perdants", "58 (37.2%)")
        
        with tab3:
            st.subheader("🔍 Analyse détaillée")
            
            # Métriques avancées
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Win Rate", "62.8%")
            with col2:
                st.metric("Profit moyen", "+1.85%")
            with col3:
                st.metric("Perte moyenne", "-1.92%")
            with col4:
                st.metric("Ratio gain/perte", "0.96")
            
            # Analyse des signaux
            st.subheader("🎯 Analyse des signaux")
            signals_df = pd.DataFrame({
                "Signal": ["CNN", "GNA", "Hybride"],
                "Précision": [76.5, 72.8, 79.2],
                "Rappel": [73.2, 70.5, 75.8],
                "F1-Score": [74.8, 71.6, 77.4]
            })
            st.dataframe(signals_df, use_container_width=True)
            
            # Graphique des signaux
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=signals_df["Signal"],
                y=signals_df["Précision"],
                name="Précision"
            ))
            fig.add_trace(go.Bar(
                x=signals_df["Signal"],
                y=signals_df["Rappel"],
                name="Rappel"
            ))
            fig.update_layout(
                title="Performance des différents signaux",
                barmode="group"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Rapport de performance
        st.subheader("📊 Rapport de performance")
        report_df = pd.DataFrame({
            "Métrique": [
                "ROI",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Calmar Ratio",
                "Max Drawdown",
                "Profit Factor",
                "Win Rate",
                "Profit moyen",
                "Perte moyenne",
                "Ratio gain/perte"
            ],
            "Valeur": [
                "+45.8%",
                "2.34",
                "2.89",
                "3.01",
                "-15.2%",
                "1.85",
                "62.8%",
                "+1.85%",
                "-1.92%",
                "0.96"
            ]
        })
        st.dataframe(report_df, use_container_width=True)

else:  # Analyse Performance
    st.title("📈 Analyse des Performances")
    
    # Sélection de la période et du modèle
    with st.expander("⚙️ Configuration de l'analyse", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            model_version = st.selectbox(
                "Version du modèle",
                ["latest", "20240319_123456", "20240318_234567"],
                help="Sélectionnez la version du modèle à analyser"
            )
        with col2:
            analysis_period = st.selectbox(
                "Période d'analyse",
                ["24h", "7j", "30j", "90j", "1an", "Tout"],
                help="Période d'analyse des performances"
            )
        with col3:
            trading_pair = st.selectbox(
                "Paire de trading",
                ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"],
                help="Paire de trading à analyser"
            )
    
    # Métriques globales
    st.subheader("📊 Métriques globales")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Précision CNN",
            "76.5%",
            "↑ 1.2%",
            help="Précision du réseau de neurones"
        )
    with col2:
        st.metric(
            "Précision GNA",
            "72.8%",
            "↑ 0.8%",
            help="Précision de l'algorithme génétique"
        )
    with col3:
        st.metric(
            "Précision Hybride",
            "79.2%",
            "↑ 1.5%",
            help="Précision du modèle hybride"
        )
    with col4:
        st.metric(
            "ROI Global",
            "+32.5%",
            "↑ 2.1%",
            help="Retour sur investissement global"
        )
    
    # Analyse détaillée
    tab1, tab2, tab3, tab4 = st.tabs(["Signaux", "Trades", "Evolution", "Comparaison"])
    
    with tab1:
        st.subheader("🎯 Distribution des signaux")
        
        # Distribution des signaux
        col1, col2 = st.columns(2)
        with col1:
            signals_dist = pd.DataFrame({
                "Signal": ["CNN", "GNA", "Hybride"],
                "Achat": [45, 42, 48],
                "Vente": [55, 58, 52]
            })
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=signals_dist["Signal"],
                y=signals_dist["Achat"],
                name="Signaux d'achat"
            ))
            fig.add_trace(go.Bar(
                x=signals_dist["Signal"],
                y=signals_dist["Vente"],
                name="Signaux de vente"
            ))
            fig.update_layout(
                title="Distribution des signaux par type",
                barmode="group"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Précision par type de signal
            precision_data = pd.DataFrame({
                "Type": ["Achat", "Vente"],
                "CNN": [78.5, 74.5],
                "GNA": [75.2, 70.4],
                "Hybride": [81.2, 77.1]
            })
            fig = go.Figure()
            for col in ["CNN", "GNA", "Hybride"]:
                fig.add_trace(go.Bar(
                    x=precision_data["Type"],
                    y=precision_data[col],
                    name=col
                ))
            fig.update_layout(
                title="Précision par type de signal",
                barmode="group"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Matrice de confusion
        st.subheader("📊 Matrices de confusion")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("CNN")
            confusion_cnn = np.array([[150, 25, 10], [20, 180, 15], [5, 15, 160]])
            fig = px.imshow(
                confusion_cnn,
                labels=dict(x="Prédiction", y="Valeur réelle"),
                title="CNN"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("GNA")
            confusion_gna = np.array([[145, 30, 10], [25, 175, 15], [10, 20, 155]])
            fig = px.imshow(
                confusion_gna,
                labels=dict(x="Prédiction", y="Valeur réelle"),
                title="GNA"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.write("Hybride")
            confusion_hybrid = np.array([[160, 20, 5], [15, 185, 10], [3, 12, 170]])
            fig = px.imshow(
                confusion_hybrid,
                labels=dict(x="Prédiction", y="Valeur réelle"),
                title="Hybride"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📝 Analyse des trades")
        
        # Statistiques des trades
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Trades total", "156")
        with col2:
            st.metric("Trades gagnants", "98 (62.8%)")
        with col3:
            st.metric("Profit moyen", "+1.85%")
        with col4:
            st.metric("Perte moyenne", "-1.92%")
        
        # Historique des trades
        trades_df = pd.DataFrame({
            "Date": pd.date_range(start="2024-01-01", periods=10),
            "Type": ["Long", "Short", "Long", "Short", "Long", "Short", "Long", "Short", "Long", "Short"],
            "Entrée": [42000, 43500, 44000, 45000, 44500, 46000, 45500, 47000, 46500, 48000],
            "Sortie": [43500, 42000, 45000, 44000, 46000, 44500, 47000, 45500, 48000, 46500],
            "Quantité": [0.1, 0.1, 0.15, 0.15, 0.1, 0.1, 0.15, 0.15, 0.1, 0.1],
            "Profit": ["+3.57%", "-3.45%", "+2.27%", "-2.22%", "+3.37%", "-3.26%", "+3.30%", "-3.19%", "+3.23%", "-3.12%"],
            "Signal CNN": [0.82, 0.15, 0.78, 0.18, 0.88, 0.12, 0.85, 0.15, 0.90, 0.10],
            "Signal GNA": [0.85, 0.20, 0.80, 0.15, 0.82, 0.18, 0.88, 0.12, 0.92, 0.08]
        })
        st.dataframe(trades_df, use_container_width=True)
        
        # Distribution des profits
        st.subheader("📊 Distribution des profits")
        profits = np.random.normal(0.02, 0.05, 1000)
        fig = px.histogram(
            x=profits,
            title="Distribution des profits",
            labels={"x": "Profit (%)", "y": "Fréquence"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📈 Evolution des performances")
        
        # Evolution des métriques
        metrics_evolution = pd.DataFrame({
            "Date": pd.date_range(start="2024-01-01", periods=30),
            "ROI": np.random.normal(0.02, 0.01, 30).cumsum(),
            "Précision": np.random.normal(0.75, 0.02, 30),
            "Win Rate": np.random.normal(0.62, 0.02, 30)
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=metrics_evolution["Date"],
            y=metrics_evolution["ROI"],
            name="ROI",
            yaxis="y"
        ))
        fig.add_trace(go.Scatter(
            x=metrics_evolution["Date"],
            y=metrics_evolution["Précision"],
            name="Précision",
            yaxis="y2"
        ))
        fig.add_trace(go.Scatter(
            x=metrics_evolution["Date"],
            y=metrics_evolution["Win Rate"],
            name="Win Rate",
            yaxis="y3"
        ))
        
        fig.update_layout(
            title="Evolution des performances",
            yaxis=dict(title="ROI (%)"),
            yaxis2=dict(
                title="Précision (%)",
                overlaying="y",
                side="right",
                range=[0, 1]
            ),
            yaxis3=dict(
                title="Win Rate (%)",
                overlaying="y",
                side="right",
                range=[0, 1],
                position=0.95
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Evolution des signaux
        st.subheader("🎯 Evolution des signaux")
        signals_evolution = pd.DataFrame({
            "Date": pd.date_range(start="2024-01-01", periods=30),
            "CNN": np.random.normal(0.75, 0.02, 30),
            "GNA": np.random.normal(0.72, 0.02, 30),
            "Hybride": np.random.normal(0.79, 0.02, 30)
        })
        
        fig = px.line(
            signals_evolution,
            x="Date",
            y=["CNN", "GNA", "Hybride"],
            title="Evolution de la précision des signaux"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("🔄 Comparaison des modèles")
        
        # Tableau comparatif
        comparison_df = pd.DataFrame({
            "Métrique": [
                "ROI",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Calmar Ratio",
                "Max Drawdown",
                "Profit Factor",
                "Win Rate",
                "Précision",
                "Rappel",
                "F1-Score"
            ],
            "CNN": [
                "+28.5%",
                "1.89",
                "2.12",
                "2.45",
                "-18.2%",
                "1.65",
                "58.5%",
                "76.5%",
                "73.2%",
                "74.8%"
            ],
            "GNA": [
                "+25.8%",
                "1.75",
                "1.98",
                "2.28",
                "-20.5%",
                "1.52",
                "55.2%",
                "72.8%",
                "70.5%",
                "71.6%"
            ],
            "Hybride": [
                "+32.5%",
                "2.34",
                "2.89",
                "3.01",
                "-15.2%",
                "1.85",
                "62.8%",
                "79.2%",
                "75.8%",
                "77.4%"
            ]
        })
        st.dataframe(comparison_df, use_container_width=True)
        
        # Graphique de comparaison
        fig = go.Figure()
        for col in ["CNN", "GNA", "Hybride"]:
            fig.add_trace(go.Bar(
                x=comparison_df["Métrique"],
                y=comparison_df[col],
                name=col
            ))
        fig.update_layout(
            title="Comparaison des performances",
            barmode="group"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Analyse des avantages
        st.subheader("💡 Analyse des avantages")
        advantages_df = pd.DataFrame({
            "Aspect": [
                "Précision globale",
                "Stabilité",
                "Adaptabilité",
                "Robustesse",
                "Performance en tendance",
                "Performance en range"
            ],
            "CNN": [
                "Bonne",
                "Moyenne",
                "Faible",
                "Moyenne",
                "Bonne",
                "Moyenne"
            ],
            "GNA": [
                "Moyenne",
                "Bonne",
                "Bonne",
                "Bonne",
                "Moyenne",
                "Bonne"
            ],
            "Hybride": [
                "Excellente",
                "Bonne",
                "Bonne",
                "Bonne",
                "Excellente",
                "Bonne"
            ]
        })
        st.dataframe(advantages_df, use_container_width=True)

# Footer avec état du système
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Version du modèle:** v2.3.1")
with col2:
    st.markdown("**État:** 🟢 En ligne")
with col3:
    st.markdown("**Dernière mise à jour:** 2024-03-19 10:30:15") 