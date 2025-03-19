import asyncio
import argparse
from datetime import datetime
import logging
from typing import List
import yaml
from pathlib import Path
import streamlit as st
from .simulation import SimulationConfig, SimulatedTrading
from .exchange_manager import ExchangeConfig
from .web_interface import show_live_trading_interface

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Charge la configuration depuis un fichier YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_simulation_cli():
    """Lance la simulation en mode ligne de commande"""
    parser = argparse.ArgumentParser(description='Simulation de trading')
    parser.add_argument(
        '--config',
        type=str,
        default='config/simulation.yaml',
        help='Chemin vers le fichier de configuration'
    )
    parser.add_argument(
        '--pairs',
        type=str,
        nargs='+',
        default=['BTC/USDT', 'ETH/USDT'],
        help='Paires de trading à simuler'
    )
    parser.add_argument(
        '--balance',
        type=float,
        default=100.0,
        help='Balance initiale en USDT'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Durée de la simulation en minutes'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Chemin vers le modèle entraîné'
    )
    
    args = parser.parse_args()
    
    # Configuration de la simulation
    config = SimulationConfig(
        initial_balance=args.balance,
        trading_pairs=args.pairs,
        start_date=datetime.now()
    )
    
    # Configuration de l'exchange
    exchange_config = ExchangeConfig(
        name='simulation',
        api_key='',
        api_secret='',
        testnet=True
    )
    
    # Création du simulateur
    simulator = SimulatedTrading(
        config=config,
        exchange_config=exchange_config,
        model_path=args.model
    )
    
    # Lancement de la simulation
    try:
        asyncio.run(simulator.start())
        
        # Attente de la durée spécifiée
        asyncio.sleep(args.duration * 60)
        
        # Arrêt de la simulation
        simulator.stop()
        
        # Affichage des résultats
        print("\nRésultats de la simulation:")
        for metric, value in simulator.performance_metrics.items():
            print(f"{metric}: {value}")
    
    except KeyboardInterrupt:
        print("\nArrêt de la simulation...")
        simulator.stop()
    
    except Exception as e:
        logger.error(f"Erreur lors de la simulation: {e}")

def run_simulation_ui():
    """Lance la simulation avec une interface Streamlit"""
    st.title("🎮 Simulation de Trading")
    
    # Configuration de la simulation
    st.sidebar.header("⚙️ Configuration")
    
    # Sélection des paires
    trading_pairs = st.sidebar.multiselect(
        "Paires de Trading",
        options=['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT'],
        default=['BTC/USDT', 'ETH/USDT']
    )
    
    # Configuration de la simulation
    initial_balance = st.sidebar.number_input(
        "Balance Initiale (USDT)",
        min_value=100.0,
        max_value=100000.0,
        value=10000.0,
        step=100.0
    )
    
    volatility = st.sidebar.slider(
        "Volatilité des Prix (%)",
        min_value=0.01,
        max_value=1.0,
        value=0.1,
        step=0.01
    )
    
    error_rate = st.sidebar.slider(
        "Taux d'Erreur (%)",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1
    )
    
    # Sélection du modèle
    model_path = st.sidebar.text_input(
        "Chemin du Modèle",
        value="models/latest.h5"
    )
    
    # Configuration
    config = SimulationConfig(
        initial_balance=initial_balance,
        trading_pairs=trading_pairs,
        price_volatility=volatility / 100,
        error_rate=error_rate / 100,
        start_date=datetime.now()
    )
    
    exchange_config = ExchangeConfig(
        name='simulation',
        api_key='',
        api_secret='',
        testnet=True
    )
    
    # Création ou récupération du simulateur
    if 'simulator' not in st.session_state:
        st.session_state.simulator = SimulatedTrading(
            config=config,
            exchange_config=exchange_config,
            model_path=model_path
        )
    
    # Contrôles
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("▶️ Démarrer Simulation", use_container_width=True):
            st.session_state['is_running'] = True
            asyncio.run(st.session_state.simulator.start())
    
    with col2:
        if st.button("⏹️ Arrêter Simulation", use_container_width=True):
            if st.session_state.get('is_running', False):
                st.session_state.simulator.stop()
                st.session_state['is_running'] = False
                st.success("Simulation arrêtée")
    
    # Affichage des métriques en temps réel
    if st.session_state.get('is_running', False):
        st.header("📊 Métriques en Temps Réel")
        
        # Balance
        if st.session_state.simulator.balances_history:
            current_balance = st.session_state.simulator.balances_history[-1]['balance']['USDT']
            initial_balance = st.session_state.simulator.balances_history[0]['balance']['USDT']
            pnl_percent = (current_balance - initial_balance) / initial_balance * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Balance USDT",
                    f"{current_balance:.2f}",
                    f"{pnl_percent:+.2f}%"
                )
            
            with col2:
                st.metric(
                    "Nombre de Trades",
                    len(st.session_state.simulator.trades)
                )
        
        # Graphique de la balance
        if len(st.session_state.simulator.balances_history) > 1:
            df = pd.DataFrame(st.session_state.simulator.balances_history)
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=[b['USDT'] for b in df['balance']],
                    name="Balance USDT",
                    line=dict(color="#2ecc71", width=2)
                )
            )
            
            fig.update_layout(
                title="Évolution de la Balance",
                xaxis_title="Temps",
                yaxis_title="USDT",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Trades récents
        if st.session_state.simulator.trades:
            st.subheader("Trades Récents")
            
            trades_df = pd.DataFrame(st.session_state.simulator.trades[-10:])
            st.dataframe(
                trades_df,
                use_container_width=True,
                hide_index=True
            )
    
    # Résultats finaux
    if not st.session_state.get('is_running', False) and st.session_state.simulator.performance_metrics:
        st.header("📈 Résultats de la Simulation")
        
        metrics = st.session_state.simulator.performance_metrics
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Rendement Total",
                f"{metrics['total_return']:.2f}%"
            )
        
        with col2:
            st.metric(
                "Win Rate",
                f"{metrics.get('win_rate', 0):.1f}%"
            )
        
        with col3:
            st.metric(
                "Durée",
                str(metrics['duration']).split('.')[0]
            )
        
        # Statistiques détaillées
        st.subheader("Statistiques Détaillées")
        
        stats_df = pd.DataFrame([{
            'Métrique': k,
            'Valeur': v
        } for k, v in metrics.items()])
        
        st.dataframe(
            stats_df,
            use_container_width=True,
            hide_index=True
        )

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Mode ligne de commande
        run_simulation_cli()
    else:
        # Mode interface graphique
        run_simulation_ui() 