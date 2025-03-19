#!/usr/bin/env python3
import argparse
import sys
import os

def setup_environment():
    """Configure l'environnement d'exécution"""
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="Trading Bot Launch Script")
    parser.add_argument("--mode", choices=["train", "backtest", "live", "ui"], default="ui",
                       help="Mode d'exécution (train/backtest/live/ui)")
    parser.add_argument("--model", choices=["hybrid", "neural", "genetic"], default="hybrid",
                       help="Type de modèle à utiliser")
    parser.add_argument("--config", type=str, default="config/trading/default.yaml",
                       help="Chemin du fichier de configuration")
    parser.add_argument("--debug", action="store_true", help="Active le mode debug")
    
    args = parser.parse_args()
    setup_environment()

    if args.mode == "train":
        from src.models.hybrid_model.model import train_model
        train_model(args.config)
    elif args.mode == "backtest":
        from src.services.backtesting.backtester import run_backtest
        run_backtest(args.config, args.model)
    elif args.mode == "live":
        from src.services.execution.execution_engine import run_live_trading
        run_live_trading(args.config, args.model)
    elif args.mode == "ui":
        from src.visualization.dashboard.main_dashboard import run_dashboard
        run_dashboard(args.debug)

if __name__ == "__main__":
    main() 