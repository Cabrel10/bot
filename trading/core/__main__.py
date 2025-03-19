"""Main entry point for the trading application.

Ce module est le point d'entrée principal de l'application de trading.
Il gère :
- Le parsing des arguments en ligne de commande
- L'initialisation du système
- La boucle principale de trading
- La gestion des erreurs et le logging
- L'arrêt propre du système
- Le monitoring des performances
- La sauvegarde et restauration d'état
"""

import sys
import argparse
import asyncio
import signal
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
import json
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import tracemalloc
import psutil
import yaml

from ..config.config_manager import ConfigManager
from ..utils.logger import TradingLogger
from ..core.data_adapter import DataAdapter
from ..core.models.model_manager import ModelManager
from ..core.risk.dynamic_risk_manager import DynamicRiskManager
from ..core.order_manager import OrderManager
from ..utils.event_manager import EventManager
from ..utils.metrics import MetricsCollector

class TradingSystem:
    """Système principal de trading."""
    
    def __init__(self, config_path: str, mode: str, debug: bool = False, 
                 log_file: Optional[str] = None, metrics_interval: int = 60):
        """
        Initialise le système de trading.
        
        Args:
            config_path: Chemin vers le fichier de configuration
            mode: Mode de fonctionnement ('training', 'prediction', 'backtest')
            debug: Active le mode debug
            log_file: Chemin vers le fichier de log (optionnel)
            metrics_interval: Intervalle de collecte des métriques en secondes
        """
        # Configuration du logging
        self.logger = TradingLogger.get_logger(
            __name__,
            level=logging.DEBUG if debug else logging.INFO,
            log_file=log_file
        )
        
        # Activation du monitoring mémoire
        if debug:
            tracemalloc.start()
        
        # Chargement de la configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load_config()
        
        # Initialisation des composants
        self.data_adapter = DataAdapter(self.config['data'])
        self.model_manager = ModelManager(self.config['models'])
        self.risk_manager = DynamicRiskManager(self.config['risk'])
        self.order_manager = OrderManager(self.config['orders'])
        
        # Collecteur de métriques
        self.metrics_collector = MetricsCollector(
            interval=metrics_interval,
            save_path=self.config.get('metrics_path', 'metrics')
        )
        
        # File d'attente d'événements
        self.event_queue = Queue()
        self.event_manager = EventManager(self.event_queue)
        
        # État du système
        self.running = False
        self.mode = mode
        self.debug = debug
        self.start_time = None
        self.error_count = 0
        self.max_errors = self.config.get('max_errors', 10)
        
        # Pool de threads pour les tâches parallèles
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.get('max_workers', 4)
        )
        
    async def initialize(self) -> None:
        """Initialise tous les composants du système."""
        try:
            self.logger.info("Initialisation du système de trading...")
            self.start_time = datetime.now()
            
            # Restauration de l'état précédent si disponible
            await self._restore_state()
            
            # Initialisation des gestionnaires
            await asyncio.gather(
                self.data_adapter.initialize(),
                self.model_manager.initialize(),
                self.risk_manager.initialize(),
                self.order_manager.initialize(),
                self.metrics_collector.initialize()
            )
            
            # Configuration des gestionnaires d'événements
            self._setup_event_handlers()
            
            # Vérification de l'état du système
            self._check_system_health()
            
            self.logger.info("Système initialisé avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation: {e}")
            if self.debug:
                self.logger.debug(traceback.format_exc())
            raise

    def _check_system_health(self) -> None:
        """Vérifie l'état du système."""
        # Vérification de la mémoire
        memory = psutil.Process().memory_info()
        memory_usage = memory.rss / 1024 / 1024  # MB
        if memory_usage > self.config.get('max_memory_mb', 1024):
            self.logger.warning(f"Usage mémoire élevé: {memory_usage:.2f} MB")
            
        # Vérification de la CPU
        cpu_percent = psutil.cpu_percent()
        if cpu_percent > self.config.get('max_cpu_percent', 80):
            self.logger.warning(f"Usage CPU élevé: {cpu_percent}%")
            
        # Vérification du disque
        disk = psutil.disk_usage('/')
        if disk.percent > self.config.get('max_disk_percent', 90):
            self.logger.warning(f"Espace disque faible: {disk.percent}%")

    async def _save_state(self) -> None:
        """Sauvegarde l'état du système."""
        state = {
            'timestamp': datetime.now().isoformat(),
            'mode': self.mode,
            'error_count': self.error_count,
            'metrics': self.metrics_collector.get_current_metrics(),
            'risk_state': await self.risk_manager.get_state(),
            'model_state': await self.model_manager.get_state()
        }
        
        state_path = Path(self.config.get('state_path', 'state'))
        state_path.mkdir(parents=True, exist_ok=True)
        
        with open(state_path / 'system_state.json', 'w') as f:
            json.dump(state, f, indent=2)

    async def _restore_state(self) -> None:
        """Restaure l'état du système."""
        state_path = Path(self.config.get('state_path', 'state'))
        state_file = state_path / 'system_state.json'
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    
                self.error_count = state.get('error_count', 0)
                await self.risk_manager.restore_state(state.get('risk_state', {}))
                await self.model_manager.restore_state(state.get('model_state', {}))
                
                self.logger.info("État du système restauré avec succès")
                
            except Exception as e:
                self.logger.error(f"Erreur lors de la restauration de l'état: {e}")

    def _setup_event_handlers(self) -> None:
        """Configure les gestionnaires d'événements."""
        self.event_manager.register('market_data', self._handle_market_data)
        self.event_manager.register('model_prediction', self._handle_prediction)
        self.event_manager.register('risk_alert', self._handle_risk_alert)
        self.event_manager.register('order_executed', self._handle_order_executed)
        self.event_manager.register('error', self._handle_error)
        
    async def _handle_market_data(self, data: Dict[str, Any]) -> None:
        """Gère les nouvelles données de marché."""
        try:
            # Mise à jour des métriques de risque
            await self.risk_manager.update_risk_metrics(data)
            
            # Génération des prédictions
            if self.mode in ['prediction', 'backtest']:
                predictions = await self.model_manager.predict(data)
                self.event_queue.put({
                    'type': 'model_prediction',
                    'data': predictions
                })
                
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement des données: {e}")
            self._handle_error(e)
            
    async def _handle_prediction(self, prediction: Dict[str, Any]) -> None:
        """Gère les nouvelles prédictions."""
        try:
            # Vérification des limites de risque
            limits = self.risk_manager.get_current_limits()
            
            # Génération des ordres
            if self._validate_prediction(prediction, limits):
                await self.order_manager.place_orders(prediction, limits)
                
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement des prédictions: {e}")
            self._handle_error(e)
            
    async def _handle_risk_alert(self, alert: Dict[str, Any]) -> None:
        """Gère les alertes de risque."""
        self.logger.warning(f"Alerte de risque reçue: {alert}")
        
        # Actions selon le type d'alerte
        if alert.get('severity') == 'high':
            await self.order_manager.close_all_positions()
            
    async def _handle_order_executed(self, order: Dict[str, Any]) -> None:
        """Gère les confirmations d'exécution d'ordres."""
        self.logger.info(f"Ordre exécuté: {order}")
        
        # Mise à jour des positions
        await self.risk_manager.update_positions(order)
        
    def _handle_error(self, error: Exception) -> None:
        """Gère les erreurs système."""
        self.logger.error(f"Erreur système: {error}")
        
        if self.debug:
            import traceback
            self.logger.debug(traceback.format_exc())
            
        # Arrêt du système si erreur critique
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            self.stop()
            
        self.error_count += 1
        if self.error_count > self.max_errors:
            self.logger.error("Nombre maximum d'erreurs atteint. Arrêt du système.")
            self.stop()
            
    def _validate_prediction(self, prediction: Dict[str, Any], limits: Dict[str, float]) -> bool:
        """Valide les prédictions selon les limites de risque."""
        # Validation à implémenter selon la logique métier
        return True
        
    async def run(self) -> None:
        """Lance la boucle principale du système."""
        self.running = True
        
        try:
            await self.initialize()
            
            self.logger.info(f"Démarrage du système en mode {self.mode}")
            
            while self.running:
                try:
                    # Traitement des événements
                    event = self.event_queue.get(timeout=1.0)
                    await self.event_manager.process_event(event)
                    
                except asyncio.TimeoutError:
                    continue
                    
                except Exception as e:
                    self.logger.error(f"Erreur dans la boucle principale: {e}")
                    if self.debug:
                        raise
                        
        except KeyboardInterrupt:
            self.logger.info("Arrêt demandé par l'utilisateur")
            
        finally:
            await self.cleanup()
            
    async def cleanup(self) -> None:
        """Nettoie les ressources du système."""
        self.logger.info("Nettoyage des ressources...")
        
        try:
            # Fermeture des connexions
            await asyncio.gather(
                self.data_adapter.cleanup(),
                self.model_manager.cleanup(),
                self.risk_manager.cleanup(),
                self.order_manager.cleanup()
            )
            
            # Fermeture du pool de threads
            self.thread_pool.shutdown(wait=True)
            
            self.logger.info("Nettoyage terminé")
            
        except Exception as e:
            self.logger.error(f"Erreur lors du nettoyage: {e}")
            
    def stop(self) -> None:
        """Arrête le système."""
        self.running = False

def parse_args() -> argparse.Namespace:
    """Parse les arguments en ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Système de trading automatique",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config/trading_config.yaml',
        help='Chemin vers le fichier de configuration'
    )
    
    parser.add_argument(
        '-m', '--mode',
        type=str,
        choices=['training', 'prediction', 'backtest'],
        default='prediction',
        help='Mode de fonctionnement du système'
    )
    
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Active le mode debug'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Niveau de logging'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Chemin vers le fichier de log'
    )
    
    parser.add_argument(
        '--metrics-interval',
        type=int,
        default=60,
        help='Intervalle de collecte des métriques (secondes)'
    )
    
    parser.add_argument(
        '--max-memory',
        type=int,
        default=1024,
        help='Limite maximale de mémoire en MB'
    )
    
    parser.add_argument(
        '--max-cpu',
        type=int,
        default=80,
        help='Limite maximale d\'utilisation CPU en %'
    )
    
    parser.add_argument(
        '--state-file',
        type=str,
        help='Chemin vers le fichier d\'état pour restauration'
    )
    
    parser.add_argument(
        '--no-restore',
        action='store_true',
        help='Désactive la restauration d\'état'
    )
    
    return parser.parse_args()

def setup_signal_handlers(system: TradingSystem) -> None:
    """Configure les gestionnaires de signaux."""
    
    def signal_handler(signum, frame):
        """Gestionnaire de signal pour arrêt propre."""
        system.logger.info(f"Signal reçu: {signal.Signals(signum).name}")
        system.stop()
        
    # Signaux d'arrêt
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Autres signaux
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, signal_handler)

async def main() -> None:
    """Point d'entrée principal."""
    args = parse_args()
    
    try:
        # Création et initialisation du système
        system = TradingSystem(
            config_path=args.config,
            mode=args.mode,
            debug=args.debug,
            log_file=args.log_file,
            metrics_interval=args.metrics_interval
        )
        
        # Configuration des gestionnaires de signaux
        setup_signal_handlers(system)
        
        # Démarrage du système
        await system.run()
        
    except Exception as e:
        logging.error(f"Erreur fatale: {e}")
        if args.debug:
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())