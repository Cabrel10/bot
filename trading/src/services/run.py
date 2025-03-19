import subprocess
import sys
from pathlib import Path
import argparse
import logging
import signal
import time

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/main.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('MainService')

class ServiceManager:
    def __init__(self):
        self.services = {}
        self.running = True
        
    def start_service(self, name, script_path):
        """Démarre un service spécifique"""
        try:
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.services[name] = process
            logger.info(f"Service {name} démarré")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du démarrage du service {name}: {str(e)}")
            return False
    
    def stop_service(self, name):
        """Arrête un service spécifique"""
        if name in self.services:
            process = self.services[name]
            process.terminate()
            process.wait()
            del self.services[name]
            logger.info(f"Service {name} arrêté")
    
    def stop_all_services(self):
        """Arrête tous les services"""
        for name in list(self.services.keys()):
            self.stop_service(name)
    
    def handle_signal(self, signum, frame):
        """Gère les signaux d'arrêt"""
        logger.info("Signal d'arrêt reçu")
        self.running = False
        self.stop_all_services()

def main():
    parser = argparse.ArgumentParser(description="Gestionnaire des services de trading")
    parser.add_argument('--services', nargs='+', default=['all'],
                       help='Services à démarrer (data, dashboard, resources, trading, ou all)')
    args = parser.parse_args()

    service_manager = ServiceManager()
    signal.signal(signal.SIGINT, service_manager.handle_signal)
    signal.signal(signal.SIGTERM, service_manager.handle_signal)

    # Définition des services disponibles
    available_services = {
        'data': 'run_data_service.py',
        'dashboard': 'rundashboard.py',
        'resources': 'run_resources_manager.py',
        'trading': 'run_trading.py'
    }

    # Démarrage des services sélectionnés
    services_to_start = (available_services.keys() 
                        if 'all' in args.services 
                        else args.services)

    for service_name in services_to_start:
        if service_name in available_services:
            script_path = Path(__file__).parent / available_services[service_name]
            service_manager.start_service(service_name, script_path)
        else:
            logger.warning(f"Service inconnu: {service_name}")

    # Boucle principale
    try:
        while service_manager.running:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur")
    finally:
        service_manager.stop_all_services()

if __name__ == "__main__":
    main() 