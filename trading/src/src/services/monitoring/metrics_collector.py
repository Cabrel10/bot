"""Service de collecte des métriques."""
import time
from prometheus_client import start_http_server, Counter, Gauge
import psutil

# Métriques Prometheus
TRADING_OPERATIONS = Counter('trading_operations_total', 'Nombre total d\'opérations de trading')
SYSTEM_MEMORY = Gauge('system_memory_usage_bytes', 'Utilisation mémoire système')
CPU_USAGE = Gauge('system_cpu_usage_percent', 'Utilisation CPU système')

def collect_system_metrics():
    """Collecte les métriques système."""
    while True:
        SYSTEM_MEMORY.set(psutil.virtual_memory().used)
        CPU_USAGE.set(psutil.cpu_percent())
        time.sleep(15)

if __name__ == '__main__':
    # Démarrage du serveur Prometheus
    start_http_server(9090)
    collect_system_metrics()
