FROM python:3.9-slim

WORKDIR /app

# Installation des dépendances minimales pour le monitoring
COPY requirements/monitoring-requirements.txt .
RUN pip install --no-cache-dir -r monitoring-requirements.txt

# Copie des fichiers nécessaires pour le monitoring
COPY trading/monitoring /app/trading/monitoring
COPY trading/utils /app/trading/utils

# Configuration du service
EXPOSE 9090

# Commande de démarrage
CMD ["python", "-m", "trading.monitoring.server"]