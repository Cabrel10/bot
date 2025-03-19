FROM trae-trading-base:latest

WORKDIR /app

# Copy and install dashboard-specific requirements
COPY requirements/dashboard-requirements.txt .
RUN pip install --no-cache-dir -r dashboard-requirements.txt

# Copy dashboard-specific files
COPY trading/visualization /app/trading/visualization
COPY trading/reporting /app/trading/reporting

# Configure service
EXPOSE 8050

# Start command
CMD ["python", "-m", "trading.visualization.dashboard"]