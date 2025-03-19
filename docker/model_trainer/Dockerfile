FROM trae-trading-base:latest

WORKDIR /app

# Copy and install model trainer-specific requirements
COPY requirements/model-requirements.txt .
RUN pip install --no-cache-dir -r model-requirements.txt

# Create model directory
RUN mkdir -p /app/models

# Copy model trainer-specific files
COPY trading/models /app/trading/models
COPY trading/training /app/trading/training
COPY trading/utils /app/trading/utils

# Configure service
EXPOSE 5001

# Start command
CMD ["python", "-m", "trading.training.trainer"]