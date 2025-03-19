FROM trae-trading-base:latest

WORKDIR /app

# Copy and install data processor-specific requirements
COPY requirements/data-requirements.txt .
RUN pip install --no-cache-dir -r data-requirements.txt

# Copy data processor-specific files
COPY trading/data /app/trading/data
COPY trading/utils /app/trading/utils

# Configure service
EXPOSE 5000

# Start command
CMD ["python", "-m", "trading.data.processor"]