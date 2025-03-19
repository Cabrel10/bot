FROM python:3.9-slim

WORKDIR /app

# Install system dependencies and cleanup in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    tzdata \
    && ln -fs /usr/share/zoneinfo/Europe/Paris /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configure pip for better reliability
RUN echo "[global]\ntimeout = 6000\nretries = 20\ndefault-timeout = 6000\ntrusted-host = pypi.org files.pythonhosted.org pypi.python.org\nindex-url = https://pypi.org/simple\nextra-index-url = https://pypi.org/simple\n" > /etc/pip.conf

# Install base Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install common dependencies
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    scipy \
    requests \
    python-dotenv \
    pyyaml

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TZ=Europe/Paris