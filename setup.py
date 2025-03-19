from setuptools import setup, find_packages

setup(
    name="trading-bot",
    version="1.0.0",
    description="Advanced Trading Bot with ML capabilities",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "scikit-learn>=0.24.0",
        "tensorflow>=2.4.0",
        "ccxt>=1.50.0",
        "ta>=0.7.0",
        "streamlit>=0.85.0",
        "plotly>=4.14.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "trading-bot=trading.main:main",
            "trading-backtest=trading.backtest:main",
        ],
    },
) 