"""
Module de constantes pour le client Binance.

Ce module définit les constantes utilisées par le client Binance,
incluant les URLs, les limites, et les configurations par défaut.
"""

# URLs de base
BINANCE_SPOT_MAINNET = "https://api.binance.com"
BINANCE_SPOT_TESTNET = "https://testnet.binance.vision"
BINANCE_FUTURES_MAINNET = "https://fapi.binance.com"
BINANCE_FUTURES_TESTNET = "https://testnet.binancefuture.com"

# URLs WebSocket
BINANCE_WS_SPOT_MAINNET = "wss://stream.binance.com:9443/ws"
BINANCE_WS_SPOT_TESTNET = "wss://testnet.binance.vision/ws"
BINANCE_WS_FUTURES_MAINNET = "wss://fstream.binance.com/ws"
BINANCE_WS_FUTURES_TESTNET = "wss://stream.binancefuture.com/ws"

# Endpoints REST
ENDPOINTS = {
    # Marché
    "exchange_info": "/api/v3/exchangeInfo",
    "ticker": "/api/v3/ticker/24hr",
    "ticker_price": "/api/v3/ticker/price",
    "ticker_book": "/api/v3/ticker/bookTicker",
    "depth": "/api/v3/depth",
    "trades": "/api/v3/trades",
    "historical_trades": "/api/v3/historicalTrades",
    "agg_trades": "/api/v3/aggTrades",
    "klines": "/api/v3/klines",
    
    # Compte
    "account": "/api/v3/account",
    "balance": "/api/v3/balance",
    "my_trades": "/api/v3/myTrades",
    
    # Ordres
    "test_order": "/api/v3/order/test",
    "order": "/api/v3/order",
    "open_orders": "/api/v3/openOrders",
    "all_orders": "/api/v3/allOrders",
    
    # Futures spécifique
    "futures_exchange_info": "/fapi/v1/exchangeInfo",
    "futures_funding_rate": "/fapi/v1/fundingRate",
    "futures_ticker": "/fapi/v1/ticker/24hr",
    "futures_ticker_price": "/fapi/v1/ticker/price",
    "futures_ticker_book": "/fapi/v1/ticker/bookTicker",
    "futures_depth": "/fapi/v1/depth",
    "futures_trades": "/fapi/v1/trades",
    "futures_historical_trades": "/fapi/v1/historicalTrades",
    "futures_agg_trades": "/fapi/v1/aggTrades",
    "futures_klines": "/fapi/v1/klines",
    "futures_continuous_klines": "/fapi/v1/continuousKlines",
    "futures_index_price_klines": "/fapi/v1/indexPriceKlines",
    "futures_mark_price_klines": "/fapi/v1/markPriceKlines",
    
    # Futures compte
    "futures_account": "/fapi/v2/account",
    "futures_balance": "/fapi/v2/balance",
    "futures_position_risk": "/fapi/v2/positionRisk",
    "futures_position": "/fapi/v2/positionSide/dual",
    "futures_leverage": "/fapi/v1/leverage",
    "futures_margin_type": "/fapi/v1/marginType",
    "futures_my_trades": "/fapi/v1/userTrades",
    
    # Futures ordres
    "futures_test_order": "/fapi/v1/order/test",
    "futures_order": "/fapi/v1/order",
    "futures_batch_orders": "/fapi/v1/batchOrders",
    "futures_open_orders": "/fapi/v1/openOrders",
    "futures_all_orders": "/fapi/v1/allOrders"
}

# Limites de taux par minute
RATE_LIMITS = {
    "IP": {
        "REQUEST_WEIGHT": 1200,
        "ORDERS": 100,
        "RAW_REQUESTS": 5000
    },
    "UID": {
        "ORDERS": 1200,
        "CANCEL_ORDERS": 1200
    }
}

# Intervalles de temps valides
VALID_INTERVALS = [
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d",
    "1w",
    "1M"
]

# Types d'ordres supportés
ORDER_TYPES = {
    "LIMIT": "LIMIT",
    "MARKET": "MARKET",
    "STOP_LOSS": "STOP_LOSS",
    "STOP_LOSS_LIMIT": "STOP_LOSS_LIMIT",
    "TAKE_PROFIT": "TAKE_PROFIT",
    "TAKE_PROFIT_LIMIT": "TAKE_PROFIT_LIMIT",
    "LIMIT_MAKER": "LIMIT_MAKER"
}

# Côtés des ordres
ORDER_SIDES = {
    "BUY": "BUY",
    "SELL": "SELL"
}

# États des ordres
ORDER_STATUS = {
    "NEW": "NEW",
    "PARTIALLY_FILLED": "PARTIALLY_FILLED",
    "FILLED": "FILLED",
    "CANCELED": "CANCELED",
    "PENDING_CANCEL": "PENDING_CANCEL",
    "REJECTED": "REJECTED",
    "EXPIRED": "EXPIRED"
}

# Time in force
TIME_IN_FORCE = {
    "GTC": "GTC",  # Good Till Cancel
    "IOC": "IOC",  # Immediate or Cancel
    "FOK": "FOK"   # Fill or Kill
}

# Types de positions (Futures)
POSITION_SIDE = {
    "BOTH": "BOTH",
    "LONG": "LONG",
    "SHORT": "SHORT"
}

# Types de marge (Futures)
MARGIN_TYPE = {
    "ISOLATED": "ISOLATED",
    "CROSSED": "CROSSED"
}

# Types de working type (Futures)
WORKING_TYPE = {
    "MARK_PRICE": "MARK_PRICE",
    "CONTRACT_PRICE": "CONTRACT_PRICE"
}

# Paramètres par défaut
DEFAULT_RECV_WINDOW = 5000  # millisecondes
DEFAULT_LIMIT = 500
MAX_LIMIT = 1000

# Configuration WebSocket
WS_CONNECTION_LIMIT = 5
WS_CONNECTION_TIMEOUT = 5  # secondes
WS_PING_INTERVAL = 20  # secondes
WS_PING_TIMEOUT = 10  # secondes
WS_CLOSE_TIMEOUT = 2  # secondes

# Configuration des requêtes HTTP
HTTP_TIMEOUT = 10  # secondes
MAX_RETRIES = 3
RETRY_DELAY = 1  # secondes
RETRY_MULTIPLIER = 2
MAX_RETRY_DELAY = 10  # secondes

# En-têtes HTTP par défaut
DEFAULT_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "binance-trader/1.0"
} 