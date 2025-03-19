"""
Module de constantes pour le client Bitget.

Ce module définit les constantes utilisées par le client Bitget,
incluant les URLs, les limites, et les configurations par défaut.
"""

# URLs de base
BITGET_SPOT_MAINNET = "https://api.bitget.com"
BITGET_SPOT_TESTNET = "https://api-simulated.bitget.com"
BITGET_FUTURES_MAINNET = "https://api.bitget.com"
BITGET_FUTURES_TESTNET = "https://api-simulated.bitget.com"

# URLs WebSocket
BITGET_WS_SPOT_MAINNET = "wss://ws.bitget.com/spot/v1/stream"
BITGET_WS_SPOT_TESTNET = "wss://ws-simulated.bitget.com/spot/v1/stream"
BITGET_WS_FUTURES_MAINNET = "wss://ws.bitget.com/mix/v1/stream"
BITGET_WS_FUTURES_TESTNET = "wss://ws-simulated.bitget.com/mix/v1/stream"

# Endpoints REST
ENDPOINTS = {
    # Marché
    "exchange_info": "/api/spot/v1/public/products",
    "ticker": "/api/spot/v1/market/ticker",
    "depth": "/api/spot/v1/market/depth",
    "trades": "/api/spot/v1/market/trades",
    "klines": "/api/spot/v1/market/candles",
    
    # Compte
    "account": "/api/spot/v1/account/assets",
    "bills": "/api/spot/v1/account/bills",
    
    # Ordres
    "order": "/api/spot/v1/trade/orders",
    "cancel_order": "/api/spot/v1/trade/cancel-order",
    "batch_orders": "/api/spot/v1/trade/batch-orders",
    "open_orders": "/api/spot/v1/trade/open-orders",
    "order_history": "/api/spot/v1/trade/history",
    
    # Futures spécifique
    "futures_exchange_info": "/api/mix/v1/market/contracts",
    "futures_ticker": "/api/mix/v1/market/ticker",
    "futures_depth": "/api/mix/v1/market/depth",
    "futures_trades": "/api/mix/v1/market/trades",
    "futures_klines": "/api/mix/v1/market/candles",
    "futures_funding_rate": "/api/mix/v1/market/funding-rate",
    
    # Futures compte
    "futures_account": "/api/mix/v1/account/account",
    "futures_positions": "/api/mix/v1/position/allPosition",
    "futures_leverage": "/api/mix/v1/account/setLeverage",
    "futures_margin_mode": "/api/mix/v1/account/setMarginMode",
    
    # Futures ordres
    "futures_order": "/api/mix/v1/order/placeOrder",
    "futures_cancel_order": "/api/mix/v1/order/cancel-order",
    "futures_batch_orders": "/api/mix/v1/order/batch-orders",
    "futures_open_orders": "/api/mix/v1/order/current",
    "futures_order_history": "/api/mix/v1/order/history"
}

# Limites de taux par minute
RATE_LIMITS = {
    "PUBLIC": {
        "REQUEST_WEIGHT": 1200,
        "ORDERS": 100
    },
    "PRIVATE": {
        "REQUEST_WEIGHT": 6000,
        "ORDERS": 300
    }
}

# Intervalles de temps valides
VALID_INTERVALS = [
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "12h",
    "1d", "3d",
    "1w",
    "1M"
]

# Types d'ordres supportés
ORDER_TYPES = {
    "LIMIT": "limit",
    "MARKET": "market",
    "POST_ONLY": "post_only",
    "FOK": "fok",
    "IOC": "ioc"
}

# Côtés des ordres
ORDER_SIDES = {
    "BUY": "buy",
    "SELL": "sell"
}

# États des ordres
ORDER_STATUS = {
    "NEW": "new",
    "PARTIALLY_FILLED": "partially_filled",
    "FILLED": "filled",
    "CANCELLED": "cancelled",
    "REJECTED": "rejected",
    "EXPIRED": "expired"
}

# Types de positions (Futures)
POSITION_SIDE = {
    "LONG": "long",
    "SHORT": "short"
}

# Types de marge (Futures)
MARGIN_TYPE = {
    "ISOLATED": "isolated",
    "CROSSED": "crossed"
}

# Configuration par défaut
DEFAULT_RECV_WINDOW = 5000  # millisecondes
DEFAULT_LIMIT = 100
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
    "Content-Type": "application/json",
    "User-Agent": "bitget-trader/1.0"
} 