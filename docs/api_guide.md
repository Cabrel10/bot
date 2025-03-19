# Guide de l'API

## 🌐 Vue d'Ensemble

### Base URL
```
http://localhost:8000/api/v1
```

## 📡 Endpoints

### Modèles

#### GET /models
Liste tous les modèles disponibles.

```bash
curl -X GET http://localhost:8000/api/v1/models
```

Réponse :
```json
{
  "models": [
    {
      "id": "neural_network_v1",
      "type": "neural_network",
      "status": "trained"
    },
    {
      "id": "genetic_algo_v1",
      "type": "genetic_algorithm",
      "status": "optimizing"
    }
  ]
}
```

#### POST /models/train
Démarre l'entraînement d'un modèle.

```bash
curl -X POST http://localhost:8000/api/v1/models/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "neural_network_v1",
    "config": {
      "epochs": 100,
      "batch_size": 32
    }
  }'
```

### Stratégies

#### GET /strategies
Liste toutes les stratégies disponibles.

```bash
curl -X GET http://localhost:8000/api/v1/strategies
```

#### POST /strategies/backtest
Lance un backtest pour une stratégie.

```bash
curl -X POST http://localhost:8000/api/v1/strategies/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_id": "trend_following_v1",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31"
  }'
```

### Données

#### GET /data/historical
Récupère les données historiques.

```bash
curl -X GET "http://localhost:8000/api/v1/data/historical?symbol=BTCUSDT&interval=1h"
```

## 🔒 Authentification

### Headers
```bash
Authorization: Bearer <votre-token>
```

### Obtention du Token
```bash
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "username": "user",
    "password": "password"
  }'
```

## 📊 Websockets

### Connexion
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/v1/market');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

### Souscription aux Mises à Jour
```javascript
ws.send(JSON.stringify({
  "action": "subscribe",
  "channel": "market_data",
  "symbols": ["BTCUSDT", "ETHUSDT"]
}));
```

## ⚡ Bonnes Pratiques

### Rate Limiting
- 60 requêtes par minute pour les endpoints publics
- 120 requêtes par minute pour les endpoints authentifiés

### Gestion des Erreurs
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Trop de requêtes",
    "details": {
      "retry_after": 60
    }
  }
}
```

### Pagination
```bash
curl -X GET "http://localhost:8000/api/v1/trades?page=2&limit=100"
```

## 🔍 Monitoring

### Statut de l'API
```bash
curl -X GET http://localhost:8000/api/v1/status
```

### Métriques
```bash
curl -X GET http://localhost:8000/api/v1/metrics
```