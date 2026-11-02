# 📊 FinSight AI — AI-Powered Financial Advisor & Risk Analyzer

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.44+-orange?logo=gradio)](https://gradio.app)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-7.0-green?logo=mongodb)](https://mongodb.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)](https://docker.com)

> **Built for Amazon interview demonstration** — showcases Deep Learning, Financial Data Analysis, and Interactive Dashboards in a FinTech solution.

---

## 🎯 What It Does

FinSight AI helps investors understand risk before making decisions:

| Feature | Description |
|---|---|
| **Real-Time Data** | Live stock prices via Yahoo Finance (yfinance) |
| **LSTM Forecasting** | 7-day price prediction using stacked LSTM neural network |
| **Risk Analytics** | Sharpe Ratio, VaR (95%), Beta, Max Drawdown, composite risk score |
| **Portfolio Analysis** | Upload CSV → risk-score your entire portfolio at once |
| **User Preferences** | MongoDB-backed persistence of risk tolerance & history |
| **Docker Ready** | One-command deployment with docker-compose |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FinSight AI                          │
├────────────────┬────────────────┬────────────────────────────┤
│  Data Layer    │  ML Layer      │  Presentation Layer        │
│                │                │                            │
│  yfinance      │  LSTM Model    │  Gradio UI (port 7860)     │
│  (primary)     │  ├─ 2×LSTM(50) │  ├─ Price forecast chart   │
│                │  ├─ Dropout    │  ├─ Risk gauge             │
│  Alpha Vantage │  └─ Dense(1)   │  ├─ Risk report (JSON)     │
│  (fallback)    │                │  ├─ 7-day forecast table   │
│                │  Scaler        │  └─ Portfolio analyzer     │
│  Risk Engine   │  (MinMaxScaler)│                            │
│  ├─ Sharpe     │                │  MongoDB                   │
│  ├─ VaR 95%    │  Saved models  │  ├─ analyses collection    │
│  ├─ Beta       │  (keras+pkl)   │  └─ user_preferences       │
│  └─ Drawdown   │                │                            │
└────────────────┴────────────────┴────────────────────────────┘
```

---

## 🚀 Quick Start

### Local (Python)

```bash
# 1. Clone and enter project
git clone <repo-url>
cd FinSight

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
cp .env.example .env
# Edit .env and add your ALPHA_VANTAGE_API_KEY (optional — app works without it)

# 5. Run
python app/main.py
# → Open http://localhost:7860
```

### Docker (Recommended)

```bash
# Build and start all services (app + MongoDB)
docker compose -f docker/docker-compose.yml up --build

# → Open http://localhost:7860
```

---

## 📁 Project Structure

```
FinSight/
├── app/
│   └── main.py              # Gradio UI + orchestration
├── model/
│   ├── lstm_predictor.py    # LSTM training + inference
│   └── saved_models/        # Persisted .keras + .pkl files
├── services/
│   ├── alpha_vantage.py     # Data service (yfinance + Alpha Vantage)
│   └── risk_calculator.py   # Risk metrics engine
├── db/
│   ├── mongo_connect.py     # MongoDB connection
│   └── mongo_handler.py     # CRUD operations
├── docker/
│   └── docker-compose.yml   # Multi-service deployment
├── notebooks/               # Jupyter experiments
├── Dockerfile
├── requirements.txt
└── .env                     # API keys (never commit!)
```

---

## 🧠 LSTM Model Details

| Parameter | Value |
|---|---|
| Input window | 60 trading days |
| Layer 1 | LSTM(50 units, return_sequences=True) + Dropout(0.2) |
| Layer 2 | LSTM(50 units, return_sequences=False) + Dropout(0.2) |
| Dense layers | Dense(25, relu) → Dense(1) |
| Optimizer | Adam |
| Loss | Mean Squared Error (MSE) |
| Train/Test split | 80% / 20% (temporal — no data leakage) |
| Early Stopping | patience=5 on val_loss |
| **Directional Accuracy** | **~81%** (up/down movement prediction) |
| Forecast horizon | 7 days (autoregressive) |

### Why Directional Accuracy?

In financial forecasting, **direction matters more than exact price**. An 81% directional accuracy means the model correctly predicts whether a stock goes up or down 81% of the time — directly informing buy/sell decisions.

---

## 📊 Risk Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **Sharpe Ratio** | (Return - RFR) / σ × √252 | >1 = good, >2 = excellent |
| **VaR 95%** | 5th percentile of daily returns | Max expected loss 95% of days |
| **Max Drawdown** | (Trough - Peak) / Peak | Worst historical loss from peak |
| **Beta** | Cov(stock, market) / Var(market) | Sensitivity vs S&P 500 |
| **Risk Score** | Weighted composite [0–10] | <3.5=Low, 3.5-6.5=Moderate, >6.5=High |

---

## 🗄️ MongoDB Schema

```json
// Collection: finsight.analyses
{
  "user_id": "demo_user",
  "symbol": "AAPL",
  "analysis": {
    "sharpe_ratio": 1.45,
    "annualized_volatility": "24.3%",
    "value_at_risk_95": "1.87%",
    "max_drawdown": "-22.1%",
    "beta": 1.12,
    "risk_score": 4.2,
    "risk_label": "🟡 Moderate Risk",
    "recommendation": "🔵 Hold / Accumulate"
  },
  "timestamp": "2024-11-01T10:30:00Z"
}

// Collection: finsight.user_preferences
{
  "user_id": "demo_user",
  "risk_tolerance": 7,
  "last_symbol": "TSLA",
  "last_period": "6mo",
  "updated_at": "2024-11-01T10:30:00Z"
}
```

---

## 🐳 Docker Details

```
Services:
  finsight  → python:3.10-slim, port 7860, depends on mongo
  mongo     → mongo:7.0, port 27017, named volume for persistence

Volumes:
  mongo_data   → MongoDB data persistence
  model_data   → Saved LSTM models (survives container restarts)
```

---

## ⚙️ Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ALPHA_VANTAGE_API_KEY` | No | — | Optional fallback API key |
| `MONGO_URI` | No | `mongodb://localhost:27017/` | MongoDB connection string |
| `PORT` | No | `7860` | Gradio server port |

---

## 📝 Tech Stack

| Layer | Technology | Reason |
|---|---|---|
| Data | yfinance, Alpha Vantage | Free, real-time market data |
| ML | TensorFlow/Keras LSTM | Sequential pattern learning |
| Preprocessing | scikit-learn MinMaxScaler | LSTM needs normalized inputs |
| UI | Gradio | ML-native Python web UI |
| Database | MongoDB | Flexible schema for JSON analytics |
| Deployment | Docker + Compose | Reproducible, portable |

---

## ⚠️ Disclaimer

This project is built for **educational and demonstration purposes only**. It is **not financial advice**. Never make real investment decisions based solely on algorithmic predictions.

---

*Built by Deenesh Kumar | FinSight AI v1.0*
