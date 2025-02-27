# Stock Price Prediction

This project is an advanced stock price prediction system using a hybrid model that integrates LSTM, Transformer, Reinforcement Learning (DQN), and XGBoost for optimal forecasting. The system also provides real-time insights and recommendations based on market trends and volatility.

## Project Structure

- **`dataset.py`**: Fetches and processes stock market data using Yahoo Finance. It applies technical indicators, normalizes data, and prepares it for model training.
- **`model.py`**: Implements a hybrid model combining LSTM, Transformer, and XGBoost with Reinforcement Learning. The model:
  - **LSTM (Long Short-Term Memory)**: Captures long-term dependencies in stock price movements.
  - **Transformer**: Incorporates attention mechanisms to weigh the importance of past data dynamically.
  - **XGBoost**: Enhances predictions by capturing non-linear relationships in stock price trends.
  - **Reinforcement Learning (DQN)**: Optimizes trading decisions by learning from historical rewards and penalties.
  - **Sentiment Analysis**: Incorporates market sentiment from financial news and search trends to adjust predictions.
  - **Adaptive Hyperparameter Tuning**: Uses Bayesian Optimization to fine-tune model parameters for optimal performance.
- **`test.py`**: Loads the trained model, fetches real-time stock data, and predicts future prices. It provides recommendations and visualizes trends.

## Installation

Ensure you have Python **3.10** installed.

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Dataset
Run `dataset.py` to fetch and preprocess stock data.

```bash
python dataset.py
```

### 2. Train Model
Run `model.py` to train and save the stock prediction model.

```bash
python model.py
```

### 3. Test & Predict
Run `test.py` to predict stock prices and get recommendations.

```bash
python test.py
```

## Features
- **Stock Data Processing**: Fetches historical stock prices and applies technical indicators.
- **Hybrid Model**: Combines LSTM, Transformer, XGBoost, and Reinforcement Learning for advanced predictions.
- **Sentiment Analysis**: Uses news sentiment and search trends for enhanced accuracy.
- **Real-Time Predictions**: Fetches current stock data and predicts future prices in USD and INR.
- **Investment Recommendations**: Provides buy/sell/hold suggestions based on market trends.
- **Market Trend Analysis**: Identifies bullish/bearish trends and suggests optimal trading strategies.
- **Risk Assessment**: Evaluates stock volatility and market conditions to enhance prediction reliability.

