import torch
import yfinance as yf
import numpy as np
import pandas as pd
import requests
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from model import HybridModel

# Load the trained model
checkpoint = torch.load("hybrid_stock_model_complete.pth", map_location=torch.device("cpu"))
scaler = checkpoint['scaler']
label_encoder = checkpoint['label_encoder']

# Initialize the model (Ensure the architecture matches your saved model)
model = HybridModel(
    input_size=30,  
    lstm_hidden=64,  
    lstm_layers=2,  
    transformer_heads=4,  
    transformer_layers=2,  
    output_size=1,  
    xgb_learning_rate=0.01,  
    xgb_max_depth=5  
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Function to fetch real-time stock data
def get_stock_data(ticker, days=200):
    stock_data = yf.download(ticker, period=f"{days}d", interval="1d")
    stock_data = stock_data[['Close']]
    stock_data.rename(columns={'Close': 'Original_Close'}, inplace=True)
    return stock_data

# Function to get real-time USD to INR exchange rate
def get_usd_to_inr():
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
        data = response.json()
        return data["rates"]["INR"]
    except:
        return 83.0  # Approximate fallback exchange rate

# Function to make predictions
def predict_stock_price(ticker):
    stock_data = get_stock_data(ticker, days=200)
    
    if stock_data.empty:
        print(f"Error: No stock data found for {ticker}")
        return

    # Scale data
    features = scaler.transform(stock_data.values.reshape(-1, 1))
    stock_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        predicted_price = model(stock_tensor).item()

    # Convert price to INR
    usd_to_inr = get_usd_to_inr()
    current_price = stock_data['Original_Close'].iloc[-1]
    predicted_price_inr = predicted_price * usd_to_inr
    current_price_inr = current_price * usd_to_inr

    # Identify trends
    highest_price = stock_data['Original_Close'].max()
    lowest_price = stock_data['Original_Close'].min()
    trend = "Bullish" if predicted_price > current_price else "Bearish"
    
    # Recommendation system
    volatility = np.std(stock_data['Original_Close'].pct_change().dropna())  
    thresholds = {
        "base_volatility_threshold": round(volatility, 4),
        "market_condition": "high" if volatility > 0.02 else "low",
        "final_threshold": round(volatility * 100, 2)
    }

    recommendation = "Buy" if predicted_price > current_price * 1.02 else "Hold" if predicted_price >= current_price else "Sell"
    reasoning = "Predicted price is significantly higher than the current price." if recommendation == "Buy" else "Market is stable; holding is recommended." if recommendation == "Hold" else "Price is expected to fall, selling is advised."

    hold_duration = 30 if recommendation == "Buy" else 0  # Suggest holding period

    # Best Buy Date (if prediction is higher than today)
    best_date = stock_data.index[-1] if recommendation == "Buy" else "N/A"

    # Print results
    print(f"\nStock: {ticker}")
    print(f"Current Price (USD): ${current_price:.2f}")
    print(f"Current Price (INR): ₹{current_price_inr:.2f}")
    print(f"Predicted Price (USD): ${predicted_price:.2f}")
    print(f"Predicted Price (INR): ₹{predicted_price_inr:.2f}")
    print(f"\nRecommendation System:")
    print(f"- Base Volatility Threshold: {thresholds['base_volatility_threshold']}")
    print(f"- Market Condition: {thresholds['market_condition'].capitalize()}")
    print(f"- Final Adaptive Threshold: {thresholds['final_threshold']}%")
    print(f"\nRecommendation: {recommendation}")
    print(f"Reasoning: {reasoning}")
    print(f"\nHold Duration Recommendation: Hold for {hold_duration} days until the predicted peak is reached.")
    print(f"\nAdditional Insights:")
    print(f"Recent Trend: {trend}")
    print(f"Highest Price (200 days): ${highest_price:.2f} (₹{highest_price * usd_to_inr:.2f})")
    print(f"Lowest Price (200 days): ${lowest_price:.2f} (₹{lowest_price * usd_to_inr:.2f})")
    print(f"Best Buy Date: {best_date}")

    # Plot prices
    plot_prices(stock_data['Original_Close'], predicted_price)

# Function to plot stock prices
def plot_prices(actual_prices, predicted_price):
    plt.figure(figsize=(10, 5))
    plt.plot(actual_prices, label="Actual Prices", color="blue")
    plt.axhline(y=predicted_price, color="red", linestyle="--", label="Predicted Price")
    plt.xlabel("Days")
    plt.ylabel("Stock Price (USD)")
    plt.title("Stock Price Prediction")
    plt.legend()
    plt.show()

# Run the test
ticker = input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT): ")
predict_stock_price(ticker)
