import numpy as np
import pandas as pd
import torch
import time
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf
from transformers import pipeline
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pytrends.request import TrendReq
from collections import deque
import random
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from bayes_opt import BayesianOptimization
import torch.nn.functional as F
from transformers import pipeline
import psutil

def get_dynamic_batch_size():
    """Estimate batch size based on available memory."""
    total_memory = psutil.virtual_memory().total / (1024 ** 3)  # Convert to GB
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GPU memory in GB
    
    if total_memory >= 16:  
        return 128  # High memory: Large batch
    elif total_memory >= 8:
        return 64   # Medium memory: Medium batch
    else:
        return 32   # Low memory: Small batch



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

load_dotenv()
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=-1)  # Use CPU

# Reinforcement Learning (Deep Q-Lea
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x):
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                           self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)
    
    @staticmethod
    def scale_noise(size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.learning_rate = 0.001
        
        self.model = self._build_noisy_net()
        self.target_model = self._build_noisy_net()
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        self.temperature = 1.0  # 🔥 Initial high temperature for more exploration
        self.temperature_decay = 0.99  # 🔽 Reduce temperature over time
        self.min_temperature = 0.1  # ❄️ Prevent it from becoming too small

    def _build_noisy_net(self):
        return nn.Sequential(
            NoisyLinear(self.state_size, 64),
            nn.ReLU(),
            NoisyLinear(64, 64),
            nn.ReLU(),
            NoisyLinear(64, self.action_size)
        )

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        with torch.no_grad():
            q_values = self.model(state)
            boltzmann_probs = torch.exp(q_values / self.temperature)
            boltzmann_probs /= boltzmann_probs.sum()
            action = torch.multinomial(boltzmann_probs, 1).item()

    # 🔽 Apply temperature decay AFTER action is chosen
        self.temperature = max(self.temperature * self.temperature_decay, self.min_temperature)

        return action
        
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.target_model(next_state))
            
            current_q = self.model(state)[action]
            
            loss = self.criterion(current_q, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=0.3)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.layer_norm(out[:, -1, :])
        out = self.fc(out)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)  # Register buffer directly

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device) 

class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, num_layers, output_size):
        super(TransformerModel, self).__init__()
        self.embed_dim = ((input_size + num_heads - 1) // num_heads * num_heads)  # Ensure divisibility
        
        # Use a linear projection instead of constant padding
        self.input_projection = nn.Linear(input_size, self.embed_dim)

        self.pos_encoder = PositionalEncoding(self.embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=num_heads, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(self.embed_dim, output_size)

    def forward(self, x):
        x = self.input_projection(x)  # Project input to embed_dim
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

    

def tune_hyperparameters():
    def objective(lr, hidden_size, dropout, batch_size, transformer_layers, xgb_learning_rate, xgb_max_depth):
        model = HybridModel(
    input_size=len(feature_columns),
    lstm_hidden=int(hidden_size),
    lstm_layers=2,
    transformer_heads=4,
    transformer_layers=int(transformer_layers),
    output_size=1,
    xgb_learning_rate=xgb_learning_rate,  # ✅ Pass optimized XGBoost learning rate
    xgb_max_depth=int(xgb_max_depth)  # ✅ Pass optimized XGBoost max depth
)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Train the model (simplified for demonstration)
        for epoch in range(5):
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        # Evaluate on validation set
        val_loss = evaluate_model(model, test_loader, criterion)
        return -val_loss  # Maximize negative validation loss

    pbounds = {
    'lr': (0.0001, 0.01),
    'hidden_size': (32, 128),
    'dropout': (0.1, 0.5),
    'batch_size': (32, 128),
    'transformer_layers': (1, 4),
    'xgb_learning_rate': (0.001, 0.1),  
    'xgb_max_depth': (3, 10)  
}


    optimizer = BayesianOptimization(f=objective, pbounds=pbounds)
    optimizer.maximize(init_points=5, n_iter=10)
    
    return optimizer.max

def fetch_historical_sentiment(ticker, date):
    """
    If no relevant news is found, estimate sentiment based on market trends.
    """
    try:
        stock_data = yf.download(ticker, start=date, end=date)  # Fetch stock data for that day
        if not stock_data.empty:
            daily_change = (stock_data['Close'][0] - stock_data['Open'][0]) / stock_data['Open'][0]
            return daily_change  # Use % price change as sentiment proxy
    except Exception as e:
        print(f"Error fetching fallback sentiment for {ticker} on {date}: {e}")

    return 0  # Default neutral sentiment if everything fails

def scrape_news_sentiment(ticker, date, max_retries=3):
    """
    Fetch sentiment for a specific stock ticker on a given date.
    Handles single-letter tickers carefully.
    """
    time.sleep(2)  # Delay between requests
    attempt = 0
    formatted_date = date.strftime('%Y-%m-%d')  # Ensure date format is correct

    # Handle single-letter stock names separately
    if len(ticker) == 1:
        print(f"⚠️ Handling single-letter ticker: {ticker}")
        search_query = f"{ticker} stock news {formatted_date}"
    else:
        search_query = f"{ticker} stock news"

    while attempt < max_retries:
        try:
            url = f"https://finance.yahoo.com/quote/{ticker}/news"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=20)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract headlines
            headlines = [item.text for item in soup.find_all('h3')]

            # Filter out irrelevant results (for single-letter stocks)
            filtered_headlines = [h for h in headlines if ticker in h.upper() or "stock" in h.lower()]

            sentiment_scores = [sentiment_pipeline(headline)[0]['score'] for headline in filtered_headlines]

            return np.mean(sentiment_scores) if sentiment_scores else 0  # Return average sentiment

        except requests.exceptions.Timeout:
            attempt += 1
            print(f"Timeout for {ticker} on {formatted_date}. Retrying {attempt}/{max_retries}...")
            time.sleep(5)

        except Exception as e:
            print(f"Error fetching sentiment for {ticker} on {formatted_date}: {e}")
            return 0  # Default to neutral if error

    return 0  # Default sentiment if all retries fail

def fetch_search_trends(ticker):
    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload(kw_list=[ticker], timeframe='today 3-m')
    interest_over_time = pytrends.interest_over_time()
    return interest_over_time[ticker].mean()  # Return average trend score

# Hybrid Model with Cross-Attention
class HybridModel(nn.Module):
    def __init__(self, input_size, lstm_hidden, lstm_layers, transformer_heads, transformer_layers, output_size, xgb_learning_rate, xgb_max_depth):
        super(HybridModel, self).__init__()
        self.lstm = LSTMModel(input_size, lstm_hidden, lstm_layers, output_size)
        self.transformer = TransformerModel(input_size, transformer_heads, transformer_layers, output_size)

        # XGBoost model as an additional predictor
        self.xgb = XGBRegressor(n_estimators=100,learning_rate=xgb_learning_rate,max_depth=int(xgb_max_depth))  # ✅ Use optimized tree depth


        self.fc = nn.Linear(output_size * 2 + 3, output_size)  # +3 for GDP, Inflation, Interest Rate
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        lstm_out = self.lstm(x)
        transformer_out = self.transformer(x)

        # Convert to NumPy for XGBoost (outside PyTorch computation graph)
        x_np = x.cpu().detach().numpy().reshape(x.shape[0], -1)
        x_np = np.nan_to_num(x_np)  # ✅ Replace NaNs with 0 to prevent crashes
        xgb_out = torch.tensor(self.xgb.predict(x_np), dtype=torch.float32).unsqueeze(1).to(x.device)

        combined = torch.cat([lstm_out, transformer_out, xgb_out, x[:, -1, -3:]], dim=1)  # Last 3 features: GDP, Inflation, Interest Rate
        return self.fc(self.dropout(combined))

def fetch_news_sentiment(date, ticker=None):
    """
    Fetch sentiment for a specific date using Google News.
    """
    try:
        formatted_date = date.strftime('%Y-%m-%d')
        search_query = f"{ticker} stock news {formatted_date}" if ticker else f"stock news {formatted_date}"
        url = f"https://www.google.com/search?q={search_query}&tbm=nws"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=20)
        soup = BeautifulSoup(response.text, 'html.parser')

        headlines = [item.text for item in soup.find_all('h3')]
        sentiment_scores = [sentiment_pipeline(headline)[0]['score'] for headline in headlines]

        if len(sentiment_scores) == 0:
            return 0  # Default neutral sentiment if no news

        return np.mean(sentiment_scores)

    except Exception as e:
        print(f"Error fetching news sentiment for {date}: {e}")
        return 0  # Default neutral sentiment if error

import os
# Load and preprocess multi-stock dataset
def fetch_macro_data(start_date, end_date):
    """
    Fetch real macroeconomic indicators from the FRED API securely.
    """
    try:
        API_KEY = os.getenv("FRED_API_KEY")  # ✅ Load API key securely

        if not API_KEY:
            raise ValueError("API key not found. Make sure it's stored in the .env file.")

        indicators = {
            "GDP_Growth": "A191RL1Q225SBEA",
            "Inflation_Rate": "CPIAUCSL",
            "Interest_Rate": "FEDFUNDS"
        }

        macro_data = pd.DataFrame({'Date': pd.date_range(start=start_date, end=end_date, freq='M')})

        for key, series_id in indicators.items():
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={API_KEY}&file_type=json"
            response = requests.get(url).json()
            df = pd.DataFrame(response['observations'])
            df['Date'] = pd.to_datetime(df['date'])
            df.rename(columns={'value': key}, inplace=True)
            df[key] = pd.to_numeric(df[key], errors='coerce')
            macro_data = pd.merge(macro_data, df[['Date', key]], on='Date', how='left')

        macro_data.fillna(method='ffill', inplace=True)

        return macro_data

    except Exception as e:
        print(f"Error fetching macro data: {e}")
        return None

def load_and_preprocess_data(file_path, chunk_size=100000):
    try:
        dataset_chunks = []
        
        # Load stock data
        for chunk in pd.read_csv(file_path, parse_dates=['Date'], chunksize=chunk_size):
            chunk = chunk.sort_values(['Ticker', 'Date'])

            # Fetch macroeconomic data for the stock date range
            macro_data = fetch_macro_data(chunk['Date'].min(), chunk['Date'].max())

            # Merge stock data with macroeconomic indicators on Date
            chunk = pd.merge(chunk, macro_data, on='Date', how='left')

            # Fill missing macro values with previous month's data
            chunk.fillna(method='ffill', inplace=True)

            dataset_chunks.append(chunk)

        dataset = pd.concat(dataset_chunks, ignore_index=True)

        # 
        label_encoder = LabelEncoder()
        dataset['Ticker'] = label_encoder.fit_transform(dataset['Ticker'])

        #
        feature_columns = ['Close', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower','Volume',
                           'Volume_SMA', 'SMA_20', 'SMA_50', 'Volatility',
                           'Daily_Return', 'Sentiment', 'Trend', 
                           'GDP_Growth', 'Inflation_Rate', 'Interest_Rate']  

        # Scale numerical features
        scaler = MinMaxScaler()
        dataset[feature_columns] = scaler.fit_transform(dataset[feature_columns])

        return dataset, scaler, label_encoder, feature_columns  

    except Exception as e:
        print(f"Error loading or preprocessing data: {e}")
        return None, None, None, None


def create_data_loaders(X, y, batch_size=64):
    batch_size = get_dynamic_batch_size() 
    num_workers = 4 if torch.cuda.is_available() else 0  # 🚀 Use multiple workers only if GPU is available
    pin_memory = True if torch.cuda.is_available() else False  # ✅ Pin memory only for GPU

    dataset = TensorDataset(X, y)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,  # 🏎️ Faster loading on GPU, safe on CPU
        pin_memory=pin_memory  # ✅ Only use pin_memory if GPU is available
    )

# Create sequences for each stock
def create_sequences(data, seq_length, feature_columns):
    sequences = []
    targets = []
    tickers = data['Ticker'].unique()
    
    for ticker in tickers:
        stock_data = data[data['Ticker'] == ticker]
        if len(stock_data) <= seq_length:
            continue  # Skip if not enough data
            
        stock_values = stock_data[feature_columns].values
        
        for i in range(len(stock_values) - seq_length):
            sequences.append(stock_values[i:i+seq_length])
            targets.append(stock_values[i+seq_length, 0])  # Predict 'Close' price
    
    if not sequences:
        raise ValueError("Not enough data to create sequences")
        
    return np.array(sequences), np.array(targets)

# Real-Time Stock Prediction
def predict_real_time(model, stock_data, scaler, stock_features, device='cpu'):
    stock_data = stock_data.fillna(0)  # Handle missing values
    
    # Safely normalize data
    features_data = stock_data[stock_features].values
    # Compute IQR
    Q1 = np.percentile(features_data, 25, axis=0)
    Q3 = np.percentile(features_data, 75, axis=0)
    IQR = Q3 - Q1

# Define acceptable range
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

# Apply IQR-based outlier filtering
    filtered_data = np.clip(features_data, lower_bound, upper_bound)

# Scale data
    normalized_data = scaler.transform(filtered_data)

    
    stock_tensor = torch.tensor(normalized_data, dtype=torch.float32).unsqueeze(0)
    stock_tensor = stock_tensor.to(device)

    model.eval()
    with torch.no_grad():
        prediction = model(stock_tensor).cpu().numpy()

    # Inverse transform the prediction
    inverse_prediction = scaler.inverse_transform(
        np.zeros((1, len(stock_features)))
    )
    inverse_prediction[0, 0] = prediction[0, 0]
    
    return inverse_prediction[0, 0]

def train_hybrid_model(model, train_loader, dqn_agent, epochs=10, device='cpu'):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    dqn_agent.model.train()

    # Prepare XGBoost training data
    xgb_features, xgb_targets = [], []

    accumulation_steps = 4  
    optimizer.zero_grad()

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)
            (loss / accumulation_steps).backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()

            # Store data for XGBoost
            xgb_features.append(data.cpu().numpy().reshape(data.shape[0], -1))
            xgb_targets.append(target.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs} complete | Avg Loss: {avg_loss:.6f}')

        if epoch % 2 == 0:
            dqn_agent.update_target_model()

    # Train XGBoost on accumulated features
    xgb_features = np.nan_to_num(np.vstack(xgb_features))  # ✅ Replace NaNs with 0
    xgb_targets = np.nan_to_num(np.vstack(xgb_targets).ravel())  # ✅ Replace NaNs with 0
    model.xgb.fit(xgb_features, xgb_targets)

    return model, dqn_agent

# Evaluate the model
def evaluate_model(model, test_loader, criterion, device='cpu'):
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Use the precomputed edge_index passed to the function
            output = model(data)
            
            loss = criterion(output, target)
            total_loss += loss.item()
            
            predictions.extend(output.cpu().numpy())
            actuals.extend(target.cpu().numpy())

    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    print(f'Evaluation Results:')
    print(f'MSE: {mse:.6f}')
    print(f'MAE: {mae:.6f}')
    print(f'R²: {r2:.6f}')
    
    return predictions, actuals

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    file_path = 'stocks_dataset.csv'
    dataset, scaler, label_encoder, feature_columns = load_and_preprocess_data(file_path)
    
    if dataset is not None and len(dataset) > 0:
        print("Data preprocessing completed successfully!")
        print(f"Dataset shape: {dataset.shape}")
        print(f"Feature columns: {feature_columns}")
        
        # Create sequences
        seq_length = 30
        try:
            X, y = create_sequences(dataset, seq_length, feature_columns)
            print(f"Created {len(X)} sequences of length {seq_length}")
            
            # Split data
            split_index = int(len(X) * 0.8)
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]
            
            print(f"Training set: {len(X_train)} sequences")
            print(f"Test set: {len(X_test)} sequences")
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train)
            X_test_tensor = torch.FloatTensor(X_test)
            y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
            y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
            
            # Create data loaders
            train_loader = create_data_loaders(X_train_tensor, y_train_tensor, batch_size=64)
            test_loader = create_data_loaders(X_test_tensor, y_test_tensor, batch_size=64)
            
            # Set device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")
            
            # Initialize model and DQN agent
            input_size = X_train.shape[2]
            best_params = tune_hyperparameters()
            print(f"Best Hyperparameters: {best_params}")

# Use optimized hyperparameters in model initialization
            model = HybridModel(
    input_size=X_train.shape[2],
    lstm_hidden=int(best_params['hidden_size']),
    lstm_layers=2,
    transformer_heads=4,
    transformer_layers=2,
    output_size=1,
    xgb_learning_rate=best_params['xgb_learning_rate'], 
    xgb_max_depth=int(best_params['xgb_max_depth']) 
).to(device)


            
            dqn_agent = DQNAgent(state_size=1, action_size=3)
            
            # Define optimizer and criterion
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # Train the model
            model, dqn_agent = train_hybrid_model(
                model=model, 
                train_loader=train_loader, 
                dqn_agent=dqn_agent, 
                epochs=10, 
                device=device
            )
            
            # Evaluate the model
            predictions, actuals = evaluate_model(
                model=model, 
                test_loader=test_loader, 
                criterion=criterion, 
                device=device
            )
            
            # Save the model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': scaler,
                'label_encoder': label_encoder
            }, 'hybrid_stock_model_complete.pth')
            
            print("Training and evaluation completed successfully!")
            
        except Exception as e:
            print(f"Error during sequence creation or training: {e}")
    else:
        print("Data preprocessing failed. Please check the dataset and file path.")