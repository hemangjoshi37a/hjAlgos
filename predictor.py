import torch
import joblib
import os
import time
import datetime
import pandas as pd
import numpy as np
from jugaad_trader import Zerodha
import pyotp
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.exception import AppwriteException
import math
import torch.nn as nn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x

# Define Stock Predictor Model
class StockPredictorModel(nn.Module):
    def __init__(self, num_stocks=50, lookback=60, num_features=5, d_model=512, nhead=8, num_layers=6, dropout=0.1, lookahead=60):
        super(StockPredictorModel, self).__init__()
        self.num_stocks = num_stocks
        self.lookback = lookback
        self.num_features = num_features
        self.d_model = d_model
        self.lookahead = lookahead

        # Input projection layer
        self.input_projection = nn.Linear(num_stocks * num_features, d_model)

        # Sinusoidal positional encoding
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=lookback)

        # Transformer encoder with dropout
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output layers
        self.output_layer_x = nn.Linear(d_model, self.num_stocks)  # For predicting X
        self.output_layer_y = nn.Linear(d_model, self.lookahead)   # For predicting Y

        # Initialize weights
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.output_layer_x.weight)
        nn.init.zeros_(self.output_layer_x.bias)
        nn.init.xavier_uniform_(self.output_layer_y.weight)
        nn.init.zeros_(self.output_layer_y.bias)

    def forward(self, x):
        # x shape: (batch_size, num_stocks, lookback, num_features)
        batch_size = x.size(0)

        # Permute x to (batch_size, lookback, num_stocks, num_features)
        x = x.permute(0, 2, 1, 3)

        # Reshape x to (batch_size, lookback, num_stocks * num_features)
        x = x.reshape(batch_size, self.lookback, self.num_stocks * self.num_features)

        # Project input features
        x = self.input_projection(x)  # Shape: (batch_size, lookback, d_model)

        # Apply positional encoding
        x = self.positional_encoding(x)  # Shape: (batch_size, lookback, d_model)

        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # Shape: (batch_size, lookback, d_model)

        # Apply layer normalization
        x = self.layer_norm(x)  # Shape: (batch_size, lookback, d_model)

        # Take the output from the last time step
        x = x[:, -1, :]  # Shape: (batch_size, d_model)

        # Apply dropout
        x = self.dropout(x)

        # Output layers
        output_x = self.output_layer_x(x)  # Shape: (batch_size, num_stocks)
        output_y = self.output_layer_y(x)  # Shape: (batch_size, lookahead)

        return output_x, output_y

# Function to setup Appwrite
def setup_appwrite():
    """Initialize and configure Appwrite client"""
    client = Client()
    
    # Load Appwrite configuration from environment variables
    appwrite_endpoint = os.getenv('APPWRITE_ENDPOINT')
    appwrite_project_id = os.getenv('APPWRITE_PROJECT_ID')
    appwrite_api_key = os.getenv('APPWRITE_API_KEY')
    
    # Validate Appwrite configuration
    if not all([appwrite_endpoint, appwrite_project_id, appwrite_api_key]):
        raise Exception("One or more Appwrite environment variables are not set")
    
    client.set_endpoint(appwrite_endpoint)
    client.set_project(appwrite_project_id)
    client.set_key(appwrite_api_key)
    
    return Databases(client)

# Function to save prediction to Appwrite database
def save_prediction_to_db(databases, prediction):
    """Save prediction to Appwrite database"""
    try:
        database_id = os.getenv('APPWRITE_DATABASE_ID')
        predictions_collection_id = 'predictions'
        
        if not database_id:
            raise Exception("APPWRITE_DATABASE_ID environment variable not set")
        
        databases.create_document(
            database_id=database_id,
            collection_id=predictions_collection_id,
            document_id='unique()',
            data={
                'stock_symbol': prediction['predicted_stock'],
                'holding_period': prediction['holding_period'],
                'user_id': 'FC5917',  # Consider loading this from .env or context if dynamic
                'prediction_time': prediction['last_candle_time'],
                'enter_price': prediction['enter_price']  # Include the enter_price value
            },
            permissions=[]
        )
        print(f"Prediction saved: {prediction}")
    except AppwriteException as e:
        print(f"Error saving prediction to database: {e}")
    except Exception as ex:
        print(f"General error: {ex}")

# Function to run continuous inference and save predictions
def run_inference(model, kite, symbol_token_map, nifty50_symbols, databases, lookback=120):
    """
    Run continuous inference and save predictions to Appwrite database.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load normalization parameters
    try:
        print("Loading mean and std...")
        mean = joblib.load('mean.pkl')
        std = joblib.load('std.pkl')
        print("Mean and std loaded successfully.")
    except Exception as e:
        print(f"Error loading mean and std: {e}")
        return

    while True:
        try:
            print("Fetching historical data...")
            data_dict = {}
            days_to_fetch = 1

            while True:
                to_date = datetime.datetime.now()
                from_date = to_date - datetime.timedelta(days=days_to_fetch)

                # Adjust to trading hours
                from_date = from_date.replace(hour=9, minute=15, second=0, microsecond=0)
                to_date = to_date.replace(hour=15, minute=30, second=0, microsecond=0)

                data_dict = {}
                data_lengths = []

                for symbol in nifty50_symbols:
                    try:
                        historical_data = kite.historical_data(
                            instrument_token=symbol_token_map[symbol],
                            from_date=from_date,
                            to_date=to_date,
                            interval='minute'
                        )
                        df = pd.DataFrame(historical_data)
                        if not df.empty:
                            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                            df = df.sort_values('date')
                            data_dict[symbol] = df
                            data_lengths.append(len(df))
                    except Exception as e:
                        print(f"Error fetching data for {symbol}: {e}")

                if data_lengths and min(data_lengths) >= lookback:
                    break
                    
                days_to_fetch += 1
                if days_to_fetch > 10:
                    print("Unable to fetch sufficient data. Retrying...")
                    time.sleep(60)
                    continue

            # Align data by date
            common_dates = set(data_dict[nifty50_symbols[0]]['date'])
            for symbol in nifty50_symbols[1:]:
                if symbol in data_dict:
                    common_dates = common_dates.intersection(set(data_dict[symbol]['date']))

            common_dates = sorted(list(common_dates))[-lookback:]
            last_candle_time = common_dates[-1].isoformat()  # Get the timestamp of the last candle

            # Prepare input data
            input_data = []
            for symbol in nifty50_symbols:
                if symbol in data_dict:
                    df = data_dict[symbol]
                    df = df[df['date'].isin(common_dates)]
                    if len(df) == lookback:
                        features = df[['open', 'high', 'low', 'close', 'volume']].values
                        input_data.append(features)

            if len(input_data) == len(nifty50_symbols):
                input_data = np.array(input_data)
                
                # Normalize data
                mean_expanded = mean[:, np.newaxis, :]
                std_expanded = std[:, np.newaxis, :]
                input_data_normalized = (input_data - mean_expanded) / (std_expanded + 1e-8)

                # Get predictions
                input_tensor = torch.tensor(input_data_normalized, dtype=torch.float32).to(device)
                input_tensor = input_tensor.unsqueeze(0)

                model.eval()
                with torch.no_grad():
                    output_x, output_y = model(input_tensor)
                    predicted_x = torch.argmax(output_x, dim=1).item()
                    predicted_y = torch.argmax(output_y, dim=1).item()

                predicted_stock = nifty50_symbols[predicted_x]
                holding_period = predicted_y + 1

                # Get the enter_price value (close price of the last candle) for the predicted stock
                df_predicted_stock = data_dict[predicted_stock]
                last_candle_date = common_dates[-1]

                df_last_candle = df_predicted_stock[df_predicted_stock['date'] == last_candle_date]
                if not df_last_candle.empty:
                    enter_price = df_last_candle['close'].iloc[0]
                else:
                    enter_price = None  # Handle case where data is missing

                # Save prediction to Appwrite with last candle time and enter_price
                prediction = {
                    'last_candle_time': last_candle_time,
                    'predicted_stock': predicted_stock,
                    'holding_period': holding_period,
                    'enter_price': enter_price  # Include enter_price in the prediction dictionary
                }
                save_prediction_to_db(databases, prediction)

            # Wait for 1 minute before next prediction
            time.sleep(60)

        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(60)
            continue

# Main function
def main():
    # Setup parameters
    lookback = 120
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Zerodha credentials from environment variables
    user_id = os.getenv('ZERODHA_USER_ID')
    password = os.getenv('ZERODHA_PASSWORD')
    totp_key = os.getenv('ZERODHA_TOTP_KEY')

    # Validate Zerodha credentials
    if not all([user_id, password, totp_key]):
        raise Exception("One or more Zerodha environment variables are not set")

    # Initialize Zerodha
    totp = pyotp.TOTP(totp_key)
    twofa = totp.now()
    kite = Zerodha(user_id=user_id, password=password, twofa=twofa)

    try:
        kite.login()
        print(f"Logged in successfully as {kite.user_id}")
    except Exception as e:
        print(f"Login failed: {e}")
        return

    # Initialize Appwrite
    try:
        databases = setup_appwrite()
    except Exception as e:
        print(f"Appwrite setup failed: {e}")
        return

    # List of Nifty 50 stock symbols
    nifty50_symbols = [
        'ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK', 
        'BAJAJ-AUTO', 'BAJAJFINSV', 'BAJFINANCE', 'BEL', 'BHARTIARTL', 
        'BPCL', 'BRITANNIA', 'CIPLA', 'COALINDIA', 'DIVISLAB', 
        'EICHERMOT', 'GRASIM', 'HCLTECH', 'HDFCBANK', 'HDFCLIFE', 
        'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK', 'INDUSINDBK', 
        'INFY', 'ITC', 'JSWSTEEL', 'KOTAKBANK', 'LT', 
        'M&M', 'MARUTI', 'NESTLEIND', 'NTPC', 'ONGC', 
        'POWERGRID', 'RELIANCE', 'SBILIFE', 'SBIN', 'SUNPHARMA', 
        'TATACONSUM', 'TATAMOTORS', 'TATASTEEL', 'TCS', 'TECHM', 
        'TITAN', 'TORNTPHARM', 'ULTRACEMCO', 'UPL', 'WIPRO'
    ]

    # Load model
    model = StockPredictorModel(
        num_stocks=len(nifty50_symbols),
        lookback=lookback,
        num_features=5,
    ).to(device)

    model_path = 'stock_predictor_model.pth'
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return

    try:
        model.load_state_dict(torch.load(model_path, map_location=device,weights_only=False))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Fetch all NSE instruments to get instrument tokens for the stocks
    try:
        instruments = kite.instruments("NSE")
        instruments_df = pd.DataFrame(instruments)

        # Filter instruments to get only Nifty 50 stocks
        nifty50_instruments = instruments_df[
            (instruments_df['tradingsymbol'].isin(nifty50_symbols)) &
            (instruments_df['segment'] == 'NSE') &
            (instruments_df['instrument_type'] == 'EQ')
        ]

        # Create a mapping from stock symbols to instrument tokens
        nifty50_tokens = nifty50_instruments[['tradingsymbol', 'instrument_token']].reset_index(drop=True)
        symbol_token_map = dict(zip(nifty50_tokens['tradingsymbol'], nifty50_tokens['instrument_token']))

        # Check if all symbols have corresponding tokens
        missing_tokens = set(nifty50_symbols) - set(symbol_token_map.keys())
        if missing_tokens:
            print(f"Missing instrument tokens for symbols: {missing_tokens}")
            return

    except Exception as e:
        print(f"Error fetching instruments: {e}")
        return

    # Run continuous inference
    run_inference(
        model=model,
        kite=kite,
        symbol_token_map=symbol_token_map,
        nifty50_symbols=nifty50_symbols,
        databases=databases,
        lookback=lookback
    )

if __name__ == '__main__':
    main()

