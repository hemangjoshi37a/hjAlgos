import os
import joblib
import time
import datetime
import pandas as pd
from tqdm import tqdm
from pandas.tseries.offsets import BDay
from jugaad_trader import Zerodha
import pyotp
import pickle
from sklearn.utils.class_weight import compute_class_weight
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from dotenv import load_dotenv

# **Data Fetching and Preparation**
load_dotenv()

# **1. Initialize and Login to Zerodha Account**
# Load Zerodha credentials from environment variables
user_id = os.getenv('ZERODHA_USER_ID')
password = os.getenv('ZERODHA_PASSWORD')
totp_key = os.getenv('ZERODHA_TOTP_KEY')

# Generate the TOTP code for two-factor authentication
totp = pyotp.TOTP(totp_key)
twofa = totp.now()

# Initialize Zerodha session
kite = Zerodha(user_id=user_id, password=password, twofa=twofa)

# Perform login
try:
    kite.login()
    print(f"Logged in successfully as {kite.user_id}")
except Exception as e:
    print(f"Login failed: {e}")
    exit(1)

# **2. Fetching Nifty 50 Stock Data**

# List of Nifty 50 stock symbols
nifty50_symbols = [
    'ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK',
    'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BPCL', 'BHARTIARTL',
    'BRITANNIA', 'CIPLA', 'COALINDIA', 'DIVISLAB', 'TORNTPHARM', 'EICHERMOT',
    'GRASIM', 'HCLTECH', 'HDFCBANK', 'HDFCLIFE', 'HINDALCO', 'HINDUNILVR',
    'ICICIBANK', 'ITC', 'INDUSINDBK', 'INFY', 'JSWSTEEL', 'KOTAKBANK',
    'LT', 'M&M', 'MARUTI', 'NTPC', 'NESTLEIND', 'ONGC', 'POWERGRID',
    'RELIANCE', 'SBILIFE', 'SBIN', 'SUNPHARMA', 'TCS', 'TATACONSUM',
    'TATAMOTORS', 'TATASTEEL', 'TECHM', 'TITAN', 'ULTRACEMCO', 'UPL',
    'WIPRO', 'HEROMOTOCO', 'BEL'
]
nifty50_symbols.sort()
print(nifty50_symbols)

# Fetch all NSE instruments to get instrument tokens for the stocks
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

# Save the symbol_token_map to a CSV file
symbol_token_df = pd.DataFrame(list(symbol_token_map.items()), columns=['Stock Symbol', 'Instrument Token'])
symbol_token_df.to_csv('symbol_token_map.csv', index=False)

print("Symbol-Token map saved to symbol_token_map.csv")
