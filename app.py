from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import threading
import time
import datetime
import os
import logging
from jugaad_trader import Zerodha
import pyotp

# Import Appwrite SDK
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.exception import AppwriteException
from appwrite.query import Query  

# Import Bokeh for plotting
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import Span

# Import python-dotenv to load .env variables
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Ensure that SECRET_KEY is set
secret_key = os.getenv('SECRET_KEY')
if secret_key:
    app.secret_key = secret_key
else:
    raise Exception("SECRET_KEY environment variable not set")

# Initialize Appwrite client
client = Client()
appwrite_endpoint = os.getenv('APPWRITE_ENDPOINT')
appwrite_project_id = os.getenv('APPWRITE_PROJECT_ID')
appwrite_api_key = os.getenv('APPWRITE_API_KEY')

if not all([appwrite_endpoint, appwrite_project_id, appwrite_api_key]):
    raise Exception("One or more Appwrite environment variables are not set")

client.set_endpoint(appwrite_endpoint)
client.set_project(appwrite_project_id)
client.set_key(appwrite_api_key)

databases = Databases(client)
database_id = os.getenv('APPWRITE_DATABASE_ID')

# Ensure that database ID is set
if not database_id:
    raise Exception("APPWRITE_DATABASE_ID environment variable not set")

users_collection_id = 'users'     # Collection for storing user info
trades_collection_id = 'trades'   # Collection for storing trade history
predictions_collection_id = 'predictions'  # Collection for storing predictions

# User Session Class
class UserSession:
    def __init__(self, user_id, quantity, fund_mode, kite):
        self.user_id = user_id
        self.quantity = int(quantity)
        self.fund_mode = fund_mode  # 'quantity' or 'funds'
        self.kite = kite
        self.trading = False
        self.thread = None
        self.current_position = None  # To track current position
        self.profile_name = None
        self.available_funds = None

    def start_trading(self):
        if not self.trading:
            self.trading = True
            self.thread = threading.Thread(target=self.run_trading_session)
            self.thread.start()

    def stop_trading(self):
        if self.trading:
            self.trading = False
            if self.thread is not None:
                self.thread.join()

    def run_trading_session(self):
        logging.info(f"Trading session started for user {self.user_id}")
        while self.trading:
            try:
                # Fetch the latest prediction
                prediction = self.get_latest_prediction()
                if prediction is None:
                    logging.info("No predictions available. Retrying in 60 seconds.")
                    time.sleep(60)
                    continue

                predicted_stock = prediction['stock_symbol']
                holding_period = prediction['holding_period']

                if self.current_position:
                    # Check if the predicted stock is different from the current position
                    if predicted_stock != self.current_position['stock']:
                        # Exit current position and enter new one
                        self.exit_position()
                        self.enter_position(predicted_stock)
                    else:
                        logging.info(f"Holding existing position in {predicted_stock}.")
                else:
                    # No current position, enter new position
                    self.enter_position(predicted_stock)

                # Wait for the holding period
                for _ in range(holding_period):
                    if not self.trading:
                        break
                    time.sleep(60)  # Wait for 1 minute

                if not self.trading:
                    break

            except Exception as e:
                logging.error(f"Error in trading session for user {self.user_id}: {e}")
                break

        # After trading session ends, exit any open position
        if self.current_position:
            self.exit_position()
            self.current_position = None
        logging.info(f"Trading session ended for user {self.user_id}")

    def get_latest_prediction(self):
        try:
            # Fetch the latest prediction from the predictions collection
            documents = databases.list_documents(
                database_id=database_id,
                collection_id=predictions_collection_id,
                queries=[
                    Query.order_desc('$createdAt'),
                    Query.limit(1)
                ]
            )
            if documents['total'] > 0:
                prediction = documents['documents'][0]
                # Check if prediction is recent
                prediction_time = datetime.datetime.fromisoformat(prediction['prediction_time'])
                now = datetime.datetime.now()
                if (now - prediction_time).total_seconds() > 300:
                    logging.info("Latest prediction is too old.")
                    return None
                return prediction
            else:
                return None
        except Exception as e:
            logging.error(f"Error fetching prediction: {e}")
            return None

    def enter_position(self, stock_symbol):
        # Calculate quantity based on funds or specified quantity
        quantity = self.calculate_quantity(stock_symbol)
        if quantity <= 0:
            logging.warning(f"Insufficient funds or invalid quantity for {stock_symbol}")
            return

        # Place buy order
        try:
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NSE,
                tradingsymbol=stock_symbol,
                transaction_type=self.kite.TRANSACTION_TYPE_BUY,
                quantity=int(quantity),
                order_type=self.kite.ORDER_TYPE_MARKET,
                product=self.kite.PRODUCT_MIS
            )
            self.current_position = {
                'stock': stock_symbol,
                'quantity': int(quantity),
                'order_id': order_id,
                'entry_time': datetime.datetime.now()
            }
            logging.info(f"Entered position in {stock_symbol} with quantity {quantity}")
        except Exception as e:
            logging.error(f"Error placing buy order for {stock_symbol}: {e}")

    def exit_position(self):
        # Place sell order to exit current position
        try:
            stock_symbol = self.current_position['stock']
            quantity = self.current_position['quantity']
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NSE,
                tradingsymbol=stock_symbol,
                transaction_type=self.kite.TRANSACTION_TYPE_SELL,
                quantity=int(quantity),
                order_type=self.kite.ORDER_TYPE_MARKET,
                product=self.kite.PRODUCT_MIS
            )
            exit_time = datetime.datetime.now()
            logging.info(f"Exited position in {stock_symbol} with quantity {quantity}")

            # Record trade details in the database
            self.record_trade(
                stock_symbol=stock_symbol,
                quantity=int(quantity),
                entry_time=self.current_position['entry_time'],
                exit_time=exit_time
            )

            # Clear current position
            self.current_position = None
        except Exception as e:
            logging.error(f"Error placing sell order for {self.current_position['stock']}: {e}")

    def calculate_quantity(self, stock_symbol):
        if self.fund_mode == 'funds':
            # Calculate quantity based on available funds
            try:
                # Get last traded price
                ltp = self.kite.ltp('NSE:' + stock_symbol)['NSE:' + stock_symbol]['last_price']
                quantity = int(self.quantity // ltp)
                return quantity
            except Exception as e:
                logging.error(f"Error fetching LTP for {stock_symbol}: {e}")
                return 0
        else:
            # Use specified quantity
            return int(self.quantity)

    def record_trade(self, stock_symbol, quantity, entry_time, exit_time):
        try:
            # Fetch trade prices
            buy_price = self.get_trade_price('BUY', stock_symbol, entry_time)
            sell_price = self.get_trade_price('SELL', stock_symbol, exit_time)
            profit = (sell_price - buy_price) * quantity

            # Save trade details to Appwrite database
            databases.create_document(
                database_id,
                trades_collection_id,
                document_id='unique()',
                data={
                    'user_id': self.user_id,
                    'stock_symbol': stock_symbol,
                    'quantity': int(quantity),
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'profit': profit,
                    'entry_time': entry_time.isoformat(),
                    'exit_time': exit_time.isoformat()
                },
                permissions=[]
            )
            logging.info(f"Trade recorded for {stock_symbol}")
        except Exception as e:
            logging.error(f"Error recording trade: {e}")

    def get_trade_price(self, transaction_type, stock_symbol, time):
        # Placeholder method to get trade price
        try:
            ltp = self.kite.ltp('NSE:' + stock_symbol)['NSE:' + stock_symbol]['last_price']
            return ltp
        except Exception as e:
            logging.error(f"Error fetching LTP for {stock_symbol}: {e}")
            return 0

    def fetch_account_details(self):
        try:
            profile = self.kite.profile()
            self.profile_name = profile.get('user_name', 'N/A')
            funds = self.kite.margins()
            self.available_funds = funds['equity']['available']['cash']
        except Exception as e:
            print(f"Error fetching account details: {e}")
            self.profile_name = 'N/A'
            self.available_funds = 'N/A'

# Dictionary to store user sessions
user_sessions = {}

# Route to display the main page
@app.route('/', methods=['GET'])
def index():
    logged_in = 'user_id' in session
    current_year = datetime.datetime.now().year
    if logged_in:
        user_id = session['user_id']
        user_session = user_sessions.get(user_id)
        trading = user_session.trading if user_session else False

        # Fetch account details
        if user_session:
            user_session.fetch_account_details()
            profile_name = user_session.profile_name
            available_funds = user_session.available_funds
            current_position = user_session.current_position
        else:
            profile_name = 'N/A'
            available_funds = 'N/A'
            current_position = None

        # Fetch latest predictions
        latest_prediction = get_latest_prediction()
        plot_script = ''
        plot_div = ''
        if latest_prediction and user_session:
            stock_symbol = latest_prediction['stock_symbol']
            initial_time_str = latest_prediction.get('initial_prediction_time')
            if initial_time_str:
                initial_time = datetime.datetime.fromisoformat(initial_time_str)
                if initial_time.tzinfo is None:
                    initial_time = initial_time.replace(tzinfo=datetime.timezone.utc)
            else:
                initial_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)

            end_time = datetime.datetime.now(datetime.timezone.utc)

            # Get instrument token for the stock
            try:
                instruments = user_session.kite.instruments("NSE")
                instrument_token = next((instrument['instrument_token'] for instrument in instruments if instrument['tradingsymbol'] == stock_symbol), None)

                if instrument_token:
                    # Fetch historical data
                    data = user_session.kite.historical_data(
                        instrument_token=instrument_token,
                        from_date=initial_time,
                        to_date=end_time,
                        interval='minute'
                    )

                    # Prepare data for Bokeh plot
                    if data:
                        dates = [d['date'] for d in data]
                        opens = [d['open'] for d in data]
                        highs = [d['high'] for d in data]
                        lows = [d['low'] for d in data]
                        closes = [d['close'] for d in data]

                        # Determine whether candles are increasing or decreasing
                        inc = [close > open_ for close, open_ in zip(closes, opens)]
                        dec = [open_ > close for open_, close in zip(opens, closes)]

                        # Width of each candle in milliseconds
                        w = 1 * 60 * 1000  # 1 minute in ms

                        # Create Bokeh plot
                        p = figure(x_axis_type="datetime", tools="pan,wheel_zoom,box_zoom,reset", width=300, height=200)
                        p.title.text = f"Candlestick chart for {stock_symbol}"

                        # Plot candlesticks
                        p.segment(dates, highs, dates, lows, color="black")
                        p.vbar(x=[d for d, inc_ in zip(dates, inc) if inc_],
                               width=w,
                               top=[close for close, inc_ in zip(closes, inc) if inc_],
                               bottom=[open_ for open_, inc_ in zip(opens, inc) if inc_],
                               fill_color="#D5E1DD", line_color="black")

                        p.vbar(x=[d for d, dec_ in zip(dates, dec) if dec_],
                               width=w,
                               top=[open_ for open_, dec_ in zip(opens, dec) if dec_],
                               bottom=[close for close, dec_ in zip(closes, dec) if dec_],
                               fill_color="#F2583E", line_color="black")

                        # Add vertical line at initial_time
                        vline = Span(location=initial_time.timestamp()*1000, dimension='height', line_color='blue', line_width=1)
                        p.add_layout(vline)

                        # Generate script and div
                        plot_script, plot_div = components(p)
                    else:
                        logging.info(f"No historical data available for {stock_symbol}")
                else:
                    logging.info(f"Instrument token not found for {stock_symbol}")

            except Exception as e:
                logging.error(f"Error generating plot: {e}")

        return render_template(
            'index.html',
            logged_in=True,
            user_id=user_id,
            quantity=int(session.get('quantity', 1)),
            fund_mode=session.get('fund_mode', 'quantity'),
            current_year=current_year,
            trading=trading,
            profile_name=profile_name,
            available_funds=available_funds,
            latest_prediction=latest_prediction,
            current_position=current_position,
            plot_script=plot_script,
            plot_div=plot_div
        )
    else:
        return render_template(
            'index.html',
            logged_in=False,
            quantity=1,
            fund_mode='quantity',
            current_year=current_year
        )

# Function to fetch the latest prediction and initial prediction time
def get_latest_prediction():
    try:
        documents = databases.list_documents(
            database_id=database_id,
            collection_id=predictions_collection_id,
            queries=[
                Query.order_desc('$createdAt'),
                Query.limit(100)
            ]
        )
        if documents['total'] > 0:
            latest_prediction = documents['documents'][0]
            current_stock_symbol = latest_prediction['stock_symbol']
            initial_time = latest_prediction['prediction_time']

            for prediction in documents['documents'][1:]:
                if prediction['stock_symbol'] == current_stock_symbol:
                    initial_time = prediction['prediction_time']
                else:
                    break

            latest_prediction['initial_prediction_time'] = initial_time

            return latest_prediction
        else:
            return None
    except Exception as e:
        logging.error(f"Error fetching prediction: {e}")
        return None

# Route to handle login
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user_id = data['user_id']
    password = data['password']
    totp_key = data['totp_key']

    # Validate credentials with Zerodha
    totp = pyotp.TOTP(totp_key)
    twofa = totp.now()

    kite = Zerodha(user_id=user_id, password=password, twofa=twofa)

    try:
        kite.login()
        logging.info(f"Logged in successfully as {kite.user_id}")

        # Check if user already exists in the database
        try:
            user_doc = databases.get_document(database_id, users_collection_id, user_id)
            # User exists, load their settings
            quantity = int(user_doc.get('quantity', 1))
            fund_mode = user_doc.get('fund_mode', 'quantity')
        except AppwriteException as e:
            if e.code == 404:
                # User does not exist, create new user with default settings
                quantity = 1
                fund_mode = 'quantity'
                databases.create_document(database_id, users_collection_id, user_id, {
                    'user_id': user_id,
                    'quantity': quantity,
                    'fund_mode': fund_mode,
                    'join_date': str(datetime.date.today())
                })
            else:
                logging.error(f"Error accessing user data: {e}")
                return jsonify({'status': 'error', 'message': 'Database error.'}), 500

        # Save user info in session
        session['user_id'] = user_id
        session['quantity'] = quantity
        session['fund_mode'] = fund_mode

        # Initialize user session
        user_session = UserSession(user_id, quantity, fund_mode, kite)
        user_sessions[user_id] = user_session

        return jsonify({'status': 'success', 'user_id': user_id, 'quantity': quantity, 'fund_mode': fund_mode})
    except Exception as e:
        logging.error(f"Login failed: {e}")
        return jsonify({'status': 'error', 'message': 'Login failed. Please check your credentials.'}), 401

# Route to handle logout
@app.route('/logout', methods=['POST'])
def logout():
    user_id = session.get('user_id')
    if user_id and user_id in user_sessions:
        user_sessions[user_id].stop_trading()
        del user_sessions[user_id]
    session.clear()
    return jsonify({'status': 'success'})

# Route to start trading
@app.route('/start', methods=['POST'])
def start_trading():
    user_id = session.get('user_id')
    if user_id and user_id in user_sessions:
        user_sessions[user_id].start_trading()
        return jsonify({'status': 'started'})
    else:
        return jsonify({'status': 'error', 'message': 'User session not found'}), 404

# Route to stop trading
@app.route('/stop', methods=['POST'])
def stop_trading():
    user_id = session.get('user_id')
    if user_id and user_id in user_sessions:
        user_sessions[user_id].stop_trading()
        return jsonify({'status': 'stopped'})
    else:
        return jsonify({'status': 'error', 'message': 'User session not found'}), 404

# Route to update quantity or funds
@app.route('/update_quantity', methods=['POST'])
def update_quantity():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'status': 'error', 'message': 'User session not found'}), 401

    data = request.get_json()
    quantity = float(data.get('quantity', 1))
    fund_mode = data.get('fund_mode', 'quantity')
    session['quantity'] = int(quantity)
    session['fund_mode'] = fund_mode

    # Update quantity/funds in the user session
    if user_id in user_sessions:
        user_sessions[user_id].quantity = int(quantity)
        user_sessions[user_id].fund_mode = fund_mode

    # Update quantity/funds in the database
    try:
        databases.update_document(database_id, users_collection_id, user_id, {
            'quantity': int(quantity),
            'fund_mode': fund_mode
        })
        return jsonify({'status': 'success'})
    except Exception as e:
        logging.error(f"Error updating quantity/funds: {e}")
        return jsonify({'status': 'error', 'message': 'Failed to update quantity/funds'}), 500

# Route to fetch trade history
@app.route('/trade_history', methods=['GET'])
def trade_history():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'status': 'error', 'message': 'User session not found'}), 401

    if user_id in user_sessions:
        user_session = user_sessions[user_id]
        try:
            trades = user_session.kite.orders()
            # Return trades as JSON
            return jsonify({'status': 'success', 'trades': trades})
        except Exception as e:
            logging.error(f"Error fetching trade history for user {user_id}: {e}")
            return jsonify({'status': 'error', 'message': 'Failed to fetch trade history'}), 500
    else:
        return jsonify({'status': 'error', 'message': 'User session not found'}), 404

# Route to fetch current position data
@app.route('/current_position', methods=['GET'])
def current_position_route():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'status': 'error', 'message': 'User session not found'}), 401

    if user_id in user_sessions:
        user_session = user_sessions[user_id]
        if user_session.current_position:
            current_position = {
                'stock': user_session.current_position['stock'],
                'quantity': user_session.current_position['quantity'],
                'order_id': user_session.current_position['order_id'],
                'entry_time': user_session.current_position['entry_time'].strftime('%Y-%m-%d %H:%M:%S')
            }
            return jsonify({'status': 'success', 'current_position': current_position})
        else:
            return jsonify({'status': 'success', 'current_position': None})
    else:
        return jsonify({'status': 'error', 'message': 'User session not found'}), 404

# Route to fetch latest prediction data
@app.route('/latest_prediction', methods=['GET'])
def latest_prediction_route():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'status': 'error', 'message': 'User session not found'}), 401

    latest_prediction = get_latest_prediction()
    if latest_prediction:
        return jsonify({'status': 'success', 'latest_prediction': latest_prediction})
    else:
        return jsonify({'status': 'error', 'message': 'No predictions available'}), 404

@app.route('/disclosure')
def disclosure():
    current_year = datetime.datetime.now().year
    return render_template('disclosure.html', current_year=current_year)

if __name__ == '__main__':
    app.run(debug=True)

