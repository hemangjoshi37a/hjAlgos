from flask import Flask, render_template, request, session, redirect, url_for, jsonify, send_from_directory
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

from binance.pay.merchant import Merchant as BClient
import uuid
import csv
import os


# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Binance Pay API credentials
binance_pay_key = os.getenv('BINANCE_PAY_KEY')
binance_pay_secret = os.getenv('BINANCE_PAY_SECRET')

# Initialize Binance Pay client
binance_pay_client = BClient(binance_pay_key, binance_pay_secret)




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

        # Fetch user's premium status
        is_premium = False
        try:
            user_doc = databases.get_document(database_id, users_collection_id, user_id)
            is_premium = user_doc.get('is_premium', False)
        except AppwriteException as e:
            logging.error(f"Error fetching user data: {e}")

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
            plot_div=plot_div,
            is_premium=is_premium
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
                    'password': password,
                    'totp_key': totp_key,
                    'quantity': quantity,
                    'join_date': str(datetime.date.today()),                    
                    'fund_mode': fund_mode
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

@app.route('/robots.txt')
def robots_txt():
    return send_from_directory(app.static_folder, 'robots.txt'), 200, {'Content-Type': 'text/plain'}

@app.route('/sitemap.xml', methods=['GET'])
def sitemap():
    return send_from_directory(app.static_folder, 'sitemap.xml'), 200, {'Content-Type': 'application/xml'}

# Route to fetch data for prediction chart
@app.route('/prediction_chart_data', methods=['GET'])
def prediction_chart_data():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'status': 'error', 'message': 'User session not found'}), 401

    stock_symbol = request.args.get('stock_symbol')
    prediction_time_str = request.args.get('prediction_time')
    holding_period = int(request.args.get('holding_period', 0))

    if not all([stock_symbol, prediction_time_str, holding_period]):
        return jsonify({'status': 'error', 'message': 'Missing parameters'}), 400

    try:
        prediction_time = datetime.datetime.fromisoformat(prediction_time_str)
        if prediction_time.tzinfo is None:
            prediction_time = prediction_time.replace(tzinfo=datetime.timezone.utc)
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Invalid prediction_time format'}), 400

    user_session = user_sessions.get(user_id)
    if not user_session:
        return jsonify({'status': 'error', 'message': 'User session not found'}), 404

    try:
        initial_time = prediction_time - datetime.timedelta(minutes=5)
        end_time = prediction_time + datetime.timedelta(minutes=holding_period + 5)

        # Get instrument token for the stock
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
            # Process data to convert datetime objects to strings
            for d in data:
                if isinstance(d['date'], datetime.datetime):
                    d['date'] = d['date'].isoformat()
            # Return data as JSON
            return jsonify({'status': 'success', 'data': data})
        else:
            logging.info(f"Instrument token not found for {stock_symbol}")
            return jsonify({'status': 'error', 'message': 'Instrument token not found'}), 404

    except Exception as e:
        logging.error(f"Error fetching historical data for chart: {e}")
        return jsonify({'status': 'error', 'message': 'Failed to fetch historical data'}), 500

# Route to fetch historical predictions
@app.route('/historical_predictions', methods=['GET'])
def historical_predictions():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'status': 'error', 'message': 'User session not found'}), 401

    # Get page number from query params, default to 1
    page = int(request.args.get('page', 1))
    per_page = 10

    try:
        # Fetch historical predictions from the predictions collection
        offset = (page - 1) * per_page
        documents = databases.list_documents(
            database_id=database_id,
            collection_id=predictions_collection_id,
            queries=[
                Query.order_desc('$createdAt'),
                Query.limit(per_page),
                Query.offset(offset)
            ]
        )
        predictions = documents['documents']
        # Return the predictions as JSON
        return jsonify({'status': 'success', 'predictions': predictions})
    except Exception as e:
        logging.error(f"Error fetching historical predictions: {e}")
        return jsonify({'status': 'error', 'message': 'Failed to fetch historical predictions'}), 500


# Example usage in the route to initiate premium purchase
@app.route('/initiate_premium_purchase', methods=['POST'])
def initiate_premium_purchase():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'status': 'error', 'message': 'User session not found'}), 401

    # Generate a unique merchant trade number
    merchant_trade_no_generator = get_next_merchant_trade_no('merchant_trade_no.csv')
    merchant_trade_no = next(merchant_trade_no_generator)

    # Define the payment parameters
    parameters = {
        "env": {"terminalType": "WEB"},
        "merchantTradeNo": merchant_trade_no,
        "orderAmount": 10.00,  # Example amount in USDT
        "currency": "USDT",
        "goods": {
            "goodsType": "01",
            "goodsCategory": "0000",
            "referenceGoodsId": "premium_version",
            "goodsName": "Premium Version",
            "goodsUnitAmount": {"currency": "USDT", "amount": 10.00},
        },
        "shipping": {
            "shippingName": {"firstName": "User", "lastName": user_id},
            "shippingAddress": {"region": "IN"},
        },
        "buyer": {"buyerName": {"firstName": "User", "lastName": user_id}},
    }

    while True:
        # Create the order
        response = binance_pay_client.new_order(parameters)
        print(response)
        if response['status'] == 'SUCCESS':
            prepay_id = response['data']['prepayId']
            checkout_url = response['data']['checkoutUrl']
            # Save the prepay ID and user ID in the database
            databases.create_document(
                database_id,
                'payment_logs',
                document_id=prepay_id,
                data={
                    'user_id': user_id,
                    'prepay_id': prepay_id,
                    'status': 'INITIAL',
                    'created_at': datetime.datetime.now().isoformat(),
                    'updated_at': datetime.datetime.now().isoformat()
                },
                permissions=[]
            )
            return jsonify({'status': 'success', 'checkoutUrl': checkout_url, 'prepayId': prepay_id})
        elif response['code'] == '400201':
            # If merchantTradeNo is invalid or duplicated, get the next one
            merchant_trade_no = next(merchant_trade_no_generator)
            parameters['merchantTradeNo'] = merchant_trade_no
        else:
            return jsonify({'status': 'error', 'message': 'Failed to initiate payment'}), 500

# Route to check payment status
@app.route('/check_payment_status', methods=['POST'])
def check_payment_status():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'status': 'error', 'message': 'User session not found'}), 401

    data = request.get_json()
    prepay_id = data.get('prepayId')

    if not prepay_id:
        return jsonify({'status': 'error', 'message': 'Missing prepayId'}), 400

    # Query the payment status
    response = binance_pay_client.get_order(prepayId=prepay_id)
    if response['status'] == 'SUCCESS':
        payment_status = response['data']['status']
        # Update the payment status in the database
        databases.update_document(
            database_id,
            'payment_logs',
            prepay_id,
            data={
                'status': payment_status,
                'updated_at': datetime.datetime.now().isoformat()
            }
        )
        if payment_status == 'PAID':
            # Update the user's premium status
            databases.update_document(
                database_id,
                users_collection_id,
                user_id,
                data={
                    'is_premium': True
                }
            )
        return jsonify({'status': payment_status})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to check payment status'}), 500

# Function to check and create the collections
def create_collections():
    collections = {
        'payment_logs': [
            ('user_id', 'string', 255, True),
            ('prepay_id', 'string', 255, True),
            ('status', 'string', 50, True),
            ('created_at', 'datetime', None, True),
            ('updated_at', 'datetime', None, True)
        ],
        'predictions': [
            ('stock_symbol', 'string', 50, True),
            ('holding_period', 'integer', None, True),
            ('prediction_time', 'datetime', None, True),
            ('ltp', 'float', None, False)
        ],
        'trades': [
            ('user_id', 'string', 255, True),
            ('stock_symbol', 'string', 50, True),
            ('quantity', 'integer', None, True),
            ('buy_price', 'float', None, True),
            ('sell_price', 'float', None, True),
            ('profit', 'float', None, True),
            ('entry_time', 'datetime', None, True),
            ('exit_time', 'datetime', None, True)
        ],
        'users': [
            ('user_id', 'string', 255, True),
            ('password', 'string', 255, True),
            ('totp_key', 'string', 255, True),
            ('quantity', 'integer', None, True),
            ('join_date', 'datetime', None, True),
            ('fund_mode', 'string', 50, True),
            ('is_premium', 'boolean', None, True)
        ],
        'extra_users': [
            ('user_id', 'string', 255, True),
            ('password', 'string', 255, True),
            ('totp_key', 'string', 255, True),
            ('is_logged_in', 'boolean', None, True)
        ]
    }

    for collection_id, attributes in collections.items():
        try:
            # Check if the collection exists
            databases.get_collection(database_id, collection_id)
            logging.info(f"{collection_id} collection already exists.")
        except AppwriteException as e:
            if e.code == 404:
                # Collection does not exist, create it
                logging.info(f"Creating {collection_id} collection.")
                databases.create_collection(
                    database_id,
                    collection_id,
                    name=collection_id.capitalize()
                )
                # Add attributes to the collection
                for attr_name, attr_type, attr_size, attr_required in attributes:
                    if attr_type == 'string':
                        databases.create_string_attribute(database_id, collection_id, attr_name, attr_size, required=attr_required)
                    elif attr_type == 'integer':
                        databases.create_integer_attribute(database_id, collection_id, attr_name, required=attr_required)
                    elif attr_type == 'float':
                        databases.create_float_attribute(database_id, collection_id, attr_name, required=attr_required)
                    elif attr_type == 'datetime':
                        databases.create_datetime_attribute(database_id, collection_id, attr_name, required=attr_required)
                    elif attr_type == 'boolean':
                        databases.create_boolean_attribute(database_id, collection_id, attr_name, required=attr_required)
                logging.info(f"{collection_id} collection created successfully.")
            else:
                logging.error(f"Error accessing database: {e}")
                raise e

def get_next_merchant_trade_no(csv_file_path):
    # Ensure the CSV file exists and has the correct format
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['merchantTradeNo'])
            writer.writerow(['1'])  # Initial value

    while True:
        # Read the last used merchantTradeNo from the CSV file
        with open(csv_file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            last_trade_no = next(reader)[0]

        # Increment the last used merchantTradeNo
        new_trade_no = str(int(last_trade_no) + 1)

        # Write the new merchantTradeNo back to the CSV file
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['merchantTradeNo'])
            writer.writerow([new_trade_no])

        # Return the new merchantTradeNo
        yield new_trade_no




# Route to add an extra user
@app.route('/add_extra_user', methods=['POST'])
def add_extra_user():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'status': 'error', 'message': 'User session not found'}), 401

    data = request.get_json()
    extra_user_id = data['extra_user_id']
    extra_password = data['extra_password']
    extra_totp_key = data['extra_totp_key']

    # Validate credentials with Zerodha
    totp = pyotp.TOTP(extra_totp_key)
    twofa = totp.now()

    kite = Zerodha(user_id=extra_user_id, password=extra_password, twofa=twofa)

    try:
        kite.login()
        logging.info(f"Extra user {extra_user_id} logged in successfully")

        # Save extra user info in the database
        databases.create_document(
            database_id,
            "extra_users",
            document_id='unique()',
            data={
                'user_id': extra_user_id,
                'password': extra_password,
                'totp_key': extra_totp_key,
                'is_logged_in': True
            },
            permissions=[]
        )

        return jsonify({'status': 'success', 'user_id': extra_user_id})
    except Exception as e:
        logging.error(f"Extra user login failed: {e}")
        return jsonify({'status': 'error', 'message': 'Login failed. Please check your credentials.'}), 401

# Route to fetch extra users for the current premium user
@app.route('/extra_users', methods=['GET'])
def extra_users():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'status': 'error', 'message': 'User session not found'}), 401

    try:
        documents = databases.list_documents(
            database_id=database_id,
            collection_id="extra_users",
            queries=[
                Query.equal('user_id', user_id)
            ]
        )
        extra_users = documents['documents']
        return jsonify({'status': 'success', 'extra_users': extra_users})
    except Exception as e:
        logging.error(f"Error fetching extra users: {e}")
        return jsonify({'status': 'error', 'message': 'Failed to fetch extra users'}), 500

# Route to start trading for an extra user
@app.route('/start_extra_user_trading', methods=['POST'])
def start_extra_user_trading():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'status': 'error', 'message': 'User session not found'}), 401

    data = request.get_json()
    extra_user_id = data['extra_user_id']

    # Fetch extra user details from the database
    try:
        extra_user_doc = databases.get_document(database_id, "extra_users", extra_user_id)
        extra_password = extra_user_doc['password']
        extra_totp_key = extra_user_doc['totp_key']

        # Validate credentials with Zerodha
        totp = pyotp.TOTP(extra_totp_key)
        twofa = totp.now()

        kite = Zerodha(user_id=extra_user_id, password=extra_password, twofa=twofa)
        kite.login()
        logging.info(f"Extra user {extra_user_id} logged in successfully")

        # Initialize user session
        user_session = UserSession(extra_user_id, 1, 'quantity', kite)
        user_sessions[extra_user_id] = user_session
        user_session.start_trading()

        return jsonify({'status': 'success'})
    except Exception as e:
        logging.error(f"Error starting trading for extra user {extra_user_id}: {e}")
        return jsonify({'status': 'error', 'message': 'Failed to start trading for extra user'}), 500

# Route to delete an extra user
@app.route('/delete_extra_user', methods=['POST'])
def delete_extra_user():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'status': 'error', 'message': 'User session not found'}), 401

    data = request.get_json()
    extra_user_id = data['extra_user_id']

    try:
        databases.delete_document(database_id, "extra_users", extra_user_id)
        if extra_user_id in user_sessions:
            user_sessions[extra_user_id].stop_trading()
            del user_sessions[extra_user_id]
        return jsonify({'status': 'success'})
    except Exception as e:
        logging.error(f"Error deleting extra user {extra_user_id}: {e}")
        return jsonify({'status': 'error', 'message': 'Failed to delete extra user'}), 500

if __name__ == '__main__':
    create_collections()
    app.run(debug=True,port=8731)

