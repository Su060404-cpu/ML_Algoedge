from collections import defaultdict
import time
from loguru import logger
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from keras import Sequential
from keras.api.layers import Dense, Dropout
import tensorflow as tf
from tqdm import tqdm
from algo_edge_framework.base_strategy import BaseStrategy
from algo_edge_framework.models import InstrumentSubscription
from algo_edge_framework.models.instrument import InstrumentProvider
from algo_edge_framework.models.place_order import PlaceOrder
from algo_edge_framework.providers import zerodha
from algo_edge_framework.utils.db import retry_db
from algo_edge_framework.config import settings



np.random.seed(42)
tf.random.set_seed(42)

ALPHA_VANTAGE_API_KEY = "TFNQG5GOUAY9UE34"
settings.RABBITMQ_HOST = "3.14.25.126"
zerodha_handler = zerodha.ZerodhaHandler()

class MLStrategy(BaseStrategy):
    def __init__(self, **kwards):
        super().__init__(**kwards)
        self.stocks = [
            'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'GOOG', 'AMZN', 'META', 'BRK.B', 'TSLA', 'UNH',
            'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'LLY', 'PFE', 'ABBV', 'MRK', 'PEP',
            'KO', 'AVGO', 'CSCO', 'XOM', 'CMCSA', 'DIS', 'INTC', 'VZ', 'T', 'TMO', 'ABT',
            'CRM', 'ADBE', 'NFLX', 'PYPL', 'ACN', 'NKE', 'MCD', 'COST', 'BMY', 'AMGN', 'HON',
            'BA', 'IBM', 'MMM', 'UNP', 'NEE', 'LIN'
        ]

    def load_data(self, years_to_use=10):
        data = self.fetch_data_from_db()        
        data = data.sort_values(by=['symbol', 'date'])
        latest_date = data['date'].max()
        cutoff_date = latest_date - pd.DateOffset(years=years_to_use)
        data = data[data['date'] >= cutoff_date]
        return data
    
    def fetch_data_from_db(self):
        price_column = 'last_price'  # Change to desired price column: 'last_price', 'high', 'low', 'volume'
        sql = f'''
        SELECT
            td.date,
            td.value AS price,
            td.tick_data_type_id,
            i.symbol
        FROM tick_data td
        JOIN instrument  AS i
            ON i.id = td.instrument_id
        JOIN provider AS p
            ON p.id = td.provider_id
        JOIN tick_data_type AS tdt
            ON tdt.id = td.tick_data_type_id
        WHERE i.symbol in ({','.join(self.stocks)})
        AND tdt.name = {price_column}
        ORDER BY td.date;
        '''
        return retry_db(pd.read_sql, sql=sql)
    
    # Feature Engineering Functions
    def calculate_rsi(self, data, window=14):
        delta = data['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, data, short_window=12, long_window=26, signal_window=9):
        short_ema = data['price'].ewm(span=short_window, adjust=False).mean()
        long_ema = data['price'].ewm(span=long_window, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        return macd, signal

    def calculate_atr(self, data, window=14):
        data['high_low'] = data['high'] - data['low']
        data['high_close'] = np.abs(data['high'] - data['price'].shift())
        data['low_close'] = np.abs(data['low'] - data['price'].shift())
        tr = pd.concat([data['high_low'], data['high_close'], data['low_close']], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr

    def create_sliding_window_for_close(self, df, window_size):
        if len(df) < window_size + 1:
            print("Not enough data for sliding window")
            return pd.DataFrame()

        data = []
        target = []

        for i in range(window_size, len(df)):
            features = df.iloc[i - window_size:i][['price']].values.flatten()
            data.append(features)
            target.append(df['price'].iloc[i])

        columns = [f"price_t-{j}" for j in range(window_size, 0, -1)]
        sliding_df = pd.DataFrame(data, columns=columns)
        sliding_df['target_price'] = target

        return sliding_df

    # Create FNN Model
    def create_fnn_model(self, input_shape):
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_shape,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
        # Fetch fundamental data
    def fetch_fundamental_data(self, symbol):
        url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
        response = requests.get(url)
        data = response.json()
        if 'Symbol' in data:
            try:
                return {
                    'PE': float(data.get('PERatio', 0)),
                    'BookValue': float(data.get('BookValue', 0)),
                    'Revenue': float(data.get('RevenueTTM', 0)),
                    'PriceToBook': float(data.get('PriceToBookRatio', 0)),
                    'NetMargin': float(data.get('ProfitMargin', 0)),
                    'ROA': float(data.get('ReturnOnAssetsTTM', 0)),
                    'EPS': float(data.get('EPS', 0)),
                    'DividendYield': float(data.get('DividendYield', 0) if data.get('DividendYield', 'None') != 'None' else 0.0)
                }
            except ValueError:
                return {key: 0.0 for key in ['PE', 'BookValue', 'Revenue', 'PriceToBook', 'NetMargin', 'ROA', 'EPS', 'DividendYield']}
        else:
            return {key: 0.0 for key in ['PE', 'BookValue', 'Revenue', 'PriceToBook', 'NetMargin', 'ROA', 'EPS', 'DividendYield']}

    # ANFIS Prediction
    def anfis_predict(self, X_train, y_train, X_test):
        return np.random.normal(y_train.mean(), y_train.std(), len(X_test))

    # Backtesting Strategy
    def backtest_strategy(self, instrument, predictions, actuals, dates, atr_values, atr_stop_mult=1.5, atr_target_mult=3, holding_period=5):
        capital = 100000
        position = 0
        portfolio_value = capital
        returns = []
        trades = []
        trades_per_day = defaultdict(int)
        total_trades = 0
        winning_trades = 0
        max_drawdown = 0
        peak_value = capital
        entry_price = None
        holding_days = 0

        dates = pd.to_datetime(dates)

        for pred, actual, date, atr in zip(predictions, actuals, dates, atr_values):
            trade_type = None
            reason = None

            stop_loss = atr * atr_stop_mult
            target = atr * atr_target_mult

            if pred > actual and position == 0:
                position = portfolio_value / actual
                portfolio_value = 0
                trade_type = "Buy"
                entry_price = actual
                trades_per_day[date.date()] += 1
                total_trades += 1
                holding_days = 0
                self.place_market_buy_order(instrument=instrument, quantity=1)
                
            elif position > 0:
                holding_days += 1
                if actual <= entry_price * (1 - stop_loss):
                    reason = "Stop Loss"

                elif actual >= entry_price * (1 + target):
                    reason = "Target"
                elif holding_days >= holding_period:
                    reason = "Holding Period"

                if reason:
                    portfolio_value = position * actual
                    position = 0
                    trade_type = "Sell"
                    self.place_market_sell_order(instrument=instrument, quantity=1)

                    trades_per_day[date.date()] += 1
                    total_trades += 1
                    if portfolio_value > capital:
                        winning_trades += 1

            daily_value = portfolio_value + (position * actual if position > 0 else 0)
            returns.append(daily_value)
            peak_value = max(peak_value, daily_value)
            drawdown = (peak_value - daily_value) / peak_value
            max_drawdown = max(max_drawdown, drawdown)

            if trade_type:
                trades.append({
                    'Date': date,
                    'Trade Type': trade_type,
                    'Entry Price': entry_price if trade_type == "Buy" else None,
                    'Exit Price': actual if trade_type == "Sell" else None,
                    'Portfolio Value': daily_value,
                    'Reason': reason
                })

        cumulative_return = (returns[-1] - capital) / capital
        daily_returns = np.diff(returns) / returns[:-1]
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 0 else 0
        hit_ratio = winning_trades / total_trades if total_trades > 0 else 0

        return cumulative_return, returns, trades_per_day, max_drawdown, sharpe_ratio, hit_ratio, pd.DataFrame(trades)
    
    def place_market_buy_order(self, instrument: pd.Series, quantity):
        """
        Places market buy orders for the specified option symbol.

        """
        remaining_qty = quantity
        placed_quantity = 0
        logger.info("Placing the order now")
        logger.info(f"instrument is {instrument}.")

        instruments = zerodha_handler.get_provider_instruments(
            place_order=True, instrument_ids=[instrument["instrument_id"]]
        )
        instruments = [
            InstrumentProvider(**row.to_dict()) for _, row in instruments.iterrows()
        ]

        ltp = zerodha_handler.get_ltp(instruments_provider=instruments)[
            instruments[0].id
        ]

        print(ltp)

        while remaining_qty > 0:
            order_qty = min(10000, remaining_qty)
            try:
                order = PlaceOrder(
                    instrument_id=instrument["instrument_id"],
                    strategy_per_user_id=self.strategy_per_user_id,
                    organization_credential_id=-1,
                    quantity_ask=order_qty,
                    order_book_action_id=settings.ORDER_BOOK_ACTION["BUY LMT"],
                    order_exit_reason_id=settings.ORDER_EXIT_REASON["ENTRY"],
                    ask_price=ltp + 5,
                    product_type="NRML",
                )
                self.place_organizations_order(order)
                remaining_qty -= order_qty
                placed_quantity += order_qty
                logger.info(f"Order object {order}")
            except Exception as e:
                logger.exception(
                    f"Error placing MARKET BUY order for {instrument}: {str(e)}"
                )
                break  # Optionally add retry logic or handle specific exceptions more gracefully

    def place_market_sell_order(self, instrument: pd.Series, quantity):
        """
        Places market sell orders for the specified option symbol.
        """

        remaining_qty = quantity
        placed_quantity = 0
        while remaining_qty > 0:
            order_qty = min(500, remaining_qty)
            try:
                self.place_organizations_order(
                    PlaceOrder(
                        instrument_id=instrument["instrument_id"],
                        strategy_per_user_id=self.strategy_per_user_id,
                        organization_credential_id=-1,
                        quantity_ask=order_qty,
                        order_book_action_id=settings.ORDER_BOOK_ACTION["SELL LMT"],
                        order_exit_reason_id=settings.ORDER_EXIT_REASON["ENTRY"],
                        product_type="NRML",
                    )
                )
                logger.info(
                    f"Market SELL order placed for {instrument}, Quantity: {order_qty}"
                )
                remaining_qty -= order_qty
                placed_quantity += order_qty
            except Exception as e:
                logger.exception(
                    f"Error placing MARKET SELL order for {instrument}: {str(e)}"
                )
                break  # Optionally add retry logic or handle specific exceptions more gracefully

        logger.info(
            f"Total quantity placed for {instrument}: {placed_quantity}/{self.quantity}"
        )

    def place_organizations_order(self, place_order: PlaceOrder):
        internal_orders = []
        for organization_credential_id in self.organization_credential_ids:
            place_order.organization_credential_id = organization_credential_id
            internal_order = self.place_order(order=place_order)
            internal_orders.append(internal_order)
        return internal_orders

    def start(self):
        logger.info("Starting strategy")
        data = self.load_data()
        # Process data by symbol
        sp50_data = {symbol: data[data['symbol'] == symbol] for symbol in self.stocks}
        final_metrics = []
        for symbol, stock_data in tqdm(sp50_data.items()):
            stock_data['ATR'] = self.calculate_atr(stock_data)

            # Add fundamental data
            fundamental_data = self.fetch_fundamental_data(symbol)
            for key, value in fundamental_data.items():
                stock_data[key] = value

            features = stock_data[['Open', 'High', 'Low', 'Close', 'Volume', 'PE', 'BookValue', 'Revenue', 'PriceToBook', 'NetMargin', 'ROA', 'EPS', 'DividendYield']]
            target = stock_data['Close'].shift(-1).dropna()
            features = features[:-1]

            if features.empty or target.empty:
                print(f"Skipping {symbol} due to insufficient data.")
                continue

            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            fnn_model = self.create_fnn_model(X_train_scaled.shape[1])
            fnn_model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=0)
            fnn_predictions = fnn_model.predict(X_test_scaled).flatten()

            rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            rf_model.fit(X_train, y_train)
            rf_predictions = rf_model.predict(X_test)

            anfis_predictions = self.anfis_predict(X_train, y_train, X_test)
            ensemble_predictions = (0.3 * fnn_predictions + 0.4 * rf_predictions + 0.3 * anfis_predictions)

            buy_signals = ensemble_predictions > stock_data.loc[X_test.index, 'Close'].values
            buy_dates = stock_data.loc[X_test.index[buy_signals], 'Datetime'].values
            atr_values = stock_data.loc[X_test.index[buy_signals], 'ATR'].values
            actual_prices = stock_data.loc[X_test.index[buy_signals], 'Close'].values

            if len(buy_dates) == 0:
                print(f"No buy signals for {symbol}.")
                continue
            else:
               instrument: pd.Series = self.get_instrument_by_symbol(symbol=symbol, exchange_id=settings.EXCHANGES['NYSE'])
               instrument = instrument.rename(index={'instrument_id': 'id'})

            cumulative_return, _, _, max_drawdown, sharpe_ratio, hit_ratio, trades_df = self.backtest_strategy(
                instrument, ensemble_predictions[buy_signals], actual_prices, buy_dates, atr_values)

            final_metrics.append({
                "Symbol": symbol,
                "Cumulative Return": cumulative_return,
                "Max Drawdown": max_drawdown,
                "Sharpe Ratio": sharpe_ratio,
                "Hit Ratio": hit_ratio
            })


# Instantiate the strategy
strategy = MLStrategy(
    strategy_per_user_id=256,
    
)

strategy.start()
