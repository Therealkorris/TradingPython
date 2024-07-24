import warnings
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import ccxt
import io
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

warnings.filterwarnings("ignore", category=FutureWarning)

def fetch_binance_data(symbol, timeframe, start_date, end_date):
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    all_data = []
    current_date = start_date
    
    while current_date < end_date:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=int(current_date.timestamp() * 1000), limit=1000)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            current_date = datetime.fromtimestamp(ohlcv[-1][0] / 1000) + timedelta(minutes=1)
            print(f"Fetched data up to {current_date}")
            time.sleep(exchange.rateLimit / 1000)  # Respect rate limits
        except ccxt.NetworkError as e:
            print(f"Network error: {e}. Retrying in 10 seconds...")
            time.sleep(10)
        except ccxt.ExchangeError as e:
            print(f"Exchange error: {e}. Retrying in 10 seconds...")
            time.sleep(10)
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
    
    df = pd.DataFrame(all_data, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]  # Remove duplicate indices
    return df

def save_data_to_csv(df, filename):
    df.to_csv(filename)
    print(f"Data saved to {filename}")

def load_data_from_csv(filename):
    df = pd.read_csv(filename, index_col='timestamp', parse_dates=True)
    return df

def ATR(high, low, close, length):
    tr = np.maximum(high - low, 
                    np.abs(high - np.roll(close, 1)),
                    np.abs(low - np.roll(close, 1)))
    return pd.Series(tr).rolling(length).mean().values

class ImprovedTrading(Strategy):
    hh_lookback = 35
    fast_length = 1
    slow_length = 9
    filter_length = 30
    atr_length = 30
    atr_multiplier = 1.3
    period_high_atr = 15

    def init(self):
        self.hh = self.I(lambda: pd.Series(self.data.Close).rolling(self.hh_lookback).max(), name='Highest High')
        self.fast = self.I(SMA, self.data.Close, self.fast_length, name='Fast SMA')
        self.slow = self.I(SMA, self.data.Close, self.slow_length, name='Slow SMA')
        self.filt = self.I(SMA, self.data.Close, self.filter_length, name='Filter SMA')
        self.atr = self.I(ATR, self.data.High, self.data.Low, self.data.Close, self.atr_length, name='ATR')
        self.hh_atr = self.I(lambda: pd.Series(self.data.Close).rolling(self.period_high_atr).max(), name='HH ATR')
        self.atr_stop = self.I(lambda: self.hh_atr - self.atr_multiplier * self.atr, name='ATR Stop')
        
        # Debug: Print column names
        print("Data columns:", self.data.df.columns)

    def next(self):
        buy_condition = (
            (self.fast[-1] > self.slow[-1]) and
            (self.fast[-1] > self.filt[-1]) and
            (self.slow[-1] > self.filt[-1]) and
            (self.data.Close[-1] > self.hh[-2])  # Using previous bar's highest high
        )
        
        sell_condition = (
            (self.fast[-1] < self.slow[-1]) or
            (self.fast[-1] < self.filt[-1]) or
            (self.slow[-1] < self.filt[-1]) or
            (self.data.Close[-1] < self.atr_stop[-1])
        )

        # Debug: Print conditions and values
        print(f"Time: {self.data.index[-1]}, Close: {self.data.Close[-1]}, Fast: {self.fast[-1]}, Slow: {self.slow[-1]}, Filter: {self.filt[-1]}")
        print(f"Buy condition: {buy_condition}, Sell condition: {sell_condition}")

        if not self.position:
            if buy_condition:
                self.buy()
                print(f"BUY signal at {self.data.index[-1]}, price: {self.data.Close[-1]}")
        elif sell_condition:
            self.position.close()
            print(f"SELL signal at {self.data.index[-1]}, price: {self.data.Close[-1]}")

def generate_trade_details(stats):
    trades = stats['_trades']
    if trades.empty:
        return pd.DataFrame()
    
    trade_details = pd.DataFrame({
        'Entry Time': trades['EntryTime'].dt.strftime('%Y-%m-%d %H:%M:%S'),
        'Exit Time': trades['ExitTime'].dt.strftime('%Y-%m-%d %H:%M:%S'),
        'Entry Price': trades['EntryPrice'].round(2),
        'Exit Price': trades['ExitPrice'].round(2),
        'Size': trades['Size'].round(4),
        'PnL': trades['PnL'].round(2),
        'Return': (trades['ReturnPct'] * 100).round(2).astype(str) + '%',
        'Duration': trades['Duration'].astype(str)
    })
    
    return trade_details.sort_values('Entry Time')

def add_trade_details_to_html(filename, trade_details):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    
    trade_table = trade_details.to_html(classes='display', table_id='trade-details', index=False)
    
    script = """
    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.25/css/jquery.dataTables.css">
    <script type="text/javascript" charset="utf8" src="https://code.jquery.com/jquery-3.5.1.js"></script>
    <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.10.25/js/jquery.dataTables.js"></script>
    <script>
    $(document).ready(function() {
        $('#trade-details').DataTable({
            "pageLength": 25,
            "order": [[ 0, "desc" ]],
            "dom": 'Bfrtip',
            "buttons": [
                'copy', 'csv', 'excel', 'pdf', 'print'
            ]
        });
    });
    </script>
    <style>
    #trade-details {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 20px;
    }
    #trade-details th, #trade-details td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    #trade-details th {
        background-color: #f2f2f2;
        color: black;
    }
    #trade-details tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    #trade-details tr:hover {
        background-color: #f5f5f5;
    }
    </style>
    """
    
    new_content = content.replace('</head>', f'{script}</head>')
    new_content = new_content.replace('</body>', f'<h2>Trade Details</h2>{trade_table}</body>')
    
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(new_content)

def run_backtest(symbol, timeframe, start_date, end_date):
    csv_filename = f"{symbol.replace('/', '_')}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    
    if not os.path.exists(csv_filename):
        data = fetch_binance_data(symbol, timeframe, start_date, end_date)
        save_data_to_csv(data, csv_filename)
    else:
        print(f"Loading data from existing file: {csv_filename}")
        data = load_data_from_csv(csv_filename)
    
    # Ensure data is within the specified date range
    data = data.loc[start_date:end_date]
    
    bt = Backtest(data, ImprovedTrading, cash=100000, commission=.002)
    stats = bt.run()
    
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Portfolio Value: $100,000")
    print(f"Final Portfolio Value: ${stats['Equity Final [$]']:.2f}")
    print(f"Total Return: {stats['Return [%]']:.2f}%")
    print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
    print(f"Total Trades: {stats['# Trades']}")
    print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
    
    bt.plot(filename='backtest_results.html', open_browser=False)
    print("Interactive chart saved as 'backtest_results.html'")
    
    trade_details = generate_trade_details(stats)
    add_trade_details_to_html('backtest_results.html', trade_details)
    print("Trade details added to 'backtest_results.html'")
    
    return bt, stats

if __name__ == '__main__':
    symbol = 'BTC/USDT'
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    timeframe = '4h'
    
    bt, stats = run_backtest(symbol, timeframe, start_date, end_date)
