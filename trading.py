import warnings
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import os, json, io, csv, itertools, time, ccxt
from datetime import datetime, timedelta
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

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
    
    trade_history = []  # Class attribute to store trades
    
    def init(self):
        ImprovedTrading.trade_history = []  # Reset trade history for each backtest
        for param, value in self._params.items():
            setattr(self, param, value)
        
        self.hh = self.I(lambda: pd.Series(self.data.Close).rolling(int(self.hh_lookback)).max(), name='Highest High')
        self.fast = self.I(SMA, self.data.Close, int(self.fast_length), name='Fast SMA')
        self.slow = self.I(SMA, self.data.Close, int(self.slow_length), name='Slow SMA')
        self.filt = self.I(SMA, self.data.Close, int(self.filter_length), name='Filter SMA')
        self.atr = self.I(ATR, self.data.High, self.data.Low, self.data.Close, int(self.atr_length), name='ATR')
        self.hh_atr = self.I(lambda: pd.Series(self.data.Close).rolling(int(self.period_high_atr)).max(), name='HH ATR')
        self.atr_stop = self.I(lambda: self.hh_atr - self.atr_multiplier * self.atr, name='ATR Stop')


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

        if not self.position:
            if buy_condition:
                self.buy()
                ImprovedTrading.trade_history.append({
                    'type': 'buy',
                    'time': self.data.index[-1].isoformat(),
                    'price': self.data.Close[-1],
                    'amount': self.position.size
                })
        elif sell_condition:
            self.position.close()
            ImprovedTrading.trade_history.append({
                'type': 'sell',
                'time': self.data.index[-1].isoformat(),
                'price': self.data.Close[-1],
                'amount': self.position.size
            })

def generate_trade_details(stats):
    trades = stats['_trades']
    if trades.empty:
        return pd.DataFrame()
    
    trade_details = pd.DataFrame({
        'Entry Time': trades['EntryTime'].dt.strftime('%Y-%m-%d %H:%M'),
        'Exit Time': trades['ExitTime'].dt.strftime('%Y-%m-%d %H:%M'),
        'Entry Price': trades['EntryPrice'].round(2),
        'Exit Price': trades['ExitPrice'].round(2),
        'Size': trades['Size'].round(4),
        'PnL': trades['PnL'].round(2),
        'Return': (trades['ReturnPct'] * 100).round(2).astype(str) + '%',
        'Duration': trades['Duration'].astype(str)
    })
    
    return trade_details.sort_values('Entry Time')

def run_backtest(symbol, timeframe, start_date, end_date, parameters):
    csv_filename = f"{symbol.replace('/', '_')}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    
    if not os.path.exists(csv_filename):
        data = fetch_binance_data(symbol, timeframe, start_date, end_date)
        data.to_csv(csv_filename)
    else:
        data = pd.read_csv(csv_filename, index_col='timestamp', parse_dates=True)
    
    data = data.loc[start_date:end_date]
    
    bt = Backtest(data, ImprovedTrading, cash=100000, commission=.002)
    stats = bt.run(**parameters)
    
    summary = {
        "Symbol": symbol,
        "Timeframe": timeframe,
        "Period": f"{start_date} to {end_date}",
        "Initial Portfolio Value": f"${100000:.2f}",
        "Final Portfolio Value": f"${stats['Equity Final [$]']:.2f}",
        "Total Return": f"{stats['Return [%]']:.2f}%",
        "Sharpe Ratio": f"{stats['Sharpe Ratio']:.2f}",
        "Max Drawdown": f"{stats['Max. Drawdown [%]']:.2f}%",
        "Total Trades": stats['# Trades'],
        "Win Rate": f"{stats['Win Rate [%]']:.2f}%"
    }
    
    chart_html = bt.plot(filename=None, open_browser=False)
    
    return summary, stats, ImprovedTrading.trade_history, chart_html

def get_parameter_ranges(range_size):
    if range_size == 'small':
        return {
            'hh_lookback': range(30, 41, 5),
            'fast_length': range(1, 4, 1),
            'slow_length': range(5, 11, 5),
            'filter_length': range(20, 31, 10),
            'atr_length': range(20, 31, 10),
            'atr_multiplier': [1.0, 1.3, 1.7],
            'period_high_atr': range(10, 16, 5)
        }
    elif range_size == 'medium':
        return {
            'hh_lookback': range(20, 51, 10),
            'fast_length': range(1, 6, 1),
            'slow_length': range(5, 16, 5),
            'filter_length': range(20, 41, 10),
            'atr_length': range(20, 41, 10),
            'atr_multiplier': [1.0, 1.3, 1.7],
            'period_high_atr': range(10, 21, 5)
        }
    elif range_size == 'large':
        return {
            'hh_lookback': range(20, 61, 5),
            'fast_length': range(1, 8, 1),
            'slow_length': range(5, 21, 2),
            'filter_length': range(20, 51, 5),
            'atr_length': range(20, 51, 5),
            'atr_multiplier': [1.0, 1.2, 1.3, 1.5, 1.7],
            'period_high_atr': range(10, 26, 3)
        }
    else:
        raise ValueError("Invalid range_size. Choose 'small', 'medium', or 'large'.")

def optimize_parameters(symbol, timeframe, start_date, end_date, range_size):
    parameter_ranges = get_parameter_ranges(range_size)
    
    combinations = list(itertools.product(*parameter_ranges.values()))
    total_combinations = len(combinations)
    results = []
    
    print(f"Total combinations to test: {total_combinations}")
    
    for i, combo in enumerate(combinations, 1):
        parameters = dict(zip(parameter_ranges.keys(), combo))
        summary, stats, trades, _ = run_backtest(symbol, timeframe, start_date, end_date, parameters)
        results.append({
            **parameters, 
            'Total Return': float(stats['Return [%]']),
            'Sharpe Ratio': float(stats['Sharpe Ratio']),
            'Max Drawdown': float(stats['Max. Drawdown [%]']),
            'Total Trades': int(stats['# Trades']),
            'Win Rate': float(stats['Win Rate [%]']),
        })
        
        if i % 10 == 0 or i == total_combinations:
            print(f"Progress: {i}/{total_combinations} combinations tested ({i/total_combinations*100:.2f}%)")
    
    results_df = pd.DataFrame(results)
    best_result = results_df.loc[results_df['Total Return'].idxmax()]
    
    # Only return the parameters that are part of the strategy
    best_parameters = {k: v for k, v in best_result.items() if k in parameter_ranges}
    
    return best_parameters, results_df

@app.route('/api/trading/optimize', methods=['POST'])
def process_optimization_request():
    data = request.json
    symbol = data['symbol']
    timeframe = data['timeframe']
    start_date = datetime.strptime(data['startDate'], '%Y-%m-%d')
    end_date = datetime.strptime(data['endDate'], '%Y-%m-%d')
    range_size = data['rangeSize']
    
    best_parameters, all_results = optimize_parameters(symbol, timeframe, start_date, end_date, range_size)
    
    # Run backtest with best parameters
    summary, stats, trade_history, chart_html = run_backtest(symbol, timeframe, start_date, end_date, best_parameters)
    
    trade_details = generate_trade_details(stats)
    
    response = {
        'bestParameters': best_parameters,
        'summary': summary,
        'tradeDetails': trade_details.to_dict(orient='records'),
        'chartHtml': chart_html,
        'allResults': all_results.to_dict(orient='records')
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
