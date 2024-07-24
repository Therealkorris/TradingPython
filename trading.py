import warnings
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import ccxt
import io
import os, json
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import itertools
import csv

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

def add_summary_and_parameters_to_html(filename, summary, parameters, trade_details, all_results):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    
    summary_table = pd.DataFrame([summary]).transpose().to_html(classes='display', table_id='summary-table', header=False)
    parameters_table = pd.DataFrame([parameters]).transpose().to_html(classes='display', table_id='parameters-table', header=False)
    trade_details_table = trade_details.to_html(classes='display', table_id='trade-details-table', index=False)
    all_results_table = all_results.to_html(classes='display', table_id='all-results-table', index=False, escape=False)
    
    tables_html = f"""
    <h2>Trade Details (Best Parameters)</h2>
    {trade_details_table}
    <h2>All Optimization Results</h2>
    {all_results_table}
    <h2>Summary</h2>
    {summary_table}
    <h2>Best Parameters</h2>
    {parameters_table}
    """
    
    style = """
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1, h2 {
            color: #333;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .dataTables_wrapper .dataTables_filter {
            float: right;
            text-align: right;
        }
        .trade-details {
            display: none;
            background-color: #f9f9f9;
            padding: 10px;
            border: 1px solid #ddd;
            margin-top: 5px;
            max-height: 300px;
            overflow-y: auto;
        }
        .trade-details table {
            width: 100%;
            border-collapse: collapse;
        }
        .trade-details th, .trade-details td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .trade-details th {
            background-color: #f2f2f2;
        }
    </style>
    """
    
    script = """
    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.25/css/jquery.dataTables.css">
    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/buttons/1.7.1/css/buttons.dataTables.min.css">
    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/colreorder/1.5.4/css/colReorder.dataTables.min.css">
    <script type="text/javascript" charset="utf8" src="https://code.jquery.com/jquery-3.5.1.js"></script>
    <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.10.25/js/jquery.dataTables.js"></script>
    <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/buttons/1.7.1/js/dataTables.buttons.min.js"></script>
    <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/buttons/1.7.1/js/buttons.colVis.min.js"></script>
    <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/colreorder/1.5.4/js/dataTables.colReorder.min.js"></script>
    <script type="text/javascript" charset="utf8" src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.1.3/jszip.min.js"></script>
    <script type="text/javascript" charset="utf8" src="https://cdnjs.cloudflare.com/ajax/libs/pdfmake/0.1.53/pdfmake.min.js"></script>
    <script type="text/javascript" charset="utf8" src="https://cdnjs.cloudflare.com/ajax/libs/pdfmake/0.1.53/vfs_fonts.js"></script>
    <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/buttons/1.7.1/js/buttons.html5.min.js"></script>
    <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/buttons/1.7.1/js/buttons.print.min.js"></script>
    <script>
    $(document).ready(function() {
        var table = $('.display').DataTable({
            dom: 'Bfrtip',
            buttons: [
                'copy', 'csv', 'excel', 'pdf', 'print',
                {
                    extend: 'colvis',
                    text: 'Show/Hide Columns'
                }
            ],
            pageLength: 25,
            colReorder: true
        });

        $('#all-results-table').on('click', '.show-trades', function(e) {
            e.preventDefault(); // Prevent scrolling to top
            var $this = $(this);
            var tradesData = $this.data('trades');
            var detailsDiv = $this.next('.trade-details');

            if (detailsDiv.is(':empty')) {
                var tableHtml = '<table><thead><tr><th>Type</th><th>Time</th><th>Price</th><th>Amount</th></tr></thead><tbody>';
                tradesData.forEach(function(trade) {
                    tableHtml += '<tr>' +
                                '<td>' + trade.type + '</td>' +
                                '<td>' + trade.time + '</td>' +
                                '<td>' + trade.price + '</td>' +
                                '<td>' + trade.amount + '</td>' +
                                '</tr>';
                });
                tableHtml += '</tbody></table>';
                detailsDiv.html(tableHtml);
            }

            detailsDiv.slideToggle(200); // Animate the show/hide
            $this.text(function(i, text) {
                return text === "Show Trades (" + tradesData.length + ")" ? "Hide Trades (" + tradesData.length + ")" : "Show Trades (" + tradesData.length + ")";
            });

            // Adjust the DataTable's row height
            table.row($this.closest('tr')).invalidate().draw();
        });
    });
    </script>
    """
    
    new_content = content.replace('</head>', f'{style}{script}</head>')
    new_content = new_content.replace('<body>', '<body><div class="container">')
    new_content = new_content.replace('</body>', f'{tables_html}</div></body>')
    
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(new_content)


def run_backtest(symbol, timeframe, start_date, end_date, parameters):
    csv_filename = f"{symbol.replace('/', '_')}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    
    if not os.path.exists(csv_filename):
        data = fetch_binance_data(symbol, timeframe, start_date, end_date)
        save_data_to_csv(data, csv_filename)
    else:
        data = load_data_from_csv(csv_filename)
    
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
    
    return summary, stats, ImprovedTrading.trade_history

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
        summary, stats, trades = run_backtest(symbol, timeframe, start_date, end_date, parameters)
        trades_json = json.dumps(trades)
        results.append({
            **parameters, 
            'Total Return': float(stats['Return [%]']),
            'Sharpe Ratio': float(stats['Sharpe Ratio']),
            'Max Drawdown': float(stats['Max. Drawdown [%]']),
            'Total Trades': int(stats['# Trades']),
            'Win Rate': float(stats['Win Rate [%]']),
            'Trades': f'<a href="#" class="show-trades" data-trades=\'{trades_json}\'>Show Trades ({len(trades)})</a><div class="trade-details"></div>'
        })
        
        if i % 10 == 0 or i == total_combinations:
            print(f"Progress: {i}/{total_combinations} combinations tested ({i/total_combinations*100:.2f}%)")
    
    results_df = pd.DataFrame(results)
    best_result = results_df.loc[results_df['Total Return'].idxmax()]
    
    return best_result.to_dict(), results_df

def save_results_to_csv(results_df, filename):
    results_df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

if __name__ == '__main__':
    symbol = 'BTC/USDT'
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    timeframe = '4h'
    
    # Choose 'small', 'medium', or 'large' for different range sizes
    range_size = 'small'
    
    best_parameters, all_results = optimize_parameters(symbol, timeframe, start_date, end_date, range_size)
    
    print("Best parameters found:")
    print(best_parameters)
    
    # Remove 'Total Return' and 'Trades' from best_parameters
    best_parameters_for_backtest = {k: v for k, v in best_parameters.items() if k not in ['Total Return', 'Trades', 'Sharpe Ratio', 'Max Drawdown', 'Total Trades', 'Win Rate']}
    
    # Convert relevant parameters to integers
    for param in ['hh_lookback', 'fast_length', 'slow_length', 'filter_length', 'atr_length', 'period_high_atr']:
        best_parameters_for_backtest[param] = int(best_parameters_for_backtest[param])
    
    summary, stats, trades = run_backtest(symbol, timeframe, start_date, end_date, best_parameters_for_backtest)
    
    print("\nBacktest results with best parameters:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    data = load_data_from_csv(f"{symbol.replace('/', '_')}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv")
    bt = Backtest(data, ImprovedTrading, cash=100000, commission=.002)
    stats = bt.run(**best_parameters_for_backtest)
    bt.plot(filename='backtest_results.html', open_browser=False)
    
    trade_details = generate_trade_details(stats)
    add_summary_and_parameters_to_html('backtest_results.html', summary, best_parameters_for_backtest, trade_details, all_results)
    
    print("Interactive chart and detailed results saved as 'backtest_results.html'")
    
    save_results_to_csv(all_results, 'optimization_results.csv')