import warnings
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import ccxt, io
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore", category=FutureWarning)

def fetch_binance_data(symbol, timeframe, since):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=int(since.timestamp() * 1000))
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def ATR(high, low, close, length):
    tr = np.maximum(high - low, 
                    np.abs(high - np.roll(close, 1)),
                    np.abs(low - np.roll(close, 1)))
    return pd.Series(tr).rolling(length).mean().values

class Trading(Strategy):
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
        self.buy_signals = self.I(lambda: np.zeros(len(self.data)), name='Buy Signals')
        self.sell_signals = self.I(lambda: np.zeros(len(self.data)), name='Sell Signals')

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

        if buy_condition:
            self.buy_signals[-1] = 1
        if sell_condition:
            self.sell_signals[-1] = 1

        if not self.position:
            if buy_condition:
                self.buy()
        elif sell_condition:
            self.position.close()

    def calculate_signal(self):
        buy_condition = (
            (self.fast > self.slow) &
            (self.fast > self.filt) &
            (self.slow > self.filt) &
            (self.data.Close > self.hh[-1])
        )
        sell_condition = (
            (self.fast < self.slow) |
            (self.fast < self.filt) |
            (self.slow < self.filt) |
            (self.data.Close < self.atr_stop)
        )
        
        signal = pd.Series(np.nan, index=self.data.index)
        signal[buy_condition] = 1   # Buy signal
        signal[sell_condition] = -1 # Sell signal
        return signal.fillna(method='ffill')

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

def calculate_sharpe_ratio(returns, risk_free_rate=0):
    excess_returns = returns - risk_free_rate
    if len(excess_returns) < 2:
        return 0
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def run_backtest(symbol, start_date, end_date, timeframe='4h'):
    data = fetch_binance_data(symbol, timeframe, start_date)
    
    bt = Backtest(data, Trading, cash=100000, commission=.002)
    stats = bt.run()
    
    # Calculate correct Win Rate
    trades = stats['_trades']
    win_rate = (trades['PnL'] > 0).mean() * 100 if len(trades) > 0 else 0

    # Calculate correct Sharpe Ratio
    returns = pd.Series(stats['_equity_curve']['Equity']).pct_change().dropna()
    sharpe_ratio = calculate_sharpe_ratio(returns)
    
    # Update the stats dictionary with the correct metrics
    stats['Win Rate [%]'] = win_rate
    stats['Sharpe Ratio'] = sharpe_ratio

    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Portfolio Value: $100,000")
    print(f"Final Portfolio Value: ${stats['Equity Final [$]']:.2f}")
    print(f"Total Return: {stats['Return [%]']:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
    print(f"Total Trades: {stats['# Trades']}")
    print(f"Win Rate: {win_rate:.2f}%")
    
    # Update the Backtest object with the correct stats
    bt._results = stats
    
    # Plot with updated metrics
    bt.plot(filename='backtest_results.html', open_browser=False)
    
    # Generate trade details
    trade_details = generate_trade_details(stats)
    
    # Add results table and trade details to the HTML
    add_results_table_to_html('backtest_results.html', stats, symbol, timeframe, start_date, end_date, trade_details)
    
    print("Interactive chart with results table and trade details saved as 'backtest_results.html'")
    
    if not trade_details.empty:
        trade_details.to_csv('trade_details.csv', index=False)
        print("Trade details saved as 'trade_details.csv'")
    else:
        print("No trades were executed in this backtest.")

    return bt, stats

def add_results_table_to_html(filename, stats, symbol, timeframe, start_date, end_date, trade_details):
    try:
        with io.open(filename, 'r', encoding='utf-8') as file:
            html_content = file.read()
    except UnicodeDecodeError:
        with io.open(filename, 'r', encoding='utf-8-sig') as file:
            html_content = file.read()
    
    # Create a table with the results summary
    results_table = go.Figure(data=[go.Table(
        header=dict(values=['Metric', 'Value'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[
            ['Symbol', 'Timeframe', 'Period', 'Initial Portfolio Value', 'Final Portfolio Value', 'Total Return', 
             'Sharpe Ratio', 'Max Drawdown', 'Total Trades', 'Win Rate'],
            [
                symbol,
                timeframe,
                f"{start_date} to {end_date}",
                f"$100,000",
                f"${stats['Equity Final [$]']:.2f}",
                f"{stats['Return [%]']:.2f}%",
                f"{stats['Sharpe Ratio']:.2f}",
                f"{stats['Max. Drawdown [%]']:.2f}%",
                f"{stats['# Trades']}",
                f"{stats['Win Rate [%]']:.2f}%"
            ]
        ],
        fill_color='lavender',
        align='left')
    )])
    
    results_table.update_layout(title_text="Backtest Results Summary")
    
    # Create trade details table
    trade_table = go.Figure(data=[go.Table(
        header=dict(values=list(trade_details.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[trade_details[col] for col in trade_details.columns],
                   fill_color='lavender',
                   align='left')
    )])
    
    trade_table.update_layout(
        title_text="Trade Details",
        updatemenus=[
            dict(
                buttons=[
                    dict(args=[{'cells.values': [trade_details[col].tolist() for col in trade_details.columns]}],
                         label="Reset",
                         method="restyle"),
                    dict(args=[{'cells.values': [trade_details.sort_values('Size', ascending=True)[col].tolist() for col in trade_details.columns]}],
                         label="Sort Size Ascending",
                         method="restyle"),
                    dict(args=[{'cells.values': [trade_details.sort_values('Size', ascending=False)[col].tolist() for col in trade_details.columns]}],
                         label="Sort Size Descending",
                         method="restyle"),
                    dict(args=[{'cells.values': [trade_details.sort_values('PnL', ascending=True)[col].tolist() for col in trade_details.columns]}],
                         label="Sort PnL Ascending",
                         method="restyle"),
                    dict(args=[{'cells.values': [trade_details.sort_values('PnL', ascending=False)[col].tolist() for col in trade_details.columns]}],
                         label="Sort PnL Descending",
                         method="restyle"),
                    dict(args=[{'cells.values': [trade_details[trade_details['PnL'] > 0][col].tolist() for col in trade_details.columns]}],
                         label="Show Profitable Trades",
                         method="restyle"),
                    dict(args=[{'cells.values': [trade_details[trade_details['PnL'] < 0][col].tolist() for col in trade_details.columns]}],
                         label="Show Loss-making Trades",
                         method="restyle"),
                ],
                direction="down",
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            )
        ]
    )
    
    # Convert the tables to HTML
    results_html = results_table.to_html(full_html=False, include_plotlyjs='cdn')
    trade_html = trade_table.to_html(full_html=False, include_plotlyjs='cdn')
    
    # Inject the table HTML into the existing file
    modified_html = html_content.replace('</body>', f'{results_html}{trade_html}</body>')
    
    with io.open(filename, 'w', encoding='utf-8') as file:
        file.write(modified_html)

def create_custom_chart(bt, stats, trade_details, symbol, timeframe, win_rate, sharpe_ratio):
    df = bt._data
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=(f'{symbol} Price', 'Volume', 'Trade Details'),
                        row_heights=[0.6, 0.2, 0.2])

    # Main price chart
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name='Price'),
                  row=1, col=1)

    # Add Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=bt.strategy.fast, name='Fast MA', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bt.strategy.slow, name='Slow MA', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bt.strategy.filt, name='Filter MA', line=dict(color='green')), row=1, col=1)

    # Add buy/sell signals and trade durations
    for i, trade in trade_details.iterrows():
        entry_time = pd.to_datetime(trade['Entry Time'])
        exit_time = pd.to_datetime(trade['Exit Time'])
        entry_price = float(trade['Entry Price'])
        exit_price = float(trade['Exit Price'])
        
        # Add buy signal
        fig.add_trace(go.Scatter(
            x=[entry_time], 
            y=[entry_price],
            mode='markers',
            marker=dict(symbol='triangle-up', size=15, color='green'),
            name=f'Buy {i}',
            showlegend=False
        ), row=1, col=1)
        
        # Add sell signal
        fig.add_trace(go.Scatter(
            x=[exit_time], 
            y=[exit_price],
            mode='markers',
            marker=dict(symbol='triangle-down', size=15, color='red'),
            name=f'Sell {i}',
            showlegend=False
        ), row=1, col=1)
        
        # Add trade duration
        fig.add_trace(go.Scatter(
            x=[entry_time, exit_time],
            y=[entry_price, exit_price],
            mode='lines',
            line=dict(color='rgba(0,0,255,0.5)', width=2),
            name=f'Trade {i}',
            showlegend=False
        ), row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'), row=2, col=1)

    # Add trade details table
    fig.add_trace(go.Table(
        header=dict(values=list(trade_details.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[trade_details[col] for col in trade_details.columns],
                   fill_color='lavender',
                   align='left')
    ), row=3, col=1)

    fig.update_layout(
        title_text=f"Backtest Results for {symbol} ({timeframe})",
        height=1200,  # Increased height to accommodate the table
        xaxis_rangeslider_visible=False,
        dragmode='zoom',
        yaxis=dict(fixedrange=False),
        yaxis2=dict(fixedrange=False)
    )

    stats_text = (
        f"Initial Portfolio Value: $100,000<br>"
        f"Final Portfolio Value: ${stats['Equity Final [$]']:.2f}<br>"
        f"Total Return: {stats['Return [%]']:.2f}%<br>"
        f"Sharpe Ratio: {sharpe_ratio:.2f}<br>"
        f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%<br>"
        f"Total Trades: {stats['# Trades']}<br>"
        f"Win Rate: {win_rate:.2f}%"
    )
    
    fig.add_annotation(text=stats_text, align='left', showarrow=False, xref='paper', yref='paper', x=1.05, y=0.8)

    # Add buttons for sorting and filtering
    updatemenus = [
        dict(
            buttons=[
                dict(args=[{'cells.values[4]': [trade_details['Size'].sort_values(ascending=True)]}],
                     label='Sort Size Ascending',
                     method='restyle'),
                dict(args=[{'cells.values[4]': [trade_details['Size'].sort_values(ascending=False)]}],
                     label='Sort Size Descending',
                     method='restyle'),
                dict(args=[{'cells.values[5]': [trade_details['PnL'].sort_values(ascending=True)]}],
                     label='Sort PnL Ascending',
                     method='restyle'),
                dict(args=[{'cells.values[5]': [trade_details['PnL'].sort_values(ascending=False)]}],
                     label='Sort PnL Descending',
                     method='restyle')
            ],
            direction='down',
            pad={'r': 10, 't': 10},
            showactive=True,
            x=0.1,
            xanchor='left',
            y=1.1,
            yanchor='top'
        ),
        dict(
            buttons=[
                dict(args=[{'cells.values': [[col for col in trade_details[trade_details['PnL'] > 0][c]] for c in trade_details.columns]}],
                     label='Show Profitable Trades',
                     method='restyle'),
                dict(args=[{'cells.values': [[col for col in trade_details[trade_details['PnL'] < 0][c]] for c in trade_details.columns]}],
                     label='Show Loss-making Trades',
                     method='restyle'),
                dict(args=[{'cells.values': [[col for col in trade_details[c]] for c in trade_details.columns]}],
                     label='Show All Trades',
                     method='restyle')
            ],
            direction='down',
            pad={'r': 10, 't': 10},
            showactive=True,
            x=0.3,
            xanchor='left',
            y=1.1,
            yanchor='top'
        )
    ]

    fig.update_layout(updatemenus=updatemenus)

    return fig

if __name__ == '__main__':
    symbol = 'BTC/USDT'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1356)
    timeframe = '4h'
    
    run_backtest(symbol, start_date, end_date, timeframe)