import pandas as pd
import yfinance as yf

from ta.volatility import BollingerBands, DonchianChannel
from ta.momentum import RSIIndicator
from ta.trend import MACD

def get_data(ticker, period):
    if period == '1d':
        interval = '5m'
    else:
        interval = '1d'
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    return df

def add_change(df):
    df['% Change'] = df['Close'].diff()
    df['% Change'] = df['% Change']/df['Close']
    df['% Change'] = df['% Change'].shift(periods=-1)

# shows volume change from previous period to current period, lagging
def add_volume(df):
    df['% Volume'] = df['Volume'].diff()
    df['% Volume'] = df['% Volume']/df['Volume']

def bollinger_bands(df):
    indicator_bb = BollingerBands(close=df["Close"])
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()
    df['bb_avg'] = indicator_bb.bollinger_mavg()
    df['bb_bbh_ind'] = indicator_bb.bollinger_hband_indicator()
    df['bb_bbl_ind'] = indicator_bb.bollinger_lband_indicator()
    df['bb_pband'] = indicator_bb.bollinger_pband()
    df['bb_wband'] = indicator_bb.bollinger_wband()

def rsi(df):
    indicator_rsi = RSIIndicator(close=df["Close"])
    df['rsi'] = indicator_rsi.rsi()
    
def macd(df):
    indicator_macd = MACD(close=df["Close"])
    df['macd'] = indicator_macd.macd()
    df['macd_diff'] = indicator_macd.macd_diff()
    df['macd_signal'] = indicator_macd.macd_signal()
    
def donchian(df):
    indicator_don = DonchianChannel(high=df["High"], low=df["Low"], close=df["Close"])
    df['don_h'] = indicator_don.donchian_channel_hband()
    df['don_l'] = indicator_don.donchian_channel_lband()
    df['don_m'] = indicator_don.donchian_channel_mband()
    df['don_p'] = indicator_don.donchian_channel_pband()
    df['don_w'] = indicator_don.donchian_channel_wband()
    
def moving_avg(df):
    df['ema_9'] = df['Close'].ewm(9).mean().shift()
    df['sma_5'] = df['Close'].rolling(5).mean().shift()
    df['sma_10'] = df['Close'].rolling(10).mean().shift()
    df['sma_15'] = df['Close'].rolling(15).mean().shift()
    df['sma_30'] = df['Close'].rolling(30).mean().shift()
    df['sma_50'] = df['Close'].rolling(50).mean().shift()

def add_all_indicators(df):
    add_change(df)
    add_volume(df)
    bollinger_bands(df)
    rsi(df)
    macd(df)
    donchian(df)
    moving_avg(df)