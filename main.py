import yfinance as yf
#import pandas as pd
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator

def bollinger_bands(df):
    indicator_bb = BollingerBands(close=df["Close"])
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()
    return df

def rsi(df):
    indicator_rsi = RSIIndicator(close=df["Close"])
    df['rsi'] = indicator_rsi.rsi()
    return df

if __name__ == '__main__':
    msft = yf.Ticker("msft")
    df = msft.history(period="5d", interval="5m")

    bollinger_bands(df)
    rsi(df)
    print(df.columns)
    print(df)