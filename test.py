import yfinance as yf
#import pandas as pd
import numpy as np
#import enum
#import matplotlib.pyplot as plt

msft = yf.Ticker("msft")
hist = msft.history(period="1d", interval="5m")

# adds new column "vwap"
hist['vwap'] = (
    np.cumsum(hist.Volume * (hist.Close + hist.High + hist.Low) / 3) /
    np.cumsum(hist.Volume))

# RSI (need around 14 data points for RSI to be viable)
delta = hist.Close.diff()
up = delta.clip(lower=0)
down = -1 * delta.clip(upper=0)
ema_up = up.ewm(com=13, adjust=False).mean()
ema_down = down.ewm(com=13, adjust=False).mean()
rs = ema_up / ema_down
hist['RSI'] = 100 - (100 / (1 + rs))

#hist = hist.iloc[14:]

print(hist)