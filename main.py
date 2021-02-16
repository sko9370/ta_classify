import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('seaborn')

def bollinger_bands(df):
    indicator_bb = BollingerBands(close=df["Close"])
    #df['bb_bbm'] = indicator_bb.bollinger_mavg()
    #df['bb_bbh'] = indicator_bb.bollinger_hband()
    #df['bb_bbl'] = indicator_bb.bollinger_lband()
    df['bb_bbh'] = indicator_bb.bollinger_hband_indicator()
    df['bb_bbl'] = indicator_bb.bollinger_lband_indicator()

def rsi(df):
    indicator_rsi = RSIIndicator(close=df["Close"])
    df['rsi'] = indicator_rsi.rsi()

def change(df):
    df['Change'] = df['Close'].diff()

def show_all():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

if __name__ == '__main__':
    msft = yf.Ticker("spy")
    df = msft.history(period="1y", interval="1d")

    change(df)
    bollinger_bands(df)
    rsi(df)

    df = df.dropna()

    show_all()
    #print(df[['Close', 'Change', 'bb_bbl', 'bb_bbm', 'bb_bbh', 'rsi']])

    #features = ['Close', 'bb_bbl', 'bb_bbm', 'bb_bbh', 'rsi']
    features = ['Close', 'bb_bbl', 'bb_bbh', 'rsi']
    X = df[features].copy()
    y = df['Change'].copy()

    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

    '''
    dtr_model = DecisionTreeRegressor()
    dtr_model.fit(train_X, train_y)
    dtr_predictions = dtr_model.predict(val_X)

    print('MAE: ', mean_absolute_error(val_y, dtr_predictions))
    print('Score: ', r2_score(val_y, dtr_predictions))
    '''

    rtr_model = RandomForestRegressor()
    rtr_model.fit(train_X, train_y)
    rtr_predictions = rtr_model.predict(val_X)

    print('MAE: ', mean_absolute_error(val_y, rtr_predictions))
    print('Score: ', r2_score(val_y, rtr_predictions))

    df['rtr'] = rtr_model.predict(df[features])

    xgb_model = XGBRegressor()
    xgb_model.fit(train_X, train_y)
    xgb_predictions = xgb_model.predict(val_X)

    print('MAE: ', mean_absolute_error(val_y, xgb_predictions))
    print('Score: ', r2_score(val_y, xgb_predictions))

    df['xgb'] = xgb_model.predict(df[features])

    plt.plot(df.Change, label='Change')
    #plt.plot(df.bb_bbl, label='bb_bbl')
    #plt.plot(df.bb_bbh, label='bb_bbh')
    plt.plot(df.rtr, label='rtr')
    plt.plot(df.xgb, label='xgb')
    plt.title('Results')
    plt.legend()
    plt.show()
