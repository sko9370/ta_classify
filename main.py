import pandas as pd
import yfinance as yf

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('seaborn')

from add_indicators import bollinger_bands, rsi, add_all_indicators

def show_all():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

def change(df):
    df['Change'] = df['Close'].diff()
    # theory that percentage change vs absolute change will work better with different stocks
    # perchange change is better
    df['Change'] = df['Change']/df['Close']
    df['Change'] = df['Change'].shift(periods=1)

def preprocess(df):
    change(df)
    add_all_indicators(df)
    return df.dropna()

def create_model(model, train_X, train_y):
    model.fit(train_X, train_y)
    return model

def run_predictions(model, val_X, val_y):
    model_predictions = model.predict(val_X)
    mae = mean_absolute_error(val_y, model_predictions)
    score = r2_score(val_y, model_predictions)
    return model_predictions, mae, score

if __name__ == '__main__':
    msft = yf.Ticker("spy")
    df = msft.history(period="1y", interval="1d")

    df = preprocess(df)

    # Close is not important
    features = ['bb_bbl', 'bb_bbh', 'rsi']
    X = df[features].copy()
    y = df['Change'].copy()

    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
    '''
    results = []
    for m in [RandomForestRegressor(), XGBRegressor()]:
        model = create_model(m, train_X, train_y)
        model_predictions, mae, score = run_predictions(model, val_X, val_y)
        results.append([model_predictions, mae, score])
    '''
    rfr_model = RandomForestRegressor()
    rfr_model.fit(train_X, train_y)
    rfr_predictions = rfr_model.predict(val_X)
    
    print('MAE: ', mean_absolute_error(val_y, rfr_predictions))
    print('Score: ', r2_score(val_y, rfr_predictions))

    df['rfr'] = rfr_model.predict(df[features])

    xgb_model = XGBRegressor()
    xgb_model.fit(train_X, train_y)
    xgb_predictions = xgb_model.predict(val_X)

    print('MAE: ', mean_absolute_error(val_y, xgb_predictions))
    print('Score: ', r2_score(val_y, xgb_predictions))

    df['xgb'] = xgb_model.predict(df[features])

    plt.plot(df.Change.iloc[30:60], label='Change')
    plt.plot(df.rfr.iloc[30:60], label='rfr')
    plt.plot(df.xgb.iloc[30:60], label='xgb')
    plt.title('Results')
    plt.legend()
    plt.show()
