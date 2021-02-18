from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator

def bollinger_bands(df):
    indicator_bb = BollingerBands(close=df["Close"])
    df['bb_bbh'] = indicator_bb.bollinger_hband_indicator()
    df['bb_bbl'] = indicator_bb.bollinger_lband_indicator()

def rsi(df):
    indicator_rsi = RSIIndicator(close=df["Close"])
    df['rsi'] = indicator_rsi.rsi()

def add_all_indicators(df):
    bollinger_bands(df)
    rsi(df)



    '''
    results = []
    for m in [RandomForestRegressor(), XGBRegressor()]:
        model = create_model(m, train_X, train_y)
        model_predictions, mae, score = run_predictions(model, val_X, val_y)
        results.append([model_predictions, mae, score])
    '''