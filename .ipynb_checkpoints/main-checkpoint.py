#!/usr/bin/env python3

import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

import data_processing
import functions

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', type = str,
        help = 'Ticker symbol in lowercase')
    parser.add_argument('period', type = str,
        help = 'Period to train model (1y, 1mo, 1w, etc)')
    parser.add_argument('-g', '--gridsearch',
        help = 'Turn on gridsearch, increases performance but decreases speed',
        action = 'store_true')
    parser.add_argument('-s', '--showgraph',
        help = 'Show graph of model performance',
        action = 'store_true')
    parser.add_argument('-v', '--verbose',
        help = 'Verbose output',
        action = 'store_true')
    args = parser.parse_args()
    return args.ticker, args.period, args.gridsearch, args.showgraph, args.verbose
    
def predict(ticker, period, gridsearch, verbosity):
    df = data_processing.get_data(ticker, period)
    data_processing.add_all_indicators(df)
    dfOriginal = df.copy()
    df = df.dropna()

    '''
    features = ['Close', '% Volume',
            'bb_bbh', 'bb_bbl', 'bb_avg', 'bb_bbh_ind', 'bb_bbl_ind',
            'bb_pband', 'bb_wband', 'rsi', 'macd', 'macd_diff', 'macd_signal',
            'don_h', 'don_l', 'don_m', 'don_p', 'don_w',
            'ema_9', 'sma_5', 'sma_10', 'sma_15', 'sma_30', 'sma_50']
    features = ['Close', '% Volume', 'macd_diff',
            'rsi', 'ema_9', 'sma_5', 'sma_50']
    '''
    features = ['Close', '% Volume',
                'bb_bbh', 'bb_bbl', 'bb_avg',
                'bb_pband', 'bb_wband', 'rsi', 'macd', 'macd_diff', 'macd_signal',
                'ema_9', 'sma_5', 'sma_10', 'sma_15', 'sma_30', 'sma_50']
    X = df[features].copy()
    y = df['% Change'].copy()
    X_train, X_val, y_train, y_val = train_test_split(X, y)
    
    model = functions.xgb_tuning(X_train, X_val, y_train, y_val, gridsearch)
    predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)
    score = r2_score(y_val, predictions)
    print('% MAE: {}'.format(mae))
    print('R2 Score: {}'.format(score))
    next = model.predict(dfOriginal.tail(1)[features])
    print('Next: {}'.format(next))
    
    model = functions.final_model(X, y, gridsearch)
    next = model.predict(dfOriginal.tail(1)[features])
    print('Next Final: {}'.format(next))
    
    return predictions

def present(predictions, graph=False):
    # optional show graph
    # optional show MAE and R2 score
    # show prediction
    pass

def main():
    ticker, period, gridsearch, graph, verbosity = parser()
    predictions = predict(ticker, period, gridsearch, verbosity)
    present(predictions, graph)
    
if __name__ == '__main__':
    main()