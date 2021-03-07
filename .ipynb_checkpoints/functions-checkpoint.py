from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def xgb_tuning(X_train, X_val, y_train, y_val, gridsearch):
    eval_set = [(X_train, y_train), (X_val, y_val)]

    if gridsearch:
        parameters = {
            'n_estimators': [100, 200, 300, 400],
            'learning_rate': [0.001, 0.005, 0.01, 0.05],
            'max_depth': [8, 10, 12, 15],
            'gamma': [0.001, 0.005, 0.01, 0.02],
        }

        model = XGBRegressor(objective='reg:squarederror')
        clf = GridSearchCV(model, parameters)
        clf.fit(X_train, y_train)

        #print(f'Best params: {clf.best_params_}')
        #print(f'Best validation score = {clf.best_score_}')

        model = XGBRegressor(**clf.best_params_, objective='reg:squarederror')
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    else:
        model = XGBRegressor(objective='reg:squarederror')
        model.fit(X_train, y_train)
    
    return model

def final_model(X, y, gridsearch):
    eval_set = [(X, y)]
    
    if gridsearch:
        parameters = {
            'n_estimators': [150, 300, 450, 600],
            'learning_rate': [0.001, 0.005, 0.01, 0.05],
            'max_depth': [8, 10, 12, 15],
            'gamma': [0.001, 0.005, 0.01, 0.02],
        }

        model = XGBRegressor(objective='reg:squarederror')
        clf = GridSearchCV(model, parameters)
        clf.fit(X, y)

        #print(f'Best params: {clf.best_params_}')
        #print(f'Best validation score = {clf.best_score_}')

        model = XGBRegressor(**clf.best_params_, objective='reg:squarederror')
        model.fit(X, y, eval_set=eval_set, verbose=False)

    else:
        model = XGBRegressor(objective='reg:squarederror')
        model.fit(X, y)
    
    return model