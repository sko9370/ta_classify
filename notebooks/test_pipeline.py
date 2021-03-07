import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -0.0001426016678061029
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=RidgeCV()),
    GradientBoostingRegressor(alpha=0.95, learning_rate=0.5, loss="lad", max_depth=9, max_features=0.2, min_samples_leaf=4, min_samples_split=9, n_estimators=100, subsample=0.05)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
