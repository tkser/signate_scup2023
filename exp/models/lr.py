import polars as pl
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_percentage_error

import warnings
warnings.filterwarnings('ignore')


class LinerRegressionModel:

    def __init__(self, train: pl.DataFrame, test: pl.DataFrame, seed: int = 0):
        self.train = train
        self.test = test
        self.seed = seed
        self.__pre_processing()
    
    def __pre_processing(self) -> None:
        self.X_train = self.train.drop("price")
        self.y_train = self.train["price"]
        self.X_test = self.test.drop("price")
        self.X_all = pl.concat([self.X_train, self.X_test]).to_numpy()

    def predict(self, n_splits = 5, label="lr"):
        self.models = []
        self.predictions = pl.DataFrame(np.zeros((self.X_all.shape[0], n_splits)), schema=[f"{label}_pred_{i}" for i in range(n_splits)])
        X, y = self.X_train.to_numpy(), self.y_train.to_numpy()
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        for i, (train_index, test_index) in enumerate(kf.split(X, y)):
            X_train, X_valid = X[train_index], X[test_index]
            y_train, y_valid = y[train_index], y[test_index]
            model = LinearRegression(n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            score = mean_absolute_percentage_error(y_valid, y_pred) # type: ignore
            print(f"Fold_{label} {i}: {score}")
            self.models.append(model)
            y_pred_all = model.predict(self.X_all)
            self.predictions = self.predictions.with_columns(
                pl.Series(y_pred_all).alias(f"{label}_pred_{i}"),
            )
        return self.predictions
