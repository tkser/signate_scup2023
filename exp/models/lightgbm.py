import pandas as pd
import numpy as np

from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error


class Model:

    def __init__(self, df: pd.DataFrame, seed = 0) -> None:
        self.df = df
        self.train = df[df["price"].notnull()]
        self.test = df[df["price"].isnull()]
        self.seed = seed
    
    def _pre_processing(self) -> None:
        self.X_train = self.train.drop(["price"], axis=1)
        self.y_train = self.train["price"]
        self.X_test = self.test.drop(["price"], axis=1)
    
    def predict(self) -> pd.Series:
        self._pre_processing()

        prediction = pd.DataFrame(np.zeros((len(self.X_test), 5)), columns=["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"])
        kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)

        for i, (tr_idx, va_idx) in enumerate(kf.split(self.X_train)):
            tr_x, va_x = self.X_train.iloc[tr_idx], self.X_train.iloc[va_idx]
            tr_y, va_y = self.y_train.iloc[tr_idx], self.y_train.iloc[va_idx]

            model = LGBMRegressor(
                n_estimators=10000,
                random_state=self.seed,
                learning_rate=0.04,
                num_leaves=70,
                max_depth=5,
                subsample=0.708,
                colsample_bytree=0.645,
                reg_alpha=0.006,
                reg_lambda=0.346,
                min_split_gain=0.0222415,
                min_child_weight=39.3259775,
                silent=-1,
                verbose=-1
            )
            model.fit(tr_x, tr_y)

            va_pred = model.predict(va_x)
            score = mean_absolute_percentage_error(va_y, va_pred) # type: ignore

            print(f'fold {i} score: {score}')

            prediction["fold_{}".format(i+1)] = model.predict(self.X_test)

        self.prediction = prediction.mean(axis=1)
        return self.prediction
