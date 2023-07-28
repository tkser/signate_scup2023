import pandas as pd
import numpy as np

import optuna

from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import make_scorer, mean_absolute_percentage_error

import warnings
warnings.filterwarnings("ignore")


class XGBModel:

    def __init__(self, df: pd.DataFrame, seed = 0) -> None:
        self.df = df
        self.seed = seed
        self._pre_processing()
    
    def _pre_processing(self) -> None:
        self.train = self.df[self.df["price"].notnull()]
        self.test = self.df[self.df["price"].isnull()]
        self.X_train = self.train.drop(["price"], axis=1)
        self.y_train = self.train["price"]
        self.X_test = self.test.drop(["price"], axis=1)
        self.X_all = pd.concat([self.X_train, self.X_test], axis=0)
    
    def _objective_trial(self, trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 1, 15),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 300),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "eta": trial.suggest_float("eta", 1e-8, 1.0, log=True)
        }

        X, y = self.X_train.values, self.y_train.values
        X_cv, X_eval, y_cv, y_eval = train_test_split(X, y, test_size=0.25, random_state=self.seed)
        kf = KFold(n_splits=5, shuffle=True, random_state=self.seed * 2)
        model = XGBRegressor(
            **params,
            random_state=self.seed,
            tree_method="gpu_hist"
        )
        scores = cross_val_score(
            model, # type: ignore
            X_cv, y_cv,
            scoring=make_scorer(mean_absolute_percentage_error, greater_is_better=False),
            fit_params={"eval_set": [(X_eval, y_eval)], "early_stopping_rounds": 100, "eval_metric": "mape", "verbose": 0},
            cv=kf
        )
        return -scores.mean()
    
    def objective(self, n_trial = 100) -> dict:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(self._objective_trial, n_trials=n_trial, show_progress_bar=True, n_jobs=-1)
        self.best_params = study.best_params
        return study.best_params
    
    def predict(self, n_splits = 5) -> pd.DataFrame:
        self.models = []
        predictions = pd.DataFrame(np.zeros((len(self.X_all), n_splits)), columns=[f"pred_{i}" for i in range(n_splits)])
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        for i, (train_index, valid_index) in enumerate(kf.split(self.X_train, self.y_train)):
            X_train, X_valid = self.X_train.iloc[train_index], self.X_train.iloc[valid_index]
            y_train, y_valid = self.y_train.iloc[train_index], self.y_train.iloc[valid_index]
            model = XGBRegressor(
                **self.best_params,
                random_state=self.seed,
                tree_method="gpu_hist"
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                early_stopping_rounds=100,
                eval_metric="mape",
                verbose=0
            )
            y_pred = model.predict(X_valid)
            score = mean_absolute_percentage_error(y_valid, y_pred) # type: ignore
            print(f"Fold_xgb {i}: {score}")
            self.models.append(model)
            predictions[f"xgb_pred_{i}"] = model.predict(self.X_all)
        return predictions
