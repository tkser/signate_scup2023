import pandas as pd
import numpy as np

import optuna

from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_percentage_error


class LGBMModel:

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
        self.X = pd.concat([self.X_train, self.X_test], axis=0)
    
    def _objective_trial(self, trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "max_depth": trial.suggest_int("max_depth", 1, 32),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 128),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True)
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=self.seed * 2)
        model = LGBMRegressor(**params, random_state=self.seed, n_estimators=10000)
        scores = cross_val_score(model, self.X_train, self.y_train, scoring=make_scorer(mean_absolute_percentage_error), cv=kf) # type: ignore
        return scores.mean()
    
    def objective(self, n_trial = 100) -> dict:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(self._objective_trial, n_trials=n_trial, show_progress_bar=True, n_jobs=-1)
        self.best_params = study.best_params
        return study.best_params
    
    def predict(self, n_splits = 5) -> pd.DataFrame:
        self.models = []
        predictions = pd.DataFrame(np.zeros((len(self.X_test), n_splits)), columns=[f"pred_{i}" for i in range(n_splits)])
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        for i, (train_index, valid_index) in enumerate(kf.split(self.X_train, self.y_train)):
            X_train, X_valid = self.X_train.iloc[train_index], self.X_train.iloc[valid_index]
            y_train, y_valid = self.y_train.iloc[train_index], self.y_train.iloc[valid_index]
            model = LGBMRegressor(**self.best_params, random_state=self.seed, n_estimators=10000)
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
            y_pred = model.predict(X_valid)
            score = mean_absolute_percentage_error(y_valid, y_pred) # type: ignore
            print(f"Fold_lgbm {i}: {score}")
            self.models.append(model)
            predictions[f"lgbm_pred_{i}"] = model.predict(self.X)
        return predictions
