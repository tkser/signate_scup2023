import pandas as pd
import numpy as np

import optuna

from rgf.sklearn import RGFRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_percentage_error

import warnings
warnings.filterwarnings("ignore")


class RGFModel:

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
            "max_leaf": trial.suggest_int("max_leaf", 1000, 10000),
            "algorithm": trial.suggest_categorical("algorithm", ["RGF", "RGF_Opt", "RGF_Sib"]),
            "test_interval": trial.suggest_int("test_interval", 100, 1000),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 100),
            "reg_depth": trial.suggest_int("reg_depth", 1, 20),
            "l2": trial.suggest_loguniform("l2", 1e-8, 1.0),
            "sl2": trial.suggest_loguniform("sl2", 1e-8, 1.0),
            "normalize": trial.suggest_categorical("normalize", [True, False])
        }

        X, y = self.X_train.values, self.y_train.values
        kf = KFold(n_splits=5, shuffle=True, random_state=self.seed * 2)
        model = RGFRegressor(
            **params,
            learning_rate=0.05
        )
        scores = cross_val_score(
            model, # type: ignore
            X, y,
            scoring=make_scorer(mean_absolute_percentage_error, greater_is_better=False),
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
        predictions = pd.DataFrame(np.zeros((len(self.X_all), n_splits)), columns=[f"rgf_pred_{i}" for i in range(n_splits)])
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        for i, (train_index, valid_index) in enumerate(kf.split(self.X_train, self.y_train)):
            X_train, X_valid = self.X_train.iloc[train_index], self.X_train.iloc[valid_index]
            y_train, y_valid = self.y_train.iloc[train_index], self.y_train.iloc[valid_index]
            model = RGFRegressor(
                **self.best_params,
                learning_rate=0.01
            )
            model.fit(
                X_train, y_train
            )
            y_pred = model.predict(X_valid)
            score = mean_absolute_percentage_error(y_valid, y_pred) # type: ignore
            print(f"Fold_rgf {i}: {score}")
            self.models.append(model)
            predictions[f"rgf_pred_{i}"] = model.predict(self.X_all)
        return predictions
