import polars as pl
import numpy as np

import optuna

from rgf.sklearn import RGFRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_percentage_error

import warnings
warnings.filterwarnings("ignore")


class RGFModel:

    def __init__(self, train: pl.DataFrame, test: pl.DataFrame, seed = 0) -> None:
        self.train = train
        self.test = test
        self.seed = seed
        self._pre_processing()
    
    def _pre_processing(self) -> None:
        self.X_train = self.train.drop("price")
        self.y_train = self.train["price"]
        self.X_test = self.test
        self.X_all = pl.concat([self.X_train, self.X_test]).to_numpy()
    
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

        X, y = self.X_train.to_numpy(), self.y_train.to_numpy()
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
    
    def predict(self, n_splits = 5) -> pl.DataFrame:
        self.models = []
        predictions = pl.DataFrame(np.zeros((self.X_all.shape[0], n_splits)), schema=[f"rgf_pred_{i}" for i in range(n_splits)])
        X, y = self.X_train.to_numpy(), self.y_train.to_numpy()
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        for i, (train_index, valid_index) in enumerate(kf.split(self.X_train, self.y_train)):
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
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
            y_pred_all = model.predict(self.X_all)
            predictions = predictions.with_columns(
                pl.Series(y_pred_all).alias(f"rgf_pred_{i}"),
            )
        return predictions
