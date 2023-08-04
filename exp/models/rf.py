import polars as pl
import numpy as np

import optuna

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_percentage_error

import warnings
warnings.filterwarnings("ignore")


class RandomForestModel:

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
            "max_depth": trial.suggest_int("max_depth", 1, 32),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 32),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 32),
            "max_features": trial.suggest_float("max_features", 0.1, 1.0),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 2, 32),
            "n_estimators": trial.suggest_int("n_estimators", 100, 10000),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False])
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=self.seed * 2)
        model = RandomForestRegressor(
            **params,
            random_state=self.seed,
            n_jobs=-1
        )
        scores = cross_val_score(
            model, # type: ignore
            self.X_train, self.y_train,
            scoring=make_scorer(mean_absolute_percentage_error, greater_is_better=False),
            cv=kf,
            n_jobs=-1,
            verbose=0
        )
        return -scores.mean()
    
    def objective(self, n_trial = 100) -> dict:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=self.seed), direction="minimize")
        study.optimize(self._objective_trial, n_trials=n_trial, show_progress_bar=True, n_jobs=-1)
        self.best_params = study.best_params
        return study.best_params
    
    def predict(self, n_splits = 5) -> pl.DataFrame:
        self.models = []
        predictions = pl.DataFrame(np.zeros((self.X_all.shape[0], n_splits)), schema=[f"rf_pred_{i}" for i in range(n_splits)])
        X, y = self.X_train.to_numpy(), self.y_train.to_numpy()
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        for i, (train_index, valid_index) in enumerate(kf.split(X, y)):
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
            model = RandomForestRegressor(
                **self.best_params,
                random_state=self.seed,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            score = mean_absolute_percentage_error(y_valid, y_pred) # type: ignore
            print(f"Fold_rf {i}: {score}")
            self.models.append(model)
            y_pred_all = model.predict(self.X_all)
            predictions = predictions.with_columns(
                pl.Series(y_pred_all).alias(f"rf_pred_{i}"),
            )
        return predictions
