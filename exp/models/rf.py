import pandas as pd
import numpy as np

import optuna

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_percentage_error


class RandomForestModel:

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
            "max_depth": trial.suggest_int("max_depth", 1, 32),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 32),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 32),
            "max_features": trial.suggest_float("max_features", 0.1, 1.0),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 2, 32),
            "n_estimators": trial.suggest_int("n_estimators", 100, 10000),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False])
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=self.seed * 2)
        model = RandomForestRegressor(**params, random_state=self.seed)
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
            model = RandomForestRegressor(**self.best_params, random_state=self.seed)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            score = mean_absolute_percentage_error(y_valid, y_pred) # type: ignore
            print(f"Fold_rf {i}: {score}")
            self.models.append(model)
            predictions[f"rf_pred_{i}"] = model.predict(self.X)
        return predictions
