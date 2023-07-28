import pandas as pd
import numpy as np

import optuna

from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import make_scorer, mean_absolute_percentage_error

import warnings
warnings.filterwarnings("ignore")


class CatBoostModel:

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
            "depth": trial.suggest_int("depth", 1, 16)
        }

        X, y = self.X_train.values, self.y_train.values
        X_cv, X_eval, y_cv, y_eval = train_test_split(X, y, test_size=0.25, random_state=self.seed)
        kf = KFold(n_splits=5, shuffle=True, random_state=self.seed * 2)
        model = CatBoostRegressor(
            **params,
            random_state=self.seed,
            learning_rate=0.05,
            eval_metric="MAPE",
            iterations=10000
        )
        scores = cross_val_score(
            model, # type: ignore
            X_cv, y_cv,
            scoring=make_scorer(mean_absolute_percentage_error, greater_is_better=False),
            cv=kf,
            n_jobs=-1,
            fit_params={"eval_set": [(X_eval, y_eval)], "early_stopping_rounds": 100, "use_best_model": True, "verbose": 0},
            verbose=0
        )
        return -scores.mean()
    
    def objective(self, n_trial = 100) -> dict:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=self.seed), direction="minimize")
        study.optimize(self._objective_trial, n_trials=n_trial, show_progress_bar=True, n_jobs=-1)
        self.best_params = study.best_params
        return study.best_params
    
    def predict(self, n_splits = 5) -> pd.DataFrame:
        self.models = []
        predictions = pd.DataFrame(np.zeros((len(self.X_all), n_splits)), columns=[f"cat_pred_{i}" for i in range(n_splits)])
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        for i, (train_index, valid_index) in enumerate(kf.split(self.X_train, self.y_train)):
            X_train, X_valid = self.X_train.iloc[train_index], self.X_train.iloc[valid_index]
            y_train, y_valid = self.y_train.iloc[train_index], self.y_train.iloc[valid_index]
            model = CatBoostRegressor(
                **self.best_params,
                random_state=self.seed,
                learning_rate=0.01,
                eval_metric="MAPE",
                iterations=10000
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                early_stopping_rounds=100,
                use_best_model=True,
                verbose=0
            )
            y_pred = model.predict(X_valid)
            score = mean_absolute_percentage_error(y_valid, y_pred) # type: ignore
            print(f"Fold_cat {i}: {score}")
            self.models.append(model)
            predictions[f"cat_pred_{i}"] = model.predict(self.X_all)
        return predictions
