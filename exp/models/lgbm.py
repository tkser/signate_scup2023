import polars as pl
import numpy as np

from typing import List

import optuna

import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_percentage_error

import warnings
warnings.filterwarnings("ignore")


class LGBMModel:

    models: List[lgb.Booster]

    def __init__(self, train: pl.DataFrame, test: pl.DataFrame, seed = 0, objective_param = "mape") -> None:
        self.train = train
        self.test = test
        self.seed = seed
        self.objective_param = objective_param
        self._pre_processing()
    
    def _pre_processing(self) -> None:
        self.X_train = self.train.drop("price")
        self.y_train = self.train["price"]
        self.X_test = self.test
        self.X_all = pl.concat([self.X_train, self.X_test]).to_numpy()
    
    def _objective(self, trial: optuna.Trial) -> float:
        params = {
            "objective": self.objective_param,
            "boosting": "gbdt",
            "metric": "mape",
            "learning_rate": 0.05,
            "verbosity": -1,
            "n_estimators": 10000,
            "random_state": self.seed,
            "n_jobs": -1,
            "importance_type": "gain",
            "num_leaves": trial.suggest_int("num_leaves", 2, 64),
            "max_depth": trial.suggest_int("max_depth", 2, 24),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 128),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_categorical("bagging_freq", [1, 5]),
        }

        X, y = self.X_train.to_numpy(), self.y_train.to_numpy()
        X_cv, X_eval, y_cv, y_eval = train_test_split(X, y, test_size=0.25, random_state=self.seed)
        
        train_data = lgb.Dataset(X_cv, label=y_cv)
        eval_data = lgb.Dataset(X_eval, label=y_eval, reference=train_data)

        model = lgb.train(
            params,
            train_set=train_data,
            valid_sets=[train_data, eval_data],
            num_boost_round=10000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
            feval=lambda y_pred, data: ("mape", mean_absolute_percentage_error(y_pred, data.get_label()), False),
        )
        scores = model.best_score["valid_1"]["mape"]
        return scores
    
    def objective(self, n_trial = 100) -> dict:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=self.seed), direction="minimize")
        study.optimize(self._objective, n_trials=n_trial, show_progress_bar=True, n_jobs=-1)
        self.best_params = study.best_params
        return study.best_params
    
    def predict(self, n_splits = 4) -> pl.DataFrame:
        self.models = []
        predictions = pl.DataFrame(np.zeros((self.X_all.shape[0], n_splits)), schema=[f"lgbm_pred_{i}" for i in range(n_splits)])
        X, y = self.X_train.to_numpy(), self.y_train.to_numpy()
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        for i, (train_index, valid_index) in enumerate(kf.split(X, y)):
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_valid, label=y_valid)

            params = {
                **self.best_params,
                "objective": self.objective_param,
                "boosting": "gbdt",
                "metric": "mape",
                "learning_rate": 0.01,
                "verbosity": -1,
                "n_estimators": 10000,
                "random_state": self.seed,
                "n_jobs": -1,
                "importance_type": "gain",
            }

            model = lgb.train(
                params,
                train_set=train_data,
                valid_sets=[train_data, valid_data],
                num_boost_round=200000,
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
            )

            y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
            score = mean_absolute_percentage_error(y_valid, y_pred)
            print(f"Fold_lgbm {i}: {score}")
            self.models.append(model)
            y_pred_all = model.predict(self.X_all, num_iteration=model.best_iteration)
            predictions = predictions.with_columns(
                pl.Series(y_pred_all).alias(f"lgbm_pred_{i}"),
            )
        return predictions

    def feature_importance(self) -> pl.DataFrame:
        importance = pl.DataFrame(self.X_train.columns, schema=["feature"])
        for i, model in enumerate(self.models):
            importance = importance.with_columns(
                pl.Series(model.feature_importance(importance_type='gain')).alias(f"lgbm_importance_{i}"),
            )
        importance = importance.with_columns(
            pl.Series(importance.mean(axis=1)).alias("lgbm_importance_mean"),
        )
        return importance