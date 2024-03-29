import polars as pl
import numpy as np

from typing import List

import optuna

from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import make_scorer, mean_absolute_percentage_error

import warnings
warnings.filterwarnings("ignore")


class LGBMModel:

    models: List[LGBMRegressor]

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
    
    def _objective_trial(self, trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 2, 40),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 2, 64),
            "subsample": trial.suggest_float("subsample", 0.1, 0.5),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "subsample_freq": trial.suggest_categorical("subsample_freq", [0, 1, 2, 5, 10]),
        }
        kf = KFold(n_splits=5, shuffle=True, random_state=self.seed * 2)
        X, y = self.X_train.clone().to_numpy(), self.y_train.clone().to_numpy()
        X_cv, X_eval, y_cv, y_eval = train_test_split(X, y, test_size=0.25, random_state=self.seed, shuffle=True)
        model = LGBMRegressor(
            **params,
            random_state=self.seed*2,
            n_estimators=10000,
            metric="mape",
            objective=self.objective_param,
            importance_type="gain",
            learning_rate=0.05,
            verbose=-1,
            n_jobs=-1
        )
        scores = cross_val_score(
            model, # type: ignore
            X_cv, y_cv,
            scoring=make_scorer(mean_absolute_percentage_error, greater_is_better=False),
            cv=kf,
            n_jobs=-1,
            verbose=0,
            fit_params={
                "eval_set": [(X_eval, y_eval)],
                "callbacks": [early_stopping(200, verbose=False), log_evaluation(0)],
                "eval_metric": "mape"
            }
        )
        return -scores.mean()
    
    def objective(self, n_trial = 100) -> dict:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=self.seed), direction="minimize")
        study.optimize(self._objective_trial, n_trials=n_trial, show_progress_bar=True, n_jobs=1)
        self.best_params = study.best_params
        return study.best_params
    
    def predict(self, n_splits = 5, col_name = "lgbm") -> pl.DataFrame:
        self.models = []
        predictions = pl.DataFrame(np.zeros((self.X_all.shape[0], n_splits)), schema=[f"{col_name}_pred_{i}" for i in range(n_splits)])
        X, y = self.X_train.to_numpy(), self.y_train.to_numpy()
        if n_splits == 1:
            X_cv, X_eval, y_cv, y_eval = train_test_split(X, y, test_size=0.25, random_state=self.seed, shuffle=True)
            model = LGBMRegressor(
                **self.best_params,
                random_state=self.seed,
                n_estimators=10000,
                metric="mape",
                objective=self.objective_param,
                importance_type="gain",
                learning_rate=0.01,
                verbose=-1
            )
            model.fit(
                X_cv, y_cv,
                eval_set=[(X_eval, y_eval)],
                callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
                eval_metric="mape"
            )
            self.models.append(model)
            y_pred_all = model.predict(self.X_all)
            predictions = predictions.with_columns(
                pl.Series(y_pred_all).alias(f"{col_name}_pred_0"),
            )
            print(f"Fold_{col_name} 0: {mean_absolute_percentage_error(y_eval, model.predict(X_eval))}") # type: ignore
            return predictions
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        scores = []
        for i, (train_index, valid_index) in enumerate(kf.split(X, y)):
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
            model = LGBMRegressor(
                **self.best_params,
                random_state=self.seed,
                n_estimators=10000,
                metric="mape",
                objective=self.objective_param,
                importance_type="gain",
                learning_rate=0.01,
                verbose=-1
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
                eval_metric="mape"
            )
            y_pred = model.predict(X_valid)
            score = mean_absolute_percentage_error(y_valid, y_pred) # type: ignore
            scores.append(score)
            print(f"Fold_{col_name} {i}: {score}")
            self.models.append(model)
            y_pred_all = model.predict(self.X_all)
            predictions = predictions.with_columns(
                pl.Series(y_pred_all).alias(f"{col_name}_pred_{i}"),
            )
        print(f"CV_{col_name}: {np.mean(scores)}")
        return predictions
    
    def feature_importance(self) -> pl.DataFrame:
        importance = pl.DataFrame(self.X_train.columns, schema=["feature"])
        for i, model in enumerate(self.models):
            importance = importance.with_columns(
                pl.Series(model.feature_importances_).alias(f"lgbm_importance_{i}"),
            )
        importance = importance.with_columns(
            pl.Series(importance[:, 1:].mean(axis=1)).alias("lgbm_importance_mean"),
        )
        return importance[["feature", "lgbm_importance_mean"]]
