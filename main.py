import os
import time
import polars as pl
import numpy as np

from sklearn.linear_model import Ridge

from exp.features import Features, FeatureSelecter
from exp.models.cat import CatBoostModel
from exp.models.lgbm import LGBMModel
from exp.models.rf import RandomForestModel
from exp.models.rgf import RGFModel
from exp.models.xgb import XGBModel

import gc
gc.enable()

import warnings
warnings.filterwarnings("ignore")


def main():
    train = pl.read_csv(os.path.join(os.path.dirname(__file__), "input/train.csv"))
    test = pl.read_csv(os.path.join(os.path.dirname(__file__), "input/test.csv"))

    features = Features(train, test)
    train, test = features.create_features()

    selecter = FeatureSelecter(train, test)

    predictions = pl.concat([train["price"].to_frame(), pl.DataFrame([None] * test.height, schema={"price": pl.Int64})])

    time_sta = time.perf_counter()
    lgbm = LGBMModel(*selecter.get_dataframe("lgbm"))
    #lgbm.objective(20)
    lgbm.best_params = {'num_leaves': 48, 'max_depth': 6, 'min_child_samples': 91, 'subsample': 0.5578230915019112, 'colsample_bytree': 0.5933052522026404, 'reg_alpha': 2.4725566626090776e-05, 'reg_lambda': 1.0114136512530978e-08, 'feature_fraction': 0.7523757350552451, 'bagging_fraction': 0.9199865329355417, 'bagging_freq': 5}
    lgbm_predictions = lgbm.predict()
    print(f"lgbm: {time.perf_counter() - time_sta}")

    time_sta = time.perf_counter()
    xgb = XGBModel(*selecter.get_dataframe("xgb"))
    #xgb.objective(20)
    xgb.best_params = {'n_estimators': 767, 'max_depth': 8, 'lambda': 1.2306916748991704e-06, 'alpha': 0.018078104089246788, 'colsample_bytree': 0.42319770953022684, 'subsample': 0.2810517802368746, 'min_child_weight': 218, 'gamma': 6.031109467976734e-08, 'eta': 0.018889170085640027}
    xgb_predictions = xgb.predict()
    print(f"xgb: {time.perf_counter() - time_sta}")

    time_sta = time.perf_counter()
    rf = RandomForestModel(*selecter.get_dataframe("rf"))
    #rf.objective(5)
    rf.best_params = {'max_depth': 9, 'min_samples_split': 11, 'min_samples_leaf': 14, 'max_features': 0.6306125661502896, 'max_leaf_nodes': 18, 'n_estimators': 8762, 'bootstrap': True}
    rf_predictions = rf.predict()
    print(f"rf: {time.perf_counter() - time_sta}")

    time_sta = time.perf_counter()
    rgf = RGFModel(*selecter.get_dataframe("rgf"))
    #rgf.objective(5)
    rgf.best_params = {'max_leaf': 8072, 'algorithm': 'RGF_Opt', 'test_interval': 142, 'min_samples_leaf': 11, 'reg_depth': 9, 'l2': 0.0002082492344277923, 'sl2': 4.2919223241162815e-07, 'normalize': False}
    rgf_predictions = rgf.predict()
    print(f"rgf: {time.perf_counter() - time_sta}")

    time_sta = time.perf_counter()
    cat = CatBoostModel(*selecter.get_dataframe("cat"))
    #cat.objective(5)
    cat.best_params = {"depth": 6}
    cat_predictions = cat.predict()
    print(f"cat: {time.perf_counter() - time_sta}")

    predictions_all = pl.concat([predictions, lgbm_predictions, xgb_predictions, rf_predictions, rgf_predictions, cat_predictions], how="horizontal")

    model = Ridge(random_state=0)
    train = predictions_all.filter(pl.col("price").is_not_null())
    test = predictions_all.filter(pl.col("price").is_null())
    X_train = train.drop("price").to_numpy()
    y_train = train["price"].to_numpy()
    X_test = test.drop("price").to_numpy()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    sub = pl.read_csv(os.path.join(os.path.dirname(__file__), "input/submit_sample.csv"), has_header=False, new_columns=["id", "price"])
    sub = sub.with_columns(pl.Series("", y_pred).alias("price"))
    sub.write_csv(os.path.join(os.path.dirname(__file__), "output/submission_te0810.csv"), has_header=False)


if __name__ == "__main__":
    time_sta_all = time.perf_counter()
    main()
    print(f"all: {time.perf_counter() - time_sta_all}")
