import polars as pl
import numpy as np

from sklearn.linear_model import Ridge

from exp.features import Features
from exp.models.cat import CatBoostModel
from exp.models.lgbm import LGBMModel
from exp.models.rf import RandomForestModel
from exp.models.rgf import RGFModel
from exp.models.xgb import XGBModel

import warnings
warnings.filterwarnings("ignore")


def main():
    train = pl.read_csv("./input/train.csv")
    test = pl.read_csv("./input/test.csv")

    features = Features(train, test)
    df = features.create_features()

    predictions = pl.DataFrame(df["price"])

    lgbm = LGBMModel(df)
    #lgbm.objective(20)
    lgbm.best_params = {'num_leaves': 48, 'max_depth': 6, 'min_child_samples': 91, 'subsample': 0.5578230915019112, 'colsample_bytree': 0.5933052522026404, 'reg_alpha': 2.4725566626090776e-05, 'reg_lambda': 1.0114136512530978e-08, 'feature_fraction': 0.7523757350552451, 'bagging_fraction': 0.9199865329355417, 'bagging_freq': 5}
    lgbm_predictions = lgbm.predict()

    xgb = XGBModel(df)
    #xgb.objective(20)
    xgb.best_params = {'n_estimators': 767, 'max_depth': 8, 'lambda': 1.2306916748991704e-06, 'alpha': 0.018078104089246788, 'colsample_bytree': 0.42319770953022684, 'subsample': 0.2810517802368746, 'min_child_weight': 218, 'gamma': 6.031109467976734e-08, 'eta': 0.018889170085640027}
    xgb_predictions = xgb.predict()

    rf = RandomForestModel(df)
    #rf.objective(5)
    rf.best_params = {'max_depth': 9, 'min_samples_split': 11, 'min_samples_leaf': 14, 'max_features': 0.6306125661502896, 'max_leaf_nodes': 18, 'n_estimators': 8762, 'bootstrap': True}
    rf_predictions = rf.predict()

    rgf = RGFModel(df)
    #rgf.objective(5)
    rgf.best_params = {'max_leaf': 8072, 'algorithm': 'RGF_Opt', 'test_interval': 142, 'min_samples_leaf': 11, 'reg_depth': 9, 'l2': 0.0002082492344277923, 'sl2': 4.2919223241162815e-07, 'normalize': False}
    rgf_predictions = rgf.predict()

    cat = CatBoostModel(df)
    #cat.objective(5)
    cat.best_params = {"depth": 6}
    cat_predictions = cat.predict()

    predictions = pl.concat([predictions, lgbm_predictions, xgb_predictions, rf_predictions, rgf_predictions, cat_predictions])

    model = Ridge(random_state=0)
    train = predictions[predictions["price"].notnull()]
    test = predictions[predictions["price"].isnull()]
    X_train = train.drop(["price"], axis=1)
    y_train = train["price"]
    X_test = test.drop(["price"], axis=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    sub = pl.read_csv("./input/submit_sample.csv")
    sub["price"] = y_pred
    sub.write_csv("./output/submit0728_fullstacking.csv", overwrite=True)


if __name__ == "__main__":
    main()
