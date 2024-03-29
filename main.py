import os
import polars as pl
import numpy as np

from exp.util.timer import Timer
from exp.features import Features, FeatureSelecter
from exp.models.cat import CatBoostModel
from exp.models._lgbm import LGBMModel
from exp.models.xgb import XGBModel

import gc
gc.enable()

import warnings
warnings.filterwarnings("ignore")

LGBM_ONLY = True
LOG_MODE = False


def main():
    with Timer("all"):
        train = pl.read_csv(os.path.join(os.path.dirname(__file__), "input/train.csv"))
        test = pl.read_csv(os.path.join(os.path.dirname(__file__), "input/test.csv"))

        features = Features(train, test, log_mode=LOG_MODE)
        train, test = features.create_features()

        selecter = FeatureSelecter(train, test)

        predictions_all = pl.concat([train["price"].to_frame(), pl.DataFrame([None] * test.height, schema={"price": pl.Float64})])

        for random_status in [0, 101, 269, 787, 983]:
            with Timer(f"lgbm01_{random_status}"):
                lgbm01 = LGBMModel(*selecter.get_dataframe("lgbm"), seed=random_status)
                lgbm01.best_params = {
                    'num_leaves': 48,
                    'max_depth': 6,
                    'min_child_samples': 91,
                    'subsample': 0.5578230915019112,
                    'colsample_bytree': 0.5933052522026404,
                    'reg_alpha': 2.4725566626090776e-05,
                    'reg_lambda': 1.0114136512530978e-08,
                    'feature_fraction': 0.7523757350552451,
                    'bagging_fraction': 0.9199865329355417,
                    'bagging_freq': 5
                }
                lgbm01_predictions = lgbm01.predict(5, f"lgbm01_{random_status}")
                predictions_all = pl.concat([predictions_all, lgbm01_predictions], how="horizontal")

            with Timer(f"lgbm02_{random_status}"):
                lgbm02 = LGBMModel(*selecter.get_dataframe("lgbm"), seed=random_status)
                lgbm02.best_params = {
                    'num_leaves': 41,
                    'max_depth': 4,
                    'min_child_samples': 65,
                    'subsample': 0.1813266686908916,
                    'colsample_bytree': 0.9997207808739403,
                    'reg_alpha': 3.8163343968470076e-06,
                    'reg_lambda': 9.185674902594394e-05,
                    'feature_fraction': 0.5180973927754882,
                    'bagging_fraction': 0.8804646505719466,
                    'bagging_freq': 1
                }
                lgbm02_predictions = lgbm02.predict(4, f"lgbm02_{random_status}")
                predictions_all = pl.concat([predictions_all, lgbm02_predictions], how="horizontal")
            
            with Timer(f"lgbm03_{random_status}"):
                lgbm03 = LGBMModel(*selecter.get_dataframe("lgbm"), seed=random_status)
                lgbm03.best_params = {
                    'num_leaves': 21,
                    'max_depth': 2,
                    'min_child_samples': 45,
                    'subsample': 0.1813266686908916,
                    'colsample_bytree': 0.9997207808739403,
                    'reg_alpha': 3.8163343968470076e-06,
                    'reg_lambda': 9.185674902594394e-05,
                    'feature_fraction': 0.5180973927754882,
                    'bagging_fraction': 0.8804646505719466,
                    'bagging_freq': 5
                }
                lgbm03_predictions = lgbm03.predict(5, f"lgbm03_{random_status}")
                predictions_all = pl.concat([predictions_all, lgbm03_predictions], how="horizontal")
            
            with Timer(f"lgbm04_{random_status}"):
                lgbm04 = LGBMModel(*selecter.get_dataframe("lgbm"), seed=random_status)
                lgbm04.best_params = {
                    'num_leaves': 60,
                    'min_child_samples': 128,
                    'subsample': 0.4557739655602947,
                    'colsample_bytree': 0.4544303943274487,
                    'reg_alpha': 0.00018337518898873693,
                    'reg_lambda': 0.06367058193742765,
                    'subsample_freq': 1
                }
                lgbm04_predictions = lgbm04.predict(1, f"lgbm04_{random_status}")
                predictions_all = pl.concat([predictions_all, lgbm04_predictions], how="horizontal")

            if not LGBM_ONLY:
                with Timer(f"xgb_{random_status}"):
                    xgb = XGBModel(*selecter.get_dataframe("xgb"))
                    #xgb.objective(20)
                    xgb.best_params = {'n_estimators': 767, 'max_depth': 8, 'lambda': 1.2306916748991704e-06, 'alpha': 0.018078104089246788, 'colsample_bytree': 0.42319770953022684, 'subsample': 0.2810517802368746, 'min_child_weight': 218, 'gamma': 6.031109467976734e-08, 'eta': 0.018889170085640027}
                    xgb_predictions = xgb.predict(5, f"xgb_{random_status}")
                    predictions_all = pl.concat([predictions_all, xgb_predictions], how="horizontal")

                with Timer(f"cat_{random_status}"):
                    cat = CatBoostModel(*selecter.get_dataframe("cat"))
                    #cat.objective(5)
                    cat.best_params = {"depth": 6}
                    cat_predictions = cat.predict(5, f"cat_{random_status}")
                    predictions_all = pl.concat([predictions_all, cat_predictions], how="horizontal")
            
        train = predictions_all.filter(pl.col("price").is_not_null())
        test = predictions_all.filter(pl.col("price").is_null()).drop("price")

        with Timer("st_lgbm"):
            #stack_lgbm = LGBMModel(train, test)
            #stack_lgbm.objective(5)
            #stack_lgbm.best_params = {'num_leaves': 44, 'max_depth': 17, 'min_child_samples': 27, 'subsample': 0.21603366788936798, 'colsample_bytree': 0.38388551583176544, 'reg_alpha': 8.122433559209657e-06, 'reg_lambda': 0.0003643964717966421, 'feature_fraction': 0.6631609080773921, 'bagging_fraction': 0.9930243028355357, 'bagging_freq': 5}
            #y_preds = stack_lgbm.predict(5, col_name="st_lgbm")
            y_pred = test.mean(axis=1).to_numpy()
        
        if LOG_MODE:
            y_pred = np.exp(y_pred)

        sub = pl.read_csv(os.path.join(os.path.dirname(__file__), "input/submit_sample.csv"), has_header=False, new_columns=["id", "price"])
        sub = sub.with_columns(pl.Series("", y_pred).alias("price"))
        sub.write_csv(os.path.join(os.path.dirname(__file__), "output/submission_te0823_1.csv"), has_header=False)


if __name__ == "__main__":
    main()
