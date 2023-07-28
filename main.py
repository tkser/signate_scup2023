import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge

from exp.features import Features
from exp.models.lgbm import LGBMModel
from exp.models.rf import RandomForestModel
from exp.models.xgb import XGBModel

import warnings
warnings.filterwarnings("ignore")


def main():
    train = pd.read_csv("./input/train.csv")
    test = pd.read_csv("./input/test.csv")

    features = Features(train, test)
    df = features.create_features()

    predictions = pd.DataFrame(df["price"])

    lgbm = LGBMModel(df)
    lgbm.objective(20)
    lgbm_predictions = lgbm.predict()

    xgb = XGBModel(df)
    xgb.objective(20)
    xgb_predictions = xgb.predict()

    rf = RandomForestModel(df)
    rf.objective(20)
    rf_predictions = rf.predict()

    predictions = pd.concat([predictions, lgbm_predictions, xgb_predictions, rf_predictions], axis=1)

    model = Ridge(random_state=0)
    train = predictions[predictions["price"].notnull()]
    test = predictions[predictions["price"].isnull()]
    X_train = train.drop(["price"], axis=1)
    y_train = train["price"]
    X_test = test.drop(["price"], axis=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    sub = pd.read_csv("./input/submit_sample.csv")
    sub["price"] = y_pred
    sub.to_csv("./output/submit0726.csv", index=False)


if __name__ == "__main__":
    main()
