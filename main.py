import pandas as pd
import numpy as np

from exp.features import Features
from exp.models.lightgbm import Model as LGBMModel


def main():
    train = pd.read_csv("./input/train.csv")
    test = pd.read_csv("./input/test.csv")

    features = Features(train, test)
    df = features.create_features()

    lgbm = LGBMModel(df)
    prediction = lgbm.predict()


if __name__ == "__main__":
    main()
