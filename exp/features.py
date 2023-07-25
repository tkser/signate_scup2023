import re
import pandas as pd
import numpy as np

import mojimoji


class Features:

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        self.train = train
        self.test = test
        self.df = pd.concat([train, test], axis=0)

    def create_features(self) -> pd.DataFrame:
        self._fill_na()
        self._pre_processing()
        self._add_features()
        self._count_encoding()
        self._label_encoding()
        self._one_hot_encoding()
        return self.df
    
    def _fill_na(self) -> None:
        # fuel
        self.df["fuel"].fillna("gas", inplace=True)
        # state
        self.df["state"].fillna("nan", inplace=True)
        # type
        self.df["type"].fillna("nan", inplace=True)
        # title_status
        self.df["title_status"].fillna("nan", inplace=True)
    
    def _pre_processing(self) -> None:
        # year
        self.df.loc[self.df["year"] >= 2030, "year"] = self.df[self.df["year"] >= 2030]["year"] - 1000
        # manufacturer
        self.df["manufacturer"] = self.df["manufacturer"].apply(lambda x: mojimoji.zen_to_han(x))
        self.df["manufacturer"] = self.df["manufacturer"].apply(lambda x: x.lower())
        # cylinders
        self.df["cylinders"].replace("other", "-1 cylinders", inplace=True)
        self.df["cylinders"] = self.df["cylinders"].str.split(" ").str[0]
        self.df["cylinders"] = self.df["cylinders"].astype(int)
        # odometer
        self.df.loc[self.df["odometer"] == -1, "odometer"] = np.nan
        self.df.loc[self.df["odometer"] == -131869, "odometer"] = 131869
        # size
        self.df["size"] = self.df["size"].str.replace("−", "-")
        self.df["size"] = self.df["size"].str.replace("ー", "-")

    def _add_features(self) -> None:
        self.df["age"] = 2023 - self.df["year"]
        self.df["odometer/age"] = self.df["odometer"] / self.df["age"]
        self.df["odometer/cylinders"] = self.df["odometer"] / self.df["cylinders"]

        self.df["manufacturer_odometer_mean"] = self.df.groupby("manufacturer")["odometer"].transform("mean")
        self.df["manufacturer_odometer_std"] = self.df.groupby("manufacturer")["odometer"].transform("std")
        self.df["manufacturer_odometer_max"] = self.df.groupby("manufacturer")["odometer"].transform("max")
        self.df["manufacturer_odometer_min"] = self.df.groupby("manufacturer")["odometer"].transform("min")
        self.df["manufacturer_odometer_diff"] = self.df["manufacturer_odometer_max"] - self.df["manufacturer_odometer_min"]

    def _label_encoding(self) -> None:
        condition_mapping = {
            "new": 5,
            "like new": 4,
            "excellent": 3,
            "good": 2,
            "salvage": 1,
            "fair": 0
        }
        self.df["condition"] = self.df["condition"].map(condition_mapping)
        size_mapping = {
            "full-size": 3,
            "mid-size": 2,
            "compact": 1,
            "sub-compact": 0
        }
        self.df["size"] = self.df["size"].map(size_mapping)
    
    def _count_encoding(self) -> None:
        count_encoding_columns = [
            "region",
            "manufacturer",
            "condition",
            "fuel",
            "title_status",
            "transmission",
            "drive",
            "size",
            "type",
            "paint_color",
            "state"
        ]
        for c in count_encoding_columns:
            self.df[f"{c}_count"] = self.df.groupby(c)["id"].transform("count")
    
    def _one_hot_encoding(self) -> None:
        encodeing_columns = [
            "region",
            "manufacturer",
            "fuel",
            "title_status",
            "transmission",
            "drive",
            "type",
            "paint_color",
            "state"
        ]
        self.df = pd.get_dummies(self.df, columns=encodeing_columns, drop_first=True, dtype=int)
        self.df.columns = [re.sub(r'[,\[\]{}:\s]', '_', c) for c in self.df.columns]


    def _remove_features(self) -> None:
        remove_columns = [
            "id"
        ]
        self.df.drop(remove_columns, axis=1, inplace=True)
