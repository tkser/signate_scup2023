import re
import pandas as pd
import numpy as np

from geopy.geocoders import Nominatim
import mojimoji
import json
import time


class Features:

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        self.train = train
        self.test = test
        self.df = pd.concat([train, test], axis=0)

    def create_features(self) -> pd.DataFrame:
        self._fill_na()
        self._pre_processing()
        self._add_features()
        self._add_geo_features()
        #self._rank_encoding()
        self._count_encoding()
        self._label_encoding()
        self._one_hot_encoding()
        self._remove_features()
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
        # odometer
        self.df.loc[self.df["odometer"] == -1, "odometer"] = np.nan
        self.df["odometer"].fillna(self.df["odometer"].median(), inplace=True)
    
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
    
    def _add_geo_features(self) -> None:
        regions = self.df["region"].unique()
        geolocator = Nominatim(user_agent="tkser@github.com")
        region_latlng = {}
        try:
            with open("../output/region_latlng.json", "r") as f:
                region_latlng = json.load(f)
        except:
            pass
        for region in regions:
            if region in region_latlng.keys():
                continue
            elif region == "high rockies":
                region_latlng[region] = {
                    "lat": 40,
                    "lng": -105,
                    "state": "Colorado"
                }
            elif region == "western slope":
                region_latlng[region] = {
                    "lat": 39,
                    "lng": -108,
                    "state": "Colorado"
                }
            elif region == "new haven":
                region_latlng[region] = {
                    "lat": 41,
                    "lng": -73,
                    "state": "Connecticut"
                }
            elif region == "hartford":
                region_latlng[region] = {
                    "lat": 41.5,
                    "lng": -73,
                    "state": "Connecticut"
                }
            else:
                print(region)
                location = geolocator.geocode(re.split(r"[,/-]", region)[0].strip(), exactly_one=True, timeout=180, language="en", addressdetails=True, country_codes=["us"])
                coordinates = (location.latitude, location.longitude)
                address = geolocator.reverse(coordinates, timeout=180, language="en", addressdetails=True, exactly_one=True, namedetails=True)
                region_latlng[region] = {
                    "lat": location.latitude,
                    "lng": location.longitude,
                    "state": address.raw["address"]["state"]
                }
                with open("../output/region_latlng.json", "w") as f:
                    json.dump(region_latlng, f)
            time.sleep(1)
        self.df["region_lat"] = self.df["region"].map(lambda x: region_latlng[x]["lat"])
        self.df["region_lng"] = self.df["region"].map(lambda x: region_latlng[x]["lng"])
        self.df["region_state"] = self.df["region"].map(lambda x: region_latlng[x]["state"])
    
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
    
    def _rank_encoding(self) -> None:
        rank_encoding_columns = [
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
        for c in rank_encoding_columns:
            self.df[f"{c}_rank"] = self.df.groupby(c)["price"].rank(method="dense", ascending=False).astype(int)
    
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
            "id",
            "region",
            "region_state"
        ]
        self.df.drop(remove_columns, axis=1, inplace=True)
