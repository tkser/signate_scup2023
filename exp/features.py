import os
import re
import time

from typing import Tuple

import polars as pl
pl.enable_string_cache(True)

import dataliner as dl

import mojimoji
from geopy.geocoders import Nominatim


class Features:

    train: pl.DataFrame
    test: pl.DataFrame

    def __init__(self, train: pl.DataFrame, test: pl.DataFrame) -> None:
        self.train = train
        self.test = test

    def create_features(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        self.__fill_na()
        self.__pre_processing()
        self.__add_features()
        self.__add_geo_features()
        self.__df_initialize()
        self.__rank_encoding()
        self.__count_encoding()
        self.__label_encoding()
        self.__target_encoding()
        self.__one_hot_encoding()
        return self.train, self.test
    
    def __df_initialize(self) -> None:
        self.train = self.train.with_columns(
            pl.col(pl.Utf8).cast(pl.Categorical)
        )
        self.test = self.test.with_columns(
            pl.col(pl.Utf8).cast(pl.Categorical)
        )
    
    def __fill_na(self) -> None:
        regions = self.train["region"].unique().to_list()
        region_state_map = {}
        for region in regions:
            state_mode = self.train.filter((pl.col("region") == region) & (pl.col("state") != "nan"))["state"].mode()
            region_state_map[region] = state_mode[0] if len(state_mode) > 0 else None
        
        self.train = self.train.with_columns(
            pl.when(pl.col("state").is_null()).then(pl.col("region").map_dict(region_state_map)).otherwise(pl.col("state")).alias("state")
        )
        self.test = self.test.with_columns(
            pl.when(pl.col("state").is_null()).then(pl.col("region").map_dict(region_state_map)).otherwise(pl.col("state")).alias("state")
        )

        self.train = self.train.with_columns(
            pl.col("fuel").fill_null("gas").alias("fuel"),
            pl.col("state").fill_null("nan").alias("state"),
            pl.col("type").fill_null("nan").alias("type"),
            pl.col("title_status").fill_null("nan").alias("title_status"),
            pl.when(pl.col("odometer") == -1).then(None).otherwise(pl.col("odometer")).fill_null(pl.col("odometer").median()).alias("odometer_f"),
        )
        self.test = self.test.with_columns(
            pl.col("fuel").fill_null("gas").alias("fuel"),
            pl.col("state").fill_null("nan").alias("state"),
            pl.col("type").fill_null("nan").alias("type"),
            pl.col("title_status").fill_null("nan").alias("title_status"),
            pl.when(pl.col("odometer") == -1).then(None).otherwise(pl.col("odometer")).fill_null(pl.col("odometer").median()).alias("odometer_f"),
        )

    def __pre_processing(self) -> None:
        self.train = self.train.with_columns(
            # year
            pl.when(pl.col("year") >= 2030).then(pl.col("year") - 1000).otherwise(pl.col("year")).alias("year"),
            # manufacturer
            pl.col("manufacturer").apply(lambda x: mojimoji.zen_to_han(x).lower().replace("α", "a").replace("ѕ", "s").replace("а", "a")).alias("manufacturer"),
            # cylinders
            pl.when(pl.col("cylinders") == "other").then("-1 cylinders").otherwise(pl.col("cylinders"))
                .apply(lambda x: re.sub(r'[^-?\d]', "", x)).cast(pl.Int8).alias("cylinders"),
            # odometer
            pl.when(pl.col("odometer") == -131869).then(131869).otherwise(pl.col("odometer")).alias("odometer"),
            # size
            pl.col("size").str.replace(r"−|ー", "-").alias("size"),
        )
        self.test = self.test.with_columns(
            # year
            pl.when(pl.col("year") >= 2030).then(pl.col("year") - 1000).otherwise(pl.col("year")).alias("year"),
            # manufacturer
            pl.col("manufacturer").apply(lambda x: mojimoji.zen_to_han(x).lower().replace("α", "a").replace("ѕ", "s").replace("а", "a")).alias("manufacturer"),
            # cylinders
            pl.when(pl.col("cylinders") == "other").then("-1 cylinders").otherwise(pl.col("cylinders"))
                .apply(lambda x: re.sub(r'[^-?\d]', "", x)).cast(pl.Int8).alias("cylinders"),
            # odometer
            pl.when(pl.col("odometer") == -131869).then(131869).otherwise(pl.col("odometer")).alias("odometer"),
            # size
            pl.col("size").str.replace(r"−|ー", "-").alias("size"),
        )

    def __add_features(self) -> None:
        self.train = self.train.with_columns(
            (2023 - pl.col("year")).alias("age"),
            (pl.col("odometer") / (2023 - pl.col("year"))).alias("odometer/age"),
            (pl.col("odometer") / pl.col("cylinders")).alias("odometer/cylinders"),
        )
        self.test = self.test.with_columns(
            (2023 - pl.col("year")).alias("age"),
            (pl.col("odometer") / (2023 - pl.col("year"))).alias("odometer/age"),
            (pl.col("odometer") / pl.col("cylinders")).alias("odometer/cylinders"),
        )

        manufacturer_agg_df = self.train.groupby("manufacturer").agg(
            pl.mean("odometer").alias("manufacturer_odometer_mean"),
            pl.std("odometer").alias("manufacturer_odometer_std"),
            pl.max("odometer").alias("manufacturer_odometer_max"),
            pl.min("odometer").alias("manufacturer_odometer_min"),
            (pl.max("odometer") - pl.min("odometer")).alias("manufacturer_odometer_diff"),
        )
        self.train = self.train.join(manufacturer_agg_df, on="manufacturer", how="left")
        self.test = self.test.join(manufacturer_agg_df, on="manufacturer", how="left")
    
    def __add_geo_features(self) -> None:
        regions = pl.concat([self.train["region"], self.test["region"]]).unique().to_list()
        geolocator = Nominatim(user_agent="tkser@github.com")
        region_csv_path = os.path.join(os.path.dirname(__file__), "../output/region_latlng.csv")
        if os.path.exists(region_csv_path):
            region_latlng_df = pl.read_csv(region_csv_path, columns=["region", "lat", "lng"])
        else:
            region_latlng_df = pl.DataFrame({
                "region": [],
                "lat": [],
                "lng": [],
            }, schema={
                "region": pl.Utf8,
                "lat": pl.Float64,
                "lng": pl.Float64,
            })
        for region in regions:
            if region in region_latlng_df["region"]:
                continue
            elif region == "high rockies":
                region_latlng_df.vstack(pl.DataFrame({
                    "region": ["high rockies"],
                    "lat": [40.0],
                    "lng": [-105.0],
                }), in_place=True)
            elif region == "western slope":
                region_latlng_df.vstack(pl.DataFrame({
                    "region": ["western slope"],
                    "lat": [39.0],
                    "lng": [-108.0],
                }), in_place=True)
            elif region == "new haven":
                region_latlng_df.vstack(pl.DataFrame({
                    "region": ["new haven"],
                    "lat": [41.0],
                    "lng": [-73.0],
                }), in_place=True)
            elif region == "hartford":
                region_latlng_df.vstack(pl.DataFrame({
                    "region": ["hartford"],
                    "lat": [41.0],
                    "lng": [-72.0],
                }), in_place=True)
            else:
                location = geolocator.geocode(re.split(r"[,/-]", region)[0].strip(), exactly_one=True, timeout=180, language="en", addressdetails=True, country_codes=["us"])
                region_latlng_df.vstack(pl.DataFrame({
                    "region": [region],
                    "lat": [location.latitude],
                    "lng": [location.longitude],
                }), in_place=True)
                region_latlng_df.write_csv(region_csv_path)
            time.sleep(1)
        self.train = self.train.join(region_latlng_df, on="region", how="left")
        self.test = self.test.join(region_latlng_df, on="region", how="left")
    
    def __label_encoding(self) -> None:
        condition_mapping = {
            "new": 5,
            "like new": 4,
            "excellent": 3,
            "good": 2,
            "salvage": 1,
            "fair": 0
        }
        size_mapping = {
            "full-size": 3,
            "mid-size": 2,
            "compact": 1,
            "sub-compact": 0
        }
        self.train = self.train.with_columns(
            pl.col("condition").map_dict(condition_mapping).alias("condition"),
            pl.col("size").map_dict(size_mapping).alias("size")
        )
        self.test = self.test.with_columns(
            pl.col("condition").map_dict(condition_mapping).alias("condition"),
            pl.col("size").map_dict(size_mapping).alias("size")
        )
    
    def __rank_encoding(self) -> None:
        rank_encoding_columns = [
            #"region",
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
            rank_df = self.train.groupby(c).count().rename({"count": f"{c}_count"})
            rank_df = rank_df.with_columns(
                pl.col(f"{c}_count").rank().alias(f"{c}_rank")
            ).select([f"{c}_rank", c])
            self.train = self.train.join(rank_df, on=c, how="left")
            self.test = self.test.join(rank_df, on=c, how="left")
    
    def __count_encoding(self) -> None:
        count_encoding_columns = [
            #"region",
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
            count_df = self.train.groupby(c).count().rename({"count": f"{c}_count"})
            self.train = self.train.join(count_df, on=c, how="left")
            self.test = self.test.join(count_df, on=c, how="left")
    
    def __target_encoding(self) -> None:
        encoding_columns = [
            "manufacturer",
            "fuel",
            "title_status",
            "transmission",
            "drive",
            "type",
            "paint_color",
            "state"
        ]
        for c in encoding_columns:
            target_df = self.train.groupby(c).agg(pl.mean("price").alias(f"{c}_mean"))
            self.train = self.train.join(target_df, on=c, how="left")
            self.test = self.test.join(target_df, on=c, how="left")
    
    def __one_hot_encoding(self) -> None:
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
        mearge_df = pl.concat([self.train.select(pl.exclude("price")), self.test]).select(["id"] + encodeing_columns)
        mearge_df = mearge_df.to_dummies(encodeing_columns, drop_first=True)
        self.train = self.train.join(mearge_df, on="id", how="left")
        self.test = self.test.join(mearge_df, on="id", how="left")
        self.train = self.train.drop(encodeing_columns)
        self.test = self.test.drop(encodeing_columns)


class FeatureSelecter:

    train: pl.DataFrame
    test: pl.DataFrame

    def __init__(self, train: pl.DataFrame, test: pl.DataFrame) -> None:
        self.train = train
        self.test = test
        self.__common_func()
    
    def __common_func(self) -> None:
        common_drop_columns = [
            "id",
            "region"
        ]
        self.train = self.train.drop(common_drop_columns)
        self.test = self.test.drop(common_drop_columns)
    
    def show_columns(self) -> None:
        print(self.test.columns)
    
    def get_dataframe(self, type: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
        return self.train, self.test
