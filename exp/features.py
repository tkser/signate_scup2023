import os
import re
import json
import time

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

    def create_features(self) -> pl.DataFrame:
        self._fill_na()
        self._pre_processing()
        #self._df_initialize()
        self._add_features()
        self._add_geo_features()
        #self._rank_encoding()
        self._count_encoding()
        self._label_encoding()
        self._one_hot_encoding()
        self._remove_features()
        return self.train, self.test
    
    def _df_initialize(self) -> None:
        self.train = self.train.with_columns(
            pl.col(pl.Utf8).cast(pl.Categorical)
        )
        self.test = self.test.with_columns(
            pl.col(pl.Utf8).cast(pl.Categorical)
        )
    
    def _fill_na(self) -> None:
        self.train = self.train.with_columns(
            pl.col("fuel").fill_null("gas").alias("fuel"),
            pl.col("state").fill_null("nan").alias("state"),
            pl.col("type").fill_null("nan").alias("type"),
            pl.col("title_status").fill_null("nan").alias("title_status"),
            pl.when(pl.col("odometer") == -1).then(None).otherwise(pl.col("odometer")).fill_null(pl.col("odometer").median()).alias("odometer"),
        )
        self.test = self.test.with_columns(
            pl.col("fuel").fill_null("gas").alias("fuel"),
            pl.col("state").fill_null("nan").alias("state"),
            pl.col("type").fill_null("nan").alias("type"),
            pl.col("title_status").fill_null("nan").alias("title_status"),
            pl.when(pl.col("odometer") == -1).then(None).otherwise(pl.col("odometer")).fill_null(pl.col("odometer").median()).alias("odometer"),
        )

    def _pre_processing(self) -> None:
        self.train = self.train.with_columns(
            # year
            pl.when(pl.col("year") >= 2030).then(pl.col("year") - 1000).otherwise(pl.col("year")).alias("year"),
            # manufacturer
            pl.col("manufacturer").apply(lambda x: mojimoji.zen_to_han(x).lower()).alias("manufacturer"),
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
            pl.col("manufacturer").apply(lambda x: mojimoji.zen_to_han(x).lower()).alias("manufacturer"),
            # cylinders
            pl.when(pl.col("cylinders") == "other").then("-1 cylinders").otherwise(pl.col("cylinders"))
                .apply(lambda x: re.sub(r'[^-?\d]', "", x)).cast(pl.Int8).alias("cylinders"),
            # odometer
            pl.when(pl.col("odometer") == -131869).then(131869).otherwise(pl.col("odometer")).alias("odometer"),
            # size
            pl.col("size").str.replace(r"−|ー", "-").alias("size"),
        )

    def _add_features(self) -> None:
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

        merge_df = pl.concat([self.train.select(pl.exclude("price")), self.test])

        manufacturer_agg_df = merge_df.groupby("manufacturer").agg(
            pl.mean("odometer").alias("manufacturer_odometer_mean"),
            pl.std("odometer").alias("manufacturer_odometer_std"),
            pl.max("odometer").alias("manufacturer_odometer_max"),
            pl.min("odometer").alias("manufacturer_odometer_min"),
            (pl.max("odometer") - pl.min("odometer")).alias("manufacturer_odometer_diff"),
        )
        self.train = self.train.join(manufacturer_agg_df, on="manufacturer", how="left")
        self.test = self.test.join(manufacturer_agg_df, on="manufacturer", how="left")
    
    def _add_geo_features(self) -> None:
        regions = pl.concat([self.train["region"], self.test["region"]]).unique().to_list()
        geolocator = Nominatim(user_agent="tkser@github.com")
        if os.path.exists("../output/region_latlng.csv"):
            region_latlng_df = pl.read_csv("../output/region_latlng.csv", columns=["region", "lat", "lng"])
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
                region_latlng_df.write_csv("../output/region_latlng.csv")
            time.sleep(1)
        self.train = self.train.join(region_latlng_df, on="region", how="left")
        self.test = self.test.join(region_latlng_df, on="region", how="left")
    
    def _label_encoding(self) -> None:
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
        merge_df = pl.concat([self.train.select(pl.exclude("price")), self.test])
        for c in rank_encoding_columns:
            rank_df = merge_df.groupby(c).count().rename({"count": f"{c}_count"})
            rank_df = rank_df.with_columns(
                pl.col(f"{c}_count").rank().alias(f"{c}_rank")
            ).select([f"{c}_rank", c])
            self.train = self.train.join(rank_df, on=c, how="left")
            self.test = self.test.join(rank_df, on=c, how="left")
    
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
        merge_df = pl.concat([self.train.select(pl.exclude("price")), self.test])
        for c in count_encoding_columns:
            count_df = merge_df.groupby(c).count().rename({"count": f"{c}_count"})
            self.train = self.train.join(count_df, on=c, how="left")
            self.test = self.test.join(count_df, on=c, how="left")
    
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
        mearge_df = pl.concat([self.train.select(pl.exclude("price")), self.test]).select(["id"] + encodeing_columns)
        mearge_df = mearge_df.to_dummies(encodeing_columns, drop_first=True)
        self.train = self.train.join(mearge_df, on="id", how="left")
        self.test = self.test.join(mearge_df, on="id", how="left")

    def _remove_features(self) -> None:
        remove_columns = [
            "id",
            "region",
            "region_state"
        ]
        self.df.drop(remove_columns, axis=1, inplace=True)
