import os
import re
import time

from typing import Tuple

import numpy as np
import polars as pl
pl.enable_string_cache(True)

import fasttext
import fasttext.util

from unidecode import unidecode
from geopy.geocoders import Nominatim

from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler


class Features:

    train: pl.DataFrame
    test: pl.DataFrame

    def __init__(self, train: pl.DataFrame, test: pl.DataFrame, log_mode = False) -> None:
        self.train = train
        self.test = test
        self.log_mode = log_mode

    def create_features(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        self.__df_initialize()
        self.__fill_na()
        self.__pre_processing()
        self.__add_features()
        self.__add_geo_features()
        self.__df_initialize2()
        self.__lat_lng_clustering()
        self.__rank_encoding()
        self.__count_encoding()
        self.__kde_encoding()
        self.__target_encoding()
        self.__label_encoding()
        self.__agg_encoding()
        self.__car_string_encoding()
        self.__one_hot_encoding()
        return self.train, self.test

    def __df_initialize(self) -> None:
        self.train = self.train.filter(pl.col("year") >= self.test["year"].min())
        if self.log_mode:
            self.train = self.train.with_columns(
                pl.col("price").log().alias("price"),
            )
    
    def __df_initialize2(self) -> None:
        self.train = self.train.with_columns(
            pl.col(pl.Utf8).cast(pl.Categorical),
            pl.col("price").cast(pl.Float64).alias("price"),
        )
        self.test = self.test.with_columns(
            pl.col(pl.Utf8).cast(pl.Categorical)
        )
    
    def __fill_na(self) -> None:
        regions = self.train["region"].unique().to_list()
        region_state_map = {
            "orthwest KS": "ks",
            "ashtabula": "oh",
            "southern WV": "wv"
        }
        for region in regions:
            state_mode = self.train.filter((pl.col("region") == region) & (pl.col("state") != "nan"))["state"].mode()
            region_state_map[region] = state_mode[0] if len(state_mode) > 0 else None
        
        self.train = self.train.with_columns(
            pl.when(pl.col("state").is_null()).then(pl.col("region").map_dict(region_state_map)).otherwise(pl.col("state")).alias("state"),
            pl.when(pl.col("odometer") == -1).then(None).otherwise(pl.col("odometer")).alias("odometer"),
        )
        self.test = self.test.with_columns(
            pl.when(pl.col("state").is_null()).then(pl.col("region").map_dict(region_state_map)).otherwise(pl.col("state")).alias("state"),
            pl.when(pl.col("odometer") == -1).then(None).otherwise(pl.col("odometer")).alias("odometer"),
        )

        self.train = self.train.with_columns(
            pl.when(pl.col("odometer") >= 1000000).then(pl.col("odometer") / 1000).otherwise(pl.col("odometer")).alias("odometer"),
        )
        self.test = self.test.with_columns(
            pl.when(pl.col("odometer") >= 1000000).then(pl.col("odometer") / 1000).otherwise(pl.col("odometer")).alias("odometer"),
        )

        self.train = self.train.with_columns(
            pl.col("fuel").fill_null("gas").alias("fuel"),
            pl.col("type").fill_null("nan").alias("type"),
            pl.col("title_status").fill_null("clean").alias("title_status"),
            pl.col("odometer").fill_null(pl.col("odometer").median()).alias("odometer_f"),
        )
        self.test = self.test.with_columns(
            pl.col("fuel").fill_null("gas").alias("fuel"),
            pl.col("type").fill_null("nan").alias("type"),
            pl.col("title_status").fill_null("clean").alias("title_status"),
            pl.col("odometer").fill_null(pl.col("odometer").median()).alias("odometer_f"),
        )
    
    def __manufacturer_clean(self, x) -> str:
        error_parse_dict = {
            "lexudz": "lexus",
            "nidzsan": "nissan",
            "nisdzan": "nissan"
        }
        x = unidecode(x).lower().replace(" ", "_").replace("-", "_")
        if x in error_parse_dict.keys():
            x = error_parse_dict[x]
        return x

    def __pre_processing(self) -> None:
        self.train = self.train.with_columns(
            # year
            pl.when(pl.col("year") >= 2030).then(pl.col("year") - 1000).otherwise(pl.col("year")).alias("year"),
            # manufacturer
            pl.col("manufacturer").apply(self.__manufacturer_clean).alias("manufacturer"),
            # cylinders
            pl.when(pl.col("cylinders") == "other").then("-1 cylinders").otherwise(pl.col("cylinders"))
                .apply(lambda x: re.sub(r'[^-?\d]', "", x)).cast(pl.Int8).alias("cylinders"),
            # odometer
            pl.when(pl.col("odometer") == -131869).then(131869).otherwise(pl.col("odometer")).alias("odometer"),
            pl.when(pl.col("odometer_f") == -131869).then(131869).otherwise(pl.col("odometer_f")).alias("odometer_f"),
            # size
            pl.col("size").str.replace(r"−|ー", "-").alias("size"),
        )
        self.test = self.test.with_columns(
            # year
            pl.when(pl.col("year") >= 2030).then(pl.col("year") - 1000).otherwise(pl.col("year")).alias("year"),
            # manufacturer
            pl.col("manufacturer").apply(self.__manufacturer_clean).alias("manufacturer"),
            # cylinders
            pl.when(pl.col("cylinders") == "other").then("-1 cylinders").otherwise(pl.col("cylinders"))
                .apply(lambda x: re.sub(r'[^-?\d]', "", x)).cast(pl.Int8).alias("cylinders"),
            # odometer
            pl.when(pl.col("odometer") == -131869).then(131869).otherwise(pl.col("odometer")).alias("odometer"),
            pl.when(pl.col("odometer_f") == -131869).then(131869).otherwise(pl.col("odometer_f")).alias("odometer_f"),
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
            pl.col("condition").map_dict(condition_mapping).alias("condition_l"),
            pl.col("size").map_dict(size_mapping).alias("size_l")
        )
        self.test = self.test.with_columns(
            pl.col("condition").map_dict(condition_mapping).alias("condition_l"),
            pl.col("size").map_dict(size_mapping).alias("size_l")
        )
    
    def __lat_lng_clustering(self) -> None:
        scaler = StandardScaler()

        scaler.fit(self.train.select(["lat", "lng"]).to_numpy())
        self.train = self.train.hstack(pl.DataFrame(scaler.transform(self.train.select(["lat", "lng"]).to_numpy()), schema=["lat_scaled", "lng_scaled"]))
        self.test = self.test.hstack(pl.DataFrame(scaler.transform(self.test.select(["lat", "lng"]).to_numpy()), schema=["lat_scaled", "lng_scaled"]))

        n_clusters = 15
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(self.train.select(["lat_scaled", "lng_scaled"]).to_numpy())

        self.train = self.train.with_columns(
            pl.Series(kmeans.labels_.astype(np.int8)).alias("lat_lng_cluster")
        )
        self.test = self.test.with_columns(
            pl.Series(kmeans.predict(self.test.select(["lat_scaled", "lng_scaled"]).to_numpy()).astype(np.int8)).alias("lat_lng_cluster")
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
        def target_encode(group_col_names, target_col_name):
            for c in group_col_names:
                target_df = self.train.groupby(c).agg(pl.mean(target_col_name).alias(f"{c}_{target_col_name}_mean"))
                self.train = self.train.join(target_df, on=c, how="left")
                self.test = self.test.join(target_df, on=c, how="left")
            
        price_encoding_columns = [
            "manufacturer",
            "fuel",
            "title_status",
            "transmission",
            "drive",
            "type",
            "paint_color",
            "state",
            "lat_lng_cluster"
        ]
        target_encode(price_encoding_columns, "price")

        price_ratio_encoding_columns = [
            "title_status",
            "state",
            "region",
            "paint_color",
            "lat_lng_cluster"
        ]
        target_encode(price_ratio_encoding_columns, "type_manufacturer_price_ratio")
    
    def __kde_encoding(self) -> None:
        def compute_type_base_value(group_col_name: str, target_col_name: str):
            type_values = []
            
            for type_name in self.train[group_col_name].unique().to_list():
                subset = self.train.filter(self.train[group_col_name] == type_name)
                prices = subset[target_col_name].to_numpy().reshape(1, -1)
                if len(prices)>10:
                    kde = KernelDensity(kernel='gaussian', bandwidth=np.std(prices)+1).fit(prices)
                    _prices = np.linspace(np.min(prices), np.max(prices), 100).reshape(1, -1)
                    dens = kde.score_samples(_prices)
                    type_values.append((type_name, prices[np.argmax(dens)]))
                else:
                    type_values.append((type_name, np.nanmean(prices)))
            
            return pl.DataFrame(type_values, schema={group_col_name: pl.Categorical, "{}_base_{}".format(group_col_name, target_col_name): pl.Float32})

        type_base_prices = compute_type_base_value("type", "price")
        self.train = self.train.join(type_base_prices, on="type", how="left")
        self.test = self.test.join(type_base_prices, on="type", how="left")

        type_base_years = compute_type_base_value("type", "year")
        self.train = self.train.join(type_base_years, on="type", how="left")
        self.test = self.test.join(type_base_years, on="type", how="left")

        type_base_odometers = compute_type_base_value("type", "odometer_f")
        self.train = self.train.join(type_base_odometers, on="type", how="left")
        self.test = self.test.join(type_base_odometers, on="type", how="left")

        manufacturer_base_prices = compute_type_base_value("manufacturer", "price")
        self.train = self.train.join(manufacturer_base_prices, on="manufacturer", how="left")
        self.test = self.test.join(manufacturer_base_prices, on="manufacturer", how="left")

        manufacturer_base_years = compute_type_base_value("manufacturer", "year")
        self.train = self.train.join(manufacturer_base_years, on="manufacturer", how="left")
        self.test = self.test.join(manufacturer_base_years, on="manufacturer", how="left")

        manufacturer_base_odometers = compute_type_base_value("manufacturer", "odometer_f")
        self.train = self.train.join(manufacturer_base_odometers, on="manufacturer", how="left")
        self.test = self.test.join(manufacturer_base_odometers, on="manufacturer", how="left")

        self.train = self.train.with_columns(
            ((pl.col("type_base_price") + pl.col("manufacturer_base_price")) / 2).alias("type_manufacturer_base_price"),
            ((pl.col("type_base_year") + pl.col("manufacturer_base_year")) / 2).alias("type_manufacturer_base_year"),
            ((pl.col("type_base_odometer_f") + pl.col("manufacturer_base_odometer_f")) / 2).alias("type_manufacturer_base_odometer"),
        )
        self.test = self.test.with_columns(
            ((pl.col("type_base_price") + pl.col("manufacturer_base_price")) / 2).alias("type_manufacturer_base_price"),
            ((pl.col("type_base_year") + pl.col("manufacturer_base_year")) / 2).alias("type_manufacturer_base_year"),
            ((pl.col("type_base_odometer_f") + pl.col("manufacturer_base_odometer_f")) / 2).alias("type_manufacturer_base_odometer"),
        )

        self.train = self.train.with_columns(
            (pl.col("price") / pl.col("type_manufacturer_base_price")).alias("type_manufacturer_price_ratio"),
            (pl.col("year") - pl.col("type_manufacturer_base_year")).alias("type_manufacturer_year_diff"),
            (pl.col("odometer_f") - pl.col("type_manufacturer_base_odometer")).alias("type_manufacturer_odometer_diff"),
            (pl.col("type_manufacturer_base_odometer") / (2023 - pl.col("type_manufacturer_base_year"))).alias("type_manufacturer_odometer/age_base"),
        )
        self.test = self.test.with_columns(
            (pl.col("year") - pl.col("type_manufacturer_base_year")).alias("type_manufacturer_year_diff"),
            (pl.col("odometer_f") - pl.col("type_manufacturer_base_odometer")).alias("type_manufacturer_odometer_diff"),
            (pl.col("type_manufacturer_base_odometer") / (2023 - pl.col("type_manufacturer_base_year"))).alias("type_manufacturer_odometer/age_base"),
        )

        self.train = self.train.with_columns(
            (pl.col("odometer/age") - pl.col("type_manufacturer_odometer/age_base")).alias("type_manufacturer_odometer/age_diff"),
        )
        self.test = self.test.with_columns(
            (pl.col("odometer/age") - pl.col("type_manufacturer_odometer/age_base")).alias("type_manufacturer_odometer/age_diff"),
        )
    
    def __agg_encoding(self) -> None:
        def agg_encode(group_col_name, target_col_name) -> None:
            agg_df = self.train.groupby(group_col_name).agg(
                pl.mean(target_col_name).alias(f"{group_col_name}_{target_col_name}_mean"),
                pl.std(target_col_name).alias(f"{group_col_name}_{target_col_name}_std"),
                pl.max(target_col_name).alias(f"{group_col_name}_{target_col_name}_max"),
                pl.min(target_col_name).alias(f"{group_col_name}_{target_col_name}_min"),
                (pl.max(target_col_name) - pl.min(target_col_name)).alias(f"{group_col_name}_{target_col_name}_diff"),
            )
            self.train = self.train.join(agg_df, on=group_col_name, how="left")
            self.test = self.test.join(agg_df, on=group_col_name, how="left")
        
        for c in ['manufacturer', 'condition', 'cylinders', 'fuel', 'drive', 'type']:
            agg_encode(c, "age")
            agg_encode(c, "odometer")

        self.train = self.train.with_columns(
            (pl.mean("age") - pl.col("type_age_mean")).alias("type_age_diff"),
            (pl.mean("odometer") - pl.col("type_odometer_mean")).alias("type_odometer_diff"),
            (pl.mean("age") - pl.col("manufacturer_age_mean")).alias("manufacturer_age_diff"),
            (pl.mean("odometer") - pl.col("manufacturer_odometer_mean")).alias("manufacturer_odometer_diff"),
        )
        self.test = self.test.with_columns(
            (pl.mean("age") - pl.col("type_age_mean")).alias("type_age_diff"),
            (pl.mean("odometer") - pl.col("type_odometer_mean")).alias("type_odometer_diff"),
            (pl.mean("age") - pl.col("manufacturer_age_mean")).alias("manufacturer_age_diff"),
            (pl.mean("odometer") - pl.col("manufacturer_odometer_mean")).alias("manufacturer_odometer_diff"),
        )
    
    def __car_string_encoding(self) -> None:
        self.train = self.train.with_columns(
            (pl.col("manufacturer") + " " + pl.col("type") + " " + pl.col("transmission") + " " + pl.col("drive") + " " + pl.col("size") + " " + pl.col("paint_color")).alias("car_string")
        )
        self.test = self.test.with_columns(
            (pl.col("manufacturer") + " " + pl.col("type") + " " + pl.col("transmission") + " " + pl.col("drive") + " " + pl.col("size") + " " + pl.col("paint_color")).alias("car_string")
        )

        model_path = os.path.join(os.path.dirname(__file__), "../output/model/cc.en.300.bin")
        if not os.path.exists(model_path):
            fasttext.util.download_model("en", if_exists="ignore")
            os.rename("cc.en.300.bin", model_path)
        ft_model = fasttext.load_model(model_path)
        ft_model = fasttext.util.reduce_model(ft_model, 50)

        self.train = self.train.with_columns(
            pl.col("car_string").apply(lambda x: ft_model.get_sentence_vector(x)).alias("car_string_vec")
        )
        self.test = self.test.with_columns(
            pl.col("car_string").apply(lambda x: ft_model.get_sentence_vector(x)).alias("car_string_vec")
        )

        self.train = self.train.with_columns(
            pl.col("car_string_vec").apply(lambda x: np.linalg.norm(x, ord=2)).alias("car_string_vec_norm2"),
            pl.col("car_string_vec").apply(lambda x: np.mean(x)).alias("car_string_vec_mean"),
        )
        self.test = self.test.with_columns(
            pl.col("car_string_vec").apply(lambda x: np.linalg.norm(x, ord=2)).alias("car_string_vec_norm2"),
            pl.col("car_string_vec").apply(lambda x: np.mean(x)).alias("car_string_vec_mean"),
        )

        for i in range(ft_model.get_dimension()):
            self.train = self.train.with_columns(
                pl.col('car_string_vec').apply(lambda x: x[i]).alias(f'car_string_vec_{i}')
            )
            self.test = self.test.with_columns(
                pl.col('car_string_vec').apply(lambda x: x[i]).alias(f'car_string_vec_{i}')
            )
    
    def __one_hot_encoding(self) -> None:
        self.train = self.train.drop("type_manufacturer_price_ratio")
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


class FeatureSelecter:

    train: pl.DataFrame
    test: pl.DataFrame

    def __init__(self, train: pl.DataFrame, test: pl.DataFrame) -> None:
        self.train = train
        self.test = test
        self.__common_func()
        self.__drop_category_columns()
    
    def __common_func(self) -> None:
        common_drop_columns = [
            "id",
            "region",
            "car_string",
            "car_string_vec"
        ]
        self.train = self.train.drop(common_drop_columns)
        self.test = self.test.drop(common_drop_columns)
    
    def __drop_category_columns(self) -> None:
        category_columns = [
            "manufacturer",
            "condition",
            "size",
            "fuel",
            "title_status",
            "transmission",
            "drive",
            "type",
            "paint_color",
            "state"
        ]
        self.train = self.train.drop(category_columns)
        self.test = self.test.drop(category_columns)
    
    def show_columns(self) -> None:
        print(self.test.columns)
    
    def get_dataframe(self, type: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
        #数値特徴量だけを抽出
        #train = self.train.select(pl.col(pl.Float32) | pl.col(pl.Int8) | pl.col(pl.Int16) | pl.col(pl.Int32) | pl.col(pl.Int64))
        #test = self.test.select(pl.col(pl.Float32) | pl.col(pl.Int8) | pl.col(pl.Int16) | pl.col(pl.Int32) | pl.col(pl.Int64))
        return self.train, self.test
