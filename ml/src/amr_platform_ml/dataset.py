from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class DatasetBundle:
    dataframe: pd.DataFrame
    feature_columns: list[str]
    target_column: str
    preprocessing_pipeline: ColumnTransformer


REQUIRED_COLUMNS = {'species'}


def build_preprocessing_pipeline(dataframe: pd.DataFrame, feature_columns: list[str]) -> ColumnTransformer:
    numeric_features = [column for column in feature_columns if pd.api.types.is_numeric_dtype(dataframe[column])]
    categorical_features = [column for column in feature_columns if column not in numeric_features]
    return ColumnTransformer(
        transformers=[
            (
                'numeric',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                ]),
                numeric_features,
            ),
            (
                'categorical',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore')),
                ]),
                categorical_features,
            ),
        ]
    )


def load_dataset(csv_path: str | Path, target_column: str) -> DatasetBundle:
    dataframe = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing:
        raise ValueError(f'Missing required dataset columns: {sorted(missing)}')
    if target_column not in dataframe.columns:
        raise ValueError(f'Target column {target_column} is not present in the dataset')

    feature_columns = [column for column in dataframe.columns if column != target_column]
    preprocessing_pipeline = build_preprocessing_pipeline(dataframe, feature_columns)
    return DatasetBundle(
        dataframe=dataframe,
        feature_columns=feature_columns,
        target_column=target_column,
        preprocessing_pipeline=preprocessing_pipeline,
    )
