import pandas as pd


NONE_COLUMNS = [
    "Alley",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
]


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove clear outliers from the training dataset.

    This function is used only for experiments.
    It should not be applied to the test dataset because test does not contain SalePrice.
    """
    df = df.copy()

    outlier_mask = (
        (df["GrLivArea"] > 4000)
        & (df["SalePrice"] < 300000)
    )

    return df.loc[~outlier_mask].copy()


def fill_domain_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values where NaN means absence of an object.

    For example:
    - missing GarageType means no garage
    - missing BsmtQual means no basement
    - missing FireplaceQu means no fireplace
    """
    df = df.copy()

    for col in NONE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    return df


def add_house_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-inspired features for house price prediction.
    """
    df = df.copy()

    df["TotalSF"] = (
        df["TotalBsmtSF"]
        + df["1stFlrSF"]
        + df["2ndFlrSF"]
    )

    df["TotalBath"] = (
        df["FullBath"]
        + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"]
        + 0.5 * df["BsmtHalfBath"]
    )

    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)

    return df


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps except outlier removal.
    """
    df = df.copy()
    df = fill_domain_missing_values(df)
    df = add_house_features(df)

    return df