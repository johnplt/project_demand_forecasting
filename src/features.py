from typing import List
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait les composantes temporelles et calendaires."""
    df = df.copy()

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

    # Indicateur de début, milieu et fin de mois (effet salaire / consommation)
    df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(int)

    return df


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    target_col: str = "Sales",
    group_col: str = "Store",
    lags: List[int] = [7, 14, 21, 28],
    windows: List[int] = [7, 14, 30],
) -> pd.DataFrame:
    df = df.sort_values(by=[group_col, "Date"]).copy()

    # 1. Lags simples
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df.groupby(group_col)[
            target_col
        ].shift(lag)

    # 2. Statistiques glissantes
    for w in windows:
        df[f"{target_col}_rolling_mean_{w}"] = df.groupby(group_col)[
            target_col
        ].transform(lambda x: x.shift(1).rolling(window=w).mean())

        df[f"{target_col}_rolling_std_{w}"] = df.groupby(group_col)[
            target_col
        ].transform(lambda x: x.shift(1).rolling(window=w).std())

    # 3. On supprime les lignes où le plus grand lag (ex: lag 28) est NaN
    max_lag = max(lags)
    df = df.dropna(subset=[f"{target_col}_lag_{max_lag}"]).reset_index(
        drop=True
    )

    return df


def prepare_full_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline complet de feature engineering."""
    df = add_calendar_features(df)
    df = add_lag_and_rolling_features(df)

    # Encodage des colonnes catégorielles simples (StoreType, Assortment)
    cat_cols = ["StoreType", "Assortment"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    return df