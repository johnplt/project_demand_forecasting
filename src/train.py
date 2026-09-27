from typing import List, Tuple, Optional
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import shap


def time_based_train_test_split(
    df: pd.DataFrame,
    cutoff_date: Optional[str] = None,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare les données chronologiquement selon une date ou un ratio."""
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    if cutoff_date is None:
        # Calcul automatique : on prend les derniers X % des dates uniques
        unique_dates = df["Date"].drop_duplicates().sort_values()
        split_idx = int(len(unique_dates) * (1 - test_ratio))
        cutoff_date = unique_dates.iloc[split_idx]

    train = df[df["Date"] < cutoff_date].copy()
    test = df[df["Date"] >= cutoff_date].copy()

    return train, test


def train_lightgbm_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "Sales",
):
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]

    model = lgb.LGBMRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # Calcul des valeurs SHAP sur un échantillon du jeu de test (pour la rapidité)
    explainer = shap.TreeExplainer(model)
    sample_size = min(1000, len(X_test))
    X_test_sample = X_test.sample(sample_size, random_state=42)
    shap_values = explainer(X_test_sample)

    return (
        model,
        pd.Series(preds, index=test_df.index),
        shap_values,
        X_test_sample,
    )